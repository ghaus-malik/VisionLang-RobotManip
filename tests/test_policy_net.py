import torch

from vlamanip.models.losses import soft_cross_entropy, soft_targets
from vlamanip.models.policy_net import PickPlaceNet


def test_forward_shapes_and_gradients():
    torch.manual_seed(0)
    net = PickPlaceNet(clip_image_dim=48, clip_text_dim=32, clip_embed_dim=24, base_channels=16, dim=64,
                       num_rotations=8, fusion_depth=1, map_size=32)
    b = 2
    maps = torch.randn(b, 4, 32, 32)
    clip_img = torch.randn(b, 48, 14, 14)
    words = torch.randn(b, 10, 32)
    mask = torch.ones(b, 10, dtype=torch.bool)
    mask[1, 6:] = False
    sentence = torch.randn(b, 24)
    pick, place = net(maps, clip_img, words, mask, sentence)
    assert pick.shape == (b, 8, 32, 32) and place.shape == (b, 8, 32, 32)
    cells = torch.tensor([[1, 5, 6], [7, 20, 3]])
    loss = soft_cross_entropy(pick, soft_targets(cells, 8, 32, 32, 1.0)) + soft_cross_entropy(place, soft_targets(cells, 8, 32, 32, 1.0))
    loss.backward()
    assert all(p.grad is not None for n, p in net.named_parameters() if "film" not in n)


def test_padded_words_are_ignored():
    torch.manual_seed(0)
    net = PickPlaceNet(clip_image_dim=48, clip_text_dim=32, clip_embed_dim=24, base_channels=16, dim=64,
                       fusion_depth=1, map_size=32).eval()
    maps, clip_img, sentence = torch.randn(1, 4, 32, 32), torch.randn(1, 48, 14, 14), torch.randn(1, 24)
    words = torch.randn(1, 10, 32)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    a = net(maps, clip_img, words, mask, sentence)[0]
    words2 = words.clone()
    words2[0, 4:] = torch.randn(6, 32) * 10
    b = net(maps, clip_img, words2, mask, sentence)[0]
    assert torch.allclose(a, b, atol=1e-5)
