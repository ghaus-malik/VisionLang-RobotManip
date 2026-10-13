import torch

from vlamanip.data.augment import augment, rot90_cells
from vlamanip.models.losses import argmax_cells, soft_cross_entropy, soft_targets

S, R = 32, 8


def one_hot_map(cells):
    m = torch.zeros(cells.shape[0], 1, S, S)
    for i, (_, r, c) in enumerate(cells.tolist()):
        m[i, 0, r, c] = 1.0
    return m


def where_is(maps):
    idx = maps[:, 0].flatten(1).argmax(1)
    return torch.stack((idx // S, idx % S), 1)


def test_rot90_cells_matches_torch_rot90():
    cells = torch.tensor([[3, 2, 7], [1, 10, 30], [0, 31, 0], [5, 4, 4]])
    for k in range(4):
        kk = torch.full((4,), k)
        rotated = torch.rot90(one_hot_map(cells), k, dims=(-2, -1))
        assert torch.equal(where_is(rotated), rot90_cells(cells, kk, S)[:, 1:])
        assert torch.equal(rot90_cells(cells, kk, S)[:, 0], cells[:, 0])


def test_augment_moves_labels_with_the_map():
    torch.manual_seed(0)
    cells = torch.randint(0, S, (64, 3))
    cells[:, 0] = torch.randint(0, R, (64,))
    maps = one_hot_map(cells)
    out, (lab,) = augment(maps, [cells], R, rot90=True, flip=True, max_shift=3)
    assert torch.equal(where_is(out), lab[:, 1:])
    assert ((lab[:, 0] >= 0) & (lab[:, 0] < R)).all()


def test_flip_negates_yaw_bins():
    cells = torch.tensor([[0, 5, 5], [1, 5, 5], [3, 5, 5], [7, 5, 5]])
    g = torch.Generator().manual_seed(0)
    flipped = None
    for _ in range(20):  # find a draw where everything is flipped
        out, (lab,) = augment(one_hot_map(cells), [cells], R, rot90=False, flip=True, generator=g)
        if (lab[:, 2] == S - 1 - 5).all():
            flipped = lab
            break
    assert flipped is not None
    assert flipped[:, 0].tolist() == [0, 7, 5, 1]


def test_soft_targets_normalised_and_peaked():
    cells = torch.tensor([[2, 10, 20], [7, 0, 0]])
    t = soft_targets(cells, R, S, S, sigma_px=1.0)
    assert torch.allclose(t.sum(dim=(1, 2, 3)), torch.ones(2))
    assert torch.equal(argmax_cells(t), cells)
    # circular in yaw: bin 0 is a neighbour of bin 7
    assert t[1, 0, 0, 0] > t[1, 5, 0, 0]


def test_cross_entropy_lower_for_correct_prediction():
    cells = torch.tensor([[2, 10, 20]])
    t = soft_targets(cells, R, S, S, sigma_px=1.0)
    good = torch.log(t + 1e-9)
    bad = torch.roll(good, 5, dims=-1)
    assert soft_cross_entropy(good, t) < soft_cross_entropy(bad, t)
