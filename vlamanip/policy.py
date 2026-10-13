"""The learned policy: camera maps + instruction -> pick and place gripper poses.

Only the top-down colour/height maps (built from the overhead RGB-D image and the known camera calibration) and
the instruction text go in. No simulator state is read here. Heights for the gripper come from the height map
at the predicted cell, through the same functions the expert uses.
"""

from __future__ import annotations

import torch

from vlamanip.expert.expert import tcp_height_pick, tcp_height_place
from vlamanip.geometry import WorkspaceGrid
from vlamanip.models.losses import argmax_cells
from vlamanip.models.policy_net import PickPlaceNet


def prepare_maps(color: torch.Tensor, height: torch.Tensor, height_scale: float) -> torch.Tensor:
    """Network input (B, 4, H, W): colour in [-1, 1] (from uint8 or [0, 1]) and scaled height."""
    color = color.float()
    if color.max() > 1.5:
        color = color / 255.0
    if height.dim() == 3:
        height = height[:, None]
    return torch.cat((color * 2 - 1, height.float() / height_scale), dim=1)


def build_net(model_cfg, clip, num_rotations: int, map_size: int) -> PickPlaceNet:
    return PickPlaceNet(
        clip_image_dim=clip.image_dim, clip_text_dim=clip.text_dim, clip_embed_dim=clip.embed_dim,
        base_channels=model_cfg.base_channels, dim=model_cfg.dim, num_rotations=num_rotations,
        fusion_depth=model_cfg.fusion_depth, map_size=map_size,
    )


def local_height(height: torch.Tensor, row: int, col: int, radius: int = 1) -> float:
    """Median height (H, W) in a (2r+1)^2 window, robust to a stray edge cell."""
    h, w = height.shape
    win = height[max(0, row - radius): min(h, row + radius + 1), max(0, col - radius): min(w, col + radius + 1)]
    return float(win.float().median())


class Policy:
    def __init__(self, net: PickPlaceNet, clip, cfg, device: str = "cuda:0"):
        self.net, self.clip, self.cfg = net.to(device).eval(), clip.to(device).eval(), cfg
        self.grid = WorkspaceGrid.from_config(cfg.workspace)
        self.device = device

    @classmethod
    def load(cls, checkpoint: str, cfg, device: str = "cuda:0") -> "Policy":
        from vlamanip.models.clip_encoder import FrozenCLIP

        state = torch.load(checkpoint, map_location=device, weights_only=True)
        clip = FrozenCLIP(cfg.model.clip_name, cfg.model.clip_image_size)
        grid = WorkspaceGrid.from_config(cfg.workspace)
        net = build_net(cfg.model, clip, grid.num_rotations, grid.height)
        net.load_state_dict(state["net"])
        return cls(net, clip, cfg, device)

    @torch.no_grad()
    def logits(self, color: torch.Tensor, height: torch.Tensor, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        maps = prepare_maps(color.to(self.device), height.to(self.device), self.cfg.model.height_scale)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            clip_img = self.clip.encode_image((maps[:, :3] + 1) / 2)
            words, mask, sentence = self.clip.encode_text(texts)
            return self.net(maps, clip_img, words, mask, sentence)

    @torch.no_grad()
    def predict(self, color: torch.Tensor, height: torch.Tensor, texts: list[str]) -> list[dict]:
        """Batch of maps (B, 3, H, W) / (B, 1, H, W) and instructions -> per sample pick and place actions.

        Each action: x, y, yaw, support_height, z_tcp (metric, env-local) and the chosen map cell.
        """
        pick_logits, place_logits = self.logits(color, height, texts)
        pick_cells, place_cells = argmax_cells(pick_logits), argmax_cells(place_logits)
        h = height[:, 0] if height.dim() == 4 else height
        out = []
        c, s = self.cfg.control, self.cfg.scene
        for i in range(len(texts)):
            acts = {}
            for key, cells in (("pick", pick_cells), ("place", place_cells)):
                rot, row, col = (int(v) for v in cells[i])
                x, y = self.grid.pixel_to_world(torch.tensor([row, col])).tolist()
                yaw = float(self.grid.bin_to_yaw(torch.tensor(rot)))
                support = local_height(h[i], row, col)
                z = tcp_height_pick(support, c) if key == "pick" else tcp_height_place(support, s, c)
                acts[key] = {"x": x, "y": y, "yaw": yaw, "support_height": support, "z_tcp": z, "cell": [rot, row, col]}
            out.append(acts)
        return out
