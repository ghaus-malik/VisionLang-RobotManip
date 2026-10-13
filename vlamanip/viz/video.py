"""Video frames: overhead + side view, instruction text, predicted pick/place markers and a success/fail banner."""

from __future__ import annotations

import math

import numpy as np
import torch

from vlamanip.geometry import quat_to_matrix

PICK_COLOR = (40, 220, 40)  # RGB
PLACE_COLOR = (255, 140, 0)
SUCCESS_COLOR = (30, 170, 60)
FAILURE_COLOR = (200, 40, 40)
HEADER_H = 64
FOOTER_H = 34


def world_to_image(points: torch.Tensor, intrinsics: torch.Tensor, cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor) -> torch.Tensor:
    """(k, 3) world points -> (k, 2) pixel (u, v) for a pinhole camera (ROS optical convention)."""
    rot = quat_to_matrix(cam_quat_ros.float())  # camera -> world
    p_cam = (points.float() - cam_pos.float()) @ rot  # world -> camera
    uvw = p_cam @ intrinsics.float().T
    return uvw[:, :2] / uvw[:, 2:3]


def make_marker(x: float, y: float, z: float, yaw: float | None, calib: dict, color, label: str) -> dict:
    """Marker for a gripper pose: centre and closing direction projected into a calibrated camera image.

    ``calib`` holds single-camera ``intrinsics`` (3, 3), ``cam_pos`` (3,) and ``cam_quat_ros`` (4,).
    The Franka fingers close along the hand y axis, which points along world (sin yaw, -cos yaw) for a top-down grasp.
    """
    pts = torch.tensor([[x, y, z], [x + 0.03 * math.sin(yaw or 0.0), y - 0.03 * math.cos(yaw or 0.0), z]])
    uv = world_to_image(pts, calib["intrinsics"].cpu(), calib["cam_pos"].cpu(), calib["cam_quat_ros"].cpu())
    yaw_px = None if yaw is None else math.atan2(float(uv[1, 1] - uv[0, 1]), float(uv[1, 0] - uv[0, 0]))
    return {"uv": uv[0].tolist(), "yaw_px": yaw_px, "color": color, "label": label}


def _text(img, text, org, scale, color, thickness=1):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_marker(img: np.ndarray, uv, yaw_px: float | None, color, label: str, size: int = 14) -> None:
    """Circle + crosshair at ``uv``; a bar shows the gripper closing direction when ``yaw_px`` is given."""
    import cv2

    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    cv2.circle(img, (u, v), size, color, 2, cv2.LINE_AA)
    cv2.line(img, (u - 4, v), (u + 4, v), color, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - 4), (u, v + 4), color, 2, cv2.LINE_AA)
    if yaw_px is not None:
        dx, dy = math.cos(yaw_px) * size * 1.6, math.sin(yaw_px) * size * 1.6
        cv2.line(img, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)), color, 2, cv2.LINE_AA)
    _text(img, label, (u + size + 3, v - size), 0.5, color, 1)


def compose_frame(
    top: np.ndarray,
    side: np.ndarray,
    instruction: str,
    status: str = "",
    markers: list[dict] | None = None,
    banner: bool | None = None,
) -> np.ndarray:
    """One video frame.

    Args:
        top, side: (S, S, 3) uint8 images of the overhead and side cameras.
        markers: dicts with ``uv`` (pixel in the top image), ``yaw_px`` (closing direction in image radians or None),
            ``color`` and ``label``.
        banner: None while running, True/False to show SUCCESS/FAILURE across the frame.
    """
    import cv2

    top = np.ascontiguousarray(top[..., :3]).copy()
    for m in markers or []:
        draw_marker(top, m["uv"], m.get("yaw_px"), m["color"], m["label"])
    s = top.shape[0]
    frame = np.full((HEADER_H + s + FOOTER_H, 2 * s, 3), 245, dtype=np.uint8)
    frame[HEADER_H:HEADER_H + s, :s] = top
    frame[HEADER_H:HEADER_H + s, s:] = np.ascontiguousarray(side[..., :3])
    frame[:HEADER_H] = (32, 36, 44)
    cv2.putText(frame, "Instruction:", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 180, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, f'"{instruction}"', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "overhead camera", (8, HEADER_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "side camera", (s + 8, HEADER_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    frame[-FOOTER_H:] = (32, 36, 44)
    cv2.putText(frame, status, (12, frame.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    if banner is not None:
        color = SUCCESS_COLOR if banner else FAILURE_COLOR
        text = "SUCCESS" if banner else "FAILURE"
        y0 = HEADER_H + s - 60  # strip along the bottom of both views, the scene stays visible
        overlay = frame.copy()
        overlay[y0:y0 + 60] = color
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cv2.putText(frame, text, ((frame.shape[1] - tw) // 2, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


class VideoWriter:
    """MP4 writer (H.264 via imageio-ffmpeg)."""

    def __init__(self, path: str, fps: int = 25):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.frames = 0

    def add(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
