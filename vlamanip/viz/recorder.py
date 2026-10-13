"""Records MP4s of selected environments while the executor runs (overhead + side view with overlays)."""

from __future__ import annotations

import os

from vlamanip.control.executor import PHASE_NAMES
from vlamanip.viz.video import PICK_COLOR, PLACE_COLOR, VideoWriter, compose_frame, make_marker

FPS = 25


def pose_markers(pick: dict, place: dict, calib: dict, pick_label: str = "pick", place_label: str = "place") -> list[dict]:
    """Markers for a pick and a place pose (dicts with x, y, yaw, support_height) in one camera's image."""
    return [
        make_marker(pick["x"], pick["y"], pick["support_height"], pick["yaw"], calib, PICK_COLOR, pick_label),
        make_marker(place["x"], place["y"], place["support_height"], place["yaw"], calib, PLACE_COLOR, place_label),
    ]


class EpisodeRecorder:
    """One video per selected environment; call :meth:`record` once per frame and :meth:`finish` at the end."""

    def __init__(self, scene, executor, env_ids: list[int], paths: list[str], texts: list[str], actions: list[dict]):
        self.scene, self.ex = scene, executor
        self.env_ids, self.paths, self.texts = list(env_ids), list(paths), list(texts)
        calib = scene.video_top_calibration()
        self.markers = []
        for e, a in zip(self.env_ids, actions):
            self.markers.append(pose_markers(a["pick"], a["place"], {k: v[e] for k, v in calib.items()}))
        for p in self.paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        self.writers = [VideoWriter(p, fps=FPS) for p in self.paths]

    def record(self, banners: list[bool] | None = None) -> None:
        top, side = self.scene.capture_video_frames()
        top, side = top.cpu().numpy(), side.cpu().numpy()
        for j, (e, w) in enumerate(zip(self.env_ids, self.writers)):
            status = f"t = {self.ex.elapsed_s[e].item():4.1f} s   phase: {PHASE_NAMES[int(self.ex.phase[e])]}"
            banner = None if banners is None else banners[j]
            w.add(compose_frame(top[e], side[e], self.texts[j], status, self.markers[j], banner))

    def finish(self, successes: list[bool], banner_s: float = 1.5) -> list[str]:
        """Show the result banner, close the files and append ``_success`` / ``_failure`` to their names."""
        for _ in range(int(banner_s * FPS)):
            self.record(successes)
        final = []
        for w, p, ok in zip(self.writers, self.paths, successes):
            w.close()
            dst = p.replace(".mp4", "_success.mp4" if ok else "_failure.mp4")
            os.replace(p, dst)
            final.append(dst)
        return final
