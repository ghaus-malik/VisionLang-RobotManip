"""Procedural USD assets: open-top bins (floor + four walls) as one rigid body with box colliders.

The files are written once into a cache folder and spawned with ``UsdFileCfg``. The bin is a single rigid body
(made kinematic when spawned) so that it can be moved to a new position at every reset.
"""

from __future__ import annotations

import os


def bin_usd_path(scene_cfg, cache_dir: str) -> str:
    """Write (if needed) and return the path of the bin USD for the configured dimensions."""
    name = "bin_{:.3f}_{:.3f}_{:.3f}_{:.3f}.usda".format(
        scene_cfg.bin_inner, scene_cfg.bin_wall_thickness, scene_cfg.bin_wall_height, scene_cfg.bin_floor_thickness
    )
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        _write_bin_usd(path, scene_cfg)
    return path


def bin_boxes(scene_cfg) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, full size) of the floor and wall boxes, in the bin frame (origin at the floor's bottom centre)."""
    inner, t = scene_cfg.bin_inner, scene_cfg.bin_wall_thickness
    h, f = scene_cfg.bin_wall_height, scene_cfg.bin_floor_thickness
    outer = inner + 2 * t
    off = inner / 2 + t / 2
    return [
        ("floor", (0.0, 0.0, f / 2), (outer, outer, f)),
        ("wall_px", (off, 0.0, h / 2), (t, outer, h)),
        ("wall_nx", (-off, 0.0, h / 2), (t, outer, h)),
        ("wall_py", (0.0, off, h / 2), (inner, t, h)),
        ("wall_ny", (0.0, -off, h / 2), (inner, t, h)),
    ]


def _write_bin_usd(path: str, scene_cfg) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # available inside Isaac Sim

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Bin")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    for name, center, size in bin_boxes(scene_cfg):
        cube = UsdGeom.Cube.Define(stage, f"/Bin/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*center))
        xform.SetScale(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
