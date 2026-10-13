"""Isaac Lab tabletop scene: table, Franka Panda, coloured cubes and bins, overhead RGB-D camera, lights.

Must be imported after the Isaac Sim app has been launched (``AppLauncher``).

All layout coordinates are local to each environment (robot base at the origin, table top at z = 0); the
scene adds the environment origins. Several environments can run in parallel for data generation.
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from vlamanip.geometry import look_at_quat_ros, quat_from_yaw
from vlamanip.sim.assets import bin_usd_path
from vlamanip.sim.layouts import Layout

COLOR_RGB = {
    "red": (0.85, 0.08, 0.08),
    "blue": (0.08, 0.25, 0.90),
    "green": (0.10, 0.70, 0.20),
    "yellow": (0.95, 0.82, 0.08),
}
TABLE_RGB = (0.62, 0.60, 0.57)
# Overhead camera looking straight down (ROS optical frame): image right = world -y, image down = world -x.
OVERHEAD_QUAT_ROS = (0.0, math.sqrt(0.5), -math.sqrt(0.5), 0.0)
DEFAULT_CACHE = os.path.expanduser("~/.cache/vlamanip")


def make_scene_cfg(
    cfg, num_envs: int, cube_colors: list[str], bin_colors: list[str], video: bool = False, cache_dir: str = DEFAULT_CACHE
) -> InteractiveSceneCfg:
    """Scene configuration with one cube per colour in ``cube_colors`` and one bin per colour in ``bin_colors``."""
    s, cam = cfg.scene, cfg.camera
    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=6.0, replicate_physics=False)

    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05))
    )
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/dome_light", spawn=sim_utils.DomeLightCfg(intensity=cam.dome_intensity, color=(0.9, 0.9, 0.9))
    )
    scene_cfg.sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=cam.sun_intensity, angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),  # 45 deg from vertical
    )
    # static table (collider only, no rigid body); its top surface is at z = 0
    sx, sy, sz = s.table_size
    scene_cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_RGB, roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(s.table_center_xy[0], s.table_center_xy[1], -sz / 2)),
    )
    scene_cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    for i, color in enumerate(cube_colors):
        setattr(
            scene_cfg,
            f"cube_{color}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/cube_{color}",
                spawn=sim_utils.CuboidCfg(
                    size=(s.cube_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=s.cube_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=s.cube_friction, dynamic_friction=s.cube_friction, friction_combine_mode="max"
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOR_RGB[color], roughness=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2 + 0.1 * i, s.cube_size / 2)),
            ),
        )
    bin_usd = bin_usd_path(s, cache_dir)
    for i, color in enumerate(bin_colors):
        r, g, b = COLOR_RGB[color]
        setattr(
            scene_cfg,
            f"bin_{color}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/bin_{color}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=bin_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8 * r, 0.8 * g, 0.8 * b), roughness=0.6),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.2 + 0.13 * i, 0.0)),
            ),
        )

    pinhole = sim_utils.PinholeCameraCfg(
        focal_length=cam.focal_length, horizontal_aperture=cam.horizontal_aperture, clipping_range=(0.05, 4.0)
    )
    scene_cfg.overhead_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        update_period=0.0,
        height=cam.height,
        width=cam.width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=pinhole,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(cam.overhead_pos), rot=OVERHEAD_QUAT_ROS, convention="ros"),
    )
    if video:
        scene_cfg.video_top_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/video_top_cam",
            update_period=0.0,
            height=cam.video_size,
            width=cam.video_size,
            data_types=["rgb"],
            spawn=pinhole,
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(cam.overhead_pos), rot=OVERHEAD_QUAT_ROS, convention="ros"),
        )
        quat = look_at_quat_ros(torch.tensor(cam.perspective_eye), torch.tensor(cam.perspective_target))
        scene_cfg.video_side_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/video_side_cam",
            update_period=0.0,
            height=cam.video_size,
            width=cam.video_size,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 6.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(cam.perspective_eye), rot=tuple(quat.tolist()), convention="ros"),
        )
    return scene_cfg


class TabletopScene:
    """Thin wrapper around ``InteractiveScene`` with layout resets, ground-truth poses and camera capture."""

    def __init__(self, cfg, num_envs: int = 1, cube_colors=None, bin_colors=None, video: bool = False, device: str = "cuda:0"):
        self.cfg = cfg
        self.cube_colors = list(cube_colors if cube_colors is not None else cfg.scene.colors)
        self.bin_colors = list(bin_colors if bin_colors is not None else cfg.scene.colors)
        self.video = video
        c = cfg.control
        sim_cfg = sim_utils.SimulationCfg(
            dt=c.sim_dt,
            render_interval=c.decimation,
            device=device,
            physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.2, friction_correlation_distance=0.00625),
            render=sim_utils.RenderCfg(antialiasing_mode=cfg.camera.antialiasing),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(make_scene_cfg(cfg, num_envs, self.cube_colors, self.bin_colors, video))
        self.sim.reset()
        self.scene.reset()  # also fills in the (static) camera poses, which cameras only read on reset
        self.device = self.sim.device
        self.num_envs = num_envs
        self.robot = self.scene["robot"]
        self.cubes = {color: self.scene[f"cube_{color}"] for color in self.cube_colors}
        self.bins = {color: self.scene[f"bin_{color}"] for color in self.bin_colors}
        self.origins = self.scene.env_origins
        self.physics_dt = self.sim.get_physics_dt()
        self.sim_time = torch.zeros(num_envs, device=self.device)

    # ------------------------------------------------------------------ stepping
    def step(self, render: bool = False) -> None:
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.physics_dt)

    def settle(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / self.physics_dt)))):
            self.step()

    # -------------------------------------------------------------------- resets
    def reset(self, layouts: list[Layout], env_ids: torch.Tensor | None = None, ready_joint_pos=None) -> None:
        """Place the objects of ``layouts[i]`` in env ``env_ids[i]`` and move the robot to its ready pose."""
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if len(layouts) != len(env_ids):
            raise ValueError("One layout per reset environment is required.")
        origins = self.origins[env_ids]
        n = len(env_ids)
        half = self.cfg.scene.cube_size / 2

        for color, cube in self.cubes.items():
            pose = torch.zeros(n, 7, device=self.device)
            for i, layout in enumerate(layouts):
                if color not in layout.cubes:
                    raise KeyError(f"Layout {i} has no {color} cube but the scene does.")
                x, y, yaw = layout.cubes[color]
                pose[i, :3] = torch.tensor([x, y, half + 0.001])
                pose[i, 3:] = quat_from_yaw(torch.tensor(float(yaw)))
            pose[:, :3] += origins
            cube.write_root_pose_to_sim(pose, env_ids)
            cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
        for color, bin_obj in self.bins.items():
            pose = torch.zeros(n, 7, device=self.device)
            pose[:, 3] = 1.0
            for i, layout in enumerate(layouts):
                pose[i, :2] = torch.tensor(layout.bins[color])
            pose[:, :3] += origins
            bin_obj.write_root_pose_to_sim(pose, env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        if ready_joint_pos is not None:
            joint_pos[:, :7] = torch.tensor(ready_joint_pos, device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.reset(env_ids)
        self.sim_time[env_ids] = 0.0

    # --------------------------------------------------------------- ground truth
    def cube_poses(self, env_id: int) -> dict[str, tuple[float, float, float, float]]:
        """colour -> (x, y, z, yaw) of each cube in env-local coordinates (ground truth)."""
        from vlamanip.geometry import yaw_from_quat

        out = {}
        for color, cube in self.cubes.items():
            p = cube.data.root_pos_w[env_id] - self.origins[env_id]
            yaw = yaw_from_quat(cube.data.root_quat_w[env_id]).item()
            out[color] = (p[0].item(), p[1].item(), p[2].item(), yaw)
        return out

    def bin_positions(self, env_id: int) -> dict[str, tuple[float, float]]:
        out = {}
        for color, bin_obj in self.bins.items():
            p = bin_obj.data.root_pos_w[env_id] - self.origins[env_id]
            out[color] = (p[0].item(), p[1].item())
        return out

    def cube_speeds(self) -> dict[str, torch.Tensor]:
        return {color: torch.linalg.norm(cube.data.root_lin_vel_w, dim=-1) for color, cube in self.cubes.items()}

    # -------------------------------------------------------------------- camera
    def render(self, frames: int = 2) -> None:
        for _ in range(frames):
            self.sim.render()

    def capture_overhead(self, render_frames: int = 3) -> dict[str, torch.Tensor]:
        """Fresh overhead RGB-D frames with calibration; positions are env-local."""
        self.render(render_frames)
        cam = self.scene["overhead_cam"]
        cam.update(0.0, force_recompute=True)
        if not (torch.isfinite(cam.data.quat_w_ros).all() and cam.data.quat_w_ros.norm(dim=-1).gt(0.5).all()):
            raise RuntimeError("Overhead camera pose is not initialised (was the scene reset?).")
        return {
            "rgb": cam.data.output["rgb"][..., :3].clone(),  # (N, H, W, 3) uint8
            "depth": cam.data.output["distance_to_image_plane"][..., 0].clone(),  # (N, H, W)
            "intrinsics": cam.data.intrinsic_matrices.clone(),
            "cam_pos": (cam.data.pos_w - self.origins).clone(),
            "cam_quat_ros": cam.data.quat_w_ros.clone(),
        }

    def video_top_calibration(self) -> dict[str, torch.Tensor]:
        """Intrinsics and env-local pose of the top video camera (to draw markers on its frames)."""
        cam = self.scene["video_top_cam"]
        return {
            "intrinsics": cam.data.intrinsic_matrices.clone(),
            "cam_pos": (cam.data.pos_w - self.origins).clone(),
            "cam_quat_ros": cam.data.quat_w_ros.clone(),
        }

    def capture_video_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, S, S, 3) uint8 frames of the top and side video cameras (requires ``video=True``)."""
        self.render(1)
        top, side = self.scene["video_top_cam"], self.scene["video_side_cam"]
        top.update(0.0, force_recompute=True)
        side.update(0.0, force_recompute=True)
        return top.data.output["rgb"][..., :3], side.data.output["rgb"][..., :3]
