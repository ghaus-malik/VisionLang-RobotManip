"""Motion execution: inverse kinematics on the Franka hand and a pick-and-place state machine.

The learned policy (or the scripted expert) only outputs two top-down gripper poses, a pick ``(x, y, yaw, z_tcp)``
and a place ``(x, y, yaw, z_tcp)``. This module turns them into joint targets:

    above pick (transit height) -> descend -> close -> lift (grasp check) -> above place (transit height)
    -> descend -> open -> retreat up -> back to the ready pose (joint space)

Cartesian waypoints are interpolated at a bounded speed and tracked with damped-least-squares IK on ``panda_hand``
(the TCP is ``tcp_offset`` along the hand z axis). Everything is batched: each environment runs its own state
machine. Must be imported after the Isaac Sim app has been launched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from vlamanip.control.waypoints import angle_error, gripper_yaw, step_angle_toward, step_toward
from vlamanip.geometry import quat_to_matrix, top_down_grasp_quat, yaw_from_quat

# phases of the pick-and-place state machine
ABOVE_PICK, DESCEND_PICK, CLOSE, LIFT, ABOVE_PLACE, DESCEND_PLACE, OPEN, RETREAT, HOME, DONE = range(10)
PHASE_NAMES = ["above_pick", "descend_pick", "close", "lift", "above_place", "descend_place", "open", "retreat", "home", "done"]
CARTESIAN_PHASES = (ABOVE_PICK, DESCEND_PICK, CLOSE, LIFT, ABOVE_PLACE, DESCEND_PLACE, OPEN, RETREAT)


class ArmController:
    """Damped-least-squares IK from a top-down TCP pose (env-local) to arm joint targets, plus gripper targets."""

    def __init__(self, scene, cfg):
        self.scene, self.c = scene, cfg.control
        robot = scene.robot
        self.robot = robot
        self.arm_ids = robot.find_joints("panda_joint[1-7]")[0]
        self.finger_ids = robot.find_joints("panda_finger_joint.*")[0]
        self.hand_id = robot.find_bodies("panda_hand")[0][0]
        self.jacobian_index = self.hand_id - 1  # fixed base: the root body has no Jacobian entry
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=scene.num_envs,
            device=scene.device,
        )
        self.ready = torch.tensor(self.c.ready_joint_pos, device=scene.device)

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Measured TCP position (N, 3, env-local) and hand yaw (N,)."""
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        tcp = hand_pos + quat_to_matrix(hand_quat)[..., :, 2] * self.c.tcp_offset
        return tcp - self.scene.origins, yaw_from_quat(hand_quat)

    def finger_width(self) -> torch.Tensor:
        """Gap between the fingers (sum of both finger joints), (N,)."""
        return self.robot.data.joint_pos[:, self.finger_ids].sum(-1)

    def arm_joint_targets(self, tcp_pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Joint targets (N, 7) that move the TCP toward ``tcp_pos`` (env-local) with the hand pointing down at ``yaw``."""
        robot = self.robot
        quat_w = top_down_grasp_quat(yaw)
        hand_pos_w = tcp_pos + self.scene.origins - quat_to_matrix(quat_w)[..., :, 2] * self.c.tcp_offset
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos, root_quat, hand_pos_w, quat_w)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos, root_quat, robot.data.body_pos_w[:, self.hand_id], robot.data.body_quat_w[:, self.hand_id]
        )
        self.ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
        jacobian = robot.root_physx_view.get_jacobians()[:, self.jacobian_index, :, self.arm_ids]
        return self.ik.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, self.arm_ids])

    def apply(self, arm_targets: torch.Tensor, finger_targets: torch.Tensor) -> None:
        self.robot.set_joint_position_target(arm_targets, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(finger_targets[:, None].expand(-1, len(self.finger_ids)), joint_ids=self.finger_ids)


class PickPlaceExecutor:
    """Batched pick-and-place state machine; one primitive per environment at a time.

    Usage::

        ex = PickPlaceExecutor(scene, cfg)
        ex.start(env_ids, pick, place)      # (n, 4) tensors of (x, y, yaw, z_tcp), env-local
        while not ex.all_done():
            ex.step()                       # one control step (``decimation`` physics steps)
        ex.grasp_ok, ex.elapsed_s, ex.timeouts
    """

    def __init__(self, scene, cfg):
        self.scene, self.c = scene, cfg.control
        self.arm = ArmController(scene, cfg)
        n, dev = scene.num_envs, scene.device
        self.n, self.device = n, dev
        self.control_dt = scene.physics_dt * self.c.decimation
        self.ws_min = torch.tensor(self.c.workspace_min, device=dev)
        self.ws_max = torch.tensor(self.c.workspace_max, device=dev)
        self.yaw_tol = math.radians(self.c.yaw_tolerance_deg)
        self.timeout_steps = int(round(self.c.waypoint_timeout_s / self.control_dt))
        self.settle_steps = int(round(self.c.gripper_settle_s / self.control_dt))

        self.phase = torch.full((n,), DONE, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.pick = torch.zeros(n, 4, device=dev)
        self.place = torch.zeros(n, 4, device=dev)
        self.cmd_pos = torch.zeros(n, 3, device=dev)  # interpolated Cartesian command
        self.cmd_yaw = torch.zeros(n, device=dev)
        self.fingers = torch.full((n,), self.c.gripper_open, device=dev)
        self.home_joints = self.arm.ready.expand(n, -1).clone()
        self.cmd_joints = self.home_joints.clone()  # interpolated joint command while returning home
        self.grasp_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.timeouts = torch.zeros(n, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------ interface
    def start(self, env_ids: torch.Tensor, pick: torch.Tensor, place: torch.Tensor) -> None:
        """Begin a pick-and-place in ``env_ids``; ``pick``/``place`` are (n, 4) = (x, y, yaw, z_tcp), env-local."""
        pick, place = pick.to(self.device).float().clone(), place.to(self.device).float().clone()
        pick[:, 2], place[:, 2] = gripper_yaw(pick[:, 2]), gripper_yaw(place[:, 2])
        lo, hi = self.ws_min, self.ws_max
        for a in (pick, place):  # never command outside the safe box
            a[:, 0] = a[:, 0].clamp(lo[0], hi[0])
            a[:, 1] = a[:, 1].clamp(lo[1], hi[1])
            a[:, 3] = a[:, 3].clamp(lo[2], hi[2])
        self.pick[env_ids], self.place[env_ids] = pick, place
        tcp, yaw = self.arm.tcp_pose()
        self.cmd_pos[env_ids], self.cmd_yaw[env_ids] = tcp[env_ids], yaw[env_ids]
        self.fingers[env_ids] = self.c.gripper_open
        self.phase[env_ids] = ABOVE_PICK
        self.phase_steps[env_ids] = 0
        self.steps[env_ids] = 0
        self.grasp_ok[env_ids] = False
        self.timeouts[env_ids] = 0

    def all_done(self) -> bool:
        return bool((self.phase == DONE).all())

    @property
    def elapsed_s(self) -> torch.Tensor:
        return self.steps.float() * self.control_dt

    # ------------------------------------------------------------------- stepping
    def _goals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cartesian goal position, yaw and maximum step length of the current phase of every env."""
        c, p, q = self.c, self.pick, self.place
        transit = torch.full_like(p[:, 0], c.transit_height)
        above_pick = torch.stack((p[:, 0], p[:, 1], torch.maximum(transit, p[:, 3] + c.approach_height)), -1)
        at_pick = p[:, [0, 1, 3]]
        above_place = torch.stack((q[:, 0], q[:, 1], torch.maximum(transit, q[:, 3] + c.approach_height)), -1)
        at_place = q[:, [0, 1, 3]]
        ph = self.phase[:, None]
        goal = torch.where(ph <= DESCEND_PICK, torch.where(ph == ABOVE_PICK, above_pick, at_pick), above_pick)
        goal = torch.where(ph == CLOSE, at_pick, goal)
        goal = torch.where(ph >= ABOVE_PLACE, above_place, goal)
        goal = torch.where((ph == DESCEND_PLACE) | (ph == OPEN), at_place, goal)
        yaw = torch.where(self.phase <= LIFT, p[:, 2], q[:, 2])
        # slower in the last centimetres above an object
        fine = (self.phase == DESCEND_PICK) | (self.phase == DESCEND_PLACE)
        step = torch.where(fine, torch.full_like(yaw, c.max_linear_step * 0.5), torch.full_like(yaw, c.max_linear_step))
        return goal, yaw, step

    def step(self, render: bool = False) -> None:
        """One control step for every env (envs in DONE hold the ready pose)."""
        c = self.c
        goal, goal_yaw, max_step = self._goals()
        cart = torch.zeros_like(self.phase, dtype=torch.bool)
        for ph in CARTESIAN_PHASES:
            cart |= self.phase == ph

        # interpolate the Cartesian command toward the goal and track it with IK
        new_cmd = self.cmd_pos + (goal - self.cmd_pos) * torch.clamp(
            max_step[:, None] / torch.linalg.norm(goal - self.cmd_pos, dim=-1, keepdim=True).clamp(min=1e-9), max=1.0
        )
        self.cmd_pos = torch.where(cart[:, None], new_cmd, self.cmd_pos)
        self.cmd_yaw = torch.where(cart, step_angle_toward(self.cmd_yaw, goal_yaw, math.radians(3.0)), self.cmd_yaw)
        arm_targets = self.arm.arm_joint_targets(self.cmd_pos, self.cmd_yaw)

        # joint-space return to the ready pose
        returning = (self.phase == HOME)[:, None]
        self.cmd_joints = torch.where(returning, step_toward(self.cmd_joints, self.home_joints, c.home_joint_step), self.cmd_joints)
        arm_targets = torch.where(cart[:, None], arm_targets, self.cmd_joints)
        arm_targets = torch.where((self.phase == DONE)[:, None], self.home_joints, arm_targets)

        self.fingers = torch.where(self.phase == CLOSE, torch.full_like(self.fingers, c.gripper_closed), self.fingers)
        self.fingers = torch.where((self.phase == OPEN) | (self.phase == DONE), torch.full_like(self.fingers, c.gripper_open), self.fingers)
        self.arm.apply(arm_targets, self.fingers)

        for i in range(c.decimation):
            self.scene.step(render=render and i == c.decimation - 1)

        active = self.phase != DONE
        self.steps += active.long()
        self.phase_steps += active.long()
        self._advance(goal, goal_yaw)

    def _advance(self, goal: torch.Tensor, goal_yaw: torch.Tensor) -> None:
        c = self.c
        tcp, yaw = self.arm.tcp_pose()
        ph = self.phase
        fine = (ph == DESCEND_PICK) | (ph == DESCEND_PLACE)
        tol = torch.where(fine, torch.full_like(yaw, c.fine_position_tolerance), torch.full_like(yaw, c.position_tolerance))
        reached = (torch.linalg.norm(tcp - goal, dim=-1) < tol) & (angle_error(yaw, goal_yaw) < self.yaw_tol)
        reached &= torch.linalg.norm(self.cmd_pos - goal, dim=-1) < 1e-6
        timed_out = self.phase_steps >= self.timeout_steps
        waited = self.phase_steps >= self.settle_steps

        nxt = ph.clone()
        moving = (ph == ABOVE_PICK) | (ph == DESCEND_PICK) | (ph == LIFT) | (ph == ABOVE_PLACE) | (ph == DESCEND_PLACE) | (ph == RETREAT)
        done_moving = moving & (reached | timed_out)
        self.timeouts += (moving & timed_out & ~reached).long()
        nxt = torch.where(done_moving, ph + 1, nxt)
        nxt = torch.where(((ph == CLOSE) | (ph == OPEN)) & waited, ph + 1, nxt)

        # grasp check after lifting: an empty gripper closes (almost) completely
        lifted = (ph == LIFT) & done_moving
        holding = self.arm.finger_width() > c.grasp_min_width
        self.grasp_ok = torch.where(lifted, holding, self.grasp_ok)
        nxt = torch.where(lifted & ~holding, torch.full_like(nxt, HOME), nxt)
        self.fingers = torch.where(lifted & ~holding, torch.full_like(self.fingers, c.gripper_open), self.fingers)

        q = self.arm.robot.data.joint_pos[:, self.arm.arm_ids]
        at_home = (q - self.home_joints).abs().max(-1).values < 0.02
        nxt = torch.where((ph == HOME) & (at_home | timed_out), torch.full_like(nxt, DONE), nxt)
        entering_home = (nxt == HOME) & (ph != HOME)
        self.cmd_joints = torch.where(entering_home[:, None], q, self.cmd_joints)

        self.phase_steps = torch.where(nxt != ph, torch.zeros_like(self.phase_steps), self.phase_steps)
        self.phase = nxt
