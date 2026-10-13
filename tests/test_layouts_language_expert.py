import math
import random

import numpy as np
import pytest

from vlamanip.evaluation.success import cube_in_bin, cube_on_cube, point_in_bin, point_on_cube
from vlamanip.expert.expert import expert_actions, tcp_height_pick, tcp_height_place
from vlamanip.language.templates import (
    PICK_PLACE,
    STACK,
    TEMPLATES,
    make_instruction,
    sample_task,
)
from vlamanip.sim.layouts import bin_half_extent, cube_half_extent, fixed_layout, sample_layout, square_gap

COLORS = ["red", "blue", "green", "yellow"]


@pytest.mark.parametrize("seed", range(50))
def test_random_layout_respects_gaps(cfg, seed):
    layout = sample_layout(np.random.default_rng(seed), cfg.scene, cfg.layout, COLORS, COLORS)
    objs = [(x, y, bin_half_extent(cfg.scene), cfg.layout.gap_bin) for x, y in layout.bins.values()]
    objs += [(x, y, cube_half_extent(cfg.scene), cfg.layout.gap_cube) for x, y, _ in layout.cubes.values()]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            assert square_gap(objs[i][:3], objs[j][:3]) >= max(objs[i][3], objs[j][3]) - 1e-9
    for x, y, yaw in layout.cubes.values():
        assert cfg.layout.region_x[0] <= x <= cfg.layout.region_x[1]
        assert 0 <= yaw < math.pi / 2


def test_layout_is_deterministic(cfg):
    a = sample_layout(np.random.default_rng(3), cfg.scene, cfg.layout, COLORS, COLORS)
    b = sample_layout(np.random.default_rng(3), cfg.scene, cfg.layout, COLORS, COLORS)
    assert a.to_dict() == b.to_dict()


def test_fixed_layouts():
    assert set(fixed_layout(1).cubes) == {"blue"} and set(fixed_layout(1).bins) == {"red"}
    assert len(fixed_layout(2).cubes) == 4


def test_heldout_phrasings_are_disjoint():
    for task in (PICK_PLACE, STACK):
        train = set(TEMPLATES[(task, "train")])
        assert not train & set(TEMPLATES[(task, "heldout")])
        assert not train & set(TEMPLATES[(task, "reordered")])


def test_reordered_phrasings_name_destination_first():
    for task in (PICK_PLACE, STACK):
        for t in TEMPLATES[(task, "reordered")]:
            assert t.index("{dst}") < t.index("{obj}")


def test_example_instructions_are_in_training_templates():
    texts = {t.format(obj="blue", dst="red") for t in TEMPLATES[(PICK_PLACE, "train")]}
    assert "Pick up the blue cube and place it in the red bin." in texts
    texts = {t.format(obj="green", dst="yellow") for t in TEMPLATES[(STACK, "train")]}
    assert "Stack the green cube on the yellow cube." in texts


def test_stack_never_targets_itself():
    rng = random.Random(0)
    for _ in range(200):
        spec = sample_task(COLORS, rng, task=STACK)
        assert spec.obj != spec.dst
    with pytest.raises(ValueError):
        make_instruction(STACK, "red", "red", rng)


def test_expert_pick_place(cfg):
    spec = make_instruction(PICK_PLACE, "blue", "red", random.Random(0))
    cubes = {"blue": (0.45, 0.12, 0.0225, 1.9)}
    pick, place = expert_actions(spec, cubes, {"red": (0.55, -0.12)}, cfg.scene, cfg.control, np.random.default_rng(0))
    assert (pick.x, pick.y) == (0.45, 0.12)
    assert pick.yaw == pytest.approx(1.9 - math.pi / 2)
    assert pick.z_tcp == pytest.approx(0.045 - cfg.control.grasp_depth)
    assert point_in_bin((place.x, place.y), (0.55, -0.12), cfg.scene)
    assert place.yaw == 0.0


def test_expert_stack(cfg):
    spec = make_instruction(STACK, "blue", "red", random.Random(0))
    cubes = {"blue": (0.45, 0.12, 0.0225, 0.2), "red": (0.5, -0.1, 0.0225, 0.5)}
    pick, place = expert_actions(spec, cubes, {}, cfg.scene, cfg.control, np.random.default_rng(0))
    assert (place.x, place.y, place.yaw) == (0.5, -0.1, 0.5)
    # held cube bottom ends place_clearance above the red cube's top face
    bottom = place.z_tcp - (cfg.scene.cube_size - cfg.control.grasp_depth)
    assert bottom == pytest.approx(0.045 + cfg.control.place_clearance)


def test_tcp_height_helpers(cfg):
    assert tcp_height_pick(0.045, cfg.control) == pytest.approx(0.045 - cfg.control.grasp_depth)
    assert tcp_height_place(0.0, cfg.scene, cfg.control) > cfg.scene.cube_size / 2


def test_success_checks(cfg):
    s, sc = cfg.scene, cfg.success
    assert cube_in_bin((0.55, -0.10, 0.03), (0.55, -0.12), s, sc)
    assert not cube_in_bin((0.55, -0.05, 0.03), (0.55, -0.12), s, sc)
    assert not cube_in_bin((0.55, -0.12, 0.08), (0.55, -0.12), s, sc)  # resting on something else
    assert cube_on_cube((0.5, 0.01, 0.0675), (0.5, 0.0, 0.0225), s, sc)
    assert not cube_on_cube((0.5, 0.03, 0.0675), (0.5, 0.0, 0.0225), s, sc)
    assert not cube_on_cube((0.5, 0.0, 0.0225), (0.6, 0.0, 0.0225), s, sc)
    assert point_on_cube((0.51, 0.0), (0.5, 0.0), s) and not point_on_cube((0.53, 0.0), (0.5, 0.0), s)
