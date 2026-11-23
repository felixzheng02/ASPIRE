"""Stretch 4 embodiment against the in-process fake bridge; no robot or ROS."""

import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from cap_harness.agent.mcp import CapMcpBridge
from cap_harness.contracts import JOINT_DIMENSIONS, MotionStrategy, Pose
from cap_harness.stretch4.client import BridgeClient
from cap_harness.stretch4.fake_bridge import FAKE_HOME, FakeBridgeServer
from cap_harness.stretch4.kinematics import JOINT_NAMES, Stretch4Kinematics
from cap_harness.stretch4.runtime import build_stretch4_runtime
from cap_harness.stretch4.station import Stretch4Station

MINK = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")


@pytest.fixture(scope="module")
def kinematics():
    return Stretch4Kinematics()


@pytest.fixture
def fake():
    server = FakeBridgeServer()
    yield server
    server.close()


def _rig(fake, kinematics, allow_motion):
    station = Stretch4Station(BridgeClient(fake.url), kinematics)
    adapter, runtime = build_stretch4_runtime(station, allow_motion=allow_motion)
    return SimpleNamespace(
        adapter=adapter,
        runtime=runtime,
        api=runtime.session.api,
        bridge=CapMcpBridge(runtime.session),
    )


@pytest.fixture
def rig(fake, kinematics):
    value = _rig(fake, kinematics, allow_motion=True)
    yield value
    value.runtime.close()
    value.adapter.close()


@pytest.fixture
def locked(fake, kinematics):
    value = _rig(fake, kinematics, allow_motion=False)
    yield value
    value.runtime.close()
    value.adapter.close()


def test_embodiment_has_eight_joints_in_model_order(kinematics):
    assert JOINT_DIMENSIONS["stretch4"] == 8
    assert (
        tuple(kinematics.model.joint(i).name for i in range(kinematics.model.njnt)) == JOINT_NAMES
    )


@pytest.mark.parametrize("seed", range(5))
def test_ik_then_fk_round_trips_within_5_mm(kinematics, seed):
    rng = np.random.default_rng(seed)
    joints = np.array(
        [
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(0.3, 1.0),
            rng.uniform(0.05, 0.45),
            rng.uniform(-0.5, 2.5),
            rng.uniform(-0.6, 1.0),
            rng.uniform(-2.0, 0.8),
        ]
    )
    position, quaternion = kinematics.forward_kinematics(joints)
    start = joints + rng.normal(0.0, 0.2, 8)
    solution = kinematics.inverse_kinematics(position, quaternion, start)
    reached, _ = kinematics.forward_kinematics(solution)
    assert np.linalg.norm(reached - position) < 0.005
    assert np.all(solution >= kinematics.lower - 1e-6) and np.all(
        solution <= kinematics.upper + 1e-6
    )


def test_base_pose_moves_grasp_site_rigidly(kinematics):
    joints = np.array([0.0, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0, 0.0])
    at_origin, _ = kinematics.forward_kinematics(joints)
    moved, _ = kinematics.forward_kinematics(
        joints + np.array([1.0, 2.0, np.pi / 2, 0, 0, 0, 0, 0])
    )
    expected = np.array([1.0 - at_origin[1], 2.0 + at_origin[0], at_origin[2]])
    np.testing.assert_allclose(moved, expected, atol=1e-9)


def test_wrist_camera_pose_follows_joints(kinematics):
    low = kinematics.camera_from_joints(np.array([0, 0, 0, 0.3, 0.1, 0, 0, 0.0]))
    high = kinematics.camera_from_joints(np.array([0, 0, 0, 0.8, 0.1, 0, 0, 0.0]))
    np.testing.assert_allclose(high[:3, 3] - low[:3, 3], [0.0, 0.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(high[:3, :3] @ high[:3, :3].T, np.eye(3), atol=1e-9)


def test_unauthorized_motion_never_reaches_bridge(locked, fake):
    state = locked.api.get_robot_state()
    target = state.joint_positions["primary"].copy()
    target[3] += 0.02
    assert locked.api.move_to_joints(target).error.code.value == "safety_interlock"
    assert locked.api.set_gripper(0.0).error.code.value == "safety_interlock"
    locked.bridge.call("cap_program_go_home", {})
    assert fake.robot.calls == []


def test_observation_has_calibrated_wrist_and_view_only_head_cameras(rig):
    observation = rig.api.get_observation()
    assert list(observation.cameras) == ["wrist"]
    wrist = observation.cameras["wrist"]
    assert wrist.depth_m.shape == wrist.rgb.shape[:2]
    assert wrist.camera_pose.frame == "odom"
    assert wrist.depth_m[0, 0] == pytest.approx(0.5)  # background; the middle has a dome
    snapshot = rig.adapter.monitor_snapshot()
    assert set(snapshot["cameras"]) == {"wrist", "head_center", "head_left", "head_right"}


def test_move_to_joints_sends_one_absolute_target(rig, fake):
    target = rig.api.get_robot_state().joint_positions["primary"].copy()
    target[3] += 0.02
    result = rig.api.move_to_joints(target)
    assert result.ok
    assert [name for name, _ in fake.robot.calls] == ["move"]
    assert fake.robot.calls[0][1]["lift"] == pytest.approx(target[3])


def test_move_to_pose_with_mink_reaches_target(rig, fake):
    pose = rig.api.get_robot_state().end_effector_poses["primary"]
    target = Pose(
        position=pose.position + [0.0, 0.05, -0.03],
        quaternion_wxyz=pose.quaternion_wxyz,
        frame="odom",
    )
    assert rig.api.move_to_pose(target, strategy=MINK).ok
    reached = rig.api.get_robot_state().end_effector_poses["primary"].position
    assert np.linalg.norm(reached - target.position) < 0.005
    assert [name for name, _ in fake.robot.calls] == ["move"]


def test_out_of_range_target_is_rejected_without_motion(rig, fake):
    target = rig.api.get_robot_state().joint_positions["primary"].copy()
    target[3] = 2.0  # lift beyond 1.2 m
    result = rig.api.move_to_joints(target)
    assert not result.ok
    assert fake.robot.joints["lift"] == pytest.approx(0.6)


def test_mcp_home_opens_gripper_then_homes(rig, fake):
    result = rig.bridge.call("cap_program_go_home", {})
    assert not result.isError
    assert [name for name, _ in fake.robot.calls] == ["gripper", "home"]
    assert fake.robot.gripper == 1.0
    for name, value in FAKE_HOME.items():
        assert fake.robot.joints[name] == pytest.approx(value)
    home = rig.adapter._adapter.go_home()
    assert home.diagnostics["home_verified"] is True


def test_stretch4_metadata_tool_is_public(rig):
    result = rig.bridge.call("cap_stretch4__get_controller_metadata", {})
    assert not result.isError
    assert "final waypoint only" in str(result.structuredContent)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_server_starts_with_stretch4_sim():
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cap_harness.agent",
            "--station",
            "stretch4",
            "--sim",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, process.stderr.read().decode()
            try:
                response = requests.get(
                    f"http://127.0.0.1:{port}/health", timeout=1, proxies={"http": ""}
                )
                if response.ok:
                    assert response.json()["ok"]
                    break
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.2)
    finally:
        process.terminate()
        process.wait(timeout=20)


@pytest.mark.parametrize("name", ["observe", "go_home", "lift_joint_move"])
def test_examples_run_on_fake_bridge(name, capsys):
    from cap_harness.stretch4.run_program import main

    assert main(["--sim", "--allow-motion", f"examples/stretch4/{name}.py"]) == 0
    assert '"code": "' not in capsys.readouterr().out  # no ApiError anywhere in the result


class _CenterSquareSegmenter:
    """Stands in for SAM3: the object is always the central 40x40 pixels."""

    def segment_text(self, observation, query):
        from cap_harness.contracts import Segmentation, SegmentationSet

        mask = np.zeros(observation.rgb.shape[:2], dtype=bool)
        rows, columns = mask.shape
        mask[rows // 2 - 20 : rows // 2 + 20, columns // 2 - 20 : columns // 2 + 20] = True
        segmentation = Segmentation(
            mask=mask, label=query, score=0.9, camera_name="wrist", frame=observation.frame
        )
        return SegmentationSet(ok=True, segmentations=(segmentation,))


def test_localize_check_moves_one_joint_at_a_time_and_returns_home(fake, kinematics):
    station = Stretch4Station(BridgeClient(fake.url), kinematics)
    adapter, runtime = build_stretch4_runtime(
        station, allow_motion=True, segmentation_provider=_CenterSquareSegmenter()
    )
    try:
        source = 'OBJECT = "mug"\n' + open("examples/stretch4/localize_check.py").read()
        execution = runtime.session.executor.execute_program(source)
    finally:
        runtime.close()
        adapter.close()
    assert execution.ok, execution.error
    result = execution.result
    assert all(result["steps"].values())
    assert [view["ok"] for view in result["views"].values()] == [True, True, True]
    assert result["largest_distance_m"] is not None
    assert set(result["images"]) == {
        f"{view}{suffix}"
        for view in ("view1_home_lift", "view2_lift_lower", "view3_roll")
        for suffix in ("", "_mask")
    }
    moves = [targets for name, targets in fake.robot.calls if name == "move"]
    assert [name for name, _ in fake.robot.calls] == ["home", "move", "move", "move", "home"]
    assert moves[0]["wrist_pitch"] == pytest.approx(0.5)
    assert moves[1]["lift"] == pytest.approx(FAKE_HOME["lift"] - 0.10)
    assert moves[2]["wrist_roll"] == pytest.approx(np.radians(20.0))
    for targets in moves:  # base, arm and yaw stay at their measured values
        for name in ("base_x", "base_y", "base_theta", "arm", "wrist_yaw"):
            assert targets[name] == pytest.approx(FAKE_HOME.get(name, 0.0))


def test_run_program_sets_variables_and_saves_images(tmp_path, capsys):
    from cap_harness.stretch4.run_program import main

    program = tmp_path / "program.py"
    program.write_text(
        'image = get_observation().cameras["wrist"].rgb\n'
        'result = {"name": NAME, "images": {"wrist": image, "mask": image[:, :, 0] > 0}}\n'
    )
    out = tmp_path / "images"
    assert main(["--sim", "--set", "NAME=red mug", "--save-images", str(out), str(program)]) == 0
    printed = capsys.readouterr().out
    assert '"name": "red mug"' in printed
    assert sorted(path.name for path in out.iterdir()) == ["mask.png", "wrist.png"]


def test_localize_check_goes_home_after_a_failed_first_view(rig, fake):
    source = 'OBJECT = "mug"\n' + open("examples/stretch4/localize_check.py").read()
    execution = rig.runtime.session.executor.execute_program(source)  # no segmenter: view 1 fails
    assert execution.ok, execution.error
    assert [name for name, _ in fake.robot.calls] == ["home", "move", "home"]
    assert set(execution.result["images"]) == {"view1_home_lift"}


class _LowScoreSegmenter(_CenterSquareSegmenter):
    def segment_text(self, observation, query):
        from dataclasses import replace

        result = super().segment_text(observation, query)
        return replace(result, segmentations=(replace(result.segmentations[0], score=0.2),))


def test_localize_check_treats_low_sam3_score_as_not_found(fake, kinematics):
    station = Stretch4Station(BridgeClient(fake.url), kinematics)
    adapter, runtime = build_stretch4_runtime(
        station, allow_motion=True, segmentation_provider=_LowScoreSegmenter()
    )
    try:
        source = 'OBJECT = "mug"\n' + open("examples/stretch4/localize_check.py").read()
        execution = runtime.session.executor.execute_program(source)
    finally:
        runtime.close()
        adapter.close()
    assert execution.ok, execution.error
    assert execution.result["views"]["view1_home_lift"] == {
        "ok": False,
        "sam3_score": 0.2,
        "error": "SAM3 score below 0.3",
    }
    assert "SAM3 score 0.2 < 0.3" in execution.stdout
    assert [name for name, _ in fake.robot.calls] == ["home", "move", "home"]


def test_wrist_joint_move_moves_pitch_then_roll_and_back(rig, fake):
    execution = rig.runtime.session.executor.execute_program(
        open("examples/stretch4/wrist_joint_move.py").read()
    )
    assert execution.ok, execution.error
    steps = execution.result["steps"]
    assert [step["ok"] for step in steps] == [True] * 4
    assert [step["commanded"] for step in steps] == pytest.approx([0.3, 0.0, 0.3, 0.0])
    assert [step["measured"] for step in steps] == pytest.approx([0.3, 0.0, 0.3, 0.0])
    moves = [targets for name, targets in fake.robot.calls if name == "move"]
    assert [(m["wrist_pitch"], m["wrist_roll"]) for m in moves] == pytest.approx(
        [(0.3, 0.0), (0.0, 0.0), (0.0, 0.3), (0.0, 0.0)]
    )
