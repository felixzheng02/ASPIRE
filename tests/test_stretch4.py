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
    np.testing.assert_allclose(wrist.depth_m, 0.5)
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
