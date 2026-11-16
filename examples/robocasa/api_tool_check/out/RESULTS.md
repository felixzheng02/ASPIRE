# RoboCasa CaP API tool check -- results table

| tool | checks | behaved | notes |
| --- | ---: | ---: | --- |
| `get_task_context` | 1 | 1 | - |
| `get_observation` | 1 | 1 | - |
| `get_robot_state` | 2 | 2 | - |
| `segment_text` | 12 | 12 | - |
| `segment_points` | 3 | 3 | - |
| `mask_to_point_cloud` | 9 | 9 | - |
| `crop_point_cloud` | 2 | 2 | - |
| `estimate_geometry` | 4 | 4 | - |
| `localize_object` | 4 | 4 | - |
| `generate_grasps` | 9 | 7 | end-to-end pick outcome, generate_grasps('fish') |
| `select_grasp` | 4 | 2 | select_grasp('pan', 'top_down'), select_grasp('plate', 'top_down') |
| `solve_ik` | 13 | 11 | solve_ik (+z 6cm), solve_ik (+x 40cm (expected out of reach)) |
| `plan_motion` | 6 | 6 | - |
| `plan_synchronized_motion` | 1 | 1 | - |
| `step` | 3 | 3 | - |
| `execute_trajectory` | 3 | 3 | - |
| `move_to_joints` | 3 | 3 | - |
| `move_to_pose` | 9 | 9 | - |
| `move_synchronized` | 1 | 0 | move_synchronized (one arm, current pose) |
| `open_gripper` | 1 | 1 | - |
| `close_gripper` | 2 | 2 | - |
| `set_gripper` | 3 | 3 | - |
| `set_grippers` | 2 | 2 | - |
| `go_home` | 1 | 1 | - |
| `robocasa.get_task_metadata` | 1 | 1 | - |
| `robocasa.get_controller_metadata` | 2 | 1 | robocasa.get_controller_metadata joint_limits vs the measured arm |
| `robocasa.get_base_pose` | 2 | 2 | - |
| `robocasa.navigate_to_pose` | 5 | 3 | robocasa.navigate_to_pose (yaw +0.30 rad), robocasa.navigate_to_pose (forward +0.30 m (counter ahead)) |
| `robocasa.base_pose_is_free` | 2 | 2 | - |
| `robocasa.plan_standoff_pose` | 2 | 2 | - |
| `robocasa.plan_approach_pose` | 2 | 2 | - |
