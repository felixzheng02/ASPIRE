# Observation only: no motion. Reports task, robot state and the wrist RGB-D view.
result = {
    "task": get_task_context(),
    "state": get_robot_state(),
    "observation": get_observation(),
}
