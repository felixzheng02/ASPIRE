# Raise the lift 2 cm, then return it to where it started. Nothing else moves.
start = get_robot_state().joint_positions["primary"].copy()
up = start.copy()
up[3] = start[3] + 0.02  # joint 3 is lift, metres
result = {"up": move_to_joints(up)}
if result["up"].ok:
    result["back"] = move_to_joints(start)
