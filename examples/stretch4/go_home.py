# Open the gripper, then Home: lift, arm and wrists to the robot's kitchen_manipulation pose. The base stays put.
result = set_gripper(1.0)
if result.ok:
    result = go_home()
