This project was designed to create a RL policy for controlling a UR3 with a Robotiq 2F 85mm Gripper. This would then be used to create a Sim2Real pipeline that would connect to the physical robot itself using ROS Noetic.


To run the RL policy code this requires Isaaclab and Isaac Sim.

To run: Copy and pace place_env_cfg.py, __init__.py, mdp and config folder in manager based RL directory in Isaac Lab.

Save the universal_robots.py in Isaaclab_assets/robots and modify the required file paths to location of UR3_Gripper.usd 


To physically control the robot itself socket connections therefore IP address are required, a VPN is required if remotely connected. 

The observed joint data from play.py will be sent to a server ur_ros_joint device that will receive the data Using the ur_ros_drivers package:
  https://github.com/UniversalRobots/Universal_Robots_ROS_Driver on the server and connect to the UR3.

Running the ur_joint_state.py using ROS will establish a server, running the play.py on the IsaacLab side will start sending data via socket connections in the VPN.
