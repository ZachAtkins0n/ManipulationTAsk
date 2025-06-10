# Introduction
This project was designed to create a RL policy for controlling a UR3 with a Robotiq 2F 85mm Gripper. This would then be used to create a Sim2Real pipeline that would connect to the physical robot itself using ROS Noetic.

![image](https://github.com/user-attachments/assets/dcc715dd-a5a0-4c5f-8e1e-6870284c8bf4)![image](https://github.com/user-attachments/assets/5bf2b561-6ad4-4501-8ef6-08ef6a2bc31d)


# Running the RL Policy
To run the RL policy code this requires Isaaclab and Isaac Sim.

To run: Copy and paste the place directory that includes:  place_env_cfg.py, __init__.py, mdp and config folder in manager based RL directory in Isaac Lab.

Save the universal_robots.py in Isaaclab_assets/robots and modify the required file paths to location of UR3_Gripper.usd 



https://github.com/user-attachments/assets/96524547-7c89-4012-a6a0-c447ba636ac0


# Sim2Real Communications 
To physically control the robot itself socket connections therefore IP address are required, a VPN is required if remotely connected. 

The observed joint data from play.py will be sent to a server ur_ros_joint device that will receive the data Using the ur_ros_drivers package:
  https://github.com/UniversalRobots/Universal_Robots_ROS_Driver on the server and connect to the UR3.

Running the ur_joint_state.py using ROS will establish a server, running the play.py on the IsaacLab side will start sending data via socket connections in the VPN.


# Outcome


https://github.com/user-attachments/assets/8eb5cac5-88ae-4459-9e26-ab69c1c40ec9

