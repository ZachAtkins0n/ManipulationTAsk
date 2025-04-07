# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.
.. code-block:: bash
    # Usage

    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

# New inclusion Author: @Zach:
    Modified the code to work exclusively with a UR3 and Robotiq 2F_140 gripper.
    Removed the default position joints to streamline the arm movement
    A communication process has been implemented, using socket connection to a server that connects to a ROS Noetic publisher:
        -> joint_state_pub
    
    Before running this code joint_state_pub the following needs to be ran in seperate terminals:
        -> roscore
        -> rosrun joint_state_pub joint_state_pub.py
        ->  ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py
    
"""


"""Launch Isaac Sim Simulator first."""


import argparse
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Diff IK controller example for remote communication")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""


import torch


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##

from isaaclab_assets import UR3_2F_140_CFG # isort:skip

import socket
HOST = ""
PORT = 65432
sock = None


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation for UR3 and Gripper
    robot = UR3_2F_140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    """Runs the simulation loop."""

    # Extract scene entities
    # note: we only do this here for readability.

    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)


    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    # Define goals for the arm
    #First three are cartesian postion, next three is eular rotation (keep the extra 0 for manipulator)
    ee_goals = [
        [0.4, 0.3, 0.35, 0, 1, 0,0],
        [0.3, -0.4, 0.28, 0, 1, 0.0,0],
        [0.35, 0, 0.4, 0.0, 1.0, 0.0,0],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # Track the given command

    current_goal_idx = 0

    # Create buffers to store actions

    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)

    ik_commands[:] = ee_goals[current_goal_idx]


    # Specify robot-specific parameters for a UR3 w/ Robitq Gripper
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"], body_names = ["robotiq_base_link"])
    # Resolving the scene entities

    robot_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] -1

    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]


    # Define simulation stepping

    sim_dt = sim.get_physics_dt()

    count = 0

    # Simulation loop

    #Attempt to connect to server
    is_connected = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST,PORT))
        is_connected = True
    except:
        print("Unable to establish a connection")


    while simulation_app.is_running():

        # Change the target location for the robot

        if count % 300 == 0:
        
            count = 0

            # reset joint state
            joint_pos = robot.data.joint_pos.clone()
            joint_vel = robot.data.joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

            diff_ik_controller.set_command(ik_commands)
            
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

        else:

            # obtain quantities from simulation

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]

            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

            root_pose_w = robot.data.root_state_w[:, 0:7]

            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            # compute frame in root frame

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
       
        # perform step
        sim.step()

        # update sim-time
        count += 1

        # update buffers
        scene.update(sim_dt)

        #If a connection is established send joint data to the server
        if is_connected:
                joint_arr = robot.data.joint_pos.cpu().flatten().tolist()
                data_str = ",".join(map(str, joint_arr))  
                sock.sendall(data_str.encode('utf-8'))  


        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])



def main():

    """Main function."""

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)



if __name__ == "__main__":

    # run the main function
    main()

    # close sim app
    simulation_app.close()