"""
FYP code to run a manipulator task for picking and placing an arm using reinforcement learning
 Set up the Isaaclab instance:
    1.  Open cmd and go to isaaclab filepath
    2.  Activate the virtual environment: isaaclabFYP\env_isaaclab\Scripts\activate
    To run custom code in the same file path:
    python scripts\zach_practice\FYP\RobotManipulation.py 

TODO: Begin developing an rl policy to move the robot end effector to a specific location
"""

#Set up the isaaclab app
import argparse
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description="Robot manipulation task")
parser.add_argument("--robot", type=str, default="UR3", help="Name of the robot")
parser.add_argument("--num_envs", type=int, default=128, help="Number of envs to spawn")
AppLauncher.add_app_launcher_args(parser)

args_cli=parser.parse_args()
app_launcher=AppLauncher(args_cli)
simulation_app=app_launcher.app

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg




from isaaclab_assets import UR3_2F_140_CFG #get the UR3 robot model 

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """ Set the scene up with a robot """

    #ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0,0.0,-1))
    )

    #lights
    dome_light=  AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75))
    )

    #arituculation robot
    robot  =UR3_2F_140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_sim(sim:sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    robot=scene["robot"]

    ee_frame_name = "wrist_3_link" #Wrist 3 link is the end effector location for the UR3
    arm_joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    gripper_joint_names = ["left_outer_finger_joint","right_outer_finger_joint","finger_joint"]
    gripper_joint_ids = robot.find_joints(gripper_joint_names)[0]
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
 # Markers

    frame_marker_cfg = FRAME_MARKER_CFG.copy()

    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    #ur3 links: 
    # ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ee_goals = [
        [0.4, 0.3, 0.2, 0.2, 0.5, 0.5,0.5],
        [0.15, -.3, 0.2, 0.2, 0.2,0.4,.5 ],
       # [0.5, 0, 0.5, 0.0, 1.0, 0,0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    robot_entity_cfg =SceneEntityCfg("robot", joint_names=[".*"], body_names=["wrist_3_link"])
    robot_entity_cfg.resolve(scene)



    ee_jacobi_idx = robot_entity_cfg.body_ids[0] -1

    sim_dt = sim.get_physics_dt()
    count =0

    while simulation_app.is_running():
        if count %150 == 0:
            #reset 
            count =0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            ik_commands[:] = ee_goals[current_goal_idx]

            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

            # reset controller
            diff_ik_controller.reset()
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

       # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        sim.step()
        count+=1
        scene.update(sim_dt)
        # obtain quantities from simulation

       # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:6]

        # update marker positions

       # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:6])

       # goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:6])



def main():
    """ Main Function"""
    sim_cfg=sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    #Main camera set up
    sim.set_camera_view([2.5,2.5,2.5], [0.0,0.0,0.0])
    
    #Scene desgin
    scene_cfg=RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO]: Scene set up is ready")
    run_sim(sim,scene)

if __name__ =="__main__":
    #run main function
    main()
    simulation_app.close()