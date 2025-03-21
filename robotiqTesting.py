"""
Controlling a robotiq manipulator
"""

import math
import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Testing the robotiq 2F-85")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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

from isaaclab_assets import UR3_2F_85_CFG #get the UR3 robot model 
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.utils import configclass

@configclass
class GripperScene(InteractiveSceneCfg):

    #ground plane
    
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

    robot: ArticulationCfg = UR3_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 150 ==0:
            count = 0
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
     #       robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
        
         # Apply random action

        # -- generate random joint efforts

       # efforts = torch.randn_like(robot.data.joint_pos) * 5.0

        # -- apply action to the robot
       # robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim=SimulationContext(sim_cfg)

    sim.set_camera_view([2.5,0.0,4.0],[0.0,0.0,2.0])

    #design scene
    scene_cfg = GripperScene(num_envs=args_cli.num_envs,env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_sim(sim,scene)

if __name__=="__main__":
    main()
    simulation_app.close()
