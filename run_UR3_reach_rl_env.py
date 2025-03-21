"""
Create the environment for a robot to reach towards a location
Using the reach_env_cfg module to aid with the RL learning process

FYP code to run a manipulator task for picking and placing an arm using reinforcement learning
 Set up the Isaaclab instance:
    1.  Open cmd and go to isaaclab filepath
    2.  Activate the virtual environment: isaaclabFYP\env_isaaclab\Scripts\activate
    To run custom code in the same file path:
    python scripts\zach_practice\FYP\RobotManipulation.py 

TODO: Begin developing an rl policy to move the robot end effector to a specific location
Running this using manager based doesn't appear to retain on re-runs therefore is best to use a direct work flow pipeline

"""

import argparse

from isaaclab.app import AppLauncher

#add argparse args
parser = argparse.ArgumentParser(description="Running the arm reach RL environment to target location")
parser.add_argument("--num_envs", type=int, default = 64)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

""" Import the manager based environment and the robot arm"""
import torch
from isaaclab.envs import ManagerBasedRLEnv

from UR3_Reach_env import UR3ReachEnv

def main():
    """Main function"""
    # create environment config
    env_cfg = UR3ReachEnv()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    #sim physics
    count =0 
    while simulation_app.is_running():
        with torch.inference_mode():
            #if reset
            if count % 300 == 0:
                count =0
                env.reset()
            
            joint_efforts = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            print("[Env 0]: Robot Joint: ", obs["policy"][0][1].item())
            count +=1
    env.close()


if __name__ == "__main__":
    #ru main function
    main()
    #close sim app
    simulation_app.close()