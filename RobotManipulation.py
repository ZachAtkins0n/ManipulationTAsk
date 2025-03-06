"""
FYP code to run a manipulator task for picking and placing an arm using reinforcement learning
 Set up the Isaaclab instance:
    1.  Open cmd and go to isaaclab filepath
    2.  Activate the virtual environment: env_isaaclab\Scripts\activate
    To run custom code in the same file path:
    python scripts\zach_practice\tutorial\CreatingPrims.py 

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
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import UR3_CFG #get the UR3 robot model 

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
    robot: ArticulationCfg =UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_sim(sim:sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    robot=scene["robot"]

 #   ur3_robot_entity= SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
   # ur3_robot_entity.resolve(scene)

    sim_dt = sim.get_physics_dt()
    count =0
    while simulation_app.is_running():
        if count %150 == 0:
            #reset 
            count =0
            robot.reset()
        
        
        sim.step()
        count+=1
        scene.update(sim_dt)


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