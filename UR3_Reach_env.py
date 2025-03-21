"""
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

UR3 reach manager-based rl environment
The end effector of the RL arm reaches towards a target location based on the general reach env class
"""

import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

#Obtain the markovian decision policy for reach tasks
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

#Get the UR3 model
from isaaclab_assets import UR3_CFG

"""
Scene component description
"""

@configclass
class UR3ReachCfg(InteractiveSceneCfg):
    """ Scene configurations """

    #Ground plane
    ground = AssetBaseCfg(
        prim_path = "/World/ground",
        spawn = sim_utils.GroundPlaneCfg(size=(100,100))
    )

    #Lights
    dome_light = AssetBaseCfg(
        prim_path = "/World/DomeLight",
        spawn = sim_utils.DomeLightCfg(color= (.9,0.9,0.9,), intensity = 1000.0)
    )

    #UR3
    robot: ArticulationCfg = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

""" Settings for MDP """

@configclass
class CommandsCfg:
    """Command Terms for the MDP"""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name = ".*",
        resampling_time_range=(4.0,4.0),
        debug_vis = True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.1),
            pos_y=(-0.1, 0.1),
            pos_z=(-0.05, 0.1),
            roll=(-3.14, 3.14),
            pitch = (-3.14,3.14),  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        )
    )

@configclass
class ActionsCfg:
    """ Action specifics for MDP"""
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=100.0)

@configclass
class ObservationsCfg:
    """Observation specs for MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events"""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.05,1),
            "velocity_range": (0.0,0.0)
        }
    )

@configclass
class RewardsCfg:
    """Reward Terms for MDP"""

    #task terms
    end_effector_position= RewTerm(
        func=mdp.position_command_error,
        weight=-.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*"]), "command_name": "ee_pose"}
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func = mdp.position_command_error_tanh,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "std": 0.1, "command_name": "ee_pose"}
    )

    end_effector_oreintation_tracking = RewTerm(
        func = mdp.orientation_command_error,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"}
    )

    #action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight = -0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""
    time_out = DoneTerm(func = mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """ Curriculum terms for MDP """
    action_rate = CurrTerm(
        func = mdp.modify_reward_weight, params = {"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func = mdp.modify_reward_weight, params = {"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


@configclass
class UR3ReachEnv(ManagerBasedRLEnvCfg):
    """ Confiuration for the reach end-effector pose tracking environment """

    #scene settings
    scene: UR3ReachCfg = UR3ReachCfg(num_envs = 4096, env_spacing=2.5)
    #Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    #MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg=TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialisation """
        self.decimation = 2
        self.sim.render_interval=self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5,3.5,3.5)
        self.sim.dt=1/60
