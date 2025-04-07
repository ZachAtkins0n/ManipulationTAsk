# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(0.8,0.5,0.6)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    #Goal pose for the block location

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(.1, 0.1), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),)
    
    goal_final_ee = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING, 
        resampling_time_range=(5.0,5.0),
        debug_vis = True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35,0.35),pos_y=(0,0),pos_z=(.2,0.2),roll=(0,0),pitch=(0,0),yaw=(0,0)
        )
    )

    block_goal = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name =MISSING,
        resampling_time_range=(5.0,5.0),
        debug_vis = True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5,0.5),pos_y=(0,0),pos_z=(0.0,0.0),roll=(0,0),pitch=(0,0),yaw=(0,0)
        )
    )
    


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        target_ee_position=ObsTerm(func=mdp.generated_commands,params={"command_name":"goal_final_ee"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.32, 0.42), "y": (-.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=0.8)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.1}, weight=12)

    #Try this one with block goal 
   # move_ee_away = RewTerm(func=mdp.move_ee_away_placed, params={"minimal_height":0.1, "command_name": "object_pose", "std":0.1}, weight = -1)
    
    #New goal should be created where the end effector should return to its position once a block reaches it's desired position
    #Two rewards
    #-> reward for the end effector reaching 'base' position' 
    #-> punish for the block being at the end effector position
    #Just need the block to be released now
    
    """
    ee_goal_loc = RewTerm(
        func=mdp.is_gripper_closed,
        params={"std": 0.08, "minimal_dist": 0.05, "command_name": "block_goal"},
        weight=-1,
    )
    """
    
   
    #TODO Use two goals. Place block meeting objective above desired location bonus reward.
    # Can utilise torch.where(block at objective, distance to this object via 1-tanh convention) 
    #TODO GOAL 2: Block gets rewarded for being at final objective only
    # If block at command goal position give reward. Encourage end effector to move to rest position via torch.where
    #TODO

    block_nearing_placement = RewTerm(
        func = mdp.is_obj_placed,
        params={"std": 0.2, "command_name": "block_goal"},
        weight = 0.8 #increase this reward

    )

    """
    block_moving_below = RewTerm(
        func = mdp.is_obj_mov_below_height,
        params={"std":0.02, "command_name": "block_goal"},
        weight = 0
    )
    """
    """Tracking the goal position of the block """
    
    
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.1, "command_name": "object_pose"},
        weight=12,
    )
    
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.1, "command_name": "object_pose"},
        weight=1,
    )
    
    object_height = RewTerm(
        func = mdp.increasing_height,
        params={"std": 0.05,},
        weight = 0.1
    )
    
    object_at_goal_loc = RewTerm(
        func=mdp.object_goal_position,
        params={"std":0.2, "command_name":"block_goal"},
        weight = 4
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

  

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 10000}
    )

    object_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_goal_tracking", "weight": 5, "num_steps":25000}
    )


    lifting_object = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "lifting_object", "weight": 8, "num_steps": 25000})
    object_height = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "object_height", "weight": -0.5, "num_steps": 25000})
    block_nearing_goal = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "block_nearing_placement", "weight": 2, "num_steps": 18000})
    object_at_goal_loc = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "object_at_goal_loc", "weight": 25, "num_steps": 18000})

##
# Environment configuration
##


@configclass
class PlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 7.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
