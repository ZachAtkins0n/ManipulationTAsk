# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    #print(env.step_dt)
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0) #May need to be conditional? Meet the threshold which is saved



#Get block name 
# if block above threshold start applying penalty to gripper closed

#Need to remodify this code. Can calc the distance of the base_link to that of the target location? 
def move_ee_away_placed(
        env: ManagerBasedRLEnv, minimal_height: float,  command_name: str, std: float,
        ee_cfg = SceneEntityCfg("ee_frame"), object_cfg: SceneEntityCfg = SceneEntityCfg("object"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Move the end effector away from the block when it's at the target location"""
    # extract the used quantities (to enable type-hinting)
    ee: RigidObject = env.scene[ee_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    obj_x_pos = object.data.root_pos_w[:,0]
    obj_y_pos = object.data.root_pos_w[:,1]
    #condition_x = torch.logical_and(obj_x_pos > des_pos_w[:,0] -0.05,  obj_x_pos < des_pos_w[:,0]+0.05)
   # condition_y = torch.logical_and(obj_y_pos > des_pos_w[:,1] -0.05,  obj_y_pos < des_pos_w[:,1]+0.05)
   # condition_xy = torch.logical_and(condition_x, condition_y)
   # condition_ee_height = torch.logical_and(condition_xy, ee.data.target_pos_w[...,0,:]<minimal_height)
    ee_pos = ee.data.target_pos_w[...,0,:]
    ee_targ_dist = torch.norm(des_pos_w- ee_pos, dim=1)
    return (1-torch.tanh(ee_targ_dist/std))

#if object is at target x and y but is above z threshold then punish? 
#Or if end effector is at goal position but too close to z punish
#Get end effector position, if end effector is near the goal height begin punishing the robot
# may allow for robot to place the block and move the end effector away
# need to be cautious of reward weight so if end effector is below goal then punish. 0.1?

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector pbject.data.root_pos_w[:,0] > minimal_location[0] and object.data.root_pos_w[:,0] < minimal_location[1]rame.data.target_pos_w[..., 0, :]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)



def is_obj_below_min_height(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
)-> torch.Tensor:
    """ Goal location for the block and the target location"""
    object: RigidObject=env.scene[object_cfg.name]
    robot: RigidObject=env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:,:3] 
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)  # Shape: [1024]

    x_pos = object.data.root_pos_w[:,0]
    y_pos = object.data.root_pos_w[:,1]
    z_pos = object.data.root_pos_w[:,2]
    condition_x = torch.logical_and(x_pos > des_pos_w[:,0] -0.05,  x_pos <des_pos_w[:,0]+0.05) 
    condition_y = torch.logical_and(y_pos > des_pos_w[:,1]-0.05, y_pos <des_pos_w[:,1]+0.05)
    condition_z = z_pos < 0.1

    condition_xy = torch.logical_and(condition_x, condition_y,)
    condition = torch.logical_and(condition_xy, condition_z,)
    

    return (torch.logical_and(distance>0.05, condition_z)) * (1-torch.tanh(distance/std))

def is_obj_above_min_height(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
)-> torch.Tensor:
    """ Goal location for the block and the target location"""
    object: RigidObject=env.scene[object_cfg.name]
    robot: RigidObject=env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:,:3] 
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)  # Shape: [1024]

    x_pos = object.data.root_pos_w[:,0]
    y_pos = object.data.root_pos_w[:,1]
    z_pos = object.data.root_pos_w[:,2]
    condition_x = torch.logical_and(x_pos > des_pos_w[:,0] -0.05,  x_pos <des_pos_w[:,0]+0.05) 
    condition_y = torch.logical_and(y_pos > des_pos_w[:,1]-0.05, y_pos <des_pos_w[:,1]+0.05)
    condition_z = z_pos > 0.1

    condition_xy = torch.logical_and(condition_x, condition_y,)
    condition = torch.logical_and(condition_xy, condition_z,)
    

    return (torch.logical_and(distance<0.05, condition_z)) * (torch.tanh(distance/std))

def is_obj_placed(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    target_threshold: float = 0.05,
    height_threshold: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing the object at the target location with smooth distance-based rewards."""
    
    # Extract object and robot data
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute goal position in world frame
    des_pos_b = command[:, :3]  # Goal position in robot frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    # Compute distances
    distance_xy = torch.norm(object.data.root_pos_w[:, :2] - des_pos_w[:, :2], dim=1)  # (x, y) distance
   # distance_z = object.data.root_pos_w[:, 2] - des_pos_w[:, 2]  # Height difference

    # Reward for reaching target (x, y)
    reward_xy = torch.exp(-distance_xy / std)  # Smooth reward for getting close
    #reward_z = torch.exp(-torch.abs(object.data.root_pos_w[:, 2] - des_pos_w[:, 2]) / std)  # Encourage correct height

    # Penalize being too high
    #penalty_z = torch.where(distance_z > height_threshold, -0.5 * torch.tanh(distance_z / std), 0.0)

   
    # Combine rewards
    reward = reward_xy #* reward_z 

    return reward

def is_obj_mov_below_height(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    target_threshold: float = 0.05,
    height_threshold: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing the object at the target location with smooth distance-based rewards."""
    
    # Extract object and robot data
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute goal position in world frame
    des_pos_b = command[:, :3]  # Goal position in robot frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    # Compute distances
    distance_xy = torch.norm(object.data.root_pos_w[:, :1] - des_pos_w[:, :1], dim=1)  # (x, y) distance
    distance_z = object.data.root_pos_w[:, 2] - des_pos_w[:, 2]  # Height difference


    condition = torch.logical_and(distance_xy > 0.05,  distance_z < height_threshold)

    # Reward for reaching target (x, y)
    reward_xy = torch.exp(-distance_xy / std)  # Smooth reward for getting close
    reward_z = torch.exp(-torch.abs(object.data.root_pos_w[:, 2] - des_pos_w[:, 2]) / std)  # Encourage correct height

    # Penalize being too high
    penalty_z = torch.where(distance_z > height_threshold, -0.5 * torch.tanh(distance_z / std), 0.0)

   
    # Combine rewards
    reward = reward_xy * reward_z 

    return torch.where(condition, (torch.tanh(distance_xy/std)), 0)




def object_too_high(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    target_threshold: float = 0.06,
    height_threshold: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing the object at the target location with smooth distance-based rewards."""
    
    # Extract object and robot data
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute goal position in world frame
    des_pos_b = command[:, :3]  # Goal position in robot frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    # Compute distances
    distance_xy = torch.norm(object.data.root_pos_w[:, :2] - des_pos_w[:, :2], dim=1)  # (x, y) distance
    distance_z = object.data.root_pos_w[:, 2] - des_pos_w[:, 2]  # Height difference

    # Reward for reaching target (x, y)
    reward_xy = torch.exp(-distance_xy / std)  # Smooth reward for getting close
    reward_z = torch.exp(-torch.abs(object.data.root_pos_w[:, 2] - des_pos_w[:, 2]) / std)  # Encourage correct height

    # Penalize being too high
    penalty_z = torch.where(distance_z > height_threshold, -0.5 * torch.tanh(distance_z / std), 0.0)

   
    # Combine rewards
    reward = reward_xy * reward_z 

    return torch.where(distance_z > height_threshold,  torch.tanh(distance_z / std), 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: list,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def increasing_height(
        env: ManagerBasedRLEnv,
        std: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w[:,2]
    return 1-torch.tanh(object_pos/std)

def ee_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_dist: float,
    command_name: str,
    robot_cfg : SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the agent for brblock distance to goal position"""
    # extract desired quantitieso
    ee:  FrameTransformer=env.scene[ee_cfg.name]
    command = env.command_manager.get_command(command_name)
    robot: RigidObject=env.scene[robot_cfg.name]

    #Potentially use the direct end effector instead? 

    # Compute desired position in world frame
    des_pos_b = command[:, :3]  # Goal position in robot base frame (Shape: [1024, 3])
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3],  # Root position (Shape: [1024, 3])
        robot.data.root_state_w[:, 3:7],  # Root orientation quaternion (Shape: [1024, 4])
        des_pos_b#.unsqueeze(1)  # Ensure correct broadcasting (Shape: [1024, 1, 3])
    )  # Expected output: [1024, 3]

    # Compute distance (Ensure correct shape)
    distance = torch.norm(ee.data.target_pos_w - des_pos_w, dim=-1)  # Shape: [1024]

    # Compute reward
    reward = 1-torch.tanh(distance/std)

    return reward[0]

def goal_location(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_location: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
)-> torch.Tensor:
    """ Goal location for the block and the target location"""
    object: RigidObject=env.scene[object_cfg.name]
    robot: RigidObject=env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:,:3] 
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)  # Shape: [1024]

    """
    x_pos = object.data.root_pos_w[:,0]
    y_pos = object.data.root_pos_w[:,1]
    z_pos = object.data.root_pos_w[:,2]
    condition_x = torch.logical_and(x_pos > des_pos_w[:,0] -0.05,  x_pos <des_pos_w[:,0]+0.05) 
    condition_y = torch.logical_and(y_pos > des_pos_w[:,1]-0.05, y_pos <des_pos_w[:,1]+0.05)
    condition_z = z_pos <= des_pos_w[:,2]+0.05
    condition_xy = torch.logical_and(condition_x, condition_y,)
    condition = torch.logical_and(condition_xy, condition_z,)
    """

    return (1-torch.tanh(distance/std))#(condition) * (1-torch.tanh(distance/std))


def object_goal_position(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        robot_cfg: SceneEntityCfg=SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg=SceneEntityCfg("object")
)-> torch.Tensor:
    
    object: RigidObject=env.scene[object_cfg.name]
    robot: RigidObject=env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:,:3] 
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)  # Shape: [1024]

    
    x_pos = object.data.root_pos_w[:,0]
    y_pos = object.data.root_pos_w[:,1]
    z_pos = object.data.root_pos_w[:,2]
    condition_x = torch.logical_and(x_pos > des_pos_w[:,0] -0.02,  x_pos <des_pos_w[:,0]+0.02) 
    condition_y = torch.logical_and(y_pos > des_pos_w[:,1]-0.02, y_pos <des_pos_w[:,1]+0.02)
    condition_z = z_pos <= des_pos_w[:,2]+0.02
    condition_xy = torch.logical_and(condition_x, condition_y,)
    condition = torch.logical_and(condition_xy, condition_z,)
    


    return torch.where(condition, 1-torch.tanh(distance/std), 0)

"""
TODO: make a new reward that rewards an increasing z before reaching the minimum and maximum 
    - Reward x and y distance decreasing
    - Punish when block z is above minimum in x and y? 
    - Cumulative reward? Higher z location before reaching goal? 
"""


"""
def object_goal_distance_xy(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_location: list,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
   
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold

    x_pos = object.data.root_pos_w[:,0]
    y_pos = object.data.root_pos_w[:,1]
    z_pos = object.data.root_pos_w[:,2]
    condition_x = torch.logical_and(x_pos > minimal_location[0],  x_pos <minimal_location[1]) 
    condition_y = torch.logical_and(y_pos > minimal_location[2], y_pos <minimal_location[3])
    condition_z = z_pos <= minimal_location[4]
    condition_xy = torch.logical_and(condition_x, condition_y,)
    condition = torch.logical_and(condition_xy, condition_z,)

    return ( condition) * (1 - torch.tanh(distance / std))

"""