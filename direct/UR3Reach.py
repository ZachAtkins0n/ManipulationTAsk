"""
Direct workflow to run UR3 reach environment

TODO: continue developing the direct UR3Reach code
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets import UR3_CFG #Import the UR3 model

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRlEnv, DirectRlEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

@configclass
class UR3ReachCfg(DirectRlEnvCfg):
    """ Design the scene and configure simulation"""


    #Spawn the scene entities

    #Ground plane
    ground = AssetBaseCfg(
        prim_path = "/World/ground",
        spawn = sim_utils.GroundPlaneCfg(size=(100,100))
    )

  
    robot =  UR3_CFG.ArticulationCfg(prim_path="{REGEX_ENV_NS}/robot",
                             spawn=sim_utils(
                                 activate_contact_sensors = False,
                                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                 disable_gravity=False,
                                 max_depenetration_velocity=5.0,
            ),
            artiuclation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count = 12, solver_velocity_iteration_count = 1
            ),
            
            init_state = ArticulationCfg.InitialStageCfg(
                joint_pos = {
                    "shoulder_pan_joint": 0.4,
                    "shoulder_lift_joint": 0.3,
                    "elbow_joint": 0.2,
                    "wrist_1_joint": 0.5,
                    "wrist_2_joint": 0.5,
                    "wrist_3_joint": 0.5
                },
            pos = (1.0,0.0,0.0),
            rot = (0.0,0,0,1),
            ),
            actuators = {
                "ur3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoudler_pan_joint","shoulder_lift_joint"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "ur3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            })
                             )
    
    cube = ArticulationCfg(
        prim_path = "/World/objects/Cube",
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1)),
        collision_props = sim_utils.CollisionPropertiesCfg(),
        mass_props = sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props = sim_utils.RigidBodyPropertiesCfg(),
        spawn = sim_utils.size(0.2,0.2,0.2),
    )

    #env
    decimation = 2
    episode_length_s=5
    action_scale = 100 
    action_space = 1
    observation_space = 4
    state_space = 0

    #sim
    sim: SimulationCfg = SimulationCfg(dt = 1/120, render_interval = decimation)

    action_scale = 7.5
    dof_velocity_scale = 0.1

    #reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05


class UR3ReachEnv(DirectRlEnv):

    cfg: UR3ReachCfg

    def __init__(self, cfg: UR3ReachCfg, render_mode: str | None=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-loca coords"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device = device)
    
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        #auxilary variarbles for computing applied action, observations and rewards
        self.robot_dof_lower_limits=  self.robot.data.soft_joint_pos_limits[0,:,0].to(device=self.device)
        self.robot_dof_upper_limits=  self.robot.data.soft_joint_pos_limits[0,:,1].to(device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/wrist_3_link")),
            self.device
        )

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos
        )

        self.hand_link_idx = self._robot.find_bodies("wrist_3_link")[0][0]
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_pose_pos = robot_local_pose_pos.repeat((self.num_envs, 1))

        self.robot_reach_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_reach_rot = torch.zeros((self.num_envs, 4), device=self.device)


        self.cube_loc_pos = torch.tensor([0.35, -.2, 0.5] , device=self.device)   #hard code the position
        self.cube_loc_rot = torch.tensor([0,0,0], device=self.device)

        self.target_pos = torch.zeros((self.num_envs,4), device=self.device)
        self.target_rot = torch.zeros((self.num_envs, 3), device=self.device)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = Articulation(self.cfg.cube)
        self.scene.articulations["robot"]=self._robot
        self.scene.articulations["cube"] = self._cube
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrian.env_spacing = self.scene.cfg.env_spacing

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        #clone and replicate
        self.scene.clone_environments(copy_from_source =False)

        #add lights
        light_cfg = sim_utils.DomeLightCfg(intensity = 2000, color =(0.75,0.75,0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0,1.0)
        targets = self.robot_dof_targets+self.robot_dof_speed_scales*self.dt.self.actions*self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets,self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cube.data.default_root_state > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        #Refersh intermedaite value after phys steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.actions,
            self._cube.data.default_root_state,
            self.robot_reach_pos,
            self.robot_reach_rot,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.action_penalty_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids = env_ids)
        self._robot.write_joint_state_to_sim(joint_pos,joint_vel, env_ids=env_ids)

        #cube state
        zeros = torch.zeros((len(env_ids), self._cube.num_joints), device = self.device)
        self._cube.write_data_to_sim(zeros,zeros,env_ids=env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled=(
            2.0*(self._robot.data.joint_pos - self.robot_dof_lower_limits) /(self.robot_dof_upper_limits - self.robot_dof_lower_limits)-1.0)
        
        to_target = self.cube_loc_pos-self.robot_reach_pos

        obs = torch.cat(
            (dof_pos_scaled,
             self._robot.data.joint_vel*self.cfg.dof_velocity_scale,
             to_target,
             self._cube.data.root_state.unsqueeze(-1))
        )
        dim=1
        return {"policy": torch.clamp(obs, -5.0,5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICIES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        cube_pos = self._cube.data.body_pos_w[env_ids, self.cube_loc_pos]
        (
            self.robot_reach_rot[env_ids],
            self.robot_reach_pos[env_ids],
            self.cube_loc_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_pose_pos[env_ids],
            cube_pos,
            self.cube_loc_pos[env_ids]
        )
    
    def _compute_rewards(
            self,
            actions,
            cube_pos,
            UR3_pos,
            UR3_rot,
            num_envs,
            dist_reward_scale,
            rot_reward_scale,
            action_penalty_scale,
            finger_reward_scale,
            joint_position
    ):
        
        #distance between hand an cube
        d = torch.norm(UR3_pos - cube_pos, p=2, dim=-1)
        dist_reward = 1/(1+d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <=0.02, dist_reward*2, dist_reward)
        
        action_penalty = torch.sum(actions**2, dim=-1)

        rewards = (dist_reward_scale*dist_reward-action_penalty*action_penalty)

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale*dist_reward()).mean()
        }

        return rewards
    
    def _compute_transform(
            self,
            hand_rot,
            hand_pos,
            UR3_local_grasp_rot,
            UR3_local_grasp_pos
    ):
        global_UR3_rot, global_UR3_pos = tf_combine(
            hand_rot,hand_pos, UR3_local_grasp_rot, UR3_local_grasp_pos
        )

        return global_UR3_rot, global_UR3_pos