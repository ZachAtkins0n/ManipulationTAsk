# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
UR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur3/ur3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.65,
            "elbow_joint": 0.56,
            "wrist_1_joint": -0.9,
            "wrist_2_joint": 1.5,
            "wrist_3_joint": 6,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

"""Configuration of UR-10 arm using implicit actuator models."""
UR3_2F_85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/zach/FYP/ManipulationtTAsk/UR3GripperV2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            
        ),
        
        activate_contact_sensors=True,
    ),
     init_state=ArticulationCfg.InitialStateCfg(
     joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.45,
            "elbow_joint": 1.15,
            "wrist_1_joint": -0.9,
            "wrist_2_joint": -1.5,
            "wrist_3_joint": 6,
            "left_inner_finger_knuckle_joint" : 0.000,
            "right_inner_finger_knuckle_joint" : 0.000,
            "left_inner_finger_joint" : 0.000,
            "right_inner_finger_joint" : 0.000,
           "right_outer_knuckle_joint" : 0.000,
            "left_outer_finger_joint":0.000,
            "right_outer_finger_joint": 0.000,
            "finger_joint" : 0.02
           
        },
     ),
    
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint","wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint", ],#["left_outer_finger_joint", "right_outer_finger_joint","left_inner_finger_knuckle_joint","right_inner_finger_knuckle_joint","left_inner_finger_joint","right_inner_finger_joint","right_outer_knuckle_joint","finger_joint" ],
            velocity_limit = 200,
            effort_limit=100.0,
            stiffness=2e11,
            damping=0,
            friction = 1
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_finger_joint", "right_outer_finger_joint","left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
           "right_outer_knuckle_joint", ],
            velocity_limit = 200,
            effort_limit=100.0,
            stiffness=10,
            damping=10,
            friction = 1
        )
    
    
    },
)

"""Configuration of UR3 with gripper 2F_140_CFG using implicit actuator models."""
UR3_2F_140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/zach/FYP/ManipulationtTAsk/UR3Gripper.usd",  #Update file path to relevant location
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            
        ),
       
        activate_contact_sensors=True,
    ),
     init_state=ArticulationCfg.InitialStateCfg(
     joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.45,
            "elbow_joint": 1.15,
            "wrist_1_joint": 4.5,
            "wrist_2_joint": -1.5,
            "wrist_3_joint": 0,
            "finger_joint":0.2,
            "right_outer_knuckle_joint":0.0,
            "left_inner_finger_joint" : 0.000,
            "right_inner_finger_joint" : 0.000,
            "left_outer_finger_joint":0.000,
            "right_outer_finger_joint": 0.000,
         
           
        
           
        },
     ),
    
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint", "elbow_joint", "wrist_1_joint","wrist_2_joint", "wrist_3_joint"],
            velocity_limit=50,
            effort_limit=80.0,
            stiffness=800.0,
            damping=40,
        ),
       
    "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit = 40,
            effort_limit=60.0,
            stiffness=2e2,
            damping=10,
            friction = 0.8,

        ),

    }
)


"""Configuration of UR3 with gripper 2F_140_CFG using implicit actuator models."""
UR3_2F_140_OFFSET_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/zach/FYP/ManipulationtTAsk/UR3Offset.usd",  #Update file path to relevant location
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            
        ),
       
        activate_contact_sensors=True,
    ),
     init_state=ArticulationCfg.InitialStateCfg(
     joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.45,
            "elbow_joint": 1.15,
            "wrist_1_joint": 4.5,
            "wrist_2_joint": -1.5,
            "wrist_3_joint": 0,
            "finger_joint":0.2,
            "right_outer_knuckle_joint":0.0,
            "left_inner_finger_joint" : 0.000,
            "right_inner_finger_joint" : 0.000,
            "left_outer_finger_joint":0.000,
            "right_outer_finger_joint": 0.000,
         
        },
     ),
    
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint", "elbow_joint", "wrist_1_joint","wrist_2_joint", "wrist_3_joint"],
            velocity_limit=50,
            effort_limit=80.0,
            stiffness=800.0,
            damping=40,
        ),
       
    "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit = 40,
            effort_limit=60.0,
            stiffness=2e2,
            damping=10,
            friction = 0.8,

        ),

    }
)
