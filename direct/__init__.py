"""
Create an import to run RL policies for running UR3 tasks
"""

import gymnasium as gym
from . import agents

# Register the environments
gym.register(
    id="UR3-Reach-Direct-v0",
    entry_point=f"{__name__}.UR3Reach:UR3ReachCfg",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:UR3ReachPPORunnerCfg"
    }
)