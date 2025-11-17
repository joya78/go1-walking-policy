"""Custom Go1 configurations."""

import gymnasium as gym

from . import agents, go1_walking_env_cfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_walking_env_cfg:UnitreeGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_walking_env_cfg:UnitreeGo1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)
