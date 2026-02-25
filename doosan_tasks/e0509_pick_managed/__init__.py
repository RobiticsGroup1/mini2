import gymnasium as gym

from .env_cfg import DoosanE0509PickEnvCfg


def env_cfg_entry_point():
    return DoosanE0509PickEnvCfg()


def sb3_cfg_entry_point():
    from .agents import sb3_ppo_e0509_cfg
    return sb3_ppo_e0509_cfg()


# Gym registration
gym.register(
    id="DoosanE0509-PickManaged-v0",
    entry_point="doosan_tasks.e0509_pick_managed.env:DoosanE0509PickEnv",
    kwargs={
        "env_cfg_entry_point": "doosan_tasks.e0509_pick_managed:env_cfg_entry_point",
        "sb3_cfg_entry_point": "doosan_tasks.e0509_pick_managed:sb3_cfg_entry_point",
    }
)
