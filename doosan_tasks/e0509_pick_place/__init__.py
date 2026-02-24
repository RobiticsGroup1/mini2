import gymnasium as gym

from .env_cfg import DoosanE0509PickPlaceEnvCfg


def env_cfg_entry_point():
    return DoosanE0509PickPlaceEnvCfg()


def sb3_cfg_entry_point():
    from .agents import sb3_ppo_e0509_cfg
    return sb3_ppo_e0509_cfg()


# Gym registration
gym.register(
    id="DoosanE0509-PickPlace-v0",
    entry_point="doosan_tasks.e0509_pick_place.env:DoosanE0509PickPlaceEnv",
)

spec = gym.spec("DoosanE0509-PickPlace-v0")

# gymnasium 버전에 따라 kwargs가 없거나 읽기 전용일 수 있어 안전하게 처리
if not hasattr(spec, "kwargs") or spec.kwargs is None:
    # 일부 버전에서는 _kwargs 사용
    if hasattr(spec, "_kwargs") and spec._kwargs is not None:
        spec._kwargs["env_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:env_cfg_entry_point"
        spec._kwargs["sb3_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:sb3_cfg_entry_point"
    else:
        # 마지막 우회: __dict__도 같이 넣어둠(디버그용)
        spec.__dict__["env_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:env_cfg_entry_point"
        spec.__dict__["sb3_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:sb3_cfg_entry_point"
else:
    spec.kwargs["env_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:env_cfg_entry_point"
    spec.kwargs["sb3_cfg_entry_point"] = "doosan_tasks.e0509_pick_place:sb3_cfg_entry_point"