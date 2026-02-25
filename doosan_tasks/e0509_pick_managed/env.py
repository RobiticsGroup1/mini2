from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from .env_cfg import DoosanE0509PickEnvCfg

class DoosanE0509PickEnv(ManagerBasedRLEnv):
    """Environment for picking objects with the Doosan E0509 robot using a manager-based approach."""
    
    cfg: DoosanE0509PickEnvCfg

    def __init__(self, cfg: DoosanE0509PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
