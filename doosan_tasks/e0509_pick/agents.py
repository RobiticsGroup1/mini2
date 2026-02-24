def sb3_ppo_e0509_cfg():
    return {
        "seed": 42,
        "policy": "MlpPolicy",
        "n_steps": 64,
        "batch_size": 131072,     # Optimized for 16k-32k environments
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_timesteps": 100_000_000,
        
        "normalize_input": True,
        "normalize_value": False,
        "clip_obs": 10.0,
    }
