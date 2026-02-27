def sb3_ppo_e0509_cfg():
    """Configuration for SB3 PPO agent optimized for stability with 4096 environments."""
    return {
        "seed": 42,
        "policy": "MlpPolicy",
        "n_steps": 24,            # Samples per env: 24 * 4096 = 98,304 total per update
        "n_minibatches": 4,       # batch_size = 98,304 / 4 = 24,576
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 5e-4,    # Reduced for stability (Prev was 1e-3)
        "clip_range": 0.2,
        "ent_coef": 0.005,        # Keep exploration
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "n_timesteps": 100_000_000,
        
        "policy_kwargs": {
            "activation_fn": "nn.ELU",
            "net_arch": [512, 256, 128],
            "ortho_init": False,
        },
        
        "normalize_input": True,
        "normalize_value": True,
        "clip_obs": 10.0,
    }
