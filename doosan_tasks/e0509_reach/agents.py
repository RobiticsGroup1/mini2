def sb3_ppo_e0509_cfg():
    # process_sb3_cfg가 num_envs에 맞게 n_steps 등을 보정해줌(train.py 내부)
    return {
        "seed": 42,
        "policy": "MlpPolicy",
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_timesteps": 2_000_000,
        
        # Optional: 관측/보상 정규화
        "normalize_input": True,
        "normalize_value": False,
        "clip_obs": 10.0,
    }