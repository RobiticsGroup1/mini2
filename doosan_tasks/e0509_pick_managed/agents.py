def sb3_ppo_e0509_cfg():
    return {
        "seed": 42,
        "policy": "MlpPolicy",
        "n_steps": 128,             # Increased from 32 for better temporal credit assignment
        "batch_size": 32768,         
        "n_epochs": 15,             
        "gamma": 0.99,
        "gae_lambda": 0.98,         # Increased from 0.95 to look further ahead
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.001,          
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_timesteps": 500_000_000, 
        
        "normalize_input": True,
        "normalize_value": True,
        "clip_obs": 10.0,
    }
