def sb3_ppo_e0509_cfg():
    return {
        "seed": 42,
        "policy": "MlpPolicy",
        # [64,64] 기본값은 5단계 순차 태스크에 너무 작음 → [256,256]으로 확장
        "policy_kwargs": {"net_arch": [256, 256]},

        # num_envs=4096 기준: rollout = 256 * 4096 = 1,048,576
        # minibatch 수 = 1,048,576 / 131,072 = 8개/epoch → 안정적
        "n_steps": 256,
        "batch_size": 131072,
        "n_epochs": 8,           # 5 → 8: 샘플 재활용 증가

        # 0.99 → 0.995: episode 600스텝에서 γ^400: 0.018 → 0.135
        # stage 3(복귀), stage 4(release) 보상이 stage 0까지 전달됨
        "gamma": 0.995,
        "gae_lambda": 0.95,

        "learning_rate": 1e-4,
        "clip_range": 0.2,

        # 0.001 → 0.005: 초반 탐색 강화 (stage 2~4 진입 기회 확보)
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,

        # 100M → 200M: 5단계 순차 태스크는 수렴에 더 많은 스텝 필요
        "n_timesteps": 200_000_000,

        "normalize_input": True,
        "normalize_value": True,
        "clip_obs": 10.0,
    }
