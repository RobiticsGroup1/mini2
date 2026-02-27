# CLAUDE.md — Doosan E0509 Isaac Lab RL Project

## 프로젝트 개요

Isaac Lab + Stable-Baselines3(PPO)를 사용하는 Doosan E0509 로봇 팔 RL 환경.
현재 구현된 태스크: `reach`, `pick`, `pick_place`

**pick 태스크 목표**: 그리퍼를 열고 물체에 접근 → 물체를 집어 들어 올림

## Doosan Robotics E0509 Joint Motion Range
- 이 모델은 모든 축이 360° 연속 회전(continuous rotation) 구조로, 기계적 한계각이 ±170° 같은 구조가 아니라, 이론적으로는 한 바퀴 이상 회전이 가능하다.

| Joint | Axis     | Motion Range |
| ----- | -------- | ------------ |
| J1    | Base     | ±360°        |
| J2    | Shoulder | ±360°        |
| J3    | Elbow    | ±360°        |
| J4    | Wrist 1  | ±360°        |
| J5    | Wrist 2  | ±360°        |
| J6    | Wrist 3  | ±360°        |



---

## 디렉토리 구조

```
mini2/
├── train_e0509.py              # 학습 실행 스크립트
├── play_e0509.py               # 체크포인트 시각화 스크립트
└── doosan_tasks/
    ├── e0509_reach/            # 도달 태스크
    ├── e0509_pick/             # 집기 태스크 (주요 작업 대상)
    │   ├── env.py              # 환경 로직 (reset, obs, reward, done)
    │   ├── env_cfg.py          # 설정값 (reward scale, threshold, 물리 설정 등)
    │   ├── agents.py           # PPO 하이퍼파라미터
    │   └── __init__.py         # Gym 등록 (id: "DoosanE0509-Pick-v0")
    └── e0509_pick_place/       # 집어서 놓기 태스크
```

---

## 실행 커맨드

```bash
# 학습 (권장)
./isaaclab.sh -p mini2/train_e0509.py --task DoosanE0509-Pick-v0 --num_envs 4096 --device cuda:0 --headless --keep_all_info

# 체크포인트에서 이어서
./isaaclab.sh -p mini2/train_e0509.py --task DoosanE0509-Pick-v0 --num_envs 4096 --device cuda:0 --headless \
  --checkpoint logs/sb3/DoosanE0509-Pick-v0/<timestamp>/model_XXXXX_steps.zip

# 시각화 (마지막 체크포인트)
./isaaclab.sh -p mini2/play_e0509.py --task "DoosanE0509-Pick-v0" --num_envs 1 --use_last_checkpoint

# 시각화 (특정 체크포인트)
./isaaclab.sh -p mini2/play_e0509.py --task "DoosanE0509-Pick-v0" --num_envs 1 --checkpoint logs/sb3/.../model.zip
```

로그 및 텐서보드: `logs/sb3/DoosanE0509-Pick-v0/<timestamp>/`

```bash
tensorboard --logdir logs/sb3/DoosanE0509-Pick-v0
```

---

## pick 태스크 아키텍처

### Stage 구조 (task_stage: 0~2)
| Stage | 이름 | 설명 | 전환/종료 조건 |
|-------|------|------|-----------|
| 0 | Approach | 그리퍼 열고 snack에 접근 | `dist_ee_snack < reach_success_dist (0.04m)` |
| 1 | Grip | 그리퍼 닫기 | `stage_timer > 0.3s` & `gripper_q.mean > 0.7` |
| 2 | Lift | snack 들어올리기 | `snack_z > lift_success_height (0.15m)` & `is_holding` → **success terminate** |

### Action Space (dim=7)
- `action[:6]`: 팔 6개 관절 **delta** 제어 (scale: 0.02 rad/step)
- `action[6]`: 그리퍼 **absolute** 제어 (-1=open → 0.0 rad, +1=close → 1.1 rad)

**Gripper Gating** (`_pre_physics_step`에서 stage별 강제 적용):
- Stage 0: 항상 열림(-1.0)
- Stage 1: 닫히는 방향만 허용 (clamp to [0, 1])
- Stage 2: 항상 닫힘(+1.0)

**관절 한계 적용**:
- `_apply_action`에서 `current_targets`를 `±2π rad`로 하드 클램프 (Doosan E0509 ±360°)
- `_get_rewards`에서 soft limit (±331°, `2π - 0.5 rad`) 초과 시 추가 벌점

### Observation Space (dim=34)
```
q(10) + qd(10) + ee_pos_l(3) + snack_pos_l(3) + home_ee_pos_l(3) + ee_to_snack_l(3) + stage(1) + timer_norm(1)
```
- 모든 위치는 env local 좌표계 (world - env_origin)
- `stage_timer`는 `episode_length_s`로 정규화하여 [0, 1] 범위

### 핵심 설계 결정사항

**`_get_rewards` 구조**:
1. 상단에서 모든 stage mask와 전환 조건을 **미리 계산** (task_stage 변경 없음)
2. 각 stage 보상을 순서대로 적용
3. 마지막에 `task_stage` 전환 실행 + 보너스 `+=` (덮어쓰기 방지)

**`home_ee_pos_l` 초기화**:
- `_reset_idx`가 아닌 `_get_observations`에서 최초 1회 캡처
- `sim.step()` 이후 호출이 보장되므로 `body_pos_w` 데이터가 유효함

**Stage 0 Reward 설계 원칙 (local optimum 방지)**:

초기 설계(`exp(-20 * dist)`)는 시작 거리(~0.35m)에서 reward ≈ 0.001로 gradient가 사실상 없었음.
로봇이 "그리퍼를 아래로 향하고 가만히 있는" local optimum에 빠지는 문제가 발생.

```
문제:  align_dot * 5.0 + 상수 4.0 = 최대 9.0  (가만히 있어도 받을 수 있음)
       exp(-20 * 0.35) * 5.0     = 0.001      (접근 reward - 사실상 0)
→ 접근하지 않는 것이 최적 전략이 됨

수정:  exp(-4 * 0.35) * 20.0    = 5.0        (시작 거리에서도 명확한 gradient)
       align_dot * 2.0           = 최대 2.0   (접근 reward가 지배적)
```

현재 Stage 0 reward 구성:
- `exp(-4 * dist) * reach_reward_scale(20)`: 주 접근 reward, 0.35m에서 5.0
- `align_dot * 2.0`: 그리퍼 하향 자세 유도 (보조)
- `-clamp(speed-0.5) * 2.0`: 고속 접근 패널티
- `-clamp(speed-0.1) * 10.0`: 근접 시(0.1m) 고속 패널티
- `-horiz_dist * 2.0`: 수직 접근 유도 (약하게)

---

## 주요 설정값 (env_cfg.py)

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `dt` | 1/120 s | 시뮬레이션 스텝 |
| `decimation` | 4 | 액션 1회당 physics step 수 |
| `episode_length_s` | 20.0 s | 에피소드 최대 길이 |
| `action_scale` | 0.02 | 팔 관절 delta 스케일 |
| `reach_success_dist` | 0.04 m | Stage 0→1 전환 거리 |
| `lift_success_height` | 0.15 m | Stage 2 성공 판정 snack z 높이 |
| `lift_reward_scale` | 100.0 | Stage 2 lift 보상 계수 |
| `reach_reward_scale` | 20.0 | Stage 0 접근 보상 계수 (local optimum 방지용으로 크게 설정) |
| `joint_limit_penalty_scale` | 5.0 | ±360° 관절 한계 소프트 벌점 계수 |

---

## 로봇/씬 구성

- **로봇**: Doosan E0509 + 2-finger gripper (`rh_l1`, `rh_l2`, `rh_r1`, `rh_r2`)
- **EE link**: `link_6`, offset (0, 0, 0.13 m)
- **Snack**: 80×44×48 mm 직육면체, 초기 위치 (0.10, 0.0, 0.059) [local]
- **Desk 상면**: z = 0.035 m [local]
- **로봇 베이스**: (-0.25, 0.0, 0.066) [local]

---

## 학습 알고리즘 (agents.py)

SB3 PPO, `MlpPolicy`, `VecNormalize` 적용

### --num_envs (실행 인자)

로봇 몇 개를 동시에 시뮬레이션할지 결정한다. Isaac Lab은 GPU 위에서 환경을 병렬 실행하므로 많을수록 같은 시간에 더 많은 경험을 수집한다. RL에서는 환경 수가 많아도 학습 품질이 떨어지지 않으며 오히려 다양한 경험이 동시에 모여 안정적이다. GPU 메모리 한계 내에서 최대한 크게 설정하는 것이 유리하다.

- 권장: `4096` (rollout 100만 개/업데이트, 약 27분에 200M 스텝 완료)
- 부족 시: `2048`로 감소

### net_arch = [256, 256]

뉴럴 네트워크의 크기. `[256, 256]`은 뉴런 256개짜리 층이 2개라는 의미.
기본값 `[64, 64]`는 5단계 순차 태스크의 복잡한 패턴을 표현하기에 부족하다.

### n_steps = 256

한 번 학습하기 전에 각 로봇이 몇 스텝 경험을 쌓을지. 총 경험 수 = `n_steps × num_envs`.
에피소드가 최대 600스텝이므로 256이면 절반 분량의 맥락을 보고 학습한다.
너무 짧으면 미래를 못 보고, 너무 길면 업데이트 빈도가 낮아진다.

### batch_size = 131,072

모아둔 경험에서 한 번에 꺼내 학습하는 양. 총 경험(1,048,576)을 이 크기로 나누면 minibatch 8개가 된다. 너무 작으면 노이즈가 많고, 너무 크면 세밀한 업데이트가 어렵다.

### n_epochs = 8

같은 경험 데이터를 몇 번 반복해서 학습할지. 높을수록 데이터를 더 잘 활용하지만 너무 높으면 오래된 데이터에 과적합된다. `clip_range`가 급격한 정책 변화를 막아주므로 8 정도가 안전하다.

### gamma = 0.995 ★ 가장 중요

미래 보상을 현재 가치로 환산하는 할인율. 이 태스크의 핵심 설정값.

```
stage 3(복귀) 보상이 400스텝 후에 발생한다면:
  gamma=0.99  → 현재 가치 = 보상 × 0.99^400 =  1.8%  (거의 사라짐)
  gamma=0.995 → 현재 가치 = 보상 × 0.995^400 = 13.5%  (신호 유지)
```

5단계 순차 태스크에서 stage 3~4의 보상이 stage 0~1 학습에 전달되려면 0.995 이상이 필요하다.

### learning_rate = 1e-4

한 번 학습할 때 파라미터를 얼마나 크게 바꿀지. VecNormalize로 입력이 정규화되어 있으므로 1e-4(=0.0001)가 안정적이다. 너무 크면 발산, 너무 작으면 수렴이 느리다.

### ent_coef = 0.005

탐험의 강도. 0이면 항상 가장 확률 높은 행동만 선택해 초반에 좋아 보이는 전략에 고착된다. 0.005면 초반에 다양한 시도를 해서 stage 2~4에 도달하는 경험을 확보할 수 있다.

### clip_range = 0.2

한 번 학습에서 정책이 얼마나 급격히 바뀔 수 있는지의 상한선. PPO의 핵심 안전장치로, 학습이 갑자기 나빠지는 것을 방지한다. 0.2는 PPO 논문의 기본값으로 대부분의 태스크에 적합하다.

### gae_lambda = 0.95

보상 추정 시 실제 경험(노이즈 있음)과 모델 예측(편향 있음)을 섞는 비율. 0.95는 두 가지를 균형 있게 섞은 표준값이다.

### n_timesteps = 200,000,000

총 학습할 경험의 수(2억). `num_envs=4096` 기준 약 27분 소요. 5단계 순차 태스크는 100M으로 부족할 수 있어 200M으로 설정.

### normalize_input / normalize_value = True

관측값(관절 각도, 위치, 타이머 등)과 가치 함수의 출력을 자동으로 0 근처로 정규화한다. 단위가 서로 다른 값들을 같은 스케일로 맞춰 네트워크가 혼동하지 않도록 한다. 항상 켜두는 것이 좋다.

---

체크포인트: 1,000 step마다 자동 저장
