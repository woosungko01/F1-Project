"""
F1 강화학습 메인
─────────────────────────────────────────────────────────────────
Speed controls:
  RENDER_EVERY  : only render every Nth episode (set high to train fast)
  RENDER_FPS    : FPS cap when rendering (0 = uncapped)

Keys:
  Q        : quit
  SPACE    : halve epsilon (force exploitation)
  F        : toggle fast-mode (skip rendering)
  +/-      : increase/decrease render_every on the fly
"""

import sys
import pygame
import numpy as np

from env.track_env import TrackEnv, RESULT_SUCCESS, RESULT_FAIL
from env.render    import Renderer
from env.car       import ACTION_STEER_OPTIONS, N_ACTIONS

# ─── 하이퍼파라미터 ───────────────────────────────────────────────
EPISODES       = 2000
MAX_STEPS      = 3000
EPSILON_START  = 1.0
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.997
ALPHA          = 0.15
GAMMA          = 0.92

# ─── 렌더링 속도 제어 ─────────────────────────────────────────────
# RENDER_EVERY=20 → 20 에피소드 중 1번만 화면 출력, 나머지는 풀속도 계산
RENDER_EVERY   = 10
RENDER_FPS     = 120    # 렌더링 시 FPS 상한 (0=무제한)

ACTION_NAMES = {
    0: "Sharp Left",
    1: "Soft  Left",
    2: "Straight",
    3: "Soft  Right",
    4: "Sharp Right",
}


def main():
    env      = TrackEnv()
    renderer = Renderer()

    epsilon       = EPSILON_START
    all_rewards   = []
    best_reward   = -1e9
    success_count = 0
    fail_count    = 0
    render_every  = RENDER_EVERY

    for episode in range(EPISODES):
        state        = env.reset()
        car          = env.car
        total_reward = 0.0
        result_str   = "ongoing"
        do_render    = (episode % render_every == 0)

        for step in range(MAX_STEPS):
            # ── pygame 이벤트 ─────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_SPACE:
                        epsilon = max(EPSILON_MIN, epsilon * 0.5)
                        print(f"  → epsilon forced to {epsilon:.4f}")
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        render_every = min(render_every + 5, 500)
                        print(f"  → render_every={render_every}")
                    if event.key == pygame.K_MINUS:
                        render_every = max(1, render_every - 5)
                        print(f"  → render_every={render_every}")

            # ── 센서 & 행동 선택 ──────────────────────────────────
            left_d, right_d, front_d = car.get_sensors()
            sensor_st = car.state_indices(left_d, right_d, front_d)
            action    = car.choose_action(sensor_st, epsilon)

            # ── 환경 스텝 ─────────────────────────────────────────
            next_state, reward, done, next_sensor_st, env_info = env.step(action)
            total_reward += reward
            result_str = env_info["result"]

            # ── Q-table 업데이트 ──────────────────────────────────
            car.update_q(sensor_st, action, reward, next_sensor_st,
                         alpha=ALPHA, gamma=GAMMA)

            # ── 렌더링 (RENDER_EVERY마다 한 번) ───────────────────
            if do_render:
                hud_info = {
                    "episode":       episode,
                    "step":          step,
                    "epsilon":       epsilon,
                    "action_name":   ACTION_NAMES[action],
                    "reward":        reward,
                    "total_reward":  round(total_reward, 2),
                    "speed":         round(car.v, 2),
                    "dist_center":   env_info["dist_center"],
                    "waypoint_idx":  env_info["waypoint_idx"],
                    "sensor_state":  sensor_st,
                    "result":        result_str,
                    "going_reverse": env_info["going_reverse"],
                    "lap_started":   env_info["lap_started"],
                    "in_start_zone": env_info["in_start_zone"],
                    "q_table":       car.q_table.copy(),
                    "render_every":  render_every,
                }
                renderer.draw(car, left_d, right_d, front_d, hud_info,
                              fps=RENDER_FPS)
            else:
                # 창 응답 유지 (vsync 없이 풀속도)
                pygame.event.pump()

            if done:
                break

        # ── 에피소드 종료 ─────────────────────────────────────────
        all_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        if env.episode_result == RESULT_SUCCESS:
            success_count += 1
        else:
            fail_count += 1

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 10 == 0:
            avg = np.mean(all_rewards[-10:])
            total_eps = success_count + fail_count
            win_rate  = success_count / total_eps * 100 if total_eps > 0 else 0
            print(
                f"[Ep {episode:4d}] "
                f"steps={step+1:4d}  "
                f"reward={total_reward:8.2f}  "
                f"avg10={avg:8.2f}  "
                f"best={best_reward:8.2f}  "
                f"eps={epsilon:.4f}  "
                f"result={result_str:<8}  "
                f"win%={win_rate:.1f}  "
                f"render_every={render_every}"
            )


if __name__ == "__main__":
    main()
