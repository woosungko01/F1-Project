import numpy as np
from env.car import Car
from env.track import (
    is_on_track, closest_point_on_centerline,
    get_centerline_direction, start_position,
    CENTERLINE_NP, TRACK_WIDTH,
    is_going_reverse, check_lap_complete,
    FINISH_LINE_IDX, FINISH_LINE_ZONE
)

# ─────────────────────────────────────────────────────────────────
#  에피소드 결과 상수
# ─────────────────────────────────────────────────────────────────
RESULT_ONGOING  = 0
RESULT_SUCCESS  = 1
RESULT_FAIL     = 2

# finish-line crossing: car must pass a virtual line perpendicular to
# the track at FINISH_LINE_IDX, moving in the FORWARD direction.
# We detect this by checking the signed projection of motion onto
# the track direction at the finish waypoint.
_FINISH_DIR = None   # computed lazily once

def _get_finish_dir():
    global _FINISH_DIR
    if _FINISH_DIR is None:
        n = len(CENTERLINE_NP)
        fi = FINISH_LINE_IDX
        nxt = (fi + 1) % n
        dx = CENTERLINE_NP[nxt, 0] - CENTERLINE_NP[fi, 0]
        dy = CENTERLINE_NP[nxt, 1] - CENTERLINE_NP[fi, 1]
        length = np.hypot(dx, dy)
        _FINISH_DIR = np.array([dx / length, dy / length])
    return _FINISH_DIR


class TrackEnv:
    def __init__(self):
        self.car = Car()
        self._prev_idx      = 0
        self._lap_started   = False
        self._prev_x        = 0.0
        self._prev_y        = 0.0
        self.episode_result = RESULT_ONGOING

    def reset(self):
        x, y, theta = start_position()
        self.car.reset_position(x, y, theta)
        self._prev_idx, _ = closest_point_on_centerline(x, y)
        self._lap_started  = False
        self._prev_x       = x
        self._prev_y       = y
        self.episode_result = RESULT_ONGOING
        return self.car.get_state_array()

    # ─────────────────────────────────────────────────────────────
    def _crossed_finish_forward(self) -> bool:
        """
        True if the car just crossed the finish line in the forward
        direction (not reverse).

        Method: the finish line is a segment perpendicular to the
        track at FINISH_LINE_IDX.  We check whether the car's
        movement vector this step has a positive dot-product with
        the track's forward direction at the finish.
        """
        n   = len(CENTERLINE_NP)
        fi  = FINISH_LINE_IDX
        fx, fy = CENTERLINE_NP[fi]

        # Perpendicular (normal) to the finish line = track forward direction
        fwd = _get_finish_dir()          # unit vector along track
        perp = np.array([-fwd[1], fwd[0]])   # perpendicular = finish-line direction

        # Signed distance of previous and current position from finish line
        prev_vec = np.array([self._prev_x - fx, self._prev_y - fy])
        cur_vec  = np.array([self.car.x  - fx, self.car.y  - fy])

        prev_side = float(np.dot(prev_vec, fwd))
        cur_side  = float(np.dot(cur_vec,  fwd))

        # Crossed = sign change (was behind, now ahead)
        crossed = (prev_side < 0) and (cur_side >= 0)

        # Also verify the car is actually near the finish line laterally
        # (within track width) to avoid false positives on the far side
        lateral = abs(float(np.dot(cur_vec, perp)))
        on_line  = lateral <= TRACK_WIDTH

        # Movement must be in forward direction (positive dot with fwd)
        motion = np.array([self.car.x - self._prev_x, self.car.y - self._prev_y])
        going_fwd = float(np.dot(motion, fwd)) > 0

        return crossed and on_line and going_fwd

    # ─────────────────────────────────────────────────────────────
    def step(self, action_idx: int):
        """
        returns: (state, reward, done, sensor_state, info)
        """
        prev_x, prev_y = self.car.x, self.car.y
        self._prev_x, self._prev_y = prev_x, prev_y

        self.car.step(action_idx)
        x, y = self.car.x, self.car.y

        cur_idx, dist_center = closest_point_on_centerline(x, y)
        n = len(CENTERLINE_NP)

        # ── lap_started: left the finish zone at least once ──────
        finish_zone_start = (FINISH_LINE_IDX - FINISH_LINE_ZONE // 2) % n
        finish_zone_end   = (FINISH_LINE_IDX + FINISH_LINE_ZONE // 2) % n
        if finish_zone_start <= finish_zone_end:
            in_start_zone = finish_zone_start <= cur_idx <= finish_zone_end
        else:
            in_start_zone = cur_idx >= finish_zone_start or cur_idx <= finish_zone_end

        if not self._lap_started and not in_start_zone:
            self._lap_started = True

        # ── termination checks ───────────────────────────────────
        going_reverse = is_going_reverse(self._prev_idx, cur_idx)
        off_track     = not is_on_track(x, y)

        # Lap complete: must have left start zone AND cross finish forward
        lap_done = self._lap_started and self._crossed_finish_forward()

        if off_track or going_reverse:
            done = True
            self.episode_result = RESULT_FAIL
            result_str = "fail"
        elif lap_done:
            done = True
            self.episode_result = RESULT_SUCCESS
            result_str = "success"
        else:
            done = False
            result_str = "ongoing"

        # ── 보상 ─────────────────────────────────────────────────
        if done and self.episode_result == RESULT_FAIL:
            reward = -30.0 if going_reverse else -20.0
        elif done and self.episode_result == RESULT_SUCCESS:
            reward = 500.0   # big lap-complete bonus
        else:
            progress = (cur_idx - self._prev_idx) % n
            if progress > n // 2:
                progress = 0
            reward = float(progress) * 0.5
            # centerline-hugging bonus
            reward += max(0.0, 1.0 - dist_center / (TRACK_WIDTH / 2))
            # speed bonus
            reward += self.car.v * 0.05
            if going_reverse:
                reward -= 5.0

        self._prev_idx = cur_idx

        state = self.car.get_state_array()
        left_d, right_d, front_d = self.car.get_sensors()
        sensor_st = self.car.state_indices(left_d, right_d, front_d)

        info = {
            "left_sensor":   round(left_d, 2),
            "right_sensor":  round(right_d, 2),
            "sensor_state":  sensor_st,
            "dist_center":   round(dist_center, 2),
            "waypoint_idx":  cur_idx,
            "result":        result_str,
            "going_reverse": going_reverse,
            "lap_started":   self._lap_started,
            "in_start_zone": in_start_zone,
        }
        return state, reward, done, sensor_st, info
