import numpy as np
from env.track import is_on_track, TRACK_WIDTH

SENSOR_ANGLE_OFFSET = np.radians(10)   # ±10° from forward
SENSOR_MAX_DIST     = 40.0
SENSOR_STEPS        = 200

# Front ray distance thresholds (tune these to your track scale)
FRONT_NEAR_THRESH = 2.5    # < 8  → near zone  (strong reaction)
FRONT_MID_THRESH  = 6.0   # < 20 → mid zone   (moderate reaction)
                            # ≥ 20 → far zone   (gentle reaction)

ACTION_STEER_OPTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
N_ACTIONS = len(ACTION_STEER_OPTIONS)

# State dimensions
N_SIDE_STATES  = 2   # 0: right wall closer, 1: left wall closer
N_FRONT_ZONES  = 3   # 0: far, 1: mid, 2: near


def _ray_distance(car_x, car_y, angle):
    for d in np.linspace(0, SENSOR_MAX_DIST, SENSOR_STEPS):
        rx = car_x + d * np.cos(angle)
        ry = car_y + d * np.sin(angle)
        if not is_on_track(rx, ry):
            return d
    return SENSOR_MAX_DIST


class Car:
    def __init__(self):
        self.max_speed    = 8.0
        self.acceleration = 1.5
        self.turn_rate    = 4.0
        self.dt           = 0.1

        # Q-table: (side_state, front_zone, action)
        self.q_table = np.zeros((N_SIDE_STATES, N_FRONT_ZONES, N_ACTIONS))

        self._last_state  = (0, 0)
        self._last_action = 0

        self.reset_position(0, 0, 0)

    def reset_position(self, x, y, theta):
        self.x     = float(x)
        self.y     = float(y)
        self.theta = float(theta)
        self.v     = 0.5

    # ── Sensors ────────────────────────────────────────────────────────────
    def get_sensors(self):
        left_dist  = _ray_distance(self.x, self.y, self.theta + SENSOR_ANGLE_OFFSET)
        right_dist = _ray_distance(self.x, self.y, self.theta - SENSOR_ANGLE_OFFSET)
        front_dist = _ray_distance(self.x, self.y, self.theta)   # straight ahead
        return left_dist, right_dist, front_dist

    # ── State encoding ─────────────────────────────────────────────────────
    def side_state_index(self, left_dist, right_dist) -> int:
        # 0: right wall closer → tend left
        # 1: left wall closer  → tend right
        return 0 if left_dist > right_dist else 1

    def front_zone_index(self, front_dist) -> int:
        # 0: far  — plenty of room, gentle correction
        # 1: mid  — getting close, moderate correction
        # 2: near — wall imminent, strong correction
        if front_dist >= FRONT_MID_THRESH:
            return 0
        elif front_dist >= FRONT_NEAR_THRESH:
            return 1
        else:
            return 2

    def state_indices(self, left_dist, right_dist, front_dist):
        return (
            self.side_state_index(left_dist, right_dist),
            self.front_zone_index(front_dist),
        )

    # ── Epsilon-greedy action selection ────────────────────────────────────
    def choose_action(self, state: tuple, epsilon: float) -> int:
        side, front = state
        if np.random.rand() < epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[side, front]))

    # ── Q-learning update ──────────────────────────────────────────────────
    def update_q(self, state, action, reward, next_state,
                 alpha=0.1, gamma=0.9):
        s, f     = state
        ns, nf   = next_state
        best_next = np.max(self.q_table[ns, nf])
        self.q_table[s, f, action] += alpha * (
            reward + gamma * best_next - self.q_table[s, f, action]
        )

    # ── Physics step ───────────────────────────────────────────────────────
    def step(self, steer_action_idx: int):
        steer  = ACTION_STEER_OPTIONS[steer_action_idx]
        self.v = min(self.v + self.acceleration * self.dt, self.max_speed)
        self.theta += steer * self.turn_rate * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

    def get_state_array(self):
        return np.array([self.x, self.y, self.theta, self.v])