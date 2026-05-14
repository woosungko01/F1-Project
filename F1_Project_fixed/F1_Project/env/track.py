import numpy as np

# ─────────────────────────────────────────────────────────────────
#  Custom F1-style track (closed polygon, defined by centerline)
#  좌표계: 월드 단위 (픽셀 ÷ 10)
#  트랙 형태: 업로드된 이미지 기반
# ─────────────────────────────────────────────────────────────────

TRACK_WIDTH = 3.5   # 트랙 폭 (양쪽 각 1.75)

# 트랙 중심선 웨이포인트 (이미지 기반 F1 서킷 레이아웃)
# 시계 반대 방향(CCW) 순서
_RAW_WAYPOINTS = [
    (5.7, 23.0),   # 0: 시작선 (start/finish)
    (5.7, 21.0),
    (5.7, 19.0),
    (5.7, 17.0),
    (5.7, 15.0),
    (5.7, 13.0),
    (5.7, 11.0),   # 왼쪽 수직 직선 구간
    (6.0,  9.0),
    (7.0,  6.5),
    (9.0,  4.0),
    (10.0, 3.625),
    (11.0, 3.25),
    (13.0, 2.5),   # 헤어핀 하단
    (18.0, 2.5),
    (25.0, 2.5),
    (33.0, 2.5),

    (41.0, 2.5),
    (41.8, 2.75),
    (42.8, 3.25),
    (43.3, 3.6),
    (43.8, 4.6),
    (44.5, 5.75),
    (44.3, 6.9),
    (43.5, 7.9),
    (42.6, 8.6),
    (41.8, 9.0),
    (41.0, 9.0),

    (37.0, 9.0),
    (31.0, 9.0),
    (25.0, 9.0),   # 중간 직선
    (22.5, 11.0),
    (21.0, 13.0),
    (19.5, 15.0),
    (18.5, 17.0),

    (17.5, 19.0),  # required start

    (16.9, 20.4),
    (16.0, 22.2),
    (14.8, 24.5),
    (13.2, 27.0),

    (11.0, 28.0),  # required peak
    (10.5, 28.0),
    (10.0, 27.8),
    (9.5, 27.4),
    (7.8, 26.2),
    (7.0, 24.8),
    (6.3, 23.8),

    (5.9, 23.3),
    (5.7, 23.0)
]

from scipy.interpolate import CubicSpline

points = np.array(_RAW_WAYPOINTS)

# Parameter t (arc-length approximation)
dist = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
t = np.concatenate([[0], np.cumsum(dist)])
t = t / t[-1]  # normalize

# Separate x, y
x = points[:, 0]
y = points[:, 1]

# Natural cubic spline (C2 continuous)
cs_x = CubicSpline(t, x, bc_type='periodic')
cs_y = CubicSpline(t, y, bc_type='periodic')

# Generate dense points
t_fine = np.linspace(0, 1, 300)

smooth_centerline = list(zip(cs_x(t_fine), cs_y(t_fine)))

# ─────────────────────────────────────────────────────────────────
#  시작선(finish line) 정의
#  웨이포인트 인덱스 0 근처 (x≈5.7, y≈23.0)
# ─────────────────────────────────────────────────────────────────
FINISH_LINE_IDX   = 0       # 중심선에서 시작/결승선에 해당하는 웨이포인트 인덱스
FINISH_LINE_ZONE  = 15      # 결승선 근방 판정 범위 (인덱스 기준)
REVERSE_THRESHOLD = 20      # 역주행 판정: 이만큼 이상 뒤로 가면 역주행

def _interpolate_waypoints(waypoints, n=400):
    """웨이포인트 사이를 선형 보간하여 촘촘한 중심선 생성."""
    pts = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        steps = max(2, int(np.hypot(x1 - x0, y1 - y0) * 5))
        for t in np.linspace(0, 1, steps, endpoint=False):
            pts.append((x0 + t*(x1-x0), y0 + t*(y1-y0)))
    return pts


CENTERLINE = _interpolate_waypoints(smooth_centerline)
CENTERLINE_NP = np.array(CENTERLINE)   # shape (N, 2)


def closest_point_on_centerline(x, y):
    """월드 좌표 (x, y)에서 가장 가까운 중심선 점과 인덱스 반환."""
    diffs = CENTERLINE_NP - np.array([x, y])
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    idx = int(np.argmin(dists))
    return idx, dists[idx]


def is_on_track(x, y):
    """트랙 위에 있으면 True."""
    _, dist = closest_point_on_centerline(x, y)
    return dist <= TRACK_WIDTH / 2


def get_centerline_direction(idx):
    """인덱스 idx에서 중심선의 진행 방향(라디안) 반환."""
    n = len(CENTERLINE_NP)
    next_idx = (idx + 1) % n
    dx = CENTERLINE_NP[next_idx, 0] - CENTERLINE_NP[idx, 0]
    dy = CENTERLINE_NP[next_idx, 1] - CENTERLINE_NP[idx, 1]
    return np.arctan2(dy, dx)


def start_position():
    """차 시작 위치와 방향 반환."""
    idx = FINISH_LINE_IDX
    x, y = CENTERLINE_NP[idx]
    theta = get_centerline_direction(idx)
    return x, y, theta


def is_going_reverse(prev_idx: int, cur_idx: int) -> bool:
    """
    역주행 여부 판정.
    backward < forward AND backward >= threshold 이면 역주행.
    """
    n = len(CENTERLINE_NP)
    forward  = (cur_idx - prev_idx) % n
    backward = (prev_idx - cur_idx) % n
    return backward < forward and backward >= REVERSE_THRESHOLD


def check_lap_complete(prev_idx: int, cur_idx: int, lap_started: bool) -> bool:
    """
    한 바퀴 완주 판정.
    - 시작 후 결승선(FINISH_LINE_IDX)을 통과하면 True.
    - lap_started: 시작 직후 결승선 zone을 벗어난 적 있는지 여부
    """
    if not lap_started:
        return False
    n = len(CENTERLINE_NP)
    # 결승선 zone 통과 여부
    finish_zone_start = (FINISH_LINE_IDX - FINISH_LINE_ZONE // 2) % n
    finish_zone_end   = (FINISH_LINE_IDX + FINISH_LINE_ZONE // 2) % n
    # cur_idx가 결승선 zone 안에 있는지
    if finish_zone_start <= finish_zone_end:
        in_zone = finish_zone_start <= cur_idx <= finish_zone_end
    else:  # wrap-around
        in_zone = cur_idx >= finish_zone_start or cur_idx <= finish_zone_end
    return in_zone
