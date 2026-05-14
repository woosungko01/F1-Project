import pygame
import numpy as np
from env.track import (
    CENTERLINE_NP, TRACK_WIDTH,
    FINISH_LINE_IDX, FINISH_LINE_ZONE,
    get_centerline_direction
)
from env.car import SENSOR_MAX_DIST, SENSOR_ANGLE_OFFSET as _SA

WIDTH, HEIGHT = 1000, 800

def _calc_transform():
    xs = CENTERLINE_NP[:, 0]
    ys = CENTERLINE_NP[:, 1]
    margin = 80
    sx = (WIDTH  - 2 * margin) / max(xs.max() - xs.min(), 1)
    sy = (HEIGHT - 2 * margin) / max(ys.max() - ys.min(), 1)
    scale = min(sx, sy)
    cx = (xs.max() + xs.min()) / 2
    cy = (ys.max() + ys.min()) / 2
    return scale, cx, cy

_SCALE, _CX, _CY = _calc_transform()

DARK_BG      = (15, 15, 20)
TRACK_MID    = (80, 80, 90)
TRACK_EDGE   = (50, 200, 120)
CAR_COLOR    = (255, 60, 60)
TRAIL_CLR    = (60, 200, 80, 180)
SENSOR_CLR   = (255, 220, 0)
WHITE        = (255, 255, 255)
CYAN         = (0, 220, 255)
YELLOW       = (255, 230, 0)
GREEN        = (80, 255, 100)
RED          = (255, 80, 80)
FINISH_COLOR = (255, 255, 0)    # 결승선: 노란색
START_ARROW  = (0, 255, 120)    # 시작 방향 화살표: 밝은 초록


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("F1 RL – Dual-Sensor Agent")
        self.clock  = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 17)
        self.font_sm = pygame.font.SysFont("Consolas", 14)

        self.trail: list = []
        self._build_track_surfaces()

    # ─── 좌표 변환 ────────────────────────────────────────────────
    def _w2s(self, x, y):
        sx = int(WIDTH  / 2 + (x - _CX) * _SCALE)
        sy = int(HEIGHT / 2 - (y - _CY) * _SCALE)
        return sx, sy

    # ─── 트랙 미리 그리기 ─────────────────────────────────────────
    def _build_track_surfaces(self):
        hw = TRACK_WIDTH / 2
        pts = CENTERLINE_NP

        self._center_pts = [self._w2s(p[0], p[1]) for p in pts]

        self._outer_pts = []
        self._inner_pts = []
        n = len(pts)
        for i in range(n):
            nxt = (i + 1) % n
            dx = pts[nxt, 0] - pts[i, 0]
            dy = pts[nxt, 1] - pts[i, 1]
            length = max(np.hypot(dx, dy), 1e-6)
            nx, ny = -dy / length, dx / length
            ox = pts[i, 0] + nx * hw
            oy = pts[i, 1] + ny * hw
            ix = pts[i, 0] - nx * hw
            iy = pts[i, 1] - ny * hw
            self._outer_pts.append(self._w2s(ox, oy))
            self._inner_pts.append(self._w2s(ix, iy))

        # ── 시작/결승선 미리 계산 ─────────────────────────────────
        # FINISH_LINE_IDX 위치에서 트랙에 수직으로 선을 그음
        fi = FINISH_LINE_IDX
        self._finish_line = (self._outer_pts[fi], self._inner_pts[fi])

        # 시작 방향 화살표
        fx, fy = CENTERLINE_NP[fi]
        theta = get_centerline_direction(fi)
        arrow_len = TRACK_WIDTH * 1.5
        ax = fx + arrow_len * np.cos(theta)
        ay = fy + arrow_len * np.sin(theta)
        self._start_arrow = (self._w2s(fx, fy), self._w2s(ax, ay))

    # ─── 결승선 그리기 ────────────────────────────────────────────
    def _draw_finish_line(self):
        p1, p2 = self._finish_line
        # 체커보드 패턴 결승선
        pygame.draw.line(self.screen, FINISH_COLOR, p1, p2, 4)

        # "START/FINISH" 라벨
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        label = self.font_sm.render("START/FINISH", True, FINISH_COLOR)
        lw, lh = label.get_size()
        self.screen.blit(label, (mid_x - lw // 2 - 30, mid_y - lh - 6))

        # 진행 방향 화살표
        p_start, p_end = self._start_arrow
        pygame.draw.line(self.screen, START_ARROW, p_start, p_end, 3)
        # 화살촉
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        length = max(np.hypot(dx, dy), 1)
        dx, dy = dx / length, dy / length
        arrow_size = 10
        tip = p_end
        left  = (int(tip[0] - arrow_size*dx + arrow_size*0.5*dy),
                 int(tip[1] - arrow_size*dy - arrow_size*0.5*dx))
        right = (int(tip[0] - arrow_size*dx - arrow_size*0.5*dy),
                 int(tip[1] - arrow_size*dy + arrow_size*0.5*dx))
        pygame.draw.polygon(self.screen, START_ARROW, [tip, left, right])

    # ─── 메인 드로우 ──────────────────────────────────────────────
    def draw(self, car, left_dist, right_dist, front_dist, info: dict, fps: int = 60):
        self.screen.fill(DARK_BG)

        # 트랙 채우기
        if len(self._outer_pts) > 2:
            pygame.draw.polygon(self.screen, (40, 42, 48),
                                self._outer_pts + self._inner_pts[::-1])
            pygame.draw.lines(self.screen, TRACK_EDGE, True, self._outer_pts, 2)
            pygame.draw.lines(self.screen, TRACK_EDGE, True, self._inner_pts, 2)

        # 중심선 점선
        for i in range(0, len(self._center_pts), 8):
            pygame.draw.circle(self.screen, (60, 60, 70),
                               self._center_pts[i], 2)

        # ── 시작/결승선 그리기 ────────────────────────────────────
        self._draw_finish_line()

        # 주행 궤적
        cx, cy = self._w2s(car.x, car.y)
        self.trail.append((cx, cy))
        if len(self.trail) > 2000:
            self.trail.pop(0)
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (40, 180, 80), False,
                              self.trail, 2)

        # 센서 레이
        self._draw_sensor(car, car.theta + _SA, left_dist, SENSOR_CLR)
        self._draw_sensor(car, car.theta - _SA, right_dist, (255, 150, 0))
        self._draw_sensor(car, car.theta, front_dist, (180, 100, 255))

        # 차량
        pygame.draw.circle(self.screen, CAR_COLOR, (cx, cy), 7)
        arrow_len = 14
        ax = cx + int(arrow_len * np.cos(car.theta))
        ay = cy - int(arrow_len * np.sin(car.theta))
        pygame.draw.line(self.screen, (255, 180, 180), (cx, cy), (ax, ay), 3)

        # ── 에피소드 결과 표시 ────────────────────────────────────
        result = info.get("result", "ongoing")
        if result == "success":
            self._draw_result_overlay("🏆  LAP COMPLETE!  SUCCESS", (50, 220, 50))
        elif result == "fail":
            reason = "REVERSE!" if info.get("going_reverse") else "OFF TRACK!"
            self._draw_result_overlay(f"✗  {reason}  FAIL", (220, 50, 50))

        # HUD
        self._draw_hud(info, left_dist, right_dist)
        self._draw_sensor_bars(left_dist, right_dist, info.get("sensor_state", 0))

        pygame.display.flip()
        if fps > 0:
            self.clock.tick(fps)

    def _draw_result_overlay(self, text, color):
        """화면 중앙 상단에 에피소드 결과 오버레이."""
        surf = self.font_lg.render(text, True, color)
        w, h = surf.get_size()
        bg = pygame.Surface((w + 20, h + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self.screen.blit(bg, (WIDTH // 2 - w // 2 - 10, 8))
        self.screen.blit(surf, (WIDTH // 2 - w // 2, 13))

    def _draw_sensor(self, car, angle, dist, color):
        sx, sy = self._w2s(car.x, car.y)
        ex = car.x + dist * np.cos(angle)
        ey = car.y + dist * np.sin(angle)
        ex_s, ey_s = self._w2s(ex, ey)
        pygame.draw.line(self.screen, color, (sx, sy), (ex_s, ey_s), 1)
        pygame.draw.circle(self.screen, color, (ex_s, ey_s), 4)

    # ─── HUD 패널 ─────────────────────────────────────────────────
    def _draw_hud(self, info: dict, left_dist, right_dist):
        panel_w, panel_h = 300, 510
        panel_x, panel_y = WIDTH - panel_w - 12, 12

        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((18, 20, 30, 200))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (60, 70, 100),
                         (panel_x, panel_y, panel_w, panel_h), 1)

        def put(text, x, y, font=None, color=WHITE):
            f = font or self.font_md
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y))

        px, py = panel_x + 12, panel_y + 10
        put("═══ F1 RL AGENT ═══", px, py, self.font_lg, CYAN)
        py += 30

        result = info.get("result", "ongoing")
        result_col = GREEN if result == "success" else (RED if result == "fail" else WHITE)

        rows = [
            ("Episode",    info.get("episode", 0),                    WHITE),
            ("Step",       info.get("step", 0),                       WHITE),
            ("Epsilon",    f"{info.get('epsilon', 1.0):.4f}",         YELLOW),
            ("Action",     info.get("action_name", "—"),              GREEN),
            ("Reward",     f"{info.get('reward', 0):+.2f}",
             GREEN if info.get("reward", 0) >= 0 else RED),
            ("TotalRew",   f"{info.get('total_reward', 0):.2f}",      CYAN),
            ("Speed",      f"{info.get('speed', 0):.2f}",             WHITE),
            ("DistCenter", f"{info.get('dist_center', 0):.2f}",       WHITE),
            ("WpIdx",      info.get("waypoint_idx", 0),               WHITE),
            ("LapStarted", str(info.get("lap_started", False)),       YELLOW),
            ("InFinZone",  str(info.get("in_start_zone", False)),     (180, 180, 255)),
            ("RESULT",     result.upper(),                            result_col),
            ("RenderEvery", info.get("render_every", 1),                   (160,160,255)),
        ]
        for label, val, col in rows:
            put(f"{label:<12}: {val}", px, py, self.font_md, col)
            py += 22

        py += 5
        put("Q-Table:", px, py, self.font_sm, CYAN)
        py += 16
        qt = info.get("q_table", None)
        if qt is not None:
            side_names = ["L>R", "L≤R"]
            zone_names = ["FAR", "MID", "NEAR"]
            zone_colors = [(120, 200, 120), (220, 180, 80), (220, 100, 100)]
            for si, sname in enumerate(side_names):
                for fi, (fname, fcol) in enumerate(zip(zone_names, zone_colors)):
                    row_str = "  ".join(f"{v:+.1f}" for v in qt[si, fi])
                    put(f"{sname}/{fname}: {row_str}", px, py, self.font_sm, fcol)
                    py += 14

    def _draw_sensor_bars(self, left_dist, right_dist, sensor_state):
        bar_y = HEIGHT - 55
        bar_h = 30
        bar_max_w = 180

        ratio_l = left_dist / SENSOR_MAX_DIST
        color_l = (int(255 * (1 - ratio_l)), int(255 * ratio_l), 60)
        pygame.draw.rect(self.screen, (30, 30, 40), (20, bar_y, bar_max_w, bar_h))
        pygame.draw.rect(self.screen, color_l,
                         (20, bar_y, int(bar_max_w * ratio_l), bar_h))
        pygame.draw.rect(self.screen, (100, 100, 120), (20, bar_y, bar_max_w, bar_h), 1)
        lbl = self.font_sm.render(f"L-Sensor: {left_dist:.1f}m", True, WHITE)
        self.screen.blit(lbl, (22, bar_y + 8))

        ratio_r = right_dist / SENSOR_MAX_DIST
        color_r = (int(255 * (1 - ratio_r)), int(255 * ratio_r), 100)
        rx = 220
        pygame.draw.rect(self.screen, (30, 30, 40), (rx, bar_y, bar_max_w, bar_h))
        pygame.draw.rect(self.screen, color_r,
                         (rx, bar_y, int(bar_max_w * ratio_r), bar_h))
        pygame.draw.rect(self.screen, (100, 100, 120), (rx, bar_y, bar_max_w, bar_h), 1)
        lbl = self.font_sm.render(f"R-Sensor: {right_dist:.1f}m", True, WHITE)
        self.screen.blit(lbl, (rx + 2, bar_y + 8))

        # unpack tuple state
        side, front_zone = sensor_state if isinstance(sensor_state, tuple) else (sensor_state, 0)

        side_strs = ["L>R → RIGHT", "L≤R → LEFT"]
        zone_strs = ["FAR", "MID", "NEAR"]
        zone_cols = [(120, 200, 120), (220, 180, 80), (220, 100, 100)]

        side_text = self.font_md.render(f"Side: {side_strs[side]}", True, YELLOW)
        zone_text = self.font_md.render(f"Front: {zone_strs[front_zone]}", True, zone_cols[front_zone])
        self.screen.blit(side_text, (430, bar_y + 2))
        self.screen.blit(zone_text, (430, bar_y + 22))
