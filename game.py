"""
game.py
=======
NeuroGlove Rehabilitation Game — Wire Loop

Controls (keyboard fallback for testing without glove)
------------------------------------------------------
    Arrow Up / Down   — manual ring control (if USE_REAL_GLOVE=False)
    R                 — restart current level
    ESC               — quit
"""

import sys
import math
import time
import random
import threading

import pygame

try:
    from data_preprocessing import SensorProcessor, SimulatedSensorProcessor
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("[game] data_preprocessing.py not found — keyboard-only mode.")

# ──────────────────────────────────────────────────────────────────────────────
# User configuration
# ──────────────────────────────────────────────────────────────────────────────

USE_REAL_GLOVE = True

# ──────────────────────────────────────────────────────────────────────────────
# Display constants
# ──────────────────────────────────────────────────────────────────────────────

SCREEN_W, SCREEN_H = 900, 600
FPS                = 60

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────

C_BG          = (15,  20,  30)
C_WIRE        = (80, 180, 220)
C_WIRE_SHADOW = (30,  70,  90)
C_RING        = (255, 200,  60)
C_RING_DANGER = (255,  80,  60)
C_TEXT        = (220, 230, 240)
C_TEXT_DIM    = (100, 120, 140)
C_ACCENT      = (50, 200, 150)
C_PANEL       = (25,  32,  48)
C_BAR_BG      = (40,  50,  70)
C_BAR_FILL    = (50, 200, 150)
C_OBSTACLE    = (220,  80,  60)

WIRE_THICKNESS  = 70
RING_RADIUS     = 8
SENSITIVITY     = 2.0
LEVEL_TIME_LIMIT = 60.0   # seconds — level 1 time limit
OBSTACLE_RADIUS  = 10     # pixels — level 2 obstacle size

# ──────────────────────────────────────────────────────────────────────────────
# Wire path generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_wire_path(difficulty: int, screen_w: int, screen_h: int):
    amplitude = 80  if difficulty == 2 else 30
    frequency = 0.012 if difficulty == 2 else 0.006
    noise_amp  = 20  if difficulty == 2 else 5

    random.seed(42)
    phase = random.uniform(0, 2 * math.pi)

    total_width = screen_w * 4
    step = 4
    points = []

    for x in range(0, total_width, step):
        sine_y  = amplitude * math.sin(frequency * x + phase)
        noise_y = noise_amp * math.sin(frequency * 3.7 * x + 1.2)
        y = screen_h // 2 + sine_y + noise_y
        y = max(WIRE_THICKNESS * 2 + RING_RADIUS + 10,
                min(screen_h - WIRE_THICKNESS * 2 - RING_RADIUS - 10, y))
        points.append((x, int(y)))

    return points


def generate_obstacles(wire_points, count=14):
    """Return list of (world_x, world_y) obstacle positions for level 2."""
    obstacles = []
    total_x = wire_points[-1][0]
    spacing = (total_x - 600) // count
    rng = random.Random(77)
    for i in range(count):
        wx = 500 + i * spacing + rng.randint(-spacing // 4, spacing // 4)
        cy = _interpolate_wire_y(wire_points, wx)
        max_off = WIRE_THICKNESS - OBSTACLE_RADIUS - 4
        wy = cy + rng.randint(-max_off, max_off)
        obstacles.append((wx, wy))
    return obstacles


# ──────────────────────────────────────────────────────────────────────────────
# Calibration screen
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationScreen:
    POSES = [
        ("closed", "Make a tight FIST",        C_RING_DANGER, -1),
        ("half",   "Half-open hand (halfway)", C_WIRE,         0),
        ("open",   "Fully OPEN hand",          C_ACCENT,      +1),
    ]

    def __init__(self, processor, font_big, font_med, font_small):
        self.proc       = processor
        self.font_big   = font_big
        self.font_med   = font_med
        self.font_small = font_small
        self.pose_idx   = 0
        self.collecting = False
        self.done       = False
        self._progress  = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if not self.collecting and not self.done:
                pose_key, _, _, _ = self.POSES[self.pose_idx]
                self.proc.record_calibration(pose_key)
                self.collecting = True
                self._progress  = 0.0

    def update(self):
        if self.collecting:
            pose_key, _, _, _ = self.POSES[self.pose_idx]
            samples = self.proc._calib_accum[pose_key]
            from data_preprocessing import CALIBRATION_SAMPLES
            self._progress = min(1.0, len(samples) / CALIBRATION_SAMPLES)
            if self._progress >= 1.0:
                self.collecting = False
                self.pose_idx  += 1
                if self.pose_idx >= len(self.POSES):
                    self.proc.finish_calibration()
                    self.done = True

    def draw(self, surface):
        surface.fill(C_BG)
        W, H = surface.get_size()

        title = self.font_big.render("Calibration", True, C_TEXT)
        surface.blit(title, title.get_rect(center=(W // 2, 70)))

        if self.done:
            msg  = self.font_med.render("Calibration complete!", True, C_ACCENT)
            hint = self.font_small.render("Press SPACE to continue", True, C_TEXT_DIM)
            surface.blit(msg,  msg.get_rect(center=(W // 2, H // 2)))
            surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 50)))
            return

        pose_key, label, colour, target = self.POSES[self.pose_idx]
        step_txt = self.font_small.render(
            f"Step {self.pose_idx + 1} / {len(self.POSES)}", True, C_TEXT_DIM)
        surface.blit(step_txt, step_txt.get_rect(center=(W // 2, 130)))

        pose_surf = self.font_med.render(label, True, colour)
        surface.blit(pose_surf, pose_surf.get_rect(center=(W // 2, H // 2 - 40)))

        bar_w, bar_h = 400, 22
        bar_x = W // 2 - bar_w // 2
        bar_y = H // 2 + 20
        pygame.draw.rect(surface, C_BAR_BG,   (bar_x, bar_y, bar_w, bar_h), border_radius=11)
        fill_w = int(bar_w * self._progress)
        if fill_w > 0:
            pygame.draw.rect(surface, C_BAR_FILL, (bar_x, bar_y, fill_w, bar_h), border_radius=11)
        pygame.draw.rect(surface, C_TEXT_DIM, (bar_x, bar_y, bar_w, bar_h), 2, border_radius=11)

        hint_txt = "Hold position…" if self.collecting else "Hold the position and press SPACE to record"
        hint = self.font_small.render(hint_txt, True, C_TEXT_DIM)
        surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 65)))

        self._draw_aperture_scale(surface, W // 2, H - 100, target)

    def _draw_aperture_scale(self, surface, cx, cy, target_ap):
        bar_w = 300
        pygame.draw.rect(surface, C_BAR_BG, (cx - bar_w // 2, cy - 8, bar_w, 16), border_radius=8)
        for ap, lbl in [(-1, "Closed"), (0, "Half"), (1, "Open")]:
            tx = cx + int(ap * bar_w // 2)
            pygame.draw.line(surface, C_TEXT_DIM, (tx, cy - 12), (tx, cy + 12), 2)
            t = self.font_small.render(lbl, True, C_TEXT_DIM)
            surface.blit(t, t.get_rect(center=(tx, cy + 26)))
        tx = cx + int(target_ap * bar_w // 2)
        pygame.draw.circle(surface, C_RING, (tx, cy), 10)


# ──────────────────────────────────────────────────────────────────────────────
# Level select screen
# ──────────────────────────────────────────────────────────────────────────────

class LevelSelectScreen:
    def __init__(self, font_big, font_med, font_small):
        self.font_big   = font_big
        self.font_med   = font_med
        self.font_small = font_small
        self.selected   = None   # set to 1 or 2 when player chooses

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.selected = 1
            elif event.key == pygame.K_2:
                self.selected = 2

    def update(self):
        pass

    def draw(self, surface):
        surface.fill(C_BG)
        W, H = surface.get_size()

        title = self.font_big.render("Choose Level", True, C_TEXT)
        surface.blit(title, title.get_rect(center=(W // 2, H // 2 - 120)))

        # Level 1 card
        l1_title = self.font_med.render("1  —  Level 1", True, C_ACCENT)
        l1_desc  = self.font_small.render("Wire loop  |  60 seconds  |  No obstacles", True, C_TEXT_DIM)
        surface.blit(l1_title, l1_title.get_rect(center=(W // 2, H // 2 - 20)))
        surface.blit(l1_desc,  l1_desc.get_rect(center=(W // 2, H // 2 + 18)))

        # Level 2 card
        l2_title = self.font_med.render("2  —  Level 2", True, C_WIRE)
        l2_desc  = self.font_small.render("Wire loop  |  No time limit  |  Obstacles inside the wire", True, C_TEXT_DIM)
        surface.blit(l2_title, l2_title.get_rect(center=(W // 2, H // 2 + 80)))
        surface.blit(l2_desc,  l2_desc.get_rect(center=(W // 2, H // 2 + 118)))

        hint = self.font_small.render("Press  1  or  2  to start", True, C_TEXT_DIM)
        surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 180)))


# ──────────────────────────────────────────────────────────────────────────────
# Ring (the player object)
# ──────────────────────────────────────────────────────────────────────────────

class Ring:
    def __init__(self):
        self.perp_off     = 0.0
        self.touching     = False
        self._flash_timer = 0.0

    def update(self, aperture: float, dt: float):
        target_off    = max(-1.0, min(1.0, aperture * SENSITIVITY))
        alpha         = min(1.0, dt * 60)
        self.perp_off = self.perp_off + alpha * (target_off - self.perp_off)
        self.touching = abs(self.perp_off) > 0.85
        if self.touching:
            self._flash_timer = 0.25
        self._flash_timer = max(0.0, self._flash_timer - dt)

    def flash(self):
        self._flash_timer = 0.25

    def screen_pos(self, wire_points, scroll_x: int, screen_w: int):
        ring_screen_x = int(screen_w * 0.20)
        world_x       = scroll_x + ring_screen_x
        centre_y      = _interpolate_wire_y(wire_points, world_x)
        offset_px     = int(self.perp_off * (WIRE_THICKNESS - RING_RADIUS - 2))
        return ring_screen_x, centre_y + offset_px

    def colour(self):
        return C_RING_DANGER if self._flash_timer > 0 else C_RING


# ──────────────────────────────────────────────────────────────────────────────
# Game scene
# ──────────────────────────────────────────────────────────────────────────────

class GameScene:
    SCROLL_SPEED_PX_S = 80

    def __init__(self, processor, difficulty, font_big, font_med, font_small):
        self.proc       = processor
        self.diff       = difficulty
        self.font_big   = font_big
        self.font_med   = font_med
        self.font_small = font_small

        self.wire_points   = generate_wire_path(difficulty, SCREEN_W, SCREEN_H)
        self.obstacles     = generate_obstacles(self.wire_points) if difficulty == 2 else []
        self.ring          = Ring()
        self.scroll_x      = 0.0
        self.elapsed       = 0.0
        self.collisions    = 0
        self.game_over     = False
        self._kb_offset    = 0.0
        self._was_touching = False

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.__init__(self.proc, self.diff,
                          self.font_big, self.font_med, self.font_small)

    def update(self, dt: float, keys):
        if self.game_over:
            return

        self.elapsed += dt

        # Level 1: 60-second time limit
        if self.diff == 1 and self.elapsed >= LEVEL_TIME_LIMIT:
            self.game_over = True
            return

        # Determine aperture
        if USE_REAL_GLOVE or SENSOR_AVAILABLE:
            state    = self.proc.get_current_state()
            aperture = state.aperture if state.calibrated else 0.0
        else:
            if keys[pygame.K_UP]:
                self._kb_offset = min(1.0, self._kb_offset + dt * 3)
            elif keys[pygame.K_DOWN]:
                self._kb_offset = max(-1.0, self._kb_offset - dt * 3)
            aperture = self._kb_offset

        # Scroll
        self.scroll_x += self.SCROLL_SPEED_PX_S * dt

        # End of wire
        if self.scroll_x + SCREEN_W >= self.wire_points[-1][0]:
            self.game_over = True
            return

        self.ring.update(aperture, dt)

        # Obstacle collision (level 2)
        obs_touching = False
        if self.diff == 2:
            rx, ry = self.ring.screen_pos(self.wire_points, int(self.scroll_x), SCREEN_W)
            for wx, wy in self.obstacles:
                sx = wx - int(self.scroll_x)
                if math.hypot(rx - sx, ry - wy) < RING_RADIUS + OBSTACLE_RADIUS:
                    obs_touching = True
                    self.ring.flash()
                    break

        touching_now = self.ring.touching or obs_touching
        if touching_now and not self._was_touching:
            self.collisions += 1
        self._was_touching = touching_now

    def draw(self, surface):
        surface.fill(C_BG)
        W, H = surface.get_size()
        sx = int(self.scroll_x)

        self._draw_wire(surface, sx, shadow=True)
        self._draw_wire(surface, sx, shadow=False)
        self._draw_obstacles(surface, sx)

        rx, ry = self.ring.screen_pos(self.wire_points, sx, W)
        pygame.draw.circle(surface, self.ring.colour(), (rx, ry), RING_RADIUS, 3)

        self._draw_hud(surface, W, H)

        if self.game_over:
            self._draw_finish(surface, W, H)

    def _draw_wire(self, surface, scroll_x, shadow=False):
        colour    = C_WIRE_SHADOW if shadow else C_WIRE
        thickness = 1 if shadow else 3
        offset_y  = 4 if shadow else 0
        pts_top, pts_bot = [], []

        for wx, wy in self.wire_points:
            sx = wx - scroll_x
            if -20 <= sx <= SCREEN_W + 20:
                pts_top.append((sx, wy - WIRE_THICKNESS + offset_y))
                pts_bot.append((sx, wy + WIRE_THICKNESS + offset_y))

        if len(pts_top) >= 2:
            pygame.draw.lines(surface, colour, False, pts_top, thickness)
            pygame.draw.lines(surface, colour, False, pts_bot, thickness)

    def _draw_obstacles(self, surface, scroll_x):
        for wx, wy in self.obstacles:
            sx = wx - scroll_x
            if -OBSTACLE_RADIUS <= sx <= SCREEN_W + OBSTACLE_RADIUS:
                pygame.draw.circle(surface, C_OBSTACLE, (sx, wy), OBSTACLE_RADIUS)
                pygame.draw.circle(surface, C_TEXT,     (sx, wy), OBSTACLE_RADIUS, 2)

    def _draw_hud(self, surface, W, H):
        # Timer — countdown for level 1, count-up for level 2
        if self.diff == 1:
            remaining   = max(0.0, LEVEL_TIME_LIMIT - self.elapsed)
            time_colour = C_RING_DANGER if remaining < 10 else C_TEXT
            t_surf = self.font_small.render(f"Time  {remaining:.1f}s", True, time_colour)
        else:
            t_surf = self.font_small.render(f"Time  {self.elapsed:.1f}s", True, C_TEXT)
        surface.blit(t_surf, (20, 16))

        col_colour = C_RING_DANGER if self.collisions > 0 else C_ACCENT
        c_surf = self.font_small.render(f"Touches  {self.collisions}", True, col_colour)
        surface.blit(c_surf, (20, 44))

        d_surf = self.font_small.render(
            f"Level {self.diff}{'  ★' * self.diff}", True, C_TEXT_DIM)
        surface.blit(d_surf, (W - d_surf.get_width() - 20, 16))

        # Aperture bar
        if SENSOR_AVAILABLE and self.proc:
            ap = self.proc.get_current_state().aperture
        else:
            ap = self._kb_offset

        bar_h = 120
        bar_w = 16
        bx    = W - 36
        by    = H // 2 - bar_h // 2
        pygame.draw.rect(surface, C_BAR_BG, (bx, by, bar_w, bar_h), border_radius=8)
        centre  = by + bar_h // 2
        fill_px = int(abs(ap) * bar_h // 2)
        if ap >= 0:
            pygame.draw.rect(surface, C_BAR_FILL, (bx, centre - fill_px, bar_w, fill_px), border_radius=6)
        else:
            pygame.draw.rect(surface, C_BAR_FILL, (bx, centre, bar_w, fill_px), border_radius=6)
        pygame.draw.rect(surface, C_TEXT_DIM, (bx, by, bar_w, bar_h), 2, border_radius=8)
        lbl = self.font_small.render("grip", True, C_TEXT_DIM)
        surface.blit(lbl, (bx - 2, by - 22))

    def _draw_finish(self, surface, W, H):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((10, 15, 25, 180))
        surface.blit(overlay, (0, 0))

        timed_out = self.diff == 1 and self.elapsed >= LEVEL_TIME_LIMIT
        title_txt = "Time's Up!" if timed_out else "Level Complete!"
        title_col = C_RING_DANGER if timed_out else C_ACCENT

        title = self.font_big.render(title_txt, True, title_col)
        surface.blit(title, title.get_rect(center=(W // 2, H // 2 - 60)))

        if self.diff == 1:
            t_txt = self.font_med.render(f"Completed in  {self.elapsed:.1f}s", True, C_TEXT)
        else:
            t_txt = self.font_med.render(f"Time: {self.elapsed:.2f}s", True, C_TEXT)
        surface.blit(t_txt, t_txt.get_rect(center=(W // 2, H // 2)))

        c_txt = self.font_med.render(f"Touches: {self.collisions}", True,
                                     C_RING_DANGER if self.collisions else C_ACCENT)
        surface.blit(c_txt, c_txt.get_rect(center=(W // 2, H // 2 + 50)))

        hint = self.font_small.render("Press R to restart  |  ESC to quit", True, C_TEXT_DIM)
        surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 110)))


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _interpolate_wire_y(wire_points, world_x: int) -> int:
    if world_x <= wire_points[0][0]:
        return wire_points[0][1]
    if world_x >= wire_points[-1][0]:
        return wire_points[-1][1]

    lo, hi = 0, len(wire_points) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if wire_points[mid][0] <= world_x:
            lo = mid
        else:
            hi = mid

    x0, y0 = wire_points[lo]
    x1, y1 = wire_points[hi]
    if x1 == x0:
        return y0
    t = (world_x - x0) / (x1 - x0)
    return int(y0 + t * (y1 - y0))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("NeuroGlove — Wire Loop Rehabilitation")
    clock = pygame.time.Clock()

    font_big   = pygame.font.SysFont("segoeui", 42, bold=True)
    font_med   = pygame.font.SysFont("segoeui", 28)
    font_small = pygame.font.SysFont("segoeui", 18)

    if SENSOR_AVAILABLE:
        processor = SensorProcessor() if USE_REAL_GLOVE else SimulatedSensorProcessor()
        processor.start()
    else:
        processor = None

    # Scene flow: calibration → level_select → game
    if SENSOR_AVAILABLE and USE_REAL_GLOVE:
        scene_name    = "calibration"
        current_scene = CalibrationScreen(processor, font_big, font_med, font_small)
    else:
        scene_name    = "level_select"
        current_scene = LevelSelectScreen(font_big, font_med, font_small)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            if scene_name == "calibration":
                current_scene.handle_event(event)
                if (current_scene.done and
                        event.type == pygame.KEYDOWN and
                        event.key == pygame.K_SPACE):
                    scene_name    = "level_select"
                    current_scene = LevelSelectScreen(font_big, font_med, font_small)

            elif scene_name == "level_select":
                current_scene.handle_event(event)
                if current_scene.selected is not None:
                    scene_name    = "game"
                    current_scene = GameScene(processor, current_scene.selected,
                                              font_big, font_med, font_small)

            elif scene_name == "game":
                current_scene.handle_event(event)

        keys = pygame.key.get_pressed()

        if scene_name == "calibration":
            current_scene.update()
        elif scene_name == "game":
            current_scene.update(dt, keys)

        current_scene.draw(screen)
        pygame.display.flip()

    if processor:
        processor.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
