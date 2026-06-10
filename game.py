"""
game.py
=======
NeuroGlove Rehabilitation Game — Wire Loop
==========================================

A Pygame-based game where the patient guides a ring along a curved wire
without touching it, using hand opening/closing to control the ring's
position.

Controls (keyboard fallback for testing without glove)
------------------------------------------------------
    Arrow Up / Down   — manual ring control (if USE_REAL_GLOVE=False)
    R                 — restart current level
    ESC               — quit

Configuration
-------------
    USE_REAL_GLOVE   : bool   — True  = BLE glove
                                False = simulated sensor (dev/demo mode)
    DIFFICULTY       : int    — 1 (gentle, mostly straight wire)
                                2 (precise control, peaks and valleys)

Run
---
    python game.py
"""

import sys
import math
import time
import random
import threading

import pygame

# Import our sensor layer — works with real glove or simulation
try:
    from sensor_processing import SensorProcessor, SimulatedSensorProcessor
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("[game] sensor_processing.py not found — keyboard-only mode.")

# ──────────────────────────────────────────────────────────────────────────────
# User configuration
# ──────────────────────────────────────────────────────────────────────────────

USE_REAL_GLOVE = False    # Set True to connect via BLE; False uses simulation
DIFFICULTY     = 1        # 1 = easy (straight), 2 = hard (peaks)

# ──────────────────────────────────────────────────────────────────────────────
# Display constants
# ──────────────────────────────────────────────────────────────────────────────

SCREEN_W, SCREEN_H = 900, 600
FPS                = 60

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette — clinical-calm with a warm accent
# ──────────────────────────────────────────────────────────────────────────────

C_BG          = (15,  20,  30)    # deep navy
C_WIRE        = (80, 180, 220)    # steel blue
C_WIRE_SHADOW = (30,  70,  90)    # dim shadow wire
C_RING        = (255, 200,  60)   # amber ring
C_RING_DANGER = (255,  80,  60)   # ring flashes red on collision
C_TEXT        = (220, 230, 240)
C_TEXT_DIM    = (100, 120, 140)
C_ACCENT      = (50, 200, 150)    # teal — success / calibration highlights
C_PANEL       = (25,  32,  48)
C_BAR_BG      = (40,  50,  70)
C_BAR_FILL    = (50, 200, 150)

WIRE_THICKNESS  = 18   # pixels — the gap the ring must pass through
RING_RADIUS     = 8

# ──────────────────────────────────────────────────────────────────────────────
# Wire path generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_wire_path(difficulty: int, screen_w: int, screen_h: int):
    """
    Returns a list of (x, y) centre-line points that define the wire.
    The ring travels along this path; the player must keep it centred.

    difficulty=1: gentle curves, stays near vertical centre
    difficulty=2: pronounced peaks and valleys, wider excursions
    """
    points = []
    cx = 0
    cy = screen_h // 2

    amplitude = 80  if difficulty == 2 else 30
    frequency = 0.012 if difficulty == 2 else 0.006
    noise_amp  = 20  if difficulty == 2 else 5

    random.seed(42)   # reproducible per difficulty level
    phase = random.uniform(0, 2 * math.pi)

    total_width = screen_w * 4   # the wire is 4× the screen width (scrolling)
    step = 4                     # pixels between control points

    for x in range(0, total_width, step):
        sine_y  = amplitude * math.sin(frequency * x + phase)
        noise_y = noise_amp * math.sin(frequency * 3.7 * x + 1.2)
        y = screen_h // 2 + sine_y + noise_y
        # Clamp well away from edges so the wire stays visible
        y = max(WIRE_THICKNESS * 2 + RING_RADIUS + 10,
                min(screen_h - WIRE_THICKNESS * 2 - RING_RADIUS - 10, y))
        points.append((x, int(y)))

    return points


# ──────────────────────────────────────────────────────────────────────────────
# Calibration screen
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationScreen:
    """
    Guides the patient through the 3-pose calibration sequence.
    Rendered inside the Pygame window.
    """

    POSES = [
        ("closed", "Make a tight FIST ✊",  C_RING_DANGER, -1),
        ("half",   "Half-open hand 🖐️ (halfway)", C_WIRE,  0),
        ("open",   "Fully OPEN hand 🖐️",    C_ACCENT,      +1),
    ]

    def __init__(self, processor, font_big, font_med, font_small):
        self.proc       = processor
        self.font_big   = font_big
        self.font_med   = font_med
        self.font_small = font_small
        self.pose_idx   = 0
        self.collecting = False
        self.done       = False
        self._progress  = 0.0   # 0.0 → 1.0 during collection

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if not self.collecting and not self.done:
                pose_key, _, _, _ = self.POSES[self.pose_idx]
                self.proc.record_calibration(pose_key)
                self.collecting = True
                self._progress  = 0.0

    def update(self):
        if self.collecting:
            status = self.proc.calibration_status()
            pose_key, _, _, _ = self.POSES[self.pose_idx]
            # Estimate progress from the internal accumulation buffer
            samples = self.proc._calib_accum[pose_key]
            from sensor_processing import CALIBRATION_SAMPLES
            self._progress = min(1.0, len(samples) / CALIBRATION_SAMPLES)

            if self._progress >= 1.0:
                self.collecting = False
                self.pose_idx  += 1
                if self.pose_idx >= len(self.POSES):
                    self.proc.finish_calibration()
                    self.done = True

    def draw(self, surface: pygame.Surface):
        surface.fill(C_BG)
        W, H = surface.get_size()

        # Title
        title = self.font_big.render("Calibration", True, C_TEXT)
        surface.blit(title, title.get_rect(center=(W // 2, 70)))

        if self.done:
            msg = self.font_med.render("✓  Calibration complete!", True, C_ACCENT)
            surface.blit(msg, msg.get_rect(center=(W // 2, H // 2)))
            hint = self.font_small.render("Press SPACE to start the game", True, C_TEXT_DIM)
            surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 50)))
            return

        # Current pose
        pose_key, label, colour, target = self.POSES[self.pose_idx]
        step_txt = self.font_small.render(
            f"Step {self.pose_idx + 1} / {len(self.POSES)}", True, C_TEXT_DIM)
        surface.blit(step_txt, step_txt.get_rect(center=(W // 2, 130)))

        pose_surf = self.font_med.render(label, True, colour)
        surface.blit(pose_surf, pose_surf.get_rect(center=(W // 2, H // 2 - 40)))

        # Progress bar
        bar_w, bar_h = 400, 22
        bar_x = W // 2 - bar_w // 2
        bar_y = H // 2 + 20
        pygame.draw.rect(surface, C_BAR_BG,   (bar_x, bar_y, bar_w, bar_h), border_radius=11)
        fill_w = int(bar_w * self._progress)
        if fill_w > 0:
            pygame.draw.rect(surface, C_BAR_FILL, (bar_x, bar_y, fill_w, bar_h), border_radius=11)
        pygame.draw.rect(surface, C_TEXT_DIM, (bar_x, bar_y, bar_w, bar_h), 2, border_radius=11)

        if self.collecting:
            hint_txt = "Hold position…"
        else:
            hint_txt = "Hold the position and press SPACE to record"
        hint = self.font_small.render(hint_txt, True, C_TEXT_DIM)
        surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 65)))

        # Aperture reference illustration
        self._draw_aperture_scale(surface, W // 2, H - 100, target)

    def _draw_aperture_scale(self, surface, cx, cy, target_ap):
        """Simple visual showing where on the [-1, +1] scale we aim."""
        bar_w = 300
        pygame.draw.rect(surface, C_BAR_BG,
                         (cx - bar_w // 2, cy - 8, bar_w, 16), border_radius=8)
        # Tick marks
        for ap, lbl in [(-1, "Closed"), (0, "Half"), (1, "Open")]:
            tx = cx + int(ap * bar_w // 2)
            pygame.draw.line(surface, C_TEXT_DIM, (tx, cy - 12), (tx, cy + 12), 2)
            t = self.font_small.render(lbl, True, C_TEXT_DIM)
            surface.blit(t, t.get_rect(center=(tx, cy + 26)))
        # Target marker
        tx = cx + int(target_ap * bar_w // 2)
        pygame.draw.circle(surface, C_RING, (tx, cy), 10)


# ──────────────────────────────────────────────────────────────────────────────
# Ring (the player object)
# ──────────────────────────────────────────────────────────────────────────────

class Ring:
    def __init__(self):
        self.path_t    = 0.0    # normalised position along path (0→1)
        self.perp_off  = 0.0    # perpendicular offset from wire centre [-1, +1]
        self.touching  = False
        self._flash_timer = 0

    def update(self, aperture: float, dt: float,
               path_speed: float, wire_points, scroll_x: int):
        """
        aperture  : [-1, +1] from SensorProcessor
        path_speed: pixels per second the wire scrolls
        """
        # Perpendicular offset directly controlled by hand aperture
        # aperture -1 (fist)  → top of the wire gap
        # aperture +1 (open)  → bottom of the wire gap
        target_off = aperture                          # already in [-1, +1]
        alpha      = min(1.0, dt * 12)                # smooth lag
        self.perp_off = self.perp_off + alpha * (target_off - self.perp_off)

        # Check collision: ring must stay within ±1.0 of centre
        self.touching = abs(self.perp_off) > 0.85

        if self.touching:
            self._flash_timer = max(0, self._flash_timer)
            self._flash_timer  = 0.25   # flash for 0.25 s

        self._flash_timer = max(0.0, self._flash_timer - dt)

    def screen_pos(self, wire_points, scroll_x: int, screen_w: int):
        """
        Returns the (x, y) pixel position of the ring on screen.
        The ring stays at a fixed x = 20% of screen width;
        the wire scrolls past it.
        """
        ring_screen_x = int(screen_w * 0.20)
        world_x = scroll_x + ring_screen_x

        # Find the two wire points bracketing world_x and interpolate y
        centre_y = _interpolate_wire_y(wire_points, world_x)

        # Perpendicular offset (vertical, since wire is mostly horizontal)
        offset_px = int(self.perp_off * (WIRE_THICKNESS - RING_RADIUS - 2))
        return ring_screen_x, centre_y + offset_px

    def colour(self):
        if self._flash_timer > 0:
            return C_RING_DANGER
        return C_RING


# ──────────────────────────────────────────────────────────────────────────────
# Game scene
# ──────────────────────────────────────────────────────────────────────────────

class GameScene:
    """Main gameplay loop."""

    SCROLL_SPEED_PX_S = 80   # pixels per second — wire scrolls left

    def __init__(self, processor, difficulty, font_big, font_med, font_small):
        self.proc       = processor
        self.diff       = difficulty
        self.font_big   = font_big
        self.font_med   = font_med
        self.font_small = font_small

        self.wire_points = generate_wire_path(difficulty, SCREEN_W, SCREEN_H)
        self.ring        = Ring()
        self.scroll_x    = 0.0
        self.elapsed     = 0.0
        self.collisions  = 0
        self.game_over   = False
        self._kb_offset  = 0.0   # keyboard fallback control

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.__init__(self.proc, self.diff,
                              self.font_big, self.font_med, self.font_small)

    def update(self, dt: float, keys):
        if self.game_over:
            return

        self.elapsed += dt

        # --- Determine aperture ---
        if USE_REAL_GLOVE or SENSOR_AVAILABLE:
            state    = self.proc.get_current_state()
            aperture = state.aperture if state.calibrated else 0.0
        else:
            # Keyboard fallback: Up arrow → open (+1), Down → closed (-1)
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

        # Update ring
        self.ring.update(aperture, dt, self.SCROLL_SPEED_PX_S,
                         self.wire_points, int(self.scroll_x))
        if self.ring.touching:
            self.collisions += 1

    def draw(self, surface: pygame.Surface):
        surface.fill(C_BG)
        W, H = surface.get_size()
        sx = int(self.scroll_x)

        # --- Draw wire (shadow + main) ---
        self._draw_wire(surface, sx, shadow=True)
        self._draw_wire(surface, sx, shadow=False)

        # --- Draw ring ---
        rx, ry = self.ring.screen_pos(self.wire_points, sx, W)
        pygame.draw.circle(surface, self.ring.colour(), (rx, ry), RING_RADIUS, 3)

        # --- HUD ---
        self._draw_hud(surface, W, H)

        if self.game_over:
            self._draw_finish(surface, W, H)

    def _draw_wire(self, surface, scroll_x, shadow=False):
        """Draw the upper and lower wire edges as polylines."""
        colour    = C_WIRE_SHADOW if shadow else C_WIRE
        thickness = 3 if not shadow else 1
        offset_y  = 4 if shadow else 0

        pts_top, pts_bot = [], []
        W = SCREEN_W

        for wx, wy in self.wire_points:
            sx = wx - scroll_x
            if -20 <= sx <= W + 20:
                pts_top.append((sx, wy - WIRE_THICKNESS + offset_y))
                pts_bot.append((sx, wy + WIRE_THICKNESS + offset_y))

        if len(pts_top) >= 2:
            pygame.draw.lines(surface, colour, False, pts_top, thickness)
            pygame.draw.lines(surface, colour, False, pts_bot, thickness)

    def _draw_hud(self, surface, W, H):
        # Time
        t_surf = self.font_small.render(f"Time  {self.elapsed:.1f}s", True, C_TEXT)
        surface.blit(t_surf, (20, 16))

        # Collisions
        col_colour = C_RING_DANGER if self.collisions > 0 else C_ACCENT
        c_surf = self.font_small.render(f"Touches  {self.collisions}", True, col_colour)
        surface.blit(c_surf, (20, 44))

        # Difficulty
        d_surf = self.font_small.render(
            f"Level {self.diff}{'  ★' * self.diff}", True, C_TEXT_DIM)
        surface.blit(d_surf, (W - d_surf.get_width() - 20, 16))

        # Aperture bar (right side)
        if SENSOR_AVAILABLE:
            state = self.proc.get_current_state()
            ap    = state.aperture
        else:
            ap = self._kb_offset

        bar_h  = 120
        bar_w  = 16
        bx     = W - 36
        by     = H // 2 - bar_h // 2
        pygame.draw.rect(surface, C_BAR_BG, (bx, by, bar_w, bar_h), border_radius=8)
        # Fill from centre outwards
        centre  = by + bar_h // 2
        fill_px = int(abs(ap) * bar_h // 2)
        if ap >= 0:
            pygame.draw.rect(surface, C_BAR_FILL,
                             (bx, centre - fill_px, bar_w, fill_px), border_radius=6)
        else:
            pygame.draw.rect(surface, C_BAR_FILL,
                             (bx, centre, bar_w, fill_px), border_radius=6)
        pygame.draw.rect(surface, C_TEXT_DIM, (bx, by, bar_w, bar_h), 2, border_radius=8)

        lbl = self.font_small.render("grip", True, C_TEXT_DIM)
        surface.blit(lbl, (bx - 2, by - 22))

    def _draw_finish(self, surface, W, H):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((10, 15, 25, 180))
        surface.blit(overlay, (0, 0))

        title = self.font_big.render("Level Complete!", True, C_ACCENT)
        surface.blit(title, title.get_rect(center=(W // 2, H // 2 - 60)))

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
    """Binary-search the wire_points list and linearly interpolate y at world_x."""
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

    # Fonts
    font_big   = pygame.font.SysFont("segoeui", 42, bold=True)
    font_med   = pygame.font.SysFont("segoeui", 28)
    font_small = pygame.font.SysFont("segoeui", 18)

    # ── Start sensor processor ──────────────────────────────────────────────
    if SENSOR_AVAILABLE:
        if USE_REAL_GLOVE:
            processor = SensorProcessor()
        else:
            processor = SimulatedSensorProcessor()
        processor.start()
    else:
        processor = None

    # ── Scene management ────────────────────────────────────────────────────
    # Scenes: "waiting" → "calibration" → "game"
    if SENSOR_AVAILABLE and not USE_REAL_GLOVE:
        # Simulation auto-calibrates — skip straight to game
        scene_name = "game"
        current_scene = GameScene(processor, DIFFICULTY, font_big, font_med, font_small)
    elif SENSOR_AVAILABLE and USE_REAL_GLOVE:
        scene_name    = "calibration"
        current_scene = CalibrationScreen(processor, font_big, font_med, font_small)
    else:
        # No sensor module at all — keyboard only game
        scene_name    = "game"
        current_scene = GameScene(None, DIFFICULTY, font_big, font_med, font_small)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0   # seconds since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Scene transitions
            if scene_name == "calibration":
                current_scene.handle_event(event)
                # Move to game when calibration is done and user presses SPACE
                if (current_scene.done and
                        event.type == pygame.KEYDOWN and
                        event.key == pygame.K_SPACE):
                    scene_name    = "game"
                    current_scene = GameScene(processor, DIFFICULTY,
                                              font_big, font_med, font_small)
            elif scene_name == "game":
                current_scene.handle_event(event)

        keys = pygame.key.get_pressed()

        # Update
        if scene_name == "calibration":
            current_scene.update()
        elif scene_name == "game":
            current_scene.update(dt, keys)

        # Draw
        current_scene.draw(screen)
        pygame.display.flip()

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if processor:
        processor.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()