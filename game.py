"""
game.py  -  Rehab Game
Pygame rehabilitation game with two modes:
  Level A : tremor assessment   (20 s, straight tunnel at max open or closed)
  Level B : range-of-motion     (30 s, tunnel moves through phases)
"""

import sys, math, time, csv, os
from datetime import datetime

import pygame
import numpy as np
from scipy.signal import find_peaks

try:
    from data_preprocessing import SensorProcessor, SimulatedSensorProcessor
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("[game] data_preprocessing not found - running in demo mode.")

# ── Configuration ──────────────────────────────────────────────────────────────
USE_REAL_GLOVE    = True

SCREEN_W, SCREEN_H = 1100, 700
FPS               = 60
AP_SCALE_PX       = 200    # pixels per aperture unit  (ap=1 -> 200 px above centre)
TUNNEL_HALF_AP    = 0.5    # tunnel half-width in aperture units
BALL_RADIUS       = 14

LEVEL_A_DURATION  = 20.0
LEVEL_B_DURATION  = 30.0

# Level B: (elapsed_s, tunnel_centre_aperture) waypoints, linearly interpolated
LEVEL_B_WAYPOINTS = [
    (0,  0.0),   # neutral
    (5,  0.0),   # stay neutral
    (10, 1.0),   # rise to max open
    (13, 0.0),   # return to neutral
    (18,-1.0),   # drop to max closed
    (20, 0.0),   # return to neutral
    (30, 0.0),   # hold neutral until end
]

TREMOR_LOW_HZ  = 4.0
TREMOR_HIGH_HZ = 7.0

FINGER_NAMES   = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
PCB_CH_LABELS  = ["ch4",  "ch5",   "ch6",    "ch9",  "ch13"]

# ── Colours ────────────────────────────────────────────────────────────────────
C_BG       = (15,  20,  30)
C_PANEL    = (25,  32,  48)
C_TEXT     = (220, 230, 240)
C_DIM      = (100, 120, 140)
C_ACCENT   = (50,  200, 150)
C_DANGER   = (255,  80,  60)
C_GOLD     = (255, 200,  60)
C_TUNNEL   = (80,  180, 220)
C_BTN      = (35,  48,  72)
C_BTN_HOV  = (55,  75, 115)
C_BTN_SEL  = (30,  110,  85)
C_BAR_BG   = (40,  50,  70)

# ── Utilities ──────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_waypoints(waypoints, t):
    if t <= waypoints[0][0]:
        return waypoints[0][1]
    if t >= waypoints[-1][0]:
        return waypoints[-1][1]
    for i in range(len(waypoints) - 1):
        t0, v0 = waypoints[i]
        t1, v1 = waypoints[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return lerp(v0, v1, frac)
    return waypoints[-1][1]

def ap_to_y(aperture, screen_h=SCREEN_H):
    """Aperture [-1..1] to screen y (positive ap = higher on screen)."""
    return int(screen_h // 2 - aperture * AP_SCALE_PX)

def export_csv(patient_name, level, finger_names, rows, timestamps):
    os.makedirs("recordings", exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe  = "".join(c for c in patient_name if c.isalnum() or c in "_-") or "patient"
    path  = f"recordings/{safe}_{level}_{ts}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + finger_names)
        for i, row in enumerate(rows):
            w.writerow([f"{timestamps[i]:.4f}"] + [f"{v:.5f}" for v in row])
    return path

def tremor_analysis(signal, sr=FPS, low=TREMOR_LOW_HZ, high=TREMOR_HIGH_HZ):
    """Return (n_tremors, dominant_hz) for the signal."""
    sig = np.array(signal, dtype=float)
    if len(sig) < 16:
        return 0, 0.0
    sig -= sig.mean()
    mag   = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
    mask  = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0, 0.0
    band_m = mag[mask]
    band_f = freqs[mask]
    thr    = np.max(mag) * 0.15
    peaks, _ = find_peaks(band_m, height=thr)
    dom_f  = float(band_f[np.argmax(band_m)])
    return len(peaks), dom_f


# ── Shared UI widgets ──────────────────────────────────────────────────────────

class Button:
    def __init__(self, rect, text, font, base_col=C_BTN, text_col=C_TEXT):
        self.rect     = pygame.Rect(rect)
        self.text     = text
        self.font     = font
        self.base_col = base_col
        self.text_col = text_col
        self._hov     = False
        self.selected = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hov = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface):
        col = C_BTN_SEL if self.selected else (C_BTN_HOV if self._hov else self.base_col)
        pygame.draw.rect(surface, col, self.rect, border_radius=10)
        pygame.draw.rect(surface, C_DIM, self.rect, 2, border_radius=10)
        t = self.font.render(self.text, True, self.text_col)
        surface.blit(t, t.get_rect(center=self.rect.center))


class TextInput:
    def __init__(self, rect, font, placeholder=""):
        self.rect        = pygame.Rect(rect)
        self.font        = font
        self.placeholder = placeholder
        self.text        = ""
        self.active      = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isprintable():
                self.text += event.unicode

    def draw(self, surface):
        border = C_ACCENT if self.active else C_DIM
        pygame.draw.rect(surface, C_PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=8)
        show  = self.text if self.text else self.placeholder
        color = C_TEXT if self.text else C_DIM
        t = self.font.render(show, True, color)
        surface.blit(t, (self.rect.x + 10, self.rect.centery - t.get_height() // 2))


# ── Hand diagram ───────────────────────────────────────────────────────────────

class HandDiagram:
    """Clickable finger-selection widget drawn as a simple hand."""

    _COLORS = [
        (255, 155,  60),  # thumb
        (80,  190, 255),  # index
        (80,  230, 130),  # middle
        (190,  90, 255),  # ring
        (255,  90, 170),  # pinky
    ]
    # (x offset from palm centre, finger rect height)
    _LAYOUT = [(-100, 75), (-55, 110), (-8, 130), (38, 108), (82, 80)]
    _FW     = 28   # finger width

    def __init__(self, cx, cy, font):
        self.cx       = cx
        self.cy       = cy   # top of palm
        self.font     = font
        self.selected = [True] * 5

        self._palm_rect  = pygame.Rect(cx - 108, cy, 216, 120)
        self._frects     = []
        self._dot_pos    = []
        for ox, fh in self._LAYOUT:
            fx = cx + ox - self._FW // 2
            fy = cy - fh
            self._frects.append(pygame.Rect(fx, fy, self._FW, fh))
            self._dot_pos.append((cx + ox, fy - 18))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, (dx, dy) in enumerate(self._dot_pos):
                if math.hypot(event.pos[0] - dx, event.pos[1] - dy) < 14:
                    self.selected[i] = not self.selected[i]

    def get_selected(self):
        return [i for i, s in enumerate(self.selected) if s]

    def draw(self, surface):
        pygame.draw.rect(surface, (55, 65, 85), self._palm_rect, border_radius=20)
        pygame.draw.rect(surface, C_DIM,        self._palm_rect, 2, border_radius=20)

        for i, (rect, (dx, dy), lbl) in enumerate(
                zip(self._frects, self._dot_pos, FINGER_NAMES)):
            col = self._COLORS[i] if self.selected[i] else (45, 55, 75)
            pygame.draw.rect(surface, col, rect, border_radius=10)
            pygame.draw.rect(surface, C_DIM, rect, 2, border_radius=10)

            dot_col = self._COLORS[i] if self.selected[i] else C_DIM
            pygame.draw.circle(surface, dot_col, (dx, dy), 12)
            pygame.draw.circle(surface, C_TEXT,   (dx, dy), 12, 2)
            if self.selected[i]:
                pygame.draw.circle(surface, C_BG if hasattr(pygame, '_dummy') else (15,20,30),
                                   (dx, dy), 5)

            lt = self.font.render(lbl, True, C_TEXT if self.selected[i] else C_DIM)
            surface.blit(lt, lt.get_rect(center=(dx, dy - 26)))


# ── Home screen ────────────────────────────────────────────────────────────────

class HomeScreen:
    def __init__(self, fonts):
        fb, fm, fs = fonts
        self.fonts       = fonts
        self.name_input  = TextInput((100, 230, 300, 44), fm, "Enter patient name")
        self.btn_records = Button((100, 130, 450, 60), "Patient Records", fs)
        self.btn_a       = Button((100,  320, 215, 76), "Game A", fm)
        self.btn_b       = Button((335, 320, 215, 76), "Game B", fm)
        self.hand        = HandDiagram(820, 390, fs)
        self.next_scene  = None

    def handle_event(self, event):
        self.name_input.handle_event(event)
        self.hand.handle_event(event)
        self.btn_records.handle_event(event)
        if self.btn_a.handle_event(event):
            self.next_scene = "game_a"
        if self.btn_b.handle_event(event):
            self.next_scene = "game_b"

    def update(self):
        pass

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        # Vertical divider
        pygame.draw.line(surface, C_DIM, (590, 40), (590, H - 40), 1)

        # Title
        title = fb.render("Rehab Game", True, C_ACCENT)
        surface.blit(title, title.get_rect(center=(295, 62)))
        sub = fs.render("Hand Rehabilitation Assessment System", True, C_DIM)
        surface.blit(sub, sub.get_rect(center=(295, 98)))

        self.btn_records.draw(surface)

        # Patient name
        lbl = fs.render("Patient Name", True, C_DIM)
        surface.blit(lbl, (80, 200))
        self.name_input.draw(surface)

        # Game buttons
        lbl2 = fs.render("Select Game Mode:", True, C_DIM)
        surface.blit(lbl2, (80, 290))
        self.btn_a.draw(surface)
        self.btn_b.draw(surface)

        da = fs.render("Tremor assessment  (20 s)", True, C_DIM)
        db = fs.render("Range of motion  (30 s)",   True, C_DIM)
        surface.blit(da, (80,  402))
        surface.blit(db, (315, 402))

        # Right panel
        rt = fm.render("Select Fingers", True, C_TEXT)
        surface.blit(rt, rt.get_rect(center=(820, 135)))
        rh = fs.render("Click circles to toggle  |  PCB: ch4 ch5 ch6 ch9 ch13", True, C_DIM)
        surface.blit(rh, rh.get_rect(center=(820, 165)))

        self.hand.draw(surface)

        sel = self.hand.get_selected()
        sel_txt = "Selected: " + (", ".join(FINGER_NAMES[i] for i in sel) if sel else "None")
        sc = C_ACCENT if sel else C_DANGER
        st = fs.render(sel_txt, True, sc)
        surface.blit(st, st.get_rect(center=(820, 580)))

        if not sel:
            warn = fs.render("Select at least one finger to continue", True, C_DANGER)
            surface.blit(warn, warn.get_rect(center=(820, 605)))


# ── Level A setup ──────────────────────────────────────────────────────────────

class LevelASetupScreen:
    def __init__(self, fonts):
        fb, fm, fs = fonts
        self.fonts   = fonts
        self.btn_open   = Button((200, 280, 240, 80), "Open Hand",   fm)
        self.btn_closed = Button((660, 280, 240, 80), "Closed Hand", fm)
        self.target  = None

    def handle_event(self, event):
        if self.btn_open.handle_event(event):
            self.target = "open"
        if self.btn_closed.handle_event(event):
            self.target = "closed"

    def update(self):
        pass

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        t = fb.render("Level A", True, C_ACCENT)
        surface.blit(t, t.get_rect(center=(W // 2, 80)))

        s = fm.render("Choose the target hand position for this session:", True, C_TEXT)
        surface.blit(s, s.get_rect(center=(W // 2, 180)))

        self.btn_open.draw(surface)
        self.btn_closed.draw(surface)

        do = fs.render("Patient holds hand OPEN  (ap = +1)", True, C_DIM)
        dc = fs.render("Patient holds FIST closed  (ap = -1)", True, C_DIM)
        surface.blit(do, do.get_rect(center=(320, 375)))
        surface.blit(dc, dc.get_rect(center=(780, 375)))

        desc = fs.render("After selecting, you will proceed to calibration.", True, C_DIM)
        surface.blit(desc, desc.get_rect(center=(W // 2, H - 60)))


# ── Calibration screen ─────────────────────────────────────────────────────────

class CalibrationScreen:
    POSES = [
        ("open",   "Fully OPEN hand",    C_ACCENT,  +1),
        ("half",   "Half-open hand",     C_TUNNEL,   0),
        ("closed", "Make a tight FIST",  C_DANGER,  -1),
    ]

    def __init__(self, processor, fonts):
        fb, fm, fs = fonts
        self.fonts      = fonts
        self.proc       = processor
        self.pose_idx   = 0
        self.collecting = False
        self.done       = False
        self._progress  = 0.0
        self.btn_record = Button((SCREEN_W // 2 - 110, 500, 220, 48),
                                 "Hold & Record", fm, C_BTN_SEL)
        self._next_after = "level_a"

    def handle_event(self, event):
        if not self.collecting and not self.done:
            if self.btn_record.handle_event(event):
                pose_key = self.POSES[self.pose_idx][0]
                self.proc.record_calibration(pose_key)
                self.collecting = True
                self._progress  = 0.0

    def update(self):
        if self.collecting:
            pose_key = self.POSES[self.pose_idx][0]
            try:
                from data_preprocessing import CALIBRATION_SAMPLES as CAL_N
            except ImportError:
                CAL_N = 100
            samples = self.proc._calib_accum[pose_key]
            self._progress = min(1.0, len(samples) / CAL_N)
            if self._progress >= 1.0:
                self.collecting = False
                self.pose_idx  += 1
                if self.pose_idx >= len(self.POSES):
                    self.proc.finish_calibration()
                    self.done = True

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        t = fb.render("Calibration", True, C_TEXT)
        surface.blit(t, t.get_rect(center=(W // 2, 65)))

        if self.done:
            msg  = fm.render("Calibration complete!", True, C_ACCENT)
            hint = fs.render("Starting game...", True, C_DIM)
            surface.blit(msg,  msg.get_rect(center=(W // 2, H // 2)))
            surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 50)))
            return

        step = fs.render(f"Step {self.pose_idx + 1} / {len(self.POSES)}", True, C_DIM)
        surface.blit(step, step.get_rect(center=(W // 2, 115)))

        _, label, colour, target = self.POSES[self.pose_idx]
        pose_s = fm.render(label, True, colour)
        surface.blit(pose_s, pose_s.get_rect(center=(W // 2, H // 2 - 70)))

        # Progress bar
        bx, by, bw, bh = W // 2 - 200, H // 2, 400, 22
        pygame.draw.rect(surface, C_BAR_BG, (bx, by, bw, bh), border_radius=11)
        fw = int(bw * self._progress)
        if fw:
            pygame.draw.rect(surface, C_ACCENT, (bx, by, fw, bh), border_radius=11)
        pygame.draw.rect(surface, C_DIM, (bx, by, bw, bh), 2, border_radius=11)

        if self.collecting:
            hint = fs.render("Hold still...", True, C_DIM)
            surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 55)))
        else:
            self.btn_record.draw(surface)
            hint = fs.render("Hold the position, then click the button", True, C_DIM)
            surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 85)))

        # Scale
        self._draw_scale(surface, W // 2, H - 85, target)

    def _draw_scale(self, surface, cx, cy, target_ap):
        bw = 300
        pygame.draw.rect(surface, C_BAR_BG, (cx - bw // 2, cy - 8, bw, 16), border_radius=8)
        for ap, lbl in [(-1, "Closed"), (0, "Half"), (1, "Open")]:
            tx = cx + int(ap * bw // 2)
            pygame.draw.line(surface, C_DIM, (tx, cy - 12), (tx, cy + 12), 2)
            t = self.fonts[2].render(lbl, True, C_DIM)
            surface.blit(t, t.get_rect(center=(tx, cy + 28)))
        tx = cx + int(target_ap * bw // 2)
        pygame.draw.circle(surface, C_GOLD, (tx, cy), 10)


# ── Shared tunnel drawing ──────────────────────────────────────────────────────

def draw_straight_tunnel(surface, tunnel_center_ap, current_ap):
    """Draw flat-line tunnel and ball. Returns True if ball is inside."""
    W, H = surface.get_size()
    cy   = H // 2

    top_ap  = tunnel_center_ap + TUNNEL_HALF_AP
    bot_ap  = tunnel_center_ap - TUNNEL_HALF_AP
    top_y   = ap_to_y(top_ap)
    bot_y   = ap_to_y(bot_ap)

    # Tunnel fill
    pygame.draw.rect(surface, (28, 42, 38),
                     pygame.Rect(0, top_y, W, bot_y - top_y))

    # Straight flat borders (no relief)
    pygame.draw.line(surface, C_TUNNEL, (0, top_y), (W, top_y), 3)
    pygame.draw.line(surface, C_TUNNEL, (0, bot_y), (W, bot_y), 3)

    # Dashed centre line
    for x in range(0, W, 22):
        pygame.draw.line(surface, (45, 75, 58), (x, ap_to_y(tunnel_center_ap)),
                         (x + 11, ap_to_y(tunnel_center_ap)), 1)

    # Ball
    ball_y  = ap_to_y(current_ap)
    inside  = top_y <= ball_y <= bot_y
    bcol    = C_GOLD if inside else C_DANGER
    pygame.draw.circle(surface, bcol,   (W // 2, ball_y), BALL_RADIUS)
    pygame.draw.circle(surface, C_TEXT, (W // 2, ball_y), BALL_RADIUS, 2)
    return inside


def draw_timer_bar(surface, elapsed, total, W, H):
    remaining = max(0.0, total - elapsed)
    bx, by, bw, bh = 40, H - 38, W - 80, 14
    pygame.draw.rect(surface, C_BAR_BG, (bx, by, bw, bh), border_radius=7)
    fw = int(bw * remaining / total)
    col = C_DANGER if remaining < 5 else C_ACCENT
    if fw:
        pygame.draw.rect(surface, col, (bx, by, fw, bh), border_radius=7)
    return remaining


# ── Level A game ───────────────────────────────────────────────────────────────

class LevelAGameScene:
    def __init__(self, processor, target_pose, patient_name, selected_fingers, fonts):
        self.proc             = processor
        self.target_pose      = target_pose
        self.patient_name     = patient_name
        self.selected_fingers = selected_fingers
        self.fonts            = fonts
        self.tunnel_center_ap = 1.0 if target_pose == "open" else -1.0
        self.elapsed          = 0.0
        self.done             = False
        self.data_rows        = []
        self.timestamps       = []
        self.csv_path         = None

    def handle_event(self, event):
        pass

    def update(self, dt):
        if self.done:
            return
        self.elapsed += dt

        if SENSOR_AVAILABLE:
            state = self.proc.get_current_state()
            row   = list(state.finger_apertures)
        else:
            row = [0.0] * max(1, len(self.selected_fingers))

        self.data_rows.append(row)
        self.timestamps.append(self.elapsed)

        if self.elapsed >= LEVEL_A_DURATION:
            self.done = True
            names = [FINGER_NAMES[i] for i in self.selected_fingers]
            self.csv_path = export_csv(self.patient_name, "levelA", names,
                                       self.data_rows, self.timestamps)

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        if SENSOR_AVAILABLE:
            state = self.proc.get_current_state()
            ap    = state.aperture if state.calibrated else 0.0
        else:
            ap = 0.0

        draw_straight_tunnel(surface, self.tunnel_center_ap, ap)

        remaining = draw_timer_bar(surface, self.elapsed, LEVEL_A_DURATION, W, H)

        mode_lbl = "Open Hand" if self.target_pose == "open" else "Closed Hand"
        title = fm.render(f"Level A  -  {mode_lbl}", True, C_TEXT)
        surface.blit(title, (20, 14))
        tcol = C_DANGER if remaining < 5 else C_TEXT
        surface.blit(fs.render(f"{remaining:.1f}s remaining", True, tcol), (20, 48))
        surface.blit(fs.render(f"Hold the ball in the tunnel  (aperture = {ap:+.2f})",
                               True, C_DIM), (20, 70))

        if self.done:
            _draw_done_overlay(surface, W, H, fb, fs)


# ── Level B game ───────────────────────────────────────────────────────────────

class LevelBGameScene:
    def __init__(self, processor, patient_name, selected_fingers, fonts):
        self.proc             = processor
        self.patient_name     = patient_name
        self.selected_fingers = selected_fingers
        self.fonts            = fonts
        self.elapsed          = 0.0
        self.done             = False
        self.data_rows        = []
        self.timestamps       = []
        self.csv_path         = None

    @property
    def tunnel_center_ap(self):
        return lerp_waypoints(LEVEL_B_WAYPOINTS, self.elapsed)

    def handle_event(self, event):
        pass

    def update(self, dt):
        if self.done:
            return
        self.elapsed += dt

        if SENSOR_AVAILABLE:
            state = self.proc.get_current_state()
            row   = list(state.finger_apertures)
        else:
            row = [0.0] * max(1, len(self.selected_fingers))

        self.data_rows.append(row)
        self.timestamps.append(self.elapsed)

        if self.elapsed >= LEVEL_B_DURATION:
            self.done = True
            names = [FINGER_NAMES[i] for i in self.selected_fingers]
            self.csv_path = export_csv(self.patient_name, "levelB", names,
                                       self.data_rows, self.timestamps)

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        if SENSOR_AVAILABLE:
            state = self.proc.get_current_state()
            ap    = state.aperture if state.calibrated else 0.0
        else:
            ap = 0.0

        draw_straight_tunnel(surface, self.tunnel_center_ap, ap)

        remaining = draw_timer_bar(surface, self.elapsed, LEVEL_B_DURATION, W, H)

        # Phase markers on timer bar
        total  = LEVEL_B_DURATION
        bar_x, bar_y, bar_w = 40, H - 38, W - 80
        for t, _ in LEVEL_B_WAYPOINTS[1:]:
            mx = bar_x + int(t / total * bar_w)
            pygame.draw.line(surface, C_DIM, (mx, bar_y - 5), (mx, bar_y + 19), 1)

        title = fm.render("Level B  -  Range of Motion", True, C_TEXT)
        surface.blit(title, (20, 14))
        tcol = C_DANGER if remaining < 5 else C_TEXT
        surface.blit(fs.render(f"{remaining:.1f}s remaining", True, tcol), (20, 48))

        tc = self.tunnel_center_ap
        phase_hint = ("Neutral" if abs(tc) < 0.15 else
                      "Reach OPEN" if tc > 0 else "Reach CLOSED")
        surface.blit(fs.render(f"Target: {phase_hint}  (tunnel centre = {tc:+.2f})",
                               True, C_DIM), (20, 70))

        if self.done:
            _draw_done_overlay(surface, W, H, fb, fs)


def _draw_done_overlay(surface, W, H, font_big, font_small):
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((10, 15, 25, 200))
    surface.blit(ov, (0, 0))
    msg  = font_big.render("Session Complete!", True, C_ACCENT)
    hint = font_small.render("Press SPACE to view results", True, C_DIM)
    surface.blit(msg,  msg.get_rect(center=(W // 2, H // 2 - 20)))
    surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 30)))


# ── Results A ──────────────────────────────────────────────────────────────────

class ResultsScreenA:
    def __init__(self, data_rows, timestamps, selected_fingers, patient_name, csv_path, fonts):
        self.fonts            = fonts
        self.selected_fingers = selected_fingers
        self.patient_name     = patient_name
        self.csv_path         = csv_path
        self.next_scene       = None

        arr   = np.array(data_rows) if data_rows else np.zeros((1, max(1, len(selected_fingers))))
        names = [FINGER_NAMES[i] for i in selected_fingers]

        self.rows = []
        for fi in range(arr.shape[1]):
            n, f = tremor_analysis(arr[:, fi].tolist())
            self.rows.append({"name": names[fi], "n": n, "hz": f})

        fb, fm, fs = fonts
        self.btn_home = Button((SCREEN_W // 2 - 100, SCREEN_H - 70, 200, 44),
                               "Back to Home", fs)

    def handle_event(self, event):
        if self.btn_home.handle_event(event):
            self.next_scene = "home"

    def update(self):
        pass

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        t = fb.render("Results  -  Level A", True, C_ACCENT)
        surface.blit(t, t.get_rect(center=(W // 2, 50)))

        s = fs.render(
            f"Patient: {self.patient_name}     Tremor detection: {TREMOR_LOW_HZ}-{TREMOR_HIGH_HZ} Hz",
            True, C_DIM)
        surface.blit(s, s.get_rect(center=(W // 2, 90)))

        # Table
        headers = ["Finger", "Tremors detected", "Dominant freq (Hz)"]
        xs      = [120, 380, 640]
        y       = 148
        for h, x in zip(headers, xs):
            surface.blit(fm.render(h, True, C_DIM), (x, y))
        pygame.draw.line(surface, C_DIM, (80, y + 36), (W - 80, y + 36), 1)
        y += 48

        for row in self.rows:
            vals = [row["name"],
                    str(row["n"]),
                    f"{row['hz']:.2f}" if row["n"] > 0 else "-"]
            colors = [C_TEXT,
                      C_DANGER if row["n"] > 0 else C_ACCENT,
                      C_TEXT]
            for val, col, x in zip(vals, colors, xs):
                surface.blit(fm.render(val, True, col), (x, y))
            pygame.draw.line(surface, C_PANEL, (80, y + 36), (W - 80, y + 36), 1)
            y += 44

        if self.csv_path:
            p = fs.render(f"Data saved: {self.csv_path}", True, C_DIM)
            surface.blit(p, p.get_rect(center=(W // 2, H - 110)))

        self.btn_home.draw(surface)


# ── Results B ──────────────────────────────────────────────────────────────────

class ResultsScreenB:
    def __init__(self, data_rows, timestamps, selected_fingers, patient_name, csv_path, fonts):
        self.fonts            = fonts
        self.selected_fingers = selected_fingers
        self.patient_name     = patient_name
        self.csv_path         = csv_path
        self.next_scene       = None

        arr   = np.array(data_rows) if data_rows else np.zeros((1, max(1, len(selected_fingers))))
        names = [FINGER_NAMES[i] for i in selected_fingers]

        total          = max(1, len(data_rows))
        self.pct_open   = 100.0 * float(np.mean(arr > 0.5))
        self.pct_closed = 100.0 * float(np.mean(arr < -0.5))

        self.finger_rows = []
        for fi in range(arr.shape[1]):
            col = arr[:, fi]
            rom = float((col.max() - col.min()) / 2.0 * 100.0)
            self.finger_rows.append({"name": names[fi], "rom": min(rom, 100.0)})

        fb, fm, fs = fonts
        self.btn_home = Button((SCREEN_W // 2 - 100, SCREEN_H - 70, 200, 44),
                               "Back to Home", fs)

    def handle_event(self, event):
        if self.btn_home.handle_event(event):
            self.next_scene = "home"

    def update(self):
        pass

    def draw(self, surface):
        fb, fm, fs = self.fonts
        surface.fill(C_BG)
        W, H = surface.get_size()

        t = fb.render("Results  -  Level B", True, C_ACCENT)
        surface.blit(t, t.get_rect(center=(W // 2, 50)))
        surface.blit(fs.render(f"Patient: {self.patient_name}", True, C_DIM),
                     pygame.Rect(0, 90, W, 0).move(0, 0).move(W // 2, 0))

        # Summary
        y = 130
        surface.blit(fm.render(f"Time hand OPEN  (>0.5):    {self.pct_open:.1f}%",
                               True, C_ACCENT), (120, y))
        surface.blit(fm.render(f"Time hand CLOSED  (<-0.5): {self.pct_closed:.1f}%",
                               True, C_DANGER), (120, y + 44))

        y += 110
        surface.blit(fm.render("Range of Motion per Finger", True, C_TEXT),
                     pygame.Rect(0, 0, W, 0).move(W // 2 - 170, y))
        y += 40
        pygame.draw.line(surface, C_DIM, (100, y + 28), (W - 100, y + 28), 1)

        headers = ["Finger", "ROM achieved  (% of full range -1 to 1)"]
        xs      = [120, 320]
        for h, x in zip(headers, xs):
            surface.blit(fs.render(h, True, C_DIM), (x, y))
        y += 38

        for row in self.finger_rows:
            surface.blit(fm.render(row["name"], True, C_TEXT), (xs[0], y))
            col = C_ACCENT if row["rom"] > 50 else C_TEXT
            surface.blit(fm.render(f"{row['rom']:.1f}%", True, col), (xs[1], y))
            bx, bw, bh = xs[1] + 130, 250, 16
            pygame.draw.rect(surface, C_BAR_BG, (bx, y + 5, bw, bh), border_radius=8)
            fw = int(bw * row["rom"] / 100.0)
            if fw:
                pygame.draw.rect(surface, col, (bx, y + 5, fw, bh), border_radius=8)
            y += 42

        if self.csv_path:
            surface.blit(fs.render(f"Data saved: {self.csv_path}", True, C_DIM),
                         (120, H - 110))

        self.btn_home.draw(surface)


# ── Dummy processor for demo mode ──────────────────────────────────────────────

class _DummyState:
    aperture         = 0.0
    finger_apertures = [0.0]
    calibrated       = True


class _DummyProc:
    _calib_accum = {
        "open":   [1] * 200,
        "half":   [1] * 200,
        "closed": [1] * 200,
    }

    def record_calibration(self, pose):
        pass

    def finish_calibration(self):
        pass

    def get_current_state(self):
        return _DummyState()

    def stop(self):
        pass


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Rehab Game")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("segoeui", 40, bold=True),
        pygame.font.SysFont("segoeui", 26),
        pygame.font.SysFont("segoeui", 18),
    )

    processor = None

    context = {
        "patient_name":     "Patient",
        "selected_fingers": [0, 1, 2, 3, 4],
        "level_a_target":   "open",
    }

    scene_name    = "home"
    current_scene = HomeScreen(fonts)

    running = True
    while running:
        dt            = min(clock.tick(FPS) / 1000.0, 0.05)
        space_pressed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    space_pressed = True
            current_scene.handle_event(event)

        # Update
        if scene_name in ("level_a_game", "level_b_game"):
            current_scene.update(dt)
        else:
            current_scene.update()

        # ── Scene transitions ──────────────────────────────────────────────────
        if scene_name == "home":
            ns  = current_scene.next_scene
            sel = current_scene.hand.get_selected()
            if ns and sel:
                context["patient_name"]     = current_scene.name_input.text or "Patient"
                context["selected_fingers"] = sel
                current_scene.next_scene    = None

                if SENSOR_AVAILABLE:
                    if processor:
                        processor.stop()
                    if USE_REAL_GLOVE:
                        processor = SensorProcessor(selected_fingers=sel)
                    else:
                        processor = SimulatedSensorProcessor(selected_fingers=sel)
                    processor.start()

                proc = processor or _DummyProc()

                if ns == "game_a":
                    scene_name    = "level_a_setup"
                    current_scene = LevelASetupScreen(fonts)
                else:
                    scene_name    = "calibration"
                    current_scene = CalibrationScreen(proc, fonts)
                    current_scene._next_after = "level_b"

        elif scene_name == "level_a_setup" and current_scene.target:
            context["level_a_target"] = current_scene.target
            proc          = processor or _DummyProc()
            scene_name    = "calibration"
            current_scene = CalibrationScreen(proc, fonts)
            current_scene._next_after = "level_a"

        elif scene_name == "calibration" and current_scene.done:
            after = current_scene._next_after
            proc  = processor or _DummyProc()
            if after == "level_a":
                scene_name    = "level_a_game"
                current_scene = LevelAGameScene(
                    proc,
                    context["level_a_target"],
                    context["patient_name"],
                    context["selected_fingers"],
                    fonts)
            else:
                scene_name    = "level_b_game"
                current_scene = LevelBGameScene(
                    proc,
                    context["patient_name"],
                    context["selected_fingers"],
                    fonts)

        elif scene_name == "level_a_game" and current_scene.done and space_pressed:
            scene_name    = "results_a"
            current_scene = ResultsScreenA(
                current_scene.data_rows,
                current_scene.timestamps,
                context["selected_fingers"],
                context["patient_name"],
                current_scene.csv_path,
                fonts)

        elif scene_name == "level_b_game" and current_scene.done and space_pressed:
            scene_name    = "results_b"
            current_scene = ResultsScreenB(
                current_scene.data_rows,
                current_scene.timestamps,
                context["selected_fingers"],
                context["patient_name"],
                current_scene.csv_path,
                fonts)

        elif scene_name in ("results_a", "results_b"):
            if current_scene.next_scene == "home":
                scene_name    = "home"
                current_scene = HomeScreen(fonts)

        # Draw
        current_scene.draw(screen)
        pygame.display.flip()

    if processor:
        processor.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
