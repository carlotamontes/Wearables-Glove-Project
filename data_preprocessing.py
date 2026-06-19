"""
sensor_processing.py
====================
Handles all BLE communication, signal processing, and hand calibration
for the NeuroGlove rehabilitation system.

Architecture
------------
- BleakClient runs in a dedicated thread / asyncio loop.
- Raw ADC values arrive via notification_handler and are placed in a
  thread-safe queue.
- SensorProcessor reads from that queue, applies a moving-average
  filter, and converts the smoothed ADC values into a normalised
  hand-aperture score in the range [-1 .. +1] using a 3-point
  calibration (closed = -1, half-open = 0, open = +1).
- The resulting HandState object is made available to the game via
  get_current_state(), which is safe to call from any thread.

Usage
-----
    from sensor_processing import SensorProcessor

    proc = SensorProcessor()
    proc.start()                    # starts BLE thread

    # --- calibration (run once per session) ---
    proc.record_calibration("closed")
    time.sleep(3)
    proc.record_calibration("half")
    time.sleep(3)
    proc.record_calibration("open")
    proc.finish_calibration()

    # --- game loop ---
    state = proc.get_current_state()
    aperture = state.aperture       # float in [-1, +1]

    proc.stop()
"""

import asyncio
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
from bleak import BleakScanner, BleakClient

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — adjust to match your ESP32 firmware
# ──────────────────────────────────────────────────────────────────────────────

DEVICE_NAME_SUBSTRING  = "group8"   # partial match, case-insensitive
CHARACTERISTIC_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# PCB channel numbers for each finger (thumb→pinky) and their payload positions.
PCB_FINGER_CHANNELS    = [9,  8, 5, 6, 10]      # PCB channel labels, thumb to pinky
FINGER_PAYLOAD_INDICES = [8,  7, 4, 5,  9]        # corresponding indices in the BLE payload

# IMU channels for tremor detection (ch15, ch16, ch17 → payload indices 14, 15, 16).
IMU_PAYLOAD_INDICES    = [14, 15, 16]
IMU_CHANNEL_LABELS     = ["ch15", "ch16", "ch17"]

# Moving-average window (number of samples).  At ~100 Hz this is 200 ms.
MOVING_AVG_WINDOW = 30

# Samples averaged when recording a calibration point
CALIBRATION_SAMPLES = 100

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HandState:
    """
    Processed hand state, updated ~100 Hz.

    aperture : float [-1 .. +1]
        -1  = fully closed fist
         0  = half-open (mid-point)
        +1  = fully open hand

    finger_apertures : list[float]
        Individual aperture for each finger, same scale.

    raw_adc : list[float]
        Raw ADC values from the 5 finger channels (before smoothing).

    timestamp : float
        time.monotonic() of this sample.

    calibrated : bool
        True once finish_calibration() has been called successfully.
    """
    aperture: float = 0.0
    finger_apertures: List[float] = field(default_factory=lambda: [0.0] * 5)
    raw_adc: List[float] = field(default_factory=lambda: [0.0] * 5)
    imu_raw: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    timestamp: float = field(default_factory=time.monotonic)
    calibrated: bool = False


@dataclass
class CalibrationData:
    """Stores mean ADC vectors for the three reference poses."""
    closed: Optional[np.ndarray] = None   # fist  → aperture = -1
    half:   Optional[np.ndarray] = None   # half  → aperture =  0
    open:   Optional[np.ndarray] = None   # open  → aperture = +1

    def is_complete(self) -> bool:
        return all(v is not None for v in [self.closed, self.half, self.open])


# ──────────────────────────────────────────────────────────────────────────────
# Core processor
# ──────────────────────────────────────────────────────────────────────────────

class SensorProcessor:
    """
    Thread-safe wrapper around BLE acquisition, smoothing, and calibration.

    selected_fingers : list of ints (0-4, thumb=0 … pinky=4)
        Subset of fingers to read. Maps to FINGER_PAYLOAD_INDICES.
        None → all five fingers.
    """

    def __init__(self, selected_fingers: Optional[List[int]] = None):
        if selected_fingers is None:
            selected_fingers = [0, 1, 2, 3, 4]
        self._selected_fingers = selected_fingers
        # Payload indices for the chosen fingers
        self._finger_indices: List[int] = [FINGER_PAYLOAD_INDICES[f] for f in selected_fingers]
        self._n_fingers = len(self._finger_indices)

        self._raw_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._ble_thread: Optional[threading.Thread] = None

        # Ring buffers — one per selected finger — for moving average
        self._buffers: List[deque] = [
            deque(maxlen=MOVING_AVG_WINDOW) for _ in self._finger_indices
        ]
        # Ring buffers for IMU channels (ch15, ch16, ch17)
        self._imu_buffers: List[deque] = [
            deque(maxlen=MOVING_AVG_WINDOW) for _ in IMU_PAYLOAD_INDICES
        ]

        # Calibration accumulation
        self._calib_data = CalibrationData()
        self._calib_accum: Dict[str, List[np.ndarray]] = {
            "closed": [], "half": [], "open": []
        }
        self._recording_pose: Optional[str] = None  # set during recording
        self._calib_spans: Optional[np.ndarray] = None  # |open - closed| per finger

        # Shared state (protected by a lock for safe cross-thread reads)
        self._state_lock = threading.Lock()
        self._state = HandState(
            finger_apertures=[0.0] * self._n_fingers,
            raw_adc=[0.0] * self._n_fingers,
            imu_raw=[0.0] * len(IMU_PAYLOAD_INDICES),
        )

        # Background processing thread
        self._proc_thread = threading.Thread(target=self._processing_loop,
                                             daemon=True, name="SensorProc")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start BLE acquisition and processing threads."""
        self._ble_thread = threading.Thread(target=self._run_ble_loop,
                                            daemon=True, name="BLE")
        self._ble_thread.start()
        self._proc_thread.start()
        print("[SensorProcessor] Started.")

    def stop(self):
        """Gracefully shut down all threads."""
        self._stop_event.set()
        if self._ble_thread:
            self._ble_thread.join(timeout=5)
        print("[SensorProcessor] Stopped.")

    def record_calibration(self, pose: str):
        """
        Begin collecting samples for a calibration pose.

        pose : "closed" | "half" | "open"

        Call this, then hold the hand still for ~CALIBRATION_SAMPLES / 100 Hz
        (≈ 1 second), then call finish_calibration() when all three poses are
        done, or call record_calibration() for the next pose.
        """
        if pose not in ("closed", "half", "open"):
            raise ValueError(f"Unknown pose '{pose}'. Use 'closed', 'half', or 'open'.")
        self._calib_accum[pose].clear()
        self._recording_pose = pose
        print(f"[Calibration] Recording pose: '{pose}' — hold still...")

    def finish_calibration(self) -> bool:
        """
        Compute calibration means from recorded samples.
        Returns True if calibration is complete and successful.
        """
        self._recording_pose = None

        for pose in ("closed", "half", "open"):
            samples = self._calib_accum[pose]
            if len(samples) < 10:
                print(f"[Calibration] Not enough samples for pose '{pose}' "
                      f"({len(samples)} collected, need ≥10). Redo calibration.")
                return False
            setattr(self._calib_data, pose, np.mean(samples, axis=0))
            print(f"[Calibration] '{pose}' → mean ADC = "
                  f"{getattr(self._calib_data, pose).round(1)}")

        if self._calib_data.is_complete():
            self._calib_spans = np.abs(self._calib_data.open - self._calib_data.closed)

            FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            MIN_SPAN = 20.0  # ADC units — below this the calibration is unreliable

            print("[Calibration] Complete.  Per-finger diagnostic:")
            any_bad = False
            for i, span in enumerate(self._calib_spans):
                fi    = self._selected_fingers[i] if i < len(self._selected_fingers) else i
                label = FINGER_LABELS[fi] if fi < len(FINGER_LABELS) else f"finger{fi}"
                cl    = self._calib_data.closed[i]
                ha    = self._calib_data.half[i]
                op    = self._calib_data.open[i]
                half_ok = (min(cl, op) <= ha <= max(cl, op))
                warn  = "" if span >= MIN_SPAN else "  *** SPAN TOO SMALL — calibration unreliable ***"
                if not half_ok:
                    warn += "  *** HALF outside [closed,open] range — redo calibration ***"
                print(f"  {label} (payload idx {FINGER_PAYLOAD_INDICES[fi]}): "
                      f"closed={cl:.1f}  half={ha:.1f}  open={op:.1f}  span={span:.1f}{warn}")
                if span < MIN_SPAN or not half_ok:
                    any_bad = True

            if any_bad:
                print("[Calibration] WARNING: one or more fingers have poor calibration. "
                      "Check the payload index mapping or redo calibration for those fingers.")

            with self._state_lock:
                self._state.calibrated = True
            return True
        return False

    def get_current_state(self) -> HandState:
        """Return a snapshot of the latest HandState. Thread-safe."""
        with self._state_lock:
            # Return a shallow copy so the caller can hold it safely
            s = self._state
            return HandState(
                aperture=s.aperture,
                finger_apertures=list(s.finger_apertures),
                raw_adc=list(s.raw_adc),
                imu_raw=list(s.imu_raw),
                timestamp=s.timestamp,
                calibrated=s.calibrated,
            )

    def calibration_status(self) -> Dict[str, bool]:
        """Returns which calibration poses have been recorded."""
        return {
            "closed": self._calib_data.closed is not None,
            "half":   self._calib_data.half   is not None,
            "open":   self._calib_data.open   is not None,
        }

    # ── BLE layer ─────────────────────────────────────────────────────────────

    def _notification_handler(self, sender, data: bytearray):
        """Runs in Bleak's event loop thread. Just enqueue the raw bytes."""
        try:
            parts = data.decode().split(',')
            values = [float(p) for p in parts]
            self._raw_queue.put((time.monotonic(), values))
        except (ValueError, UnicodeDecodeError) as exc:
            print(f"[BLE] Parse error: {exc}")

    def _run_ble_loop(self):
        asyncio.run(self._ble_main())

    async def _ble_main(self):
        print("[BLE] Scanning…")
        devices = await BleakScanner.discover(timeout=10)
        device = next(
            (d for d in devices
             if d.name and DEVICE_NAME_SUBSTRING in d.name.lower()),
            None
        )
        if not device:
            print("[BLE] Device not found. Is the ESP32 powered on and advertising?")
            return

        print(f"[BLE] Connecting to {device.name} ({device.address})…")
        async with BleakClient(device.address) as client:
            print(f"[BLE] Connected ✓")
            await client.start_notify(CHARACTERISTIC_UUID, self._notification_handler)
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)
            await client.stop_notify(CHARACTERISTIC_UUID)
        print("[BLE] Disconnected.")

    # ── Processing loop ───────────────────────────────────────────────────────

    def _processing_loop(self):
        """
        Runs in a dedicated thread.
        Drains _raw_queue, smooths each channel, applies calibration,
        and updates _state.
        """
        while not self._stop_event.is_set():
            # Drain all pending samples
            batch = []
            try:
                while True:
                    batch.append(self._raw_queue.get_nowait())
            except queue.Empty:
                pass

            if not batch:
                time.sleep(0.005)
                continue

            for ts, values in batch:
                min_needed = max(max(self._finger_indices), max(IMU_PAYLOAD_INDICES)) + 1
                if len(values) < min_needed:
                    continue
                finger_raw = np.array([values[i] for i in self._finger_indices],
                                      dtype=float)
                imu_raw    = np.array([values[i] for i in IMU_PAYLOAD_INDICES],
                                      dtype=float)

                # Update finger ring buffers
                for idx, val in enumerate(finger_raw):
                    self._buffers[idx].append(val)

                # Update IMU ring buffers
                for idx, val in enumerate(imu_raw):
                    self._imu_buffers[idx].append(val)

                # Compute smoothed finger values (moving average)
                smoothed = np.array([
                    np.mean(buf) if buf else val
                    for buf, val in zip(self._buffers, finger_raw)
                ])

                # Accumulate for calibration recording
                if self._recording_pose is not None:
                    pool = self._calib_accum[self._recording_pose]
                    if len(pool) < CALIBRATION_SAMPLES:
                        pool.append(smoothed.copy())
                    if len(pool) >= CALIBRATION_SAMPLES:
                        print(f"[Calibration] Done collecting '{self._recording_pose}'.")
                        self._recording_pose = None

                # Compute aperture
                aperture, finger_apertures = self._compute_aperture(smoothed)

                # Update shared state
                with self._state_lock:
                    self._state.raw_adc          = finger_raw.tolist()
                    self._state.imu_raw          = imu_raw.tolist()
                    self._state.aperture         = aperture
                    self._state.finger_apertures = finger_apertures
                    self._state.timestamp        = ts

            time.sleep(0.001)

    # ── Calibration math ──────────────────────────────────────────────────────

    def _compute_aperture(self, smoothed: np.ndarray):
        """
        Map smoothed ADC vector → scalar aperture in [-1, +1].

        With 3-point calibration (closed=-1, half=0, open=+1) we use
        piecewise linear interpolation per finger, then average.

        Without calibration, returns 0.0 for everything.
        """
        if not self._calib_data.is_complete():
            return 0.0, [0.0] * self._n_fingers

        closed_vec = self._calib_data.closed
        half_vec   = self._calib_data.half
        open_vec   = self._calib_data.open

        MIN_SPAN = 20.0

        finger_apertures = []
        for i in range(self._n_fingers):
            v      = smoothed[i]
            v_cl   = closed_vec[i]
            v_half = half_vec[i]
            v_op   = open_vec[i]
            span   = abs(v_op - v_cl)

            # Skip fingers with near-zero span — interpolation is meaningless
            # and noise would produce random ±1 values.
            if span < MIN_SPAN:
                finger_apertures.append(0.0)
                continue

            # Clamp v_half to [v_lo, v_hi] so the piecewise branches stay
            # monotone even if calibration captured a slightly wrong pose.
            v_lo   = min(v_op, v_cl)
            v_hi   = max(v_op, v_cl)
            v_half = float(np.clip(v_half, v_lo, v_hi))
            v      = float(np.clip(v, v_lo, v_hi))

            ap = _piecewise_interp(v, v_cl, v_half, v_op)
            finger_apertures.append(float(np.clip(ap, -1.0, 1.0)))

        # Weighted average by calibration span: fingers with larger open-closed
        # range are more reliable and contribute more to the scalar aperture.
        if self._calib_spans is not None and self._calib_spans.sum() > 0:
            aperture = float(np.average(finger_apertures, weights=self._calib_spans))
        else:
            aperture = float(np.mean(finger_apertures))
        return aperture, finger_apertures


# ──────────────────────────────────────────────────────────────────────────────
# Utility: piecewise linear interpolation
# ──────────────────────────────────────────────────────────────────────────────

def _piecewise_interp(v: float,
                      v_closed: float,
                      v_half:   float,
                      v_open:   float) -> float:
    """
    Map a single ADC value to an aperture score in [-1, +1] using the
    three calibration landmarks:

        v_closed → -1    (fist)
        v_half   →  0    (half-open)
        v_open   → +1    (open)

    Between the landmarks we interpolate linearly.
    Outside the range we extrapolate but the caller clips to [-1, +1].

    This works regardless of whether ADC goes up or down with bending
    (depends on voltage-divider orientation), because we always use the
    patient's own calibration values.
    """
    # Lower half: closed → half
    span_lo = v_half - v_closed
    span_hi = v_open  - v_half

    if v_closed <= v <= v_half or v_half <= v <= v_closed:
        # Between closed and half
        if span_lo == 0:
            return -1.0
        return -1.0 + (v - v_closed) / span_lo   # maps to [-1, 0]
    else:
        # Between half and open
        if span_hi == 0:
            return 0.0
        return 0.0 + (v - v_half) / span_hi       # maps to [0, +1]


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo / debug mode (run without a real glove using simulated data)
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedSensorProcessor(SensorProcessor):
    """
    Drop-in replacement for SensorProcessor that generates a sinusoidal
    fake hand signal instead of connecting via BLE.
    Useful for game development without the physical glove.
    """

    def start(self):
        self._proc_thread.start()
        sim_thread = threading.Thread(target=self._simulate, daemon=True,
                                      name="Simulate")
        sim_thread.start()
        # Pre-fill calibration so the game can run immediately
        self._auto_calibrate()
        print("[SimulatedSensor] Started with simulated sinusoidal hand signal.")

    def _simulate(self):
        t = 0.0
        n_ch = max(max(FINGER_PAYLOAD_INDICES), max(IMU_PAYLOAD_INDICES)) + 1
        while not self._stop_event.is_set():
            # Slow sine: full open→close cycle every 4 seconds
            value = 512 + 200 * np.sin(2 * np.pi * t / 4.0)
            noise = np.random.normal(0, 5, size=n_ch)
            values = [value + noise[i] for i in range(n_ch)]
            self._raw_queue.put((time.monotonic(), values))
            t += 0.01
            time.sleep(0.01)

    def _auto_calibrate(self):
        """Set calibration values without needing real sensor samples."""
        n = self._n_fingers
        self._calib_data.closed = np.array([312.0] * n)
        self._calib_data.half   = np.array([512.0] * n)
        self._calib_data.open   = np.array([712.0] * n)
        with self._state_lock:
            self._state.calibrated = True
        print("[SimulatedSensor] Auto-calibration done (closed=312, half=512, open=712).")


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== NeuroGlove sensor_processing.py — self-test ===")
    print("Using SIMULATED sensor data (no BLE hardware required).\n")

    proc = SimulatedSensorProcessor()
    proc.start()

    print("Monitoring for 10 seconds…  (Ctrl-C to stop early)\n")
    try:
        for _ in range(100):
            state = proc.get_current_state()
            bar_len = int((state.aperture + 1) / 2 * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"\r  aperture: [{bar}] {state.aperture:+.3f}   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\n\nDone.")
    proc.stop()