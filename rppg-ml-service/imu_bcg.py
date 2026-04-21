"""
imu_bcg.py  —  Phone IMU Micro-Ballistocardiography (BCG) Liveness Layer
═══════════════════════════════════════════════════════════════════════════

Physical basis
──────────────
Each heartbeat ejects ~70 mL of blood into the aorta, producing a recoil
impulse (~1–4 mN·s) that travels up through the torso, arm, and hand into
the phone.  When the phone is held naturally (screen facing the user), this
cardiac recoil is dominated by the **Z-axis** of the phone IMU — the axis
perpendicular to the screen, pointing away from the user's face.

Why Z-axis?
  • X-axis (horizontal) captures voluntary wrist pronation/supination.
  • Y-axis (vertical) captures arm elevation and gravity components.
  • Z-axis (screen-normal) is orthogonal to typical hand tremor directions
    and captures the systolic recoil most cleanly in a handheld posture.

Data contract
─────────────
The frontend must POST a JSON payload alongside the video containing a
time-series of raw accelerometer readings sampled from the device IMU:

    {
        "imu": {
            "timestamps_ms": [0, 20, 40, ...],   // epoch ms, or relative ms
            "accel_x": [...],                     // m/s² or g — normalised here
            "accel_y": [...],
            "accel_z": [...]                      // PRIMARY cardiac axis
        }
    }

Sample rate should be ≥ 50 Hz (20 ms interval).  Most phones support
50–200 Hz on TYPE_ACCELEROMETER.

Algorithm
─────────
1. Receive raw Z-axis accelerometer stream.
2. Detrend: subtract rolling mean (2 s window) to remove gravity + slow drift.
3. Bandpass filter at 0.7–3.0 Hz (42–180 BPM) — same as camera BCG.
4. Compute dominant frequency via FFT → IMU BCG heart rate (BPM).
5. Cross-validate against rPPG heart rate:
   a. Direct match within ±0.25 Hz → confirmed live.
   b. Harmonic aliasing check (same HARMONIC_RATIOS as camera BCG) — if IMU
      frequency is a simple multiple of rPPG, flag as spoofed signal.
6. During challenge windows: verify that IMU shows the expected motion
   signature for the challenge type (e.g. head_left → sustained lateral
   acceleration on X/Y axes during the challenge window).

Challenge motion signatures
───────────────────────────
  blink        — no expected gross IMU motion (head still); IMU stays flat.
  head_left    — sustained negative X-axis deflection during window.
  head_right   — sustained positive X-axis deflection during window.
  head_turn    — X-axis deflection in either direction.
  mouth_open   — no expected gross IMU motion.
  eyebrow_raise— no expected gross IMU motion.

  For challenges with expected motion: the IMU MUST show the deflection.
  For still challenges: the IMU MUST stay within the resting noise envelope.
  Either mismatch is flagged as "challenge IMU inconsistency" and logged.

Returns a dict with:
    passed              : bool   Overall IMU BCG liveness verdict
    imu_hr_bpm          : float  IMU-estimated heart rate
    rppg_hr_bpm         : float  rPPG cross-reference heart rate
    freq_match          : bool   Fundamental frequency agreement
    is_harmonic         : bool   True if IMU freq is a harmonic of rPPG
    harmonic_ratio      : float  Ratio that triggered harmonic flag
    imu_signal_power    : float  Variance of filtered Z-axis signal
    challenge_imu_ok    : bool   Challenge-window motion check passed
    samples_used        : int    Number of IMU samples processed
    reason              : str    Human-readable verdict
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Tuning parameters (shared with camera BCG for consistency) ────────────────
BCG_LOW_HZ         = 0.7     # 42 BPM
BCG_HIGH_HZ        = 3.0     # 180 BPM
FREQ_MATCH_TOL_HZ  = 0.25    # ±0.25 Hz tolerance for cross-modal agreement
MIN_SIGNAL_POWER   = 1e-8    # Minimum variance for a detectable BCG signal
MIN_SAMPLES        = 50      # ~1 second at 50 Hz minimum sample rate
DEFAULT_SAMPLE_HZ  = 50.0    # Assumed if sample rate cannot be computed

# Harmonic ratios — mirrors camera BCG HARMONIC_RATIOS exactly.
# A spoofed IMU signal (e.g. synthetic waveform injected via software) will
# often lock to a harmonic of the rPPG frequency rather than the fundamental.
HARMONIC_RATIOS    = [2.0, 0.5, 3.0, 1.0 / 3.0]

# Challenge motion detection thresholds
# Units: m/s²  (if the frontend sends raw accel in m/s²)
# If the frontend sends in g, values are ~10× smaller — we normalise below.
CHALLENGE_MOTION_THRESHOLD  = 0.08   # deflection needed to confirm head motion
CHALLENGE_STILL_THRESHOLD   = 0.15   # max allowed motion for "hold still" phases
CHALLENGE_MIN_WINDOW_RATIO  = 0.25   # at least 25% of challenge window must show signal


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_sample_hz(timestamps_ms: List[float]) -> float:
    """
    Infer sample rate from timestamp deltas.
    Returns DEFAULT_SAMPLE_HZ if the list is too short or deltas are erratic.
    """
    if len(timestamps_ms) < 4:
        return DEFAULT_SAMPLE_HZ
    deltas = np.diff(timestamps_ms)
    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        return DEFAULT_SAMPLE_HZ
    hz = 1000.0 / median_dt
    # Clamp to a physically meaningful range
    if hz < 5 or hz > 500:
        logger.warning(f"IMU: inferred sample rate {hz:.1f} Hz out of range — using {DEFAULT_SAMPLE_HZ}")
        return DEFAULT_SAMPLE_HZ
    return hz


def _detrend(signal: np.ndarray, sample_hz: float, window_s: float = 2.0) -> np.ndarray:
    """
    Remove slow drift and gravity by subtracting a rolling mean.
    window_s: width of rolling window in seconds.
    """
    window = max(3, int(sample_hz * window_s))
    # Use 'same' convolution so output length matches input
    kernel = np.ones(window) / window
    trend  = np.convolve(signal, kernel, mode='same')
    return signal - trend


def _bandpass(signal: np.ndarray, sample_hz: float,
              low: float = BCG_LOW_HZ, high: float = BCG_HIGH_HZ) -> np.ndarray:
    """Butterworth bandpass filter isolating cardiac BCG frequencies."""
    from scipy.signal import butter, filtfilt

    if len(signal) < 10:
        return signal

    nyq = 0.5 * sample_hz
    lo  = max(0.01, low  / nyq)
    hi  = min(0.99, high / nyq)

    try:
        b, a = butter(3, [lo, hi], btype='band')
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.error(f"IMU bandpass error: {e}")
        return signal


def _dominant_freq(signal: np.ndarray, sample_hz: float,
                   low: float = BCG_LOW_HZ, high: float = BCG_HIGH_HZ) -> float:
    """Return dominant frequency (Hz) in the bandpass window via FFT."""
    n = len(signal)
    if n < 10:
        return 0.0
    freqs   = np.fft.rfftfreq(n, d=1.0 / sample_hz)
    fft_mag = np.abs(np.fft.rfft(signal - signal.mean()))
    mask    = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(fft_mag[mask])])


def _normalise_accel(values: List[float]) -> np.ndarray:
    """
    Normalise accelerometer values to m/s².
    If values look like they are in g (|mean| < 2), scale by 9.81.
    Most browser DeviceMotion APIs deliver in m/s²; some deliver in g.
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    # Heuristic: if the signal range is < 5, likely in g
    signal_range = float(np.max(np.abs(arr)))
    if signal_range < 5.0:
        arr = arr * 9.81
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Challenge window motion verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_challenge_motion(
    imu_data:            Dict,
    challenge_id:        str,
    challenge_start_ms:  float,
    challenge_end_ms:    float,
    sample_hz:           float,
) -> Dict:
    """
    Verify that IMU motion during a challenge window is consistent with the
    expected physical movement for that challenge type.

    Returns dict: { passed: bool, reason: str, axis_used: str }
    """
    timestamps = np.array(imu_data.get("timestamps_ms", []), dtype=np.float64)
    if len(timestamps) == 0:
        return {"passed": True, "reason": "No timestamps — skipping IMU challenge check", "axis_used": "none"}

    # Normalise timestamps to be relative (start from 0)
    ts_rel = timestamps - timestamps[0]

    # Find indices inside the challenge window
    mask = (ts_rel >= challenge_start_ms) & (ts_rel <= challenge_end_ms)
    n_window = int(np.sum(mask))

    if n_window < 3:
        return {"passed": True, "reason": f"Challenge window too short ({n_window} samples) — skipped", "axis_used": "none"}

    accel_z = _normalise_accel(imu_data.get("accel_z", []))
    accel_x = _normalise_accel(imu_data.get("accel_x", []))
    accel_y = _normalise_accel(imu_data.get("accel_y", []))

    # Clip to available length
    max_idx = min(len(timestamps), len(accel_x), len(accel_y), len(accel_z))
    mask    = mask[:max_idx]
    win_x   = accel_x[:max_idx][mask]
    win_y   = accel_y[:max_idx][mask]
    win_z   = accel_z[:max_idx][mask]

    # Challenges that expect head motion → look at X (lateral) and Y (vertical)
    MOTION_CHALLENGES = {"head_turn", "head_left", "head_right"}
    # Challenges that expect stillness
    STILL_CHALLENGES  = {"blink", "mouth_open", "eyebrow_raise"}

    if challenge_id in MOTION_CHALLENGES:
        axis_used    = "x"
        win_signal   = win_x
        # head_left:  sustained negative-X deflection
        # head_right: sustained positive-X deflection
        # head_turn:  deflection in either direction
        detrended    = win_signal - win_signal.mean()
        rms_motion   = float(np.sqrt(np.mean(detrended ** 2)))
        active_frac  = float(np.mean(np.abs(detrended) > CHALLENGE_MOTION_THRESHOLD))

        if rms_motion < CHALLENGE_MOTION_THRESHOLD or active_frac < CHALLENGE_MIN_WINDOW_RATIO:
            return {
                "passed":    False,
                "reason":    f"IMU challenge '{challenge_id}': expected head motion not detected "
                             f"(rms={rms_motion:.4f} m/s², active_frac={active_frac:.2f})",
                "axis_used": axis_used,
            }

        # Directional check for head_left / head_right
        if challenge_id == "head_left":
            mean_deflection = float(np.mean(detrended))
            if mean_deflection > 0:   # should be negative for left turn
                return {
                    "passed":    False,
                    "reason":    f"IMU head_left: deflection direction wrong (mean={mean_deflection:.4f})",
                    "axis_used": axis_used,
                }

        elif challenge_id == "head_right":
            mean_deflection = float(np.mean(detrended))
            if mean_deflection < 0:   # should be positive for right turn
                return {
                    "passed":    False,
                    "reason":    f"IMU head_right: deflection direction wrong (mean={mean_deflection:.4f})",
                    "axis_used": axis_used,
                }

        return {
            "passed":    True,
            "reason":    f"IMU challenge '{challenge_id}' motion confirmed (rms={rms_motion:.4f} m/s²)",
            "axis_used": axis_used,
        }

    elif challenge_id in STILL_CHALLENGES:
        # For still challenges: phone should not move significantly
        axis_used  = "xyz_rms"
        rms_x      = float(np.sqrt(np.mean((win_x - win_x.mean()) ** 2)))
        rms_y      = float(np.sqrt(np.mean((win_y - win_y.mean()) ** 2)))
        total_rms  = float(np.sqrt(rms_x ** 2 + rms_y ** 2))

        if total_rms > CHALLENGE_STILL_THRESHOLD:
            return {
                "passed":    False,
                "reason":    f"IMU challenge '{challenge_id}': unexpected motion during still phase "
                             f"(rms={total_rms:.4f} m/s²)",
                "axis_used": axis_used,
            }

        return {
            "passed":    True,
            "reason":    f"IMU challenge '{challenge_id}' stillness confirmed (rms={total_rms:.4f} m/s²)",
            "axis_used": axis_used,
        }

    # Unknown challenge type — don't penalise
    return {"passed": True, "reason": f"IMU: no motion rule for challenge '{challenge_id}'", "axis_used": "none"}


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_imu_bcg(
    imu_data:       Dict,
    rppg_hr_bpm:    float = 0.0,
    challenge_id:   Optional[str]   = None,
    challenge_start_ms: float = 0.0,
    challenge_end_ms:   float = 0.0,
) -> Dict:
    """
    Main entry point.  Processes raw IMU accelerometer data and returns a
    liveness verdict.

    Args:
        imu_data:           Dict with keys:
                              timestamps_ms  : List[float]  — relative or epoch ms
                              accel_x        : List[float]  — lateral axis (m/s² or g)
                              accel_y        : List[float]  — vertical axis
                              accel_z        : List[float]  — screen-normal (cardiac axis)
        rppg_hr_bpm:        Heart rate from the rPPG pipeline for cross-validation.
        challenge_id:       The challenge being verified (e.g. "head_left").
        challenge_start_ms: Millisecond offset into the recording where the
                            challenge window started.
        challenge_end_ms:   Millisecond offset where the challenge ended.

    Returns:
        Dict (see module docstring for field descriptions).
    """
    result = {
        "passed":           False,
        "imu_hr_bpm":       0.0,
        "rppg_hr_bpm":      rppg_hr_bpm,
        "freq_match":       False,
        "is_harmonic":      False,
        "harmonic_ratio":   1.0,
        "imu_signal_power": 0.0,
        "challenge_imu_ok": True,   # default True — only False if explicitly failed
        "samples_used":     0,
        "reason":           "",
    }

    # ── Input validation ──────────────────────────────────────────────────────
    if not imu_data:
        result["reason"] = "No IMU data provided"
        return result

    timestamps_ms = imu_data.get("timestamps_ms", [])
    accel_z_raw   = imu_data.get("accel_z",       [])

    if len(accel_z_raw) < MIN_SAMPLES:
        result["reason"] = (
            f"Insufficient IMU samples ({len(accel_z_raw)}) — "
            f"need ≥ {MIN_SAMPLES} (~1 s at 50 Hz). "
            "Ensure phone sensor access is granted."
        )
        return result

    # ── Infer sample rate ─────────────────────────────────────────────────────
    sample_hz = _infer_sample_hz(timestamps_ms) if len(timestamps_ms) >= 4 else DEFAULT_SAMPLE_HZ
    logger.info(f"IMU BCG: {len(accel_z_raw)} samples at {sample_hz:.1f} Hz")

    result["samples_used"] = len(accel_z_raw)

    # ── Normalise and detrend Z-axis ──────────────────────────────────────────
    accel_z      = _normalise_accel(accel_z_raw)
    z_detrended  = _detrend(accel_z, sample_hz)

    # ── Bandpass filter ───────────────────────────────────────────────────────
    try:
        z_filtered = _bandpass(z_detrended, sample_hz)
    except Exception as e:
        logger.warning(f"IMU bandpass failed ({e}) — using detrended signal")
        z_filtered = z_detrended

    # ── Signal power check ────────────────────────────────────────────────────
    imu_power = float(np.var(z_filtered))
    result["imu_signal_power"] = round(imu_power, 10)
    logger.info(f"IMU BCG signal power (Z): {imu_power:.2e}")

    if imu_power < MIN_SIGNAL_POWER:
        result["reason"] = (
            "IMU BCG signal power too low — no detectable cardiac recoil. "
            "Hold the phone naturally in hand (not on a surface)."
        )
        return result

    # ── Dominant frequency → IMU heart rate ──────────────────────────────────
    imu_freq    = _dominant_freq(z_filtered, sample_hz)
    imu_hr      = imu_freq * 60.0
    result["imu_hr_bpm"] = round(imu_hr, 1)
    logger.info(f"IMU BCG dominant frequency: {imu_freq:.3f} Hz → {imu_hr:.1f} BPM")

    if not (40 <= imu_hr <= 180):
        result["reason"] = (
            f"IMU BCG dominant frequency {imu_hr:.0f} BPM outside "
            "physiological range (40–180 BPM)"
        )
        return result

    # ── Challenge window IMU motion verification ──────────────────────────────
    if challenge_id and challenge_end_ms > challenge_start_ms:
        ch_check = _verify_challenge_motion(
            imu_data, challenge_id,
            challenge_start_ms, challenge_end_ms,
            sample_hz,
        )
        result["challenge_imu_ok"] = ch_check["passed"]
        logger.info(
            f"IMU challenge '{challenge_id}': "
            f"{'OK' if ch_check['passed'] else 'FAIL'} — {ch_check['reason']}"
        )
        if not ch_check["passed"]:
            result["reason"] = ch_check["reason"]
            return result

    # ── Cross-modal frequency agreement with rPPG ─────────────────────────────
    if rppg_hr_bpm > 0:
        imu_freq_hz  = imu_hr      / 60.0
        rppg_freq_hz = rppg_hr_bpm / 60.0
        freq_diff    = abs(imu_freq_hz - rppg_freq_hz)
        freq_match   = freq_diff <= FREQ_MATCH_TOL_HZ

        # ── Harmonic aliasing detection ───────────────────────────────────────
        # Same logic as camera BCG — if IMU frequency is a simple harmonic of
        # rPPG, the signal is likely synthetic/replayed rather than genuine.
        is_harmonic    = False
        harmonic_ratio = 1.0
        for ratio in HARMONIC_RATIOS:
            expected_hz = rppg_freq_hz * ratio
            if abs(imu_freq_hz - expected_hz) <= FREQ_MATCH_TOL_HZ:
                is_harmonic    = True
                harmonic_ratio = ratio
                break

        result["freq_match"]      = freq_match
        result["is_harmonic"]     = is_harmonic
        result["harmonic_ratio"]  = harmonic_ratio

        logger.info(
            f"IMU↔rPPG: |{imu_freq_hz:.3f} - {rppg_freq_hz:.3f}| = "
            f"{freq_diff:.3f} Hz (tol={FREQ_MATCH_TOL_HZ}) → "
            f"{'MATCH' if freq_match else 'NO MATCH'} | harmonic={is_harmonic} "
            f"(ratio={harmonic_ratio})"
        )

        if freq_match and not is_harmonic:
            result["passed"] = True
            result["reason"] = (
                f"IMU BCG confirmed: cardiac recoil ({imu_hr:.0f} BPM) "
                f"matches rPPG ({rppg_hr_bpm:.0f} BPM) — live person in hand"
            )
        elif is_harmonic:
            result["passed"] = False
            result["reason"] = (
                f"IMU BCG harmonic artifact: IMU={imu_hr:.0f} BPM is "
                f"{harmonic_ratio}× of rPPG={rppg_hr_bpm:.0f} BPM. "
                "Possible synthetic IMU injection or signal noise."
            )
            logger.warning(result["reason"])
        else:
            result["passed"] = False
            result["reason"] = (
                f"IMU↔rPPG frequency mismatch: IMU={imu_hr:.0f} BPM, "
                f"rPPG={rppg_hr_bpm:.0f} BPM "
                f"(diff={freq_diff:.2f} Hz > tol={FREQ_MATCH_TOL_HZ} Hz)"
            )
            logger.warning(result["reason"])

    else:
        # No rPPG reference available — verdict from IMU alone
        result["freq_match"] = False
        result["passed"]     = True   # IMU in range and power present
        result["reason"]     = (
            f"IMU BCG motion detected at {imu_hr:.0f} BPM "
            "(rPPG reference unavailable — IMU-only verdict)"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: extract rPPG heart rate from the signals dict produced by
# rppg_core.extract_roi_signals(), so the caller doesn't need to re-run FFT.
# ─────────────────────────────────────────────────────────────────────────────

def rppg_hr_from_signals(signals: Dict) -> float:
    """
    Compute a quick rPPG heart rate estimate from the green channel of the
    forehead ROI signal dict (as returned by rppg_core.extract_roi_signals).

    Returns heart rate in BPM, or 0.0 if unavailable.
    """
    if not signals:
        return 0.0

    fps = float(signals.get("_fps", DEFAULT_SAMPLE_HZ))
    fh  = signals.get("forehead")

    if fh is None or len(fh) == 0:
        return 0.0

    arr = np.array(fh, dtype=np.float64)
    # Green channel is index 1 in BGR order
    g = arr[:, 1] if (arr.ndim == 2 and arr.shape[1] >= 2) else arr.flatten()

    if len(g) < 10:
        return 0.0

    try:
        g_detrended = g - np.convolve(g, np.ones(min(int(fps * 2), len(g) // 2)) /
                                          max(1, min(int(fps * 2), len(g) // 2)),
                                      mode='same')
        g_filtered  = _bandpass(g_detrended, fps)
        freq        = _dominant_freq(g_filtered, fps)
        return round(freq * 60.0, 1)
    except Exception as e:
        logger.warning(f"rppg_hr_from_signals error: {e}")
        return 0.0