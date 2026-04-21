"""
anti_spoofing.py  —  STRICT MODE
─────────────────────────────────
Key changes vs previous version:
  1. Challenge is now MANDATORY — if challenge was required and failed,
     liveness is rejected regardless of other layers.
  2. Error fallback changed from True → False. An exception during analysis
     is NOT evidence of a live user; fail closed.
  3. rPPG coherence threshold raised: must be in [0.15, 0.95] to count as
     a passing layer (the old > 0.05 let near-zero screen signals through).
  4. Hard-block extended: coherence > 0.90 (was 0.98) triggers screen-replay
     rejection — perfect screen correlation rarely drops below 0.90.
  5. Requires 2/3 layers, but challenge MUST be one of those 2 if it was
     requested. BCG alone + rPPG cannot bypass a failed challenge.
  6. IMU BCG added as Layer 4 (optional) — phone in-hand Z-axis cardiac
     recoil cross-validated against rPPG.  Harmonic aliasing detection
     applied identically to camera BCG.  When present, required threshold
     stays at 2/4 but IMU BCG failure triggers an explicit warning.
"""
import numpy as np
from scipy.signal import butter, filtfilt
import logging
import traceback
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _safe_fps(signals, default: float = 30.0) -> float:
    """
    Extract the validated FPS that rppg_core stamped into the signals dict.
    Falls back to `default` if missing or invalid.
    """
    if isinstance(signals, dict):
        fps = signals.get("_fps", default)
        if isinstance(fps, (int, float)) and 5 < fps <= 120:
            return float(fps)
    return default


def filter_signal(signal, fps: float = 30.0):
    """Bandpass filter isolating heart-rate frequencies (0.7–3.0 Hz)."""
    if signal is None or len(signal) == 0:
        return np.array([])
    if len(signal) < 10:
        return signal

    if fps <= 0 or fps > 120:
        logger.warning(f"filter_signal: invalid fps={fps}, defaulting to 30")
        fps = 30.0

    g = signal[:, 1] if (len(signal.shape) > 1 and signal.shape[1] >= 2) else signal
    if len(g) < 9:
        return g

    nyq = 0.5 * fps
    lo  = max(0.01, 0.7 / nyq)
    hi  = min(0.99, 3.0 / nyq)

    try:
        b, a = butter(3, [lo, hi], btype='band')
        return filtfilt(b, a, g)
    except Exception as e:
        logger.error(f"Filter error: {e}")
        return g


def analyze_liveness(
    signals,
    fps: float = 30.0,
    challenge_result:  Optional[Dict] = None,
    bcg_result:        Optional[Dict] = None,
    imu_bcg_result:    Optional[Dict] = None,
    challenge_was_required: bool = True,
) -> tuple:
    """
    Combined liveness check — STRICT MODE.

    Decision rules:
      • If challenge_was_required and challenge FAILED → REJECT immediately.
        A recorded video cannot respond to a random challenge it never saw.
      • Require 2/3 layers to pass overall (camera BCG, rPPG, challenge).
      • IMU BCG is Layer 4 (optional).  When present and passing it adds
        one extra layer.  When present and failing (harmonic artifact or
        challenge motion mismatch) it is logged as a warning and does NOT
        count as a passing layer — it does not hard-block on its own.
      • rPPG coherence must be in [0.15, 0.95] to count as a passing layer
        (near-zero = no pulse signal; near-1.0 = perfect screen correlation).
      • Hard-block on coherence > 0.90 (screen replay) or < -0.30 (anti-phase).
      • On ANY unhandled exception → return FAIL (fail closed, not open).

    Returns: (is_real: bool, coherence_score: float, reason: str)
    """
    try:
        fps = _safe_fps(signals, default=fps)

        # ── Layer 1: rPPG coherence ───────────────────────────────────────────
        coherence  = 0.0
        rppg_valid = False

        if signals:
            roi_keys = ["forehead", "left_cheek", "right_cheek"]
            valid = all(
                roi in signals
                and signals[roi] is not None
                and len(signals[roi]) > 0
                for roi in roi_keys
            )
            if valid:
                fh = filter_signal(signals["forehead"],   fps)
                lc = filter_signal(signals["left_cheek"], fps)

                if len(fh) > 0 and len(lc) > 0:
                    ml = min(len(fh), len(lc))
                    if ml >= 5:
                        fh, lc = fh[:ml], lc[:ml]
                        if np.std(fh) > 0 and np.std(lc) > 0:
                            mat = np.corrcoef(fh, lc)
                            coherence  = 0.0 if np.isnan(mat[0, 1]) else float(mat[0, 1])
                            rppg_valid = True

        logger.info(f"rPPG coherence: {coherence:.4f} (valid={rppg_valid}, fps={fps})")

        # ── HARD-BLOCK 1: Screen replay ───────────────────────────────────────
        # Lowered threshold from 0.98 → 0.90.
        # Real skin inter-ROI coherence rarely exceeds 0.85 due to melanin
        # variation, micro-expressions and lighting gradients.
        # A static screen or looped video produces near-perfect correlation.
        if coherence > 0.90:
            return False, coherence, (
                f"Screen Replay Detected (coherence={coherence:.3f} > 0.90). "
                f"Please present your live face."
            )

        # ── HARD-BLOCK 2: Anti-phase signal ───────────────────────────────────
        if rppg_valid and coherence < -0.30:
            return False, coherence, (
                f"Anti-phase rPPG signal detected (coherence={coherence:.3f}). "
                f"Likely a screen or photo replay."
            )

        # ── Layer 2: Challenge-Response ───────────────────────────────────────
        challenge_ok = False
        challenge_attempted = challenge_result is not None

        if challenge_attempted:
            challenge_ok = challenge_result.get("passed", False)
            logger.info(f"Challenge: {'PASSED' if challenge_ok else 'FAILED'}")

        # MANDATORY CHALLENGE GATE
        # If a challenge was required (login flow) and it definitively failed,
        # reject immediately. This is the primary spoof guard:
        # a pre-recorded video cannot know which random challenges were issued.
        if challenge_was_required and challenge_attempted and not challenge_ok:
            reason_detail = challenge_result.get("reason", "unknown") if challenge_result else "not completed"
            logger.warning(f"MANDATORY challenge FAILED: {reason_detail}")
            return False, coherence, (
                f"Liveness Rejected — required challenge not completed. "
                f"({reason_detail})"
            )

        # ── Layer 3: Camera BCG ───────────────────────────────────────────────
        bcg_ok = False
        if bcg_result is not None:
            bcg_ok = bcg_result.get("passed", False)
            logger.info(f"Camera BCG: {'PASSED' if bcg_ok else 'FAILED/weak'}")

        # ── Layer 4: IMU BCG (optional — phone in-hand Z-axis cardiac recoil) ─
        # When the frontend sends accelerometer data, this layer provides a
        # physically independent cross-check: a camera replay cannot produce
        # a real cardiac recoil signal in the phone's IMU.
        imu_bcg_ok      = False
        imu_bcg_present = imu_bcg_result is not None
        if imu_bcg_present:
            imu_bcg_ok   = imu_bcg_result.get("passed", False)
            imu_harmonic = imu_bcg_result.get("is_harmonic", False)
            logger.info(
                f"IMU BCG: {'PASSED' if imu_bcg_ok else 'FAILED'} "
                f"(hr={imu_bcg_result.get('imu_hr_bpm', 0):.1f} BPM, "
                f"harmonic={imu_harmonic}, "
                f"challenge_motion_ok={imu_bcg_result.get('challenge_imu_ok', True)})"
            )
            if imu_harmonic:
                logger.warning(
                    f"IMU BCG HARMONIC ARTIFACT: ratio="
                    f"{imu_bcg_result.get('harmonic_ratio', '?')} — "
                    "possible synthetic IMU injection or signal noise"
                )

        # ── DECISION LOGIC ─────────────────────────────────────────────────────
        # rPPG counts only if coherence is in a physiologically plausible range.
        # Too low  (< 0.15) = noise / no signal.
        # Too high (> 0.90) = already hard-blocked above.
        rppg_passes = rppg_valid and (0.15 <= coherence <= 0.90)

        layers_passed = 0
        if rppg_passes:
            layers_passed += 1
        if challenge_ok:
            layers_passed += 1
        if bcg_ok:
            layers_passed += 1
        if imu_bcg_present and imu_bcg_ok:
            layers_passed += 1

        total_layers = 3 + (1 if imu_bcg_present else 0)

        logger.info(
            f"Layers passed: {layers_passed}/{total_layers}  "
            f"(rPPG={rppg_passes}, challenge={challenge_ok}, "
            f"camera_BCG={bcg_ok}, "
            f"IMU_BCG={imu_bcg_ok if imu_bcg_present else 'N/A'})"
        )

        # Require at least 2 layers (same bar regardless of IMU presence).
        # If a challenge was required, it MUST be one of the passing layers —
        # BCG + rPPG cannot silently bypass a failed challenge.
        if layers_passed >= 2:
            if challenge_was_required and not challenge_ok:
                # This path should be unreachable (caught above), but be safe.
                return False, coherence, "Liveness Rejected — challenge required but not confirmed"
            imu_note = ""
            if imu_bcg_present:
                imu_note = f", IMU BCG={'✓' if imu_bcg_ok else '✗ (warning only)'}"
            return True, coherence, (
                f"Liveness Confirmed ({layers_passed}/{total_layers} layers{imu_note})"
            )

        if layers_passed == 0:
            return False, coherence, "No physiological signal detected"

        return False, coherence, (
            f"Insufficient liveness evidence ({layers_passed}/{total_layers} layers). "
            f"Ensure good lighting and complete all prompts."
        )

    except Exception as e:
        logger.error(f"analyze_liveness error: {traceback.format_exc()}")
        # FAIL CLOSED — an exception is not evidence of a live user.
        return False, 0.0, f"Liveness analysis error — please retry. ({str(e)})"