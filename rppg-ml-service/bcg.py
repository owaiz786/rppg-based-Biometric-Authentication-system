"""
bcg.py  —  Micro-Ballistocardiography (BCG) Liveness Detection
═══════════════════════════════════════════════════════════════
Each heartbeat ejects blood into the aorta, creating a tiny recoil force
that moves the head ~1-3 pixels vertically at the heart rate frequency.

Algorithm
─────────
1. Detect stable facial landmark points in frame 0 (nose bridge + forehead).
2. Track those points frame-to-frame with Lucas-Kanade sparse optical flow.
3. Record the mean vertical (Y) displacement between consecutive frames.
4. Bandpass-filter the displacement signal at 0.7–3.0 Hz (42–180 BPM).
5. Find the dominant frequency via FFT.
6. Independently estimate heart rate from rPPG (green channel FFT).
7. If both frequencies agree within ±0.4 Hz  →  BCG confirmed (live person).
   A printed photo / screen cannot produce correlated sub-pixel motion.

Returns a dict with:
    bcg_hr_bpm      : float   BCG-estimated heart rate
    rppg_hr_bpm     : float   rPPG-estimated heart rate (from green channel)
    freq_match      : bool    True if the two estimates agree within tolerance
    bcg_signal_power: float   Power of the filtered BCG signal (>0 = motion present)
    passed          : bool    Overall BCG liveness verdict
    reason          : str
"""

import cv2
import numpy as np
import logging
import traceback
from typing import Dict

logger = logging.getLogger(__name__)

# ── Landmark indices used as optical-flow tracking points ────────────────────
# Nose bridge + inner brow points — rigid, minimal expression change
TRACK_LANDMARK_INDICES = [
    6,    # nose bridge top
    197,  # nose bridge middle
    4,    # nose tip
    168,  # between brows
    8,    # mid forehead
    9,    # upper forehead
]

# ── Tuning parameters ────────────────────────────────────────────────────────
BCG_LOW_HZ        = 0.7    # 42 BPM
BCG_HIGH_HZ       = 3.0    # 180 BPM
# Tightened from 0.40 → 0.25 Hz.
# At 0.40 Hz the check was accepting BCG readings that were harmonics
# of the true rPPG frequency (e.g. rPPG=67 BPM, BCG=134 BPM, diff=1.1 Hz
# in freq space — which the old forgiving fallback let through).
FREQ_MATCH_TOL_HZ = 0.25
MIN_SIGNAL_POWER  = 1e-8   # Very sensitive — catches weak heartbeat motion
MIN_FRAMES        = 30     # ~1 second at 30 fps

# Harmonic ratios to check for aliasing artifacts.
# A spoofed video often causes BCG to lock onto the 2nd or 0.5× harmonic
# of the rPPG frequency rather than the fundamental.
HARMONIC_RATIOS   = [2.0, 0.5, 3.0, 1.0/3.0]  # multiples to flag as harmonic

# ── Lucas-Kanade parameters ───────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)


def _safe_fps(raw_fps: float) -> float:
    """
    Clamp FPS to a sane range.
    Browser WebM files often report 0 or 1000 — default to 30 in those cases.
    """
    if raw_fps and 5 < raw_fps <= 120:
        return float(raw_fps)
    logger.warning(f"BCG: unreliable FPS ({raw_fps}) — defaulting to 30")
    return 30.0


def _bandpass(signal: np.ndarray, fps: float,
              low: float = BCG_LOW_HZ, high: float = BCG_HIGH_HZ) -> np.ndarray:
    """Bandpass filter for heart-rate frequencies."""
    from scipy.signal import butter, filtfilt

    if len(signal) < 10:
        return signal

    nyq = 0.5 * fps
    lo  = max(0.01, low  / nyq)
    hi  = min(0.99, high / nyq)

    try:
        b, a = butter(3, [lo, hi], btype='band')
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.error(f"BCG bandpass error: {e}")
        return signal


def _dominant_freq(signal: np.ndarray, fps: float,
                   low: float = BCG_LOW_HZ, high: float = BCG_HIGH_HZ) -> float:
    """Return the dominant frequency (Hz) in the bandpass window via FFT."""
    n = len(signal)
    if n < 10:
        return 0.0

    freqs   = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(signal - signal.mean()))

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0

    peak_idx = np.argmax(fft_mag[mask])
    return float(freqs[mask][peak_idx])


def analyze_bcg(video_path: str) -> Dict:
    """
    Main entry point.  Processes a pre-recorded video and returns the BCG
    liveness verdict dict (see module docstring for field descriptions).
    FORGIVING MODE — errs on side of accepting real users.
    """
    result = {
        "passed":           False,
        "bcg_hr_bpm":       0.0,
        "rppg_hr_bpm":      0.0,
        "freq_match":       False,
        "bcg_signal_power": 0.0,
        "frames_tracked":   0,
        "reason":           "",
    }

    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,       # tracking mode for video
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    except Exception:
        result["reason"] = f"MediaPipe init failed: {traceback.format_exc()}"
        return result

    # FIX: Try multiple backends — browser WebM needs FFMPEG
    cap = None
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            break

    if not cap or not cap.isOpened():
        result["reason"] = "Could not open video"
        face_mesh.close()
        return result

    # FIX: Sanitize FPS before using it in any signal-processing calculations
    fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"BCG: video fps={fps:.1f}")

    # ── Phase 1: collect optical-flow displacement signal ────────────────────
    prev_gray  = None
    track_pts  = None          # shape (N,1,2) float32
    vert_disps = []            # per-frame mean vertical displacement (px)
    green_vals = []            # per-frame mean green channel over face ROI

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Resize to at most 640 wide — speeds up MediaPipe and LK flow
        h_orig, w_orig = frame.shape[:2]
        if w_orig > 640:
            scale = 640.0 / w_orig
            frame = cv2.resize(frame, (640, int(h_orig * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm_result = face_mesh.process(rgb)

        if not lm_result.multi_face_landmarks:
            prev_gray = gray.copy()
            track_pts = None    # reset if face lost
            continue

        lms  = lm_result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # ── Initialise / reinitialise tracking points ─────────────────────
        # FIX: Removed the `continue` that was skipping the reinit frame's
        # displacement entirely. We now reinit but still process the frame.
        refresh = (track_pts is None) or (frame_idx % 90 == 0)
        if refresh:
            pts = np.array(
                [[lms[i].x * w, lms[i].y * h] for i in TRACK_LANDMARK_INDICES],
                dtype=np.float32
            ).reshape(-1, 1, 2)
            track_pts = pts
            prev_gray = gray.copy()
            # On a hard reinit (track_pts was None) we cannot compute flow yet
            if not refresh or track_pts is None:
                continue
            # On a periodic refresh we fall through and use the new points
            # in the next frame — skip this frame's flow measurement
            if frame_idx % 90 == 0:
                prev_gray = gray.copy()
                continue

        # ── Lucas-Kanade optical flow ─────────────────────────────────────
        if prev_gray is None:
            prev_gray = gray.copy()
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, track_pts, None, **LK_PARAMS
        )

        good_prev = track_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) < 2:
            prev_gray = gray.copy()
            continue

        # Ensure 2-D arrays
        good_prev = good_prev.reshape(-1, 2)
        good_next = good_next.reshape(-1, 2)

        if good_prev.shape[1] < 2 or good_next.shape[1] < 2:
            logger.warning(
                f"BCG: unexpected shape — prev:{good_prev.shape}, next:{good_next.shape}"
            )
            prev_gray = gray.copy()
            continue

        try:
            dy = float(np.mean(good_next[:, 1] - good_prev[:, 1]))
        except IndexError as e:
            logger.error(
                f"BCG index error: {e}, shapes: prev={good_prev.shape}, next={good_next.shape}"
            )
            prev_gray = gray.copy()
            continue

        vert_disps.append(dy)

        track_pts = good_next.reshape(-1, 1, 2)
        prev_gray = gray.copy()

        # ── Green channel for rPPG reference ─────────────────────────────
        nose_pts = np.array(
            [[int(lms[i].x * w), int(lms[i].y * h)] for i in [6, 168, 4]],
            dtype=np.int32
        )
        x, y, bw, bh = cv2.boundingRect(nose_pts)
        x  = max(0, x - 10)
        y  = max(0, y - 10)
        bw = min(bw + 20, w - x)
        bh = min(bh + 20, h - y)
        roi = frame[y:y + bh, x:x + bw]
        if roi.size > 0:
            green_vals.append(float(np.mean(roi[:, :, 1])))  # BGR → G at index 1

    cap.release()
    face_mesh.close()

    result["frames_tracked"] = len(vert_disps)
    logger.info(f"BCG: {len(vert_disps)} displacement frames, {len(green_vals)} green frames")

    if len(vert_disps) < MIN_FRAMES:
        result["reason"] = (
            f"Too few tracked frames ({len(vert_disps)}) — "
            f"need at least {MIN_FRAMES} (~1 s). Hold camera steady."
        )
        return result

    # ── Phase 2: BCG signal analysis ─────────────────────────────────────────
    dy_arr = np.array(vert_disps, dtype=np.float64)

    # Detrend — remove slow head-pose drift
    window_size = min(int(fps * 2), len(dy_arr) // 2)
    if window_size > 1:
        dy_detrended = dy_arr - np.convolve(
            dy_arr, np.ones(window_size) / window_size, mode='same'
        )
    else:
        dy_detrended = dy_arr

    try:
        bcg_filtered = _bandpass(dy_detrended, fps)
    except Exception:
        logger.warning("BCG bandpass failed — using raw detrended signal")
        bcg_filtered = dy_detrended

    bcg_power = float(np.var(bcg_filtered))
    result["bcg_signal_power"] = round(bcg_power, 8)
    logger.info(f"BCG signal power: {bcg_power:.2e}")

    if bcg_power < MIN_SIGNAL_POWER:
        if bcg_power > 1e-9:
            logger.warning("BCG signal very weak but continuing analysis")
        else:
            result["reason"] = (
                "BCG signal power too low — no detectable heartbeat motion. "
                "Possible still image or very short clip."
            )
            return result

    bcg_freq = _dominant_freq(bcg_filtered, fps)
    bcg_hr   = bcg_freq * 60.0
    result["bcg_hr_bpm"] = round(bcg_hr, 1)
    logger.info(f"BCG dominant frequency: {bcg_freq:.3f} Hz → {bcg_hr:.1f} BPM")

    # ── Phase 3: rPPG reference heart rate ───────────────────────────────────
    rppg_hr = 0.0
    if len(green_vals) >= MIN_FRAMES:
        g_arr = np.array(green_vals, dtype=np.float64)
        try:
            g_filtered = _bandpass(g_arr, fps)
            rppg_freq  = _dominant_freq(g_filtered, fps)
            rppg_hr    = rppg_freq * 60.0
        except Exception:
            logger.warning("rPPG reference bandpass failed")

    result["rppg_hr_bpm"] = round(rppg_hr, 1)
    logger.info(f"rPPG reference: {rppg_hr:.1f} BPM")

    # ── Phase 4: frequency agreement check ───────────────────────────────────
    if rppg_hr > 0:
        bcg_freq_hz  = bcg_hr  / 60.0
        rppg_freq_hz = rppg_hr / 60.0
        freq_diff    = abs(bcg_freq_hz - rppg_freq_hz)
        freq_match   = freq_diff <= FREQ_MATCH_TOL_HZ

        # ── Harmonic aliasing detection ───────────────────────────────────
        # Spoofed video often causes BCG to lock onto a harmonic of the rPPG
        # frequency rather than the fundamental (e.g. rPPG=67 BPM, BCG=134 BPM).
        # Passing that off as "BCG in physiological range" is a false pass.
        # We now check whether BCG is simply a harmonic/subharmonic of rPPG.
        is_harmonic = False
        harmonic_ratio = 1.0
        for ratio in HARMONIC_RATIOS:
            expected_hz = rppg_freq_hz * ratio
            if abs(bcg_freq_hz - expected_hz) <= FREQ_MATCH_TOL_HZ:
                is_harmonic = True
                harmonic_ratio = ratio
                break

        result["freq_match"] = freq_match
        logger.info(
            f"Freq diff: |{bcg_freq_hz:.3f} - {rppg_freq_hz:.3f}| = {freq_diff:.3f} Hz "
            f"(tol={FREQ_MATCH_TOL_HZ}) → {'MATCH' if freq_match else 'NO MATCH'} | "
            f"harmonic={is_harmonic} (ratio={harmonic_ratio})"
        )

        if freq_match and not is_harmonic:
            # True agreement at the fundamental frequency
            result["passed"] = True
            result["reason"] = (
                f"BCG confirmed: heartbeat motion ({bcg_hr:.0f} BPM) "
                f"matches rPPG ({rppg_hr:.0f} BPM)"
            )
        elif is_harmonic:
            # BCG is a harmonic artifact — characteristic of screen replay
            result["passed"] = False
            result["reason"] = (
                f"BCG harmonic artifact detected: BCG={bcg_hr:.0f} BPM is "
                f"{harmonic_ratio}× of rPPG={rppg_hr:.0f} BPM. "
                f"Possible screen replay or signal noise."
            )
            logger.warning(result["reason"])
        else:
            # Frequencies don't match at all — fail, no forgiving fallback
            result["passed"] = False
            result["reason"] = (
                f"BCG/rPPG frequency mismatch: BCG={bcg_hr:.0f} BPM, "
                f"rPPG={rppg_hr:.0f} BPM (diff={freq_diff:.2f} Hz > tol={FREQ_MATCH_TOL_HZ} Hz)"
            )
            logger.warning(result["reason"])
    else:
        # No rPPG reference — verdict from BCG alone
        if 40 <= bcg_hr <= 180:
            result["passed"] = True
            result["reason"] = (
                f"BCG motion detected at {bcg_hr:.0f} BPM "
                f"(rPPG reference unavailable — BCG-only verdict)"
            )
        else:
            result["passed"] = False
            result["reason"] = (
                f"BCG dominant frequency {bcg_hr:.0f} BPM outside "
                f"physiological range (40–180 BPM)"
            )

    return result