import numpy as np
from scipy.signal import butter, filtfilt
import logging
import traceback

logger = logging.getLogger(__name__)


def filter_signal(signal, fps=30):
    """Bandpass filter to isolate heart rate frequencies (0.7–3.0 Hz)."""
    if signal is None or len(signal) == 0:
        return np.array([])
    if len(signal) < 10:
        return signal

    g_signal = signal[:, 1] if (len(signal.shape) > 1 and signal.shape[1] >= 2) else signal

    if len(g_signal) < 9:
        return g_signal

    nyquist = 0.5 * fps
    low = max(0.01, 0.7 / nyquist)
    high = min(0.99, 3.0 / nyquist)

    try:
        b, a = butter(3, [low, high], btype='band')
        return filtfilt(b, a, g_signal)
    except Exception as e:
        logger.error(f"Filter error: {e}")
        return g_signal


def analyze_liveness(signals, fps=30):
    try:
        if not signals:
            return False, 0.0, "No signals provided"

        for roi in ["forehead", "left_cheek", "right_cheek"]:
            if roi not in signals or signals[roi] is None or len(signals[roi]) == 0:
                return False, 0.0, f"Missing or empty {roi} signal"

        fh = filter_signal(signals["forehead"], fps)
        lc = filter_signal(signals["left_cheek"], fps)
        rc = filter_signal(signals["right_cheek"], fps)

        if len(fh) == 0 or len(lc) == 0:
            return False, 0.0, "Signal filtering failed"

        min_len = min(len(fh), len(lc), len(rc) if len(rc) > 0 else len(fh))
        if min_len < 5:
            return False, 0.0, f"Too few frames: {min_len}"

        fh, lc = fh[:min_len], lc[:min_len]

        if np.std(fh) == 0 or np.std(lc) == 0:
            coherence = 0.0
        else:
            mat = np.corrcoef(fh, lc)
            coherence = 0.0 if np.isnan(mat[0, 1]) else float(mat[0, 1])

        logger.info(f"Coherence score: {coherence:.4f}")

        if coherence > 0.95:
            return False, coherence, "Screen Replay Detected (Uniform Spatial Coherence)"
        elif coherence < 0.05:
            return False, coherence, "No valid physiological signal (Printed Photo / Noise)"
        else:
            return True, coherence, "Liveness Confirmed"

    except Exception as e:
        logger.error(f"analyze_liveness error: {e}\n{traceback.format_exc()}")
        return False, 0.0, f"Analysis error: {str(e)}"