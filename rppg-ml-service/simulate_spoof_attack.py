"""
simulate_spoof_attack.py
════════════════════════
Simulates a video-replay spoof attack against the liveness pipeline
and generates a set of matplotlib visual reports saved as PNG files.

Usage
─────
    python simulate_spoof_attack.py path/to/your_video.mp4

Output (saved next to the video file)
──────────────────────────────────────
    spoof_report_01_rppg_signals.png      — raw & filtered rPPG waveforms per ROI
    spoof_report_02_rppg_spectrum.png     — FFT frequency spectrum per ROI
    spoof_report_03_bcg_motion.png        — BCG optical-flow displacement + spectrum
    spoof_report_04_layer_scores.png      — weighted layer scoring breakdown
    spoof_report_05_forensic_summary.png  — one-page verdict + all key numbers

Place this file in the SAME folder as:
    rppg_core.py  anti_spoofing.py  bcg.py  challenge_response.py
"""

import sys
import os
import time
import textwrap
import numpy as np

# ── terminal colours (still used for progress prints) ───────────────────────
def _c(code, t):
    return t if os.name == "nt" else f"\033[{code}m{t}\033[0m"
RED    = lambda t: _c("31;1", t)
GREEN  = lambda t: _c("32;1", t)
YELLOW = lambda t: _c("33;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)

def _step(msg): print(f"  {DIM('...')}  {msg}")
def _ok(msg):   print(f"  {GREEN('OK')}   {msg}")
def _warn(msg): print(f"  {YELLOW('!')}    {msg}")
def _fail(msg): print(f"  {RED('FAIL')} {msg}")


# ── signal helpers ───────────────────────────────────────────────────────────
def _bandpass(sig, fps, lo=0.7, hi=3.0):
    from scipy.signal import butter, filtfilt
    if len(sig) < 10:
        return sig
    nyq = 0.5 * fps
    b, a = butter(3, [max(0.01, lo / nyq), min(0.99, hi / nyq)], btype="band")
    return filtfilt(b, a, sig)

def _fft_spectrum(sig, fps, lo=0.5, hi=4.0):
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag   = np.abs(np.fft.rfft(sig - sig.mean()))
    mask  = (freqs >= lo) & (freqs <= hi)
    return freqs[mask], mag[mask]

def _dominant_hz(sig, fps, lo=0.7, hi=3.0):
    freqs, mag = _fft_spectrum(sig, fps, lo, hi)
    return float(freqs[np.argmax(mag)]) if len(freqs) else 0.0

def _coherence_pairs(fh, lc, rc):
    corrs = []
    for a, b in [(fh, lc), (fh, rc), (lc, rc)]:
        if np.std(a) > 0 and np.std(b) > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(r):
                corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0

def _harmonic_check(bcg_hz, rppg_hz, tol=0.25):
    for ratio in [2.0, 0.5, 3.0, 1 / 3]:
        if abs(bcg_hz - rppg_hz * ratio) <= tol:
            return True, ratio
    return False, 1.0


# ── matplotlib style ─────────────────────────────────────────────────────────
def _style():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":  "#0f1117",
        "axes.facecolor":    "#1a1d27",
        "axes.edgecolor":    "#2e3148",
        "axes.labelcolor":   "#c8ccd8",
        "axes.titlecolor":   "#e8eaf0",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "axes.titleweight":  "bold",
        "axes.grid":         True,
        "grid.color":        "#2e3148",
        "grid.linewidth":    0.6,
        "xtick.color":       "#888ba0",
        "ytick.color":       "#888ba0",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        "#c8ccd8",
        "lines.linewidth":   1.4,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        140,
    })
    return plt

ROI_COLOURS = {
    "forehead":    "#2dd4a0",   # teal
    "left_cheek":  "#5b9cf6",   # blue
    "right_cheek": "#f47560",   # coral
}
PASS_CLR  = "#2dd4a0"
FAIL_CLR  = "#f04747"
WARN_CLR  = "#f5a623"
NEUT_CLR  = "#5b9cf6"


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — rPPG waveforms  (raw green channel + bandpass filtered)
# ════════════════════════════════════════════════════════════════════════════
def fig_rppg_signals(signals, filtered, fps, coherence, out_path):
    plt = _style()
    import matplotlib.pyplot as mpl_plt
    rois   = ["forehead", "left_cheek", "right_cheek"]
    labels = ["Forehead", "Left cheek", "Right cheek"]

    fig, axes = mpl_plt.subplots(3, 2, figsize=(13, 8))
    fig.suptitle("Figure 1 — rPPG signal per ROI  (raw green channel vs bandpass filtered)",
                 color="#e8eaf0", fontsize=12, fontweight="bold", y=1.01)

    for row, (roi, label) in enumerate(zip(rois, labels)):
        clr = ROI_COLOURS[roi]
        arr = signals.get(roi)
        if arr is None or len(arr) == 0:
            continue
        arr = np.array(arr)
        g   = arr[:, 1] if arr.ndim > 1 else arr
        t   = np.arange(len(g)) / fps

        # Left: raw
        ax_raw = axes[row, 0]
        ax_raw.plot(t, g, color=clr, alpha=0.85, linewidth=1.2)
        ax_raw.set_title(f"{label} — raw green channel")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Mean pixel intensity")

        # Right: filtered
        ax_f = axes[row, 1]
        filt = filtered.get(roi)
        if filt is not None:
            ax_f.plot(t[:len(filt)], filt, color=clr, linewidth=1.4)
            ax_f.axhline(0, color="#444", linewidth=0.6, linestyle="--")
        ax_f.set_title(f"{label} — bandpass filtered (0.7–3.0 Hz)")
        ax_f.set_xlabel("Time (s)")
        ax_f.set_ylabel("Amplitude")

    # Coherence annotation
    coh_clr = FAIL_CLR if coherence < -0.30 else (WARN_CLR if coherence < 0.05 else PASS_CLR)
    coh_txt = (
        f"Mean inter-ROI coherence: {coherence:.4f}   "
        f"{'[HARD BLOCK — anti-phase]' if coherence < -0.30 else '[negative — screen noise]' if coherence < 0 else '[OK]'}"
    )
    fig.text(0.5, -0.01, coh_txt, ha="center", fontsize=9,
             color=coh_clr, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    mpl_plt.close(fig)
    _ok(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — rPPG frequency spectrum
# ════════════════════════════════════════════════════════════════════════════
def fig_rppg_spectrum(filtered, hr_per_roi, fps, coherence, out_path):
    plt = _style()
    import matplotlib.pyplot as mpl_plt

    rois   = ["forehead", "left_cheek", "right_cheek"]
    labels = ["Forehead", "Left cheek", "Right cheek"]

    fig, axes = mpl_plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Figure 2 — rPPG FFT frequency spectrum per ROI",
                 color="#e8eaf0", fontsize=12, fontweight="bold")

    for ax, roi, label in zip(axes, rois, labels):
        clr  = ROI_COLOURS[roi]
        filt = filtered.get(roi)
        if filt is None or len(filt) < 10:
            ax.set_title(f"{label}\n(no data)")
            continue
        freqs, mag = _fft_spectrum(filt, fps, lo=0.5, hi=4.0)
        bpm        = freqs * 60.0
        ax.fill_between(bpm, mag, alpha=0.3, color=clr)
        ax.plot(bpm, mag, color=clr, linewidth=1.5)

        # Cardiac band shading
        ax.axvspan(42, 180, alpha=0.07, color=PASS_CLR, label="Cardiac band")

        # Peak
        peak_bpm = hr_per_roi.get(roi, 0.0)
        if peak_bpm > 0:
            peak_idx = np.argmin(np.abs(bpm - peak_bpm))
            ax.axvline(peak_bpm, color=WARN_CLR, linewidth=1.2, linestyle="--")
            ax.annotate(f"{peak_bpm:.0f} BPM",
                        xy=(peak_bpm, mag[peak_idx]),
                        xytext=(peak_bpm + 8, mag[peak_idx] * 0.85),
                        fontsize=8, color=WARN_CLR,
                        arrowprops=dict(arrowstyle="->", color=WARN_CLR, lw=0.8))

        ax.set_title(f"{label}")
        ax.set_xlabel("Heart rate (BPM)")
        ax.set_ylabel("FFT magnitude")
        ax.legend(fontsize=7, loc="upper right")

    coh_clr = FAIL_CLR if coherence < 0 else PASS_CLR
    fig.text(0.5, -0.04,
             f"Coherence {coherence:.4f} — ROI peaks should align for a live face; "
             f"divergent peaks indicate screen noise",
             ha="center", fontsize=9, color=coh_clr)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    mpl_plt.close(fig)
    _ok(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — BCG micro-motion displacement + spectrum
# ════════════════════════════════════════════════════════════════════════════
def fig_bcg(video_path, bcg_result, rppg_hr, fps, out_path):
    """Re-run the BCG optical flow to collect the raw displacement signal."""
    import cv2

    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)
    except Exception as e:
        _warn(f"MediaPipe unavailable for BCG figure: {e}")
        return

    TRACK_IDX = [6, 197, 4, 168, 8, 9]
    LK = dict(winSize=(15, 15), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    vert_disps = []
    prev_gray = None
    track_pts = None
    fidx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            prev_gray = gray.copy(); track_pts = None; continue

        lms = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        if track_pts is None or fidx % 90 == 0:
            pts = np.array([[lms[i].x * w, lms[i].y * h] for i in TRACK_IDX],
                           dtype=np.float32).reshape(-1, 1, 2)
            track_pts = pts; prev_gray = gray.copy(); continue

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, track_pts, None, **LK)
        good_p = track_pts[status.flatten() == 1].reshape(-1, 2)
        good_n = nxt[status.flatten() == 1].reshape(-1, 2)
        if len(good_p) >= 2:
            vert_disps.append(float(np.mean(good_n[:, 1] - good_p[:, 1])))
        track_pts = good_n.reshape(-1, 1, 2)
        prev_gray = gray.copy()

    cap.release()
    face_mesh.close()

    if len(vert_disps) < 10:
        _warn("Too few BCG frames to plot")
        return

    dy  = np.array(vert_disps)
    t   = np.arange(len(dy)) / fps

    # Bandpass
    try:
        dy_filt = _bandpass(dy, fps)
    except Exception:
        dy_filt = dy

    # Spectrum
    freqs_raw, mag_raw   = _fft_spectrum(dy,      fps, lo=0.3, hi=4.0)
    freqs_filt, mag_filt = _fft_spectrum(dy_filt, fps, lo=0.3, hi=4.0)
    bpm_raw  = freqs_raw  * 60.0
    bpm_filt = freqs_filt * 60.0

    bcg_hr   = bcg_result.get("bcg_hr_bpm", 0.0)
    is_harm, ratio = _harmonic_check(bcg_hr / 60.0, rppg_hr / 60.0) if rppg_hr > 0 else (False, 1.0)

    plt = _style()
    import matplotlib.pyplot as mpl_plt

    fig, axes = mpl_plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle("Figure 3 — BCG micro-motion (head displacement via optical flow)",
                 color="#e8eaf0", fontsize=12, fontweight="bold")

    # 1 raw displacement
    ax = axes[0, 0]
    ax.plot(t, dy, color=NEUT_CLR, alpha=0.7, linewidth=1.0, label="Raw displacement")
    ax.axhline(0, color="#444", linewidth=0.6, linestyle="--")
    ax.set_title("Raw vertical head displacement")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (px)")
    ax.legend(fontsize=7)

    # 2 filtered displacement
    ax = axes[0, 1]
    ax.plot(t[:len(dy_filt)], dy_filt, color="#7F77DD", linewidth=1.4, label="Bandpass filtered")
    ax.axhline(0, color="#444", linewidth=0.6, linestyle="--")
    ax.set_title("Filtered displacement (0.7–3.0 Hz cardiac band)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)

    # 3 raw spectrum
    ax = axes[1, 0]
    ax.fill_between(bpm_raw, mag_raw, alpha=0.25, color=NEUT_CLR)
    ax.plot(bpm_raw, mag_raw, color=NEUT_CLR, linewidth=1.3)
    ax.axvspan(42, 180, alpha=0.08, color=PASS_CLR, label="Cardiac band")
    if bcg_hr > 0:
        ax.axvline(bcg_hr, color=WARN_CLR, linewidth=1.4, linestyle="--",
                   label=f"BCG peak: {bcg_hr:.0f} BPM")
    if rppg_hr > 0:
        ax.axvline(rppg_hr, color=PASS_CLR, linewidth=1.4, linestyle=":",
                   label=f"rPPG ref: {rppg_hr:.0f} BPM")
    ax.set_title("BCG frequency spectrum (raw)")
    ax.set_xlabel("Heart rate (BPM)")
    ax.set_ylabel("FFT magnitude")
    ax.legend(fontsize=7)

    # 4 filtered spectrum + harmonic annotation
    ax = axes[1, 1]
    ax.fill_between(bpm_filt, mag_filt, alpha=0.25, color="#7F77DD")
    ax.plot(bpm_filt, mag_filt, color="#7F77DD", linewidth=1.4)
    ax.axvspan(42, 180, alpha=0.08, color=PASS_CLR)
    if bcg_hr > 0:
        ax.axvline(bcg_hr, color=WARN_CLR, linewidth=1.4, linestyle="--",
                   label=f"BCG peak: {bcg_hr:.0f} BPM")
    if rppg_hr > 0:
        ax.axvline(rppg_hr, color=PASS_CLR, linewidth=1.4, linestyle=":",
                   label=f"rPPG ref: {rppg_hr:.0f} BPM")
    ax.set_title("BCG frequency spectrum (filtered)")
    ax.set_xlabel("Heart rate (BPM)")
    ax.set_ylabel("FFT magnitude")

    if is_harm:
        ax.set_title(f"BCG filtered — HARMONIC ARTIFACT x{ratio}")
        ann = f"BCG {bcg_hr:.0f} = {ratio}x rPPG {rppg_hr:.0f} BPM\nScreen refresh aliasing"
        ax.annotate(ann, xy=(bcg_hr, 0), xytext=(bcg_hr + 10, max(mag_filt) * 0.5),
                    fontsize=8, color=FAIL_CLR,
                    arrowprops=dict(arrowstyle="->", color=FAIL_CLR, lw=0.9))
    ax.legend(fontsize=7)

    # Power / signal quality annotation
    power = bcg_result.get("bcg_signal_power", 0.0)
    fig.text(0.5, -0.02,
             f"BCG signal power: {power:.3e}   |   Frames tracked: {len(vert_disps)}   |   "
             f"Freq match: {bcg_result.get('freq_match', False)}   |   "
             f"Harmonic artifact: {is_harm}",
             ha="center", fontsize=9,
             color=FAIL_CLR if is_harm else (WARN_CLR if not bcg_result.get("freq_match") else PASS_CLR))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    mpl_plt.close(fig)
    _ok(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Layer scoring breakdown
# ════════════════════════════════════════════════════════════════════════════
def fig_layer_scores(rppg_s, bcg_s, chal_s, weighted, coherence,
                     bcg_hr, rppg_hr, bcg_result, chal_result,
                     is_real, reason, out_path):
    plt = _style()
    import matplotlib.pyplot as mpl_plt
    import matplotlib.patches as mpatches

    THRESHOLD = 0.55
    W = [0.35, 0.35, 0.30]
    scores  = [rppg_s, bcg_s, chal_s]
    labels  = ["rPPG\n(weight 35%)", "BCG\n(weight 35%)", "Challenge\n(weight 30%)"]
    colours = [
        PASS_CLR if rppg_s >= 0.55 else (WARN_CLR if rppg_s >= 0.3 else FAIL_CLR),
        PASS_CLR if bcg_s  >= 0.55 else (WARN_CLR if bcg_s  >= 0.3 else FAIL_CLR),
        PASS_CLR if chal_s >= 0.55 else (WARN_CLR if chal_s >= 0.3 else FAIL_CLR),
    ]

    fig = mpl_plt.figure(figsize=(13, 7))
    fig.suptitle("Figure 4 — Liveness layer scoring breakdown",
                 color="#e8eaf0", fontsize=12, fontweight="bold")

    # ── Left: horizontal bar chart ──────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    y = np.arange(3)
    bars = ax1.barh(y, scores, color=colours, height=0.45, zorder=3)
    ax1.barh(y, [1.0]*3, color="#2e3148", height=0.45, zorder=2)   # background track
    ax1.axvline(THRESHOLD, color=WARN_CLR, linewidth=1.4, linestyle="--",
                label=f"Pass threshold ({THRESHOLD})", zorder=4)
    ax1.axvline(weighted, color=PASS_CLR if is_real else FAIL_CLR,
                linewidth=2.0, linestyle="-",
                label=f"Weighted total ({weighted:.2f})", zorder=5)

    for bar_, score in zip(bars, scores):
        ax1.text(min(score + 0.03, 0.97), bar_.get_y() + bar_.get_height() / 2,
                 f"{score:.2f}", va="center", fontsize=10, fontweight="bold",
                 color="#e8eaf0")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Score (0 = spoof, 1 = live)")
    ax1.set_title("Per-layer score vs threshold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    # ── Right: waterfall / contribution chart ───────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    contribs = [w * s for w, s in zip(W, scores)]
    contrib_labels = [
        f"rPPG × 0.35 = {contribs[0]:.3f}",
        f"BCG × 0.35 = {contribs[1]:.3f}",
        f"Chal × 0.30 = {contribs[2]:.3f}",
    ]
    ccolours = [
        PASS_CLR if c >= 0.55 * w else (WARN_CLR if c >= 0.3 * w else FAIL_CLR)
        for c, w in zip(contribs, W)
    ]
    bars2 = ax2.bar(contrib_labels, contribs, color=ccolours, width=0.5, zorder=3)
    ax2.axhline(THRESHOLD, color=WARN_CLR, linewidth=1.4, linestyle="--",
                label=f"Threshold contribution reference")
    ax2.axhline(weighted, color=PASS_CLR if is_real else FAIL_CLR,
                linewidth=2.0, linestyle="-",
                label=f"Weighted total = {weighted:.3f}")

    for bar_, val in zip(bars2, contribs):
        ax2.text(bar_.get_x() + bar_.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#e8eaf0")

    ax2.set_ylim(0, max(0.75, max(contribs) + 0.12))
    ax2.set_title("Weighted contribution per layer")
    ax2.set_ylabel("Contribution to total score")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", labelsize=8)

    # Final verdict strip
    vclr = PASS_CLR if is_real else FAIL_CLR
    vtxt = "LIVENESS CONFIRMED" if is_real else "SPOOF DETECTED — LOGIN BLOCKED"
    fig.text(0.5, -0.03, f"{vtxt}   |   Score {weighted:.3f}  (threshold {THRESHOLD})",
             ha="center", fontsize=11, fontweight="bold", color=vclr)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    mpl_plt.close(fig)
    _ok(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Forensic summary (one-page report)
# ════════════════════════════════════════════════════════════════════════════
def fig_summary(video_path, fps, coherence, rppg_hr, hr_per_roi,
                bcg_result, chal_result, chal_detail,
                rppg_s, bcg_s, chal_s, weighted,
                is_real, reason, is_harmonic, harm_ratio,
                out_path):
    plt = _style()
    import matplotlib.pyplot as mpl_plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    THRESHOLD = 0.55
    fig = mpl_plt.figure(figsize=(14, 10))
    fig.suptitle("Figure 5 — Forensic liveness analysis summary",
                 color="#e8eaf0", fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    vclr = PASS_CLR if is_real else FAIL_CLR
    vtxt = "LIVENESS CONFIRMED" if is_real else "SPOOF DETECTED"

    # ── Top row: big metric cards ────────────────────────────────────────────
    metrics = [
        ("rPPG coherence", f"{coherence:.4f}",
         PASS_CLR if 0.05 < coherence < 0.98 else (WARN_CLR if coherence >= 0 else FAIL_CLR),
         "anti-phase" if coherence < -0.3 else ("negative" if coherence < 0 else
          "near-zero" if coherence < 0.05 else "OK")),
        ("BCG heart rate", f"{bcg_result.get('bcg_hr_bpm', 0):.0f} BPM",
         WARN_CLR if is_harmonic else (PASS_CLR if bcg_result.get("freq_match") else WARN_CLR),
         f"x{harm_ratio} harmonic" if is_harmonic else ("freq match" if bcg_result.get("freq_match") else "mismatch")),
        ("rPPG heart rate", f"{rppg_hr:.0f} BPM",
         NEUT_CLR, "forehead green channel"),
        ("Weighted score", f"{weighted:.3f}",
         PASS_CLR if weighted >= THRESHOLD else FAIL_CLR,
         f"threshold {THRESHOLD}"),
        ("Frames analysed", f"{bcg_result.get('frames_tracked', 0)}",
         NEUT_CLR, "BCG tracking"),
        ("BCG power", f"{bcg_result.get('bcg_signal_power', 0):.2e}",
         PASS_CLR if bcg_result.get("bcg_signal_power", 0) > 1e-8 else FAIL_CLR,
         "signal energy"),
    ]

    for idx, (title, val, clr, sub) in enumerate(metrics):
        row, col = 0, idx % 3
        if idx == 3:
            row = 1
        ax = fig.add_subplot(gs[row, col % 3])
        ax.set_facecolor("#1a1d27")
        ax.axis("off")
        ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=9,
                color="#888ba0", transform=ax.transAxes)
        ax.text(0.5, 0.38, val, ha="center", va="center", fontsize=22,
                fontweight="bold", color=clr, transform=ax.transAxes)
        ax.text(0.5, 0.10, sub, ha="center", va="center", fontsize=8,
                color="#5a5d70", transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3148")
            spine.set_linewidth(0.8)
            spine.set_visible(True)

    # ── Middle row col 0-1: ROI HR comparison bar chart ─────────────────────
    ax_hr = fig.add_subplot(gs[1, :2])
    roi_names = ["Forehead", "Left cheek", "Right cheek"]
    roi_hrs   = [hr_per_roi.get(r, 0) for r in ["forehead", "left_cheek", "right_cheek"]]
    roi_clrs  = list(ROI_COLOURS.values())
    xpos = np.arange(3)
    ax_hr.bar(xpos, roi_hrs, color=roi_clrs, width=0.4, zorder=3)
    if rppg_hr > 0:
        ax_hr.axhline(rppg_hr, color=WARN_CLR, linewidth=1.4, linestyle="--",
                      label=f"Forehead reference ({rppg_hr:.0f} BPM)")
    ax_hr.set_xticks(xpos)
    ax_hr.set_xticklabels(roi_names)
    ax_hr.set_ylabel("Estimated heart rate (BPM)")
    ax_hr.set_title("rPPG heart rate estimate per ROI\n(aligned = physiological; divergent = screen noise)")
    ax_hr.set_ylim(0, max(roi_hrs + [90]) * 1.25)
    for x, hr_v in zip(xpos, roi_hrs):
        ax_hr.text(x, hr_v + 1, f"{hr_v:.0f}", ha="center", fontsize=9,
                   color="#e8eaf0", fontweight="bold")
    ax_hr.legend(fontsize=8)

    # ── Middle row col 2: challenge result table ─────────────────────────────
    ax_c = fig.add_subplot(gs[1, 2])
    ax_c.axis("off")
    ax_c.set_title("Challenge results", fontsize=10, color="#e8eaf0", fontweight="bold")
    rows_data = []
    for name, passed in chal_detail.items():
        rows_data.append([name.replace("_", " ").title(),
                          "PASS" if passed else "FAIL"])
    rows_data.append(["—", "—"])
    rows_data.append(["Overall", "PASS" if chal_result.get("passed") else "FAIL"])
    tbl = ax_c.table(cellText=rows_data,
                     colLabels=["Challenge", "Result"],
                     cellLoc="center", loc="center",
                     bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1d27")
        cell.set_edgecolor("#2e3148")
        cell.set_text_props(color="#c8ccd8")
        if col == 1 and row > 0:
            txt = cell.get_text().get_text()
            if txt == "PASS":
                cell.set_text_props(color=PASS_CLR, fontweight="bold")
            elif txt == "FAIL":
                cell.set_text_props(color=FAIL_CLR, fontweight="bold")
        if row == 0:
            cell.set_facecolor("#2e3148")

    # ── Bottom row: scoring waterfall + verdict text ─────────────────────────
    ax_w = fig.add_subplot(gs[2, :2])
    layers     = ["rPPG\n(×0.35)", "BCG\n(×0.35)", "Challenge\n(×0.30)", "Weighted\ntotal"]
    raw_scores = [rppg_s, bcg_s, chal_s, weighted]
    wts        = [0.35, 0.35, 0.30, 1.0]
    contribs   = [rppg_s*0.35, bcg_s*0.35, chal_s*0.30, weighted]
    bar_clrs   = [
        PASS_CLR if s >= 0.55 else (WARN_CLR if s >= 0.3 else FAIL_CLR)
        for s in raw_scores
    ]
    bar_clrs[-1] = PASS_CLR if is_real else FAIL_CLR

    xb = np.arange(4)
    b  = ax_w.bar(xb[:3], contribs[:3], color=bar_clrs[:3], width=0.45, zorder=3)
    ax_w.bar([3], [weighted], color=bar_clrs[-1], width=0.45, zorder=3)
    ax_w.axhline(THRESHOLD, color=WARN_CLR, linewidth=1.4, linestyle="--",
                 label=f"Pass threshold ({THRESHOLD})")
    ax_w.set_xticks(xb)
    ax_w.set_xticklabels(layers)
    ax_w.set_ylim(0, 0.82)
    ax_w.set_ylabel("Score contribution")
    ax_w.set_title("Weighted score contribution per layer")
    ax_w.legend(fontsize=8)
    for x, val, rs in zip(xb, contribs, raw_scores):
        ax_w.text(x, val + 0.01, f"{val:.3f}\n(raw {rs:.2f})",
                  ha="center", fontsize=8, color="#e8eaf0")

    # Verdict panel
    ax_v = fig.add_subplot(gs[2, 2])
    ax_v.axis("off")
    ax_v.set_facecolor("#1a1d27")
    rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                   boxstyle="round,pad=0.04",
                                   linewidth=2, edgecolor=vclr,
                                   facecolor="#1a1d27",
                                   transform=ax_v.transAxes)
    ax_v.add_patch(rect)
    ax_v.text(0.5, 0.78, "VERDICT", ha="center", va="center",
              fontsize=9, color="#888ba0", transform=ax_v.transAxes)
    ax_v.text(0.5, 0.58, vtxt, ha="center", va="center",
              fontsize=12, fontweight="bold", color=vclr, transform=ax_v.transAxes,
              wrap=True)
    ax_v.text(0.5, 0.35, f"Score: {weighted:.3f}",
              ha="center", va="center", fontsize=11, color="#c8ccd8",
              transform=ax_v.transAxes)

    # Key stop reason
    stop = "Hard-block: anti-phase rPPG" if coherence < -0.3 else \
           "Hard-block: BCG harmonic" if is_harmonic else \
           f"Score {weighted:.2f} < {THRESHOLD}" if not is_real else "All layers passed"
    for i, line in enumerate(textwrap.wrap(stop, 20)):
        ax_v.text(0.5, 0.18 - i * 0.10, line, ha="center", va="center",
                  fontsize=8, color=vclr, transform=ax_v.transAxes)

    # Footer
    fname = os.path.basename(video_path)
    fig.text(0.5, -0.02,
             f"Video: {fname}   |   FPS: {fps:.0f}   |   "
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             ha="center", fontsize=8, color="#5a5d70")

    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    mpl_plt.close(fig)
    _ok(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def run(video_path: str):

    if not os.path.exists(video_path):
        print(RED(f"File not found: {video_path}"))
        sys.exit(1)

    # Output folder = same directory as the video
    out_dir  = os.path.dirname(os.path.abspath(video_path))
    basename = "spoof_report"

    def out(n, name):
        return os.path.join(out_dir, f"{basename}_{n:02d}_{name}.png")

    print(BOLD("\n  Spoof Attack Simulation — Visual Report Generator"))
    print(f"  Video : {video_path}")
    print(f"  Output: {out_dir}\n")

    # ── Import pipeline modules ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from rppg_core          import extract_roi_signals
        from anti_spoofing      import analyze_liveness
        from bcg                import analyze_bcg
        from challenge_response import analyze_challenges
    except ImportError as e:
        print(RED(f"Import error: {e}"))
        print(DIM("Place this script in the same folder as rppg_core.py etc."))
        sys.exit(1)

    # ── Layer 1: rPPG ────────────────────────────────────────────────────────
    _step("Extracting rPPG signals (MediaPipe FaceMesh)...")
    t0 = time.time()
    signals, frame = extract_roi_signals(video_path)
    print(f"       done in {time.time()-t0:.1f}s")

    if signals is None or frame is None:
        _fail("No face detected — cannot continue.")
        sys.exit(1)

    fps  = float(signals.get("_fps", 30.0))
    rois = ["forehead", "left_cheek", "right_cheek"]

    filtered   = {}
    hr_per_roi = {}
    for roi in rois:
        arr = signals.get(roi)
        if arr is None or len(arr) == 0:
            continue
        arr = np.array(arr)
        g   = arr[:, 1] if arr.ndim > 1 else arr
        try:
            filt = _bandpass(g, fps)
            filtered[roi]   = filt
            hr_per_roi[roi] = _dominant_hz(filt, fps) * 60.0
        except Exception:
            pass

    coherence = 0.0
    rppg_valid = False
    rppg_hr    = 0.0
    if all(r in filtered for r in rois):
        ml = min(len(filtered[r]) for r in rois)
        if ml >= 10:
            fh = filtered["forehead"][:ml]
            lc = filtered["left_cheek"][:ml]
            rc = filtered["right_cheek"][:ml]
            coherence  = _coherence_pairs(fh, lc, rc)
            rppg_valid = True
            rppg_hr    = _dominant_hz(fh, fps) * 60.0

    # ── Layer 2: BCG ─────────────────────────────────────────────────────────
    _step("Running BCG optical flow analysis...")
    t0 = time.time()
    bcg = analyze_bcg(video_path)
    print(f"       done in {time.time()-t0:.1f}s")

    bcg_hr     = bcg.get("bcg_hr_bpm",      0.0)
    power      = bcg.get("bcg_signal_power", 0.0)
    freq_match = bcg.get("freq_match",       False)
    bcg_passed = bcg.get("passed",           False)

    is_harmonic, harm_ratio = False, 1.0
    if rppg_hr > 0 and bcg_hr > 0:
        is_harmonic, harm_ratio = _harmonic_check(bcg_hr / 60.0, rppg_hr / 60.0)

    # ── Layer 3: Challenges ──────────────────────────────────────────────────
    _step("Analysing challenge response...")
    t0 = time.time()
    test_challenges = ["blink", "head_turn"]
    chal        = analyze_challenges(video_path, test_challenges, fps=fps)
    print(f"       done in {time.time()-t0:.1f}s")
    chal_detail = chal.get("challenges", {})

    # ── Combined liveness ────────────────────────────────────────────────────
    _step("Computing combined liveness score...")
    is_real, score, reason = analyze_liveness(
        signals, fps=fps, challenge_result=chal, bcg_result=bcg)

    # Reconstruct per-layer scores (mirrors anti_spoofing.py logic)
    if not rppg_valid:
        rppg_s = 0.0
    elif coherence > 0.98 or coherence < -0.30:
        rppg_s = 0.0
    elif coherence < 0.0:
        rppg_s = max(0.0, 0.3 + coherence)
    elif coherence < 0.05:
        rppg_s = 0.1
    else:
        rppg_s = 0.5 + 0.5 * min(1.0, (coherence - 0.05) / 0.75)

    if power < 1e-9:
        bcg_s = 0.0
    elif is_harmonic:
        bcg_s = 0.05
    elif not (40 <= bcg_hr <= 180):
        bcg_s = 0.1
    elif freq_match:
        bcg_s = 1.0
    elif bcg_passed:
        bcg_s = 0.6
    else:
        bcg_s = 0.3

    total_c = len(chal_detail)
    pass_c  = sum(1 for v in chal_detail.values() if v)
    inds    = chal.get("spoof_indicators", [])
    chal_s  = max(0.0, (pass_c / total_c if total_c > 0 else 0.0) - min(0.3, len(inds) * 0.1))

    W_RPPG, W_BCG, W_CHAL = 0.35, 0.35, 0.30
    weighted = W_RPPG * rppg_s + W_BCG * bcg_s + W_CHAL * chal_s

    verdict = "SPOOF DETECTED" if not is_real else "LIVENESS CONFIRMED"
    clr     = RED if not is_real else GREEN
    print(f"\n  {clr(verdict)}  —  score {weighted:.3f}  —  {reason[:80]}\n")

    # ── Generate figures ─────────────────────────────────────────────────────
    _step("Generating Figure 1: rPPG waveforms...")
    fig_rppg_signals(signals, filtered, fps, coherence,
                     out(1, "rppg_signals"))

    _step("Generating Figure 2: rPPG frequency spectrum...")
    fig_rppg_spectrum(filtered, hr_per_roi, fps, coherence,
                      out(2, "rppg_spectrum"))

    _step("Generating Figure 3: BCG micro-motion...")
    fig_bcg(video_path, bcg, rppg_hr, fps,
            out(3, "bcg_motion"))

    _step("Generating Figure 4: layer scoring...")
    fig_layer_scores(rppg_s, bcg_s, chal_s, weighted, coherence,
                     bcg_hr, rppg_hr, bcg, chal, chal_detail,
                     is_real, reason,
                     out(4, "layer_scores"))

    _step("Generating Figure 5: forensic summary...")
    fig_summary(video_path, fps, coherence, rppg_hr, hr_per_roi,
                bcg, chal, chal_detail,
                rppg_s, bcg_s, chal_s, weighted,
                is_real, reason, is_harmonic, harm_ratio,
                out(5, "forensic_summary"))

    print(f"\n  {GREEN('All 5 figures saved to:')}  {out_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(BOLD("\nUsage:"))
        print("  python simulate_spoof_attack.py path/to/video.mp4\n")
        print(DIM("  Supported: .mp4  .webm  .avi  .mov"))
        sys.exit(0)
    run(sys.argv[1])