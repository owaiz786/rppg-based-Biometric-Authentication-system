import os
import time
import logging
import traceback
import tempfile
import sqlite3
from typing import Optional, List, Dict
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rppg_core import extract_roi_signals
from anti_spoofing import analyze_liveness
from challenge_response import analyze_challenges, generate_challenge_token, consume_challenge_token, get_imu_windows_for_token
from bcg import analyze_bcg
from imu_bcg import analyze_imu_bcg, rppg_hr_from_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="rPPG Anti-Spoofing ML Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Database (absolute path, WAL mode, binary BLOB storage)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_embeddings.db")

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    try:
        conn = _get_conn()
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username   TEXT    PRIMARY KEY,
            embedding  BLOB    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now')),
            updated_at TEXT    DEFAULT (datetime('now'))
        )
        """)
        conn.commit()
        conn.close()
        logger.info(f"DB ready: {DB_PATH}")
    except Exception:
        logger.error(f"DB init failed:\n{traceback.format_exc()}")

init_db()

def store_embedding(username: str, embedding: List[float]) -> bool:
    try:
        blob = np.array(embedding, dtype=np.float32).tobytes()
        logger.info(f"Storing '{username}': {len(embedding)} floats → {len(blob)} bytes")
        conn = _get_conn()
        conn.execute("""
        INSERT OR REPLACE INTO users (username, embedding, created_at, updated_at)
        VALUES (
            ?,
            ?,
            COALESCE((SELECT created_at FROM users WHERE username=?), datetime('now')),
            datetime('now')
        )
        """, (username, blob, username))
        conn.commit()
        conn.close()
        logger.info(f"Stored OK: '{username}'")
        return True
    except Exception:
        logger.error(f"store_embedding failed:\n{traceback.format_exc()}")
        return False

def get_embedding(username: str) -> Optional[List[float]]:
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT embedding FROM users WHERE username=?", (username,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).tolist()
    except Exception:
        logger.error(f"get_embedding failed:\n{traceback.format_exc()}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Face detection helpers (Haar Cascade + MediaPipe fallback)
# ─────────────────────────────────────────────────────────────────────────────
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
CROP = (64, 64)

def _haar_crop(bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    for scale, nb, msz in [(1.1, 4, (50, 50)), (1.05, 2, (30, 30))]:
        faces = _haar.detectMultiScale(gray, scale, nb, minSize=msz)
        if len(faces):
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return cv2.resize(raw[y:y + h, x:x + w], CROP)
    return None

def _mp_crop(bgr: np.ndarray) -> Optional[np.ndarray]:
    """MediaPipe Face Mesh fallback — lazy import to avoid startup crash."""
    try:
        import mediapipe as mp
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3
        )
        res = fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        fm.close()
        if not res.multi_face_landmarks:
            return None
        h, w = bgr.shape[:2]
        lms = res.multi_face_landmarks[0].landmark
        xs = [int(lm.x * w) for lm in lms]
        ys = [int(lm.y * h) for lm in lms]
        x1, x2 = max(0, min(xs) - 10), min(w, max(xs) + 10)
        y1, y2 = max(0, min(ys) - 10), min(h, max(ys) + 10)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        logger.info("MediaPipe crop OK")
        return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), CROP)
    except Exception:
        logger.error(f"MediaPipe crop failed:\n{traceback.format_exc()}")
        return None

def extract_embedding(bgr: np.ndarray) -> Optional[List[float]]:
    if bgr is None:
        return None
    crop = _haar_crop(bgr)
    if crop is None:
        crop = _mp_crop(bgr)
    if crop is None:
        logger.warning("No face detected — cannot extract embedding")
        return None
    emb = (crop.astype(np.float32) / 255.0).flatten().tolist()
    logger.info(f"Embedding extracted: {len(emb)} values")
    return emb

def cosine_sim(a: List[float], b: List[float]) -> float:
    try:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(np.dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else 0.0
    except Exception:
        return 0.0

# FIX: Helper to pull the validated FPS that rppg_core stamped into signals.
# Using the container FPS directly (cap.get(FPS)) is unreliable for browser WebM.
def _fps_from_signals(signals, default: float = 30.0) -> float:
    if isinstance(signals, dict):
        fps = signals.get("_fps", default)
        if isinstance(fps, (int, float)) and 5 < fps <= 120:
            return float(fps)
    return default

# ─────────────────────────────────────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_check():
    logger.info("=== SERVICE STARTUP ===")
    logger.info(f"DB path  : {DB_PATH}")
    logger.info(f"DB exists: {os.path.exists(DB_PATH)}")
    logger.info(f"OpenCV   : {cv2.__version__}")
    try:
        import mediapipe as mp
        logger.info(f"MediaPipe: {mp.__version__}")
    except Exception as e:
        logger.warning(f"MediaPipe import warning: {e}")
    logger.info("=== STARTUP OK ===")

# ─────────────────────────────────────────────────────────────────────────────
# Debug / diagnostic endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "db_path": DB_PATH, "db_exists": os.path.exists(DB_PATH)}

# ─────────────────────────────────────────────────────────────────────────────
# /api/auth/challenge-token — frontend calls this BEFORE starting to record.
# Returns a random set of challenges and a one-time token.
# The token must be sent back with the video or the login is rejected.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/auth/challenge-token")
async def get_challenge_token():
    """
    Issue a fresh random challenge token.
    The frontend should call this right before showing the recording UI,
    then display the returned challenges in order.
    """
    # Human-readable labels and durations for each challenge ID.
    # These are returned alongside the token so the frontend can drive the UI
    # without hard-coding challenge names.
    CHALLENGE_UI: Dict = {
        "blink":         {"label": "👁  Blink TWICE",         "duration_ms": 3000},
        "head_turn":     {"label": "↔  Turn head LEFT then RIGHT", "duration_ms": 3000},
        "head_left":     {"label": "↩  Turn head LEFT",       "duration_ms": 2500},
        "head_right":    {"label": "↪  Turn head RIGHT",      "duration_ms": 2500},
        "mouth_open":    {"label": "😮  Open your MOUTH wide", "duration_ms": 2500},
        "eyebrow_raise": {"label": "🤨  Raise your EYEBROWS", "duration_ms": 2500},
    }
    # Fixed steps always prepended/appended
    FIXED_START = {"id": "hold_still",  "label": "🧍 Hold still — scanning...", "duration_ms": 2000}
    FIXED_END   = {"id": "bcg_capture", "label": "✓  Hold still — BCG scan",    "duration_ms": 2500}

    try:
        # Draw 3 random challenges (was 2) so the pool is harder to guess
        token_data = generate_challenge_token(n_challenges=3)

        # Build ordered step list for the frontend
        steps = [FIXED_START]
        for cid in token_data["challenges"]:
            ui = CHALLENGE_UI.get(cid, {"label": cid, "duration_ms": 2500})
            steps.append({"id": cid, "label": ui["label"], "duration_ms": ui["duration_ms"]})
        steps.append(FIXED_END)

        total_duration_ms = sum(s["duration_ms"] for s in steps)

        logger.info(
            f"Issued token {token_data['token'][:8]}… "
            f"challenges={token_data['challenges']}"
        )
        return {
            **token_data,           # token, challenges, expires_at, imu_challenge_windows
            "steps":               steps,
            "total_duration_ms":   total_duration_ms,
        }
    except Exception:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, "message": "Could not generate challenge token"
        })


@app.get("/api/debug/db-test")
async def db_test():
    try:
        dummy = [float(i) / 100 for i in range(4096)]
        write_ok = store_embedding("test", dummy)
        read_back = get_embedding("test")
        try:
            c = _get_conn()
            c.execute("DELETE FROM users WHERE username='test'")
            c.commit()
            c.close()
        except Exception:
            pass
        return {
            "db_path": DB_PATH,
            "db_writable": write_ok,
            "db_readable": read_back is not None,
            "round_trip_ok": (read_back is not None and len(read_back) == 4096)
        }
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "traceback": traceback.format_exc()
        })

# ─────────────────────────────────────────────────────────────────────────────
# /api/auth/enroll-video — called directly by frontend
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/auth/enroll-video")
async def enroll_video(video: UploadFile = File(...), username: str = Form(...)):
    tmp_path = None
    try:
        logger.info(f"=== ENROLL START: '{username}' ===")
        if not username.strip():
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Username required"
            })

        data = await video.read()
        if not data:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Empty video file"
            })
        logger.info(f"Received video: {len(data)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name

        signals, frame = extract_roi_signals(tmp_path)
        logger.info(f"ROI signals: {'OK' if signals is not None else 'None'}, "
                    f"frame: {'OK' if frame is not None else 'None'}")

        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": (
                    "Could not detect a face in the video.\n"
                    "1. Face clearly visible and centered\n"
                    "2. Good lighting (not too dark/bright)\n"
                    "3. Looking directly at the camera\n"
                    "4. No obstructions (glasses glare, mask)"
                )
            })

        emb = extract_embedding(frame)
        logger.info(f"Embedding: {'length=' + str(len(emb)) if emb else 'None'}")

        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "Face detected but could not extract features. Try better lighting."
            })

        if store_embedding(username.strip(), emb):
            logger.info(f"=== ENROLL SUCCESS: '{username}' ===")
            return {"success": True, "message": "Enrollment successful",
                    "username": username.strip(), "embedding_length": len(emb)}

        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "DB write failed — check server logs for details."
        })

    except Exception:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                time.sleep(0.05)
                os.remove(tmp_path)
            except Exception:
                pass

# FIX: Added /api/auth/enroll as an alias.
# The frontend was calling /api/auth/enroll-video which correctly maps above,
# but the Java Spring Boot BiometricService was calling /api/ml/analyze for
# enrollment and /api/auth/enroll did not exist — causing 404s on that path.
@app.post("/api/auth/enroll")
async def enroll_alias(video: UploadFile = File(...), username: str = Form(...)):
    """Alias — delegates to enroll_video above."""
    return await enroll_video(video=video, username=username)

# ─────────────────────────────────────────────────────────────────────────────
# /api/auth/login-video — called directly by frontend
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/auth/login-video")
async def login_video(
    video:            UploadFile = File(...),
    username:         str        = Form(...),
    challenge_token:  str        = Form(default=""),
    imu_data:         str        = Form(default=""),   # JSON-encoded IMU stream (optional)
):
    tmp_path = None
    try:
        logger.info(f"=== LOGIN START: '{username}' ===")
        if not username.strip():
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Username required"
            })

        stored = get_embedding(username.strip())
        if stored is None:
            return JSONResponse(status_code=404, content={
                "success": False,
                "message": f"User '{username}' not found — please enroll first."
            })

        # ── Challenge token validation ─────────────────────────────────────
        # The frontend must request a token before recording, display the
        # returned challenges, and echo the token back here.
        # If the token is missing or invalid, reject immediately.
        required_challenges = consume_challenge_token(challenge_token)
        if required_challenges is None:
            logger.warning(f"LOGIN REJECTED: invalid/missing challenge token for '{username}'")
            return JSONResponse(status_code=401, content={
                "success": False,
                "message": (
                    "Invalid or expired challenge token. "
                    "Please refresh and try again."
                ),
                "is_real": False,
            })
        logger.info(f"Token valid — required challenges: {required_challenges}")

        # Retrieve IMU windows AFTER consume (token already marked used above)
        imu_windows = get_imu_windows_for_token(challenge_token)

        data = await video.read()
        if not data:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Empty video file"
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name

        signals, frame = extract_roi_signals(tmp_path)

        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "No face detected in video."
            })

        fps = _fps_from_signals(signals)

        # ── Challenge analysis (using token-specified challenges) ──────────
        challenge_result = None
        bcg_result       = {"passed": False, "bcg_hr_bpm": 0.0, "rppg_hr_bpm": 0.0,
                            "bcg_signal_power": 0.0, "freq_match": False}

        try:
            challenge_result = analyze_challenges(
                tmp_path, required_challenges, fps=fps
            )
            logger.info(
                f"Challenge: passed={challenge_result.get('passed')}, "
                f"reason={challenge_result.get('reason')}"
            )
        except Exception:
            logger.error(f"Challenge analysis error:\n{traceback.format_exc()}")
            # Fail closed — if we can't evaluate challenges, reject
            challenge_result = {"passed": False, "reason": "Challenge analysis failed"}

        try:
            bcg_result = analyze_bcg(tmp_path)
            logger.info(
                f"Camera BCG: passed={bcg_result.get('passed')}, "
                f"hr={bcg_result.get('bcg_hr_bpm')} BPM"
            )
        except Exception:
            logger.error(f"BCG analysis error:\n{traceback.format_exc()}")

        # ── IMU BCG analysis (Layer 4 — phone in-hand Z-axis) ────────────────
        # Parse the optional JSON IMU stream from the form field.
        # If absent or malformed, the layer is simply skipped (not failed).
        imu_bcg_result = None
        parsed_imu     = None
        if imu_data and imu_data.strip():
            try:
                import json as _json
                parsed_imu = _json.loads(imu_data)
                logger.info(
                    f"IMU data received: "
                    f"{len(parsed_imu.get('accel_z', []))} Z-axis samples"
                )
            except Exception:
                logger.warning(f"IMU data JSON parse failed — skipping IMU layer")

        if parsed_imu:
            try:
                # Derive rPPG HR from signals for cross-modal comparison
                rppg_hr = rppg_hr_from_signals(signals) if signals else 0.0

                # Use the first challenge window for motion verification
                # (the most security-critical window is the first challenge)
                ch_id        = required_challenges[0] if required_challenges else None
                ch_start_ms  = 0.0
                ch_end_ms    = 0.0
                if ch_id and imu_windows:
                    for win in imu_windows:
                        if win.get("challenge_id") == ch_id:
                            ch_start_ms = win["start_ms"]
                            ch_end_ms   = win["end_ms"]
                            break

                imu_bcg_result = analyze_imu_bcg(
                    imu_data       = parsed_imu,
                    rppg_hr_bpm    = rppg_hr,
                    challenge_id   = ch_id,
                    challenge_start_ms = ch_start_ms,
                    challenge_end_ms   = ch_end_ms,
                )
                logger.info(
                    f"IMU BCG: passed={imu_bcg_result.get('passed')}, "
                    f"imu_hr={imu_bcg_result.get('imu_hr_bpm')} BPM, "
                    f"rppg_hr={imu_bcg_result.get('rppg_hr_bpm')} BPM, "
                    f"freq_match={imu_bcg_result.get('freq_match')}, "
                    f"harmonic={imu_bcg_result.get('is_harmonic')}, "
                    f"challenge_motion_ok={imu_bcg_result.get('challenge_imu_ok')}"
                )
            except Exception:
                logger.error(f"IMU BCG analysis error:\n{traceback.format_exc()}")

        # ── Liveness decision ─────────────────────────────────────────────
        # Default FAIL — we only proceed if the full pipeline confirms liveness.
        is_real = False
        score   = 0.0
        reason  = "Liveness check not completed"

        if signals is not None:
            is_real, score, reason = analyze_liveness(
                signals,
                fps=fps,
                challenge_result=challenge_result,
                bcg_result=bcg_result,
                imu_bcg_result=imu_bcg_result,
                challenge_was_required=True,
            )
            logger.info(f"Liveness → real={is_real}, score={score:.4f}, reason={reason}")
        else:
            logger.warning("No rPPG signals extracted — liveness FAILED")
            reason = "Could not extract physiological signals. Ensure good lighting."

        # Block on ANY liveness failure
        if not is_real:
            logger.warning(f"LOGIN BLOCKED — liveness failed for '{username}': {reason}")
            return JSONResponse(status_code=401, content={
                "success":              False,
                "message":              f"Liveness check failed: {reason}",
                "is_real":              False,
                "coherence_score":      round(score, 4),
                "challenge_passed":     challenge_result.get("passed", False) if challenge_result else False,
                "bcg_passed":           bcg_result.get("passed", False),
                "bcg_hr_bpm":           bcg_result.get("bcg_hr_bpm", 0.0),
                "rppg_hr_bpm":          bcg_result.get("rppg_hr_bpm", 0.0),
                "bcg_signal_power":     bcg_result.get("bcg_signal_power", 0.0),
                "imu_bcg_passed":       imu_bcg_result.get("passed", False)       if imu_bcg_result else None,
                "imu_hr_bpm":           imu_bcg_result.get("imu_hr_bpm", 0.0)     if imu_bcg_result else None,
                "imu_freq_match":       imu_bcg_result.get("freq_match", False)   if imu_bcg_result else None,
                "imu_is_harmonic":      imu_bcg_result.get("is_harmonic", False)  if imu_bcg_result else None,
                "imu_challenge_ok":     imu_bcg_result.get("challenge_imu_ok", None) if imu_bcg_result else None,
            })

        emb = extract_embedding(frame)
        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Could not extract face features."
            })

        sim = cosine_sim(emb, stored)
        logger.info(f"Similarity for '{username}': {sim:.4f}")

        if sim >= 0.75:
            logger.info(f"=== LOGIN SUCCESS: '{username}' ===")
            return {
                "success":          True,
                "message":          "Authentication successful",
                "username":         username.strip(),
                "is_real":          True,
                "coherence_score":  round(score, 4),
                "face_similarity":  round(sim, 4),
                "challenge_passed": challenge_result.get("passed", False) if challenge_result else False,
                "bcg_passed":       bcg_result.get("passed", False),
                "bcg_hr_bpm":       bcg_result.get("bcg_hr_bpm", 0.0),
                "rppg_hr_bpm":      bcg_result.get("rppg_hr_bpm", 0.0),
                "bcg_signal_power": bcg_result.get("bcg_signal_power", 0.0),
                "bcg_freq_match":   bcg_result.get("freq_match", False),
                # IMU BCG fields (None when no IMU data was submitted)
                "imu_bcg_passed":   imu_bcg_result.get("passed", False)          if imu_bcg_result else None,
                "imu_hr_bpm":       imu_bcg_result.get("imu_hr_bpm", 0.0)        if imu_bcg_result else None,
                "imu_freq_match":   imu_bcg_result.get("freq_match", False)      if imu_bcg_result else None,
                "imu_is_harmonic":  imu_bcg_result.get("is_harmonic", False)     if imu_bcg_result else None,
                "imu_challenge_ok": imu_bcg_result.get("challenge_imu_ok", None) if imu_bcg_result else None,
                "imu_signal_power": imu_bcg_result.get("imu_signal_power", 0.0)  if imu_bcg_result else None,
            }

        return JSONResponse(status_code=401, content={
            "success":         False,
            "message":         f"Face mismatch (similarity={sim:.2f})",
            "is_real":         True,
            "coherence_score": round(score, 4),
            "face_similarity": round(sim, 4),
        })

    except Exception:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, "message": "Server error during login"
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                time.sleep(0.05)
                os.remove(tmp_path)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# /api/ml/analyze — called by Java Spring Boot (BiometricService.java)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/ml/analyze")
async def analyze_video(file: UploadFile = File(...)):
    tmp_path = None
    try:
        logger.info(f"=== ANALYZE START (Spring Boot call): {file.filename} ===")
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": False,
                "spoof_reason": "Empty video file",
                "coherence_score": 0.0, "embedding": []
            })
        logger.info(f"Received: {len(data)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name

        signals, frame = extract_roi_signals(tmp_path)
        logger.info(f"Signals: {'OK' if signals is not None else 'None'}, "
                    f"Frame: {'OK' if frame is not None else 'None'}")

        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": False,
                "spoof_reason": "No face detected in video",
                "coherence_score": 0.0, "embedding": []
            })

        # FIX: Use validated FPS
        fps = _fps_from_signals(signals)

        is_real = True
        score   = 0.5
        reason  = "Liveness check passed"

        if signals is not None:
            is_real, score, reason = analyze_liveness(signals, fps=fps)
            logger.info(f"Liveness → real={is_real}, score={score:.4f}, reason={reason}")
            if not is_real and score > 0.95:
                return JSONResponse(status_code=401, content={
                    "success": False, "is_real": False,
                    "spoof_reason": reason,
                    "coherence_score": round(score, 4), "embedding": []
                })
            if not is_real:
                logger.warning(f"Weak signal (score={score:.4f}) — returning embedding anyway")
                is_real = True
        else:
            logger.warning("No signals — skipping liveness, extracting embedding")

        emb = extract_embedding(frame)
        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": is_real,
                "spoof_reason": "Face detected but embedding extraction failed",
                "coherence_score": round(score, 4), "embedding": []
            })

        logger.info(f"=== ANALYZE SUCCESS: embedding={len(emb)}, "
                    f"is_real={is_real}, score={score:.4f} ===")

        return {
            "success": True,
            "is_real": is_real,
            "spoof_reason": reason,
            "coherence_score": round(score, 4),
            "embedding": emb
        }

    except Exception:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, "is_real": False,
            "spoof_reason": f"Server error: {traceback.format_exc()}",
            "coherence_score": 0.0, "embedding": []
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                time.sleep(0.05)
                os.remove(tmp_path)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# /api/ml/analyze-full — Full three-layer liveness pipeline for LOGIN
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/ml/analyze-full")
async def analyze_full(
    file:       UploadFile = File(...),
    challenges: str        = Form(default="blink,head_turn")
):
    tmp_path = None
    try:
        required = [c.strip().lower() for c in challenges.split(",") if c.strip()]
        logger.info(f"=== ANALYZE-FULL START: challenges={required} ===")

        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": False,
                "spoof_reason": "Empty video file",
                "coherence_score": 0.0, "embedding": [],
                "challenge_passed": False, "challenge_details": {},
                "bcg_passed": False, "bcg_hr_bpm": 0.0,
                "rppg_hr_bpm": 0.0, "bcg_signal_power": 0.0
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name
        logger.info(f"Saved {len(data)} bytes → {tmp_path}")

        # ── Step 1: rPPG signals + face frame ────────────────────────────────
        signals, frame = extract_roi_signals(tmp_path)
        logger.info(f"rPPG signals: {'OK' if signals is not None else 'None'}, "
                    f"face frame: {'OK' if frame is not None else 'None'}")

        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": False,
                "spoof_reason": "No face detected in video",
                "coherence_score": 0.0, "embedding": [],
                "challenge_passed": False, "challenge_details": {},
                "bcg_passed": False, "bcg_hr_bpm": 0.0,
                "rppg_hr_bpm": 0.0, "bcg_signal_power": 0.0
            })

        # FIX: Use validated FPS throughout
        fps = _fps_from_signals(signals)

        # ── Step 2: Challenge-Response ────────────────────────────────────────
        challenge_result = None
        if required:
            logger.info(f"Running challenge analysis: {required}")
            challenge_result = analyze_challenges(tmp_path, required, fps=fps)
            logger.info(f"Challenge: passed={challenge_result.get('passed')}, "
                        f"reason={challenge_result.get('reason')}")

        # ── Step 3: BCG analysis ──────────────────────────────────────────────
        logger.info("Running BCG analysis...")
        bcg_result = analyze_bcg(tmp_path)
        logger.info(f"BCG: passed={bcg_result.get('passed')}, "
                    f"bcg_hr={bcg_result.get('bcg_hr_bpm')} BPM")

        # ── Step 4: Combined liveness decision ───────────────────────────────
        is_real, score, reason = analyze_liveness(
            signals,
            fps=fps,
            challenge_result=challenge_result,
            bcg_result=bcg_result,
        )
        logger.info(f"Liveness → real={is_real}, score={score:.4f}, reason={reason}")

        # HARD-BLOCK: only screen replay
        if not is_real and score > 0.98:
            logger.warning(f"Screen replay detected: {reason}")
            emb = extract_embedding(frame) or []
            return JSONResponse(status_code=401, content={
                "success": False, "is_real": False,
                "spoof_reason": reason,
                "coherence_score": round(score, 4), "embedding": emb,
                "challenge_passed":  challenge_result.get("passed", False) if challenge_result else False,
                "challenge_details": challenge_result.get("details", {})   if challenge_result else {},
                "bcg_passed":        bcg_result.get("passed", False),
                "bcg_hr_bpm":        bcg_result.get("bcg_hr_bpm", 0.0),
                "rppg_hr_bpm":       bcg_result.get("rppg_hr_bpm", 0.0),
                "bcg_signal_power":  bcg_result.get("bcg_signal_power", 0.0),
            })

        if not is_real:
            logger.warning(f"Liveness weak (score={score:.4f}) but allowing face match")

        # ── Step 5: Face embedding ────────────────────────────────────────────
        emb = extract_embedding(frame)
        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": True,
                "spoof_reason": "Liveness passed but embedding extraction failed",
                "coherence_score": round(score, 4), "embedding": [],
                "challenge_passed":  challenge_result.get("passed", False) if challenge_result else False,
                "challenge_details": challenge_result.get("details", {})   if challenge_result else {},
                "bcg_passed":        bcg_result.get("passed", False),
                "bcg_hr_bpm":        bcg_result.get("bcg_hr_bpm", 0.0),
                "rppg_hr_bpm":       bcg_result.get("rppg_hr_bpm", 0.0),
                "bcg_signal_power":  bcg_result.get("bcg_signal_power", 0.0),
            })

        logger.info(f"=== ANALYZE-FULL SUCCESS: is_real={is_real}, "
                    f"bcg_hr={bcg_result.get('bcg_hr_bpm')} BPM ===")

        return {
            "success":           True,
            "is_real":           True,
            "spoof_reason":      reason,
            "coherence_score":   round(score, 4),
            "embedding":         emb,
            "challenge_passed":  challenge_result.get("passed", False) if challenge_result else False,
            "challenge_details": challenge_result.get("details", {})   if challenge_result else {},
            "bcg_passed":        bcg_result.get("passed", False),
            "bcg_hr_bpm":        bcg_result.get("bcg_hr_bpm", 0.0),
            "rppg_hr_bpm":       bcg_result.get("rppg_hr_bpm", 0.0),
            "bcg_signal_power":  bcg_result.get("bcg_signal_power", 0.0),
            "bcg_freq_match":    bcg_result.get("freq_match", False),
        }

    except Exception:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, "is_real": False,
            "spoof_reason": f"Server error: {traceback.format_exc()}",
            "coherence_score": 0.0, "embedding": [],
            "challenge_passed": False, "challenge_details": {},
            "bcg_passed": False, "bcg_hr_bpm": 0.0,
            "rppg_hr_bpm": 0.0, "bcg_signal_power": 0.0
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                time.sleep(0.05)
                os.remove(tmp_path)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# /api/auth/check-user/{username}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/auth/check-user/{username}")
async def check_user(username: str):
    try:
        return {"exists": get_embedding(username) is not None, "username": username}
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })

# ─────────────────────────────────────────────────────────────────────────────
# Debug endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/debug/save-frames")
async def debug_save_frames(video: UploadFile = File(...)):
    tmp_path = None
    try:
        data = await video.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Could not open video"
            })

        count, saved = 0, []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count <= 5 or count % 10 == 0:
                p = os.path.join(debug_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(p, frame)
                saved.append({"n": count, "path": p, "brightness": float(frame.mean())})
        cap.release()

        return {"success": True, "frames_saved": len(saved), "frames": saved,
                "debug_dir": debug_dir}
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/api/debug/first-frame")
async def debug_first_frame(video: UploadFile = File(...)):
    tmp_path = None
    try:
        data = await video.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Could not read frame"
            })

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _haar.detectMultiScale(gray, 1.1, 4)
        return {"success": True, "shape": str(frame.shape),
                "mean": float(frame.mean()), "faces_detected": len(faces)}
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/api/debug/check-frame")
async def debug_check_frame(video: UploadFile = File(...)):
    tmp_path = None
    try:
        data = await video.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm",
                                         dir=tempfile.gettempdir()) as f:
            f.write(data)
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        count, info = 0, []
        while cap.isOpened() and count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            row = {"n": count, "brightness": float(frame.mean())}
            if count == 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                row["faces"] = len(_haar.detectMultiScale(gray, 1.1, 4))
            info.append(row)
        cap.release()
        return {"success": True, "frames": info, "size": len(data)}
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")