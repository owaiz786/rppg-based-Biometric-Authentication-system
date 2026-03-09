import os
import time
import logging
import traceback
import tempfile
import sqlite3
from typing import Optional, List

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from rppg_core import extract_roi_signals
from anti_spoofing import analyze_liveness

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

# ---------------------------------------------------------------------------
# Database (absolute path, WAL mode, binary BLOB storage)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Face detection helpers (fully local — Haar Cascade + MediaPipe fallback)
# ---------------------------------------------------------------------------
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
    # FIX: use explicit None checks — 'or' on numpy arrays raises ValueError
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


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Debug / diagnostic endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "db_path": DB_PATH, "db_exists": os.path.exists(DB_PATH)}


@app.get("/api/debug/db-test")
async def db_test():
    try:
        dummy = [float(i) / 100 for i in range(4096)]
        write_ok = store_embedding("__test__", dummy)
        read_back = get_embedding("__test__")
        try:
            c = _get_conn()
            c.execute("DELETE FROM users WHERE username='__test__'")
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


# ---------------------------------------------------------------------------
# /api/auth/enroll-video  — called directly by frontend
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# /api/auth/login-video  — called directly by frontend
# ---------------------------------------------------------------------------
@app.post("/api/auth/login-video")
async def login_video(video: UploadFile = File(...), username: str = Form(...)):
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

        # Run liveness if signals available
        if signals is not None:
            is_real, score, reason = analyze_liveness(signals, fps=30)
            logger.info(f"Liveness → real={is_real}, score={score:.4f}, reason={reason}")
            # Only block definite screen replay (score > 0.95)
            # Weak signal (score < 0.05) = poor conditions, NOT a spoof
            if not is_real and score > 0.95:
                return JSONResponse(status_code=401, content={
                    "success": False, "message": f"Spoof detected: {reason}",
                    "coherence_score": round(score, 4), "is_real": False
                })
            if not is_real and score < 0.05:
                logger.warning(f"Weak rPPG signal (score={score:.4f}) — face match only")
        else:
            is_real, score, reason = False, 0.0, "No signal"
            logger.warning("No signals — face match only")

        emb = extract_embedding(frame)
        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "Could not extract face features."
            })

        sim = cosine_sim(emb, stored)
        logger.info(f"Similarity for '{username}': {sim:.4f}")

        if sim >= 0.85:
            logger.info(f"=== LOGIN SUCCESS: '{username}' ===")
            return {"success": True, "message": "Authentication successful",
                    "username": username.strip(), "is_real": is_real,
                    "coherence_score": round(score, 4), "face_similarity": round(sim, 4)}

        return JSONResponse(status_code=401, content={
            "success": False, "message": f"Face mismatch (similarity={sim:.2f})",
            "is_real": is_real, "coherence_score": round(score, 4),
            "face_similarity": round(sim, 4)
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


# ---------------------------------------------------------------------------
# /api/ml/analyze  — called by Java Spring Boot (BiometricService.java)
#
# IMPORTANT: accepts field name "file" (matches body.add("file", ...) in Java)
# Returns PythonResponse-compatible JSON matching PythonResponse.java DTO:
#   { success, is_real, spoof_reason, coherence_score, embedding }
#
# embedding is ALWAYS returned when success=true so Java can save to Neon DB.
# Only hard-blocks on screen replay (score > 0.95). Weak signal is NOT a spoof.
# ---------------------------------------------------------------------------
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

        # Step 1: extract rPPG signals + first confirmed face frame
        signals, frame = extract_roi_signals(tmp_path)
        logger.info(f"Signals: {'OK' if signals is not None else 'None'}, "
                    f"Frame: {'OK' if frame is not None else 'None'}")

        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": False,
                "spoof_reason": "No face detected in video",
                "coherence_score": 0.0, "embedding": []
            })

        # Step 2: liveness analysis
        if signals is not None:
            is_real, score, reason = analyze_liveness(signals, fps=30)
            logger.info(f"Liveness → real={is_real}, score={score:.4f}, reason={reason}")
            # Hard-block only on screen replay
            if not is_real and score > 0.95:
                return JSONResponse(status_code=401, content={
                    "success": False, "is_real": False,
                    "spoof_reason": reason,
                    "coherence_score": round(score, 4), "embedding": []
                })
            # Weak signal — not a spoof, fall through to embedding
            if not is_real and score < 0.05:
                logger.warning(f"Weak signal (score={score:.4f}) — returning embedding anyway")
        else:
            is_real, score, reason = False, 0.0, "Weak signal — face match only"
            logger.warning("No signals — skipping liveness, extracting embedding")

        # Step 3: extract face embedding — always returned on success
        emb = extract_embedding(frame)
        if emb is None:
            return JSONResponse(status_code=400, content={
                "success": False, "is_real": is_real,
                "spoof_reason": "Face detected but embedding extraction failed",
                "coherence_score": round(score, 4), "embedding": []
            })

        logger.info(f"=== ANALYZE SUCCESS: embedding={len(emb)}, "
                    f"is_real={is_real}, score={score:.4f} ===")

        # Return shape matches PythonResponse.java exactly:
        # success, is_real, spoof_reason, coherence_score, embedding
        return {
            "success": True,
            "is_real": is_real,
            "spoof_reason": reason,
            "coherence_score": round(score, 4),
            "embedding": emb        # 4096 floats → Java saves to Neon PostgreSQL
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


@app.get("/api/auth/check-user/{username}")
async def check_user(username: str):
    try:
        return {"exists": get_embedding(username) is not None, "username": username}
    except Exception:
        return JSONResponse(status_code=500, content={
            "success": False, "message": traceback.format_exc()
        })


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