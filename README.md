# rPPG-Based Biometric Authentication System

A hardware-free, multi-layer liveness detection and face authentication system that uses a standard webcam to verify that the person logging in is genuinely alive — not a photograph, video replay, or deepfake. Built with Python FastAPI, Spring Boot, and Next.js, deployed on AWS.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
  - [1. Python ML Service](#1-python-ml-service)
  - [2. Spring Boot Gateway](#2-spring-boot-gateway)
  - [3. Next.js Frontend](#3-nextjs-frontend)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [The Four-Layer Anti-Spoofing Pipeline](#the-four-layer-anti-spoofing-pipeline)
- [Troubleshooting](#troubleshooting)

---

## How It Works

Traditional face-recognition login is trivially defeated by holding up a photo. This system prevents that by verifying **physiological liveness** before any identity check is performed.

**Enrollment** — the user records a short webcam clip. The system extracts a face embedding and stores it.

**Login** — the server issues a one-time challenge token (e.g. "blink, then turn your head"). The user records themselves completing the challenges. The system then:

1. **rPPG coherence check** — extracts subtle green-channel colour fluctuations from three facial skin regions (forehead, left cheek, right cheek) driven by blood flow. A live person's signal is correlated across all three regions. A photo has no signal; a screen replay has near-perfect correlation (which is itself a hard-block).
2. **Challenge-response verification** — confirms the user completed the randomly selected gestures (blink, head turn) within the correct time windows. The token is single-use, so a recorded replay of a previous session always fails.
3. **BCG (Ballistocardiogram) cross-validation** — measures subtle frame-to-frame head micro-motion caused by each heartbeat. The dominant frequency must match the rPPG-derived heart rate, providing a second independent physiological signal.
4. **Face identity matching** — if at least two of the three core layers pass, a 4096-element cosine-similarity face embedding is compared against the stored enrollment embedding (threshold ≥ 0.75).

---

## Architecture

```
┌─────────────────────┐     HTTPS/REST      ┌──────────────────────┐     HTTP :8000      ┌─────────────────────────┐
│  Next.js Frontend   │ ──────────────────► │  Spring Boot Gateway │ ──────────────────► │  Python FastAPI ML      │
│  (port 3000)        │ ◄────────────────── │  (port 8080)         │ ◄────────────────── │  Service (port 8000)    │
│                     │                     │                      │                     │                         │
│  • Webcam capture   │                     │  • REST orchestration│                     │  • rPPG signal extract  │
│  • Challenge UI     │                     │  • Cosine-sim match  │                     │  • BCG analysis         │
│  • Liveness display │                     │  • JPA / PostgreSQL  │                     │  • Challenge verify     │
└─────────────────────┘                     └──────────────────────┘                     │  • Face embedding       │
                                                       │                                 │  • SQLite WAL cache     │
                                                       ▼                                 └─────────────────────────┘
                                            ┌──────────────────────┐
                                            │  AWS Neon PostgreSQL │
                                            │  (serverless)        │
                                            └──────────────────────┘
```

The frontend never talks to the ML service directly — all requests go through the Spring Boot gateway, which handles user persistence and forwards video to the ML service for analysis.

---

## Project Structure

```
rppg-based-Biometric-Authentication-system-main/
│
├── rppg-ml-service/               # Python FastAPI ML service
│   ├── main.py                    # API endpoints, face embedding, SQLite DB
│   ├── rppg_core.py               # MediaPipe ROI extraction, FPS guard
│   ├── anti_spoofing.py           # Butterworth filter, coherence scoring, liveness decision
│   ├── challenge_response.py      # Token generation, blink/head-turn detection
│   ├── bcg.py                     # Ballistocardiogram micro-motion analysis
│   ├── test.py                    # Unit tests
│   └── test_webcam.py             # Live webcam smoke test
│
├── biometric/biometric/           # Spring Boot API gateway (Java 17)
│   ├── src/main/java/com/yourapp/biometric/
│   │   ├── controller/
│   │   │   └── AuthController.java        # /api/auth/* REST endpoints
│   │   ├── service/
│   │   │   └── BiometricService.java      # Cosine similarity, Python proxy
│   │   ├── model/
│   │   │   └── User.java                  # JPA entity (username + embedding)
│   │   ├── repository/
│   │   │   └── UserRepository.java        # Spring Data JPA
│   │   └── dto/
│   │       └── PythonResponse.java        # ML service response DTO
│   └── src/main/resources/
│       └── application.yml                # DB config, port, multipart limits
│
├── components/                    # Next.js React components
│   ├── biometric-auth.tsx         # Main state machine (idle → challenge → record → result)
│   ├── biometric-webcam-area.tsx  # getUserMedia / MediaRecorder webcam feed
│   ├── biometric-controls.tsx     # Enroll / Login buttons
│   ├── biometric-feedback.tsx     # Success / failure messaging
│   ├── biometric-telemetry.tsx    # rPPG HR, BCG HR, coherence, cosine sim display
│   ├── biometric-instruction.tsx  # Timed challenge instruction overlay
│   └── biometric-header.tsx      # Page header
│
├── app/                           # Next.js App Router
│   ├── page.tsx                   # Root page (mounts BiometricAuth)
│   └── layout.tsx                 # Root layout + theme provider
│
├── requirements.txt               # Python dependencies (pinned versions)
├── package.json                   # Node.js dependencies
└── .gitignore
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10.x | Other versions may break the pinned ML deps |
| Java | 17+ | Required by Spring Boot 3 |
| Maven | 3.8+ | Or use the included `mvnw` wrapper |
| Node.js | 18+ | For Next.js 16 |
| npm / pnpm | any | `pnpm-lock.yaml` is included |
| Webcam | any RGB camera | ≥ 15 fps at ≥ 320×240 resolution |
| PostgreSQL | Neon serverless | Or any Postgres — update `application.yml` |

---

## Setup and Installation

### 1. Python ML Service

```bash
cd rppg-ml-service

# Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install pinned dependencies (order matters for numpy/mediapipe compatibility)
pip install --upgrade pip
pip install -r ../requirements.txt
```

> **Note on versions**: The `requirements.txt` pins specific versions that are known to work together (`numpy==1.24.3`, `mediapipe==0.10.7`, `tensorflow==2.13.0`). Do not upgrade these individually — they have tight cross-dependencies.

**Verify the installation:**

```bash
python -c "import cv2, mediapipe, scipy, fastapi; print('All imports OK')"
```

---

### 2. Spring Boot Gateway

```bash
cd biometric/biometric

# Build (skip tests on first run)
./mvnw clean package -DskipTests

# Or on Windows
mvnw.cmd clean package -DskipTests
```

**Update the database connection** in `src/main/resources/application.yml` before running (see [Configuration](#configuration)).

---

### 3. Next.js Frontend

```bash
# From the project root
npm install
# or
pnpm install
```

---

## Configuration

### Database (`biometric/biometric/src/main/resources/application.yml`)

The default config points to a shared Neon PostgreSQL instance. Replace it with your own:

```yaml
spring:
  datasource:
    url: jdbc:postgresql://<your-neon-host>/neondb?sslmode=require
    username: <your-username>
    password: <your-password>
  jpa:
    hibernate:
      ddl-auto: update          # Creates/updates the users table automatically
  servlet:
    multipart:
      max-file-size: 15MB
      max-request-size: 20MB

server:
  port: 8080
```

To use a local Postgres instead:

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/biometric_db
    username: postgres
    password: yourpassword
```

### CORS (`rppg-ml-service/main.py`)

The ML service allows requests from `localhost:3000`, `3001`, and `3002` by default. If your frontend runs on a different port, add it:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://your-port-here"],
    ...
)
```

The Spring Boot gateway also has a `@CrossOrigin(origins = "http://localhost:3000")` annotation on `AuthController.java` — update this if your frontend URL changes.

### ML Service Thresholds

Key thresholds in `rppg-ml-service/`:

| Parameter | File | Default | Effect |
|---|---|---|---|
| Cosine similarity | `main.py` | `0.75` | Minimum face match score to authenticate |
| rPPG coherence lower bound | `anti_spoofing.py` | `0.15` | Below this → no physiological signal detected |
| rPPG coherence hard-block | `anti_spoofing.py` | `0.90` | Above this → screen replay detected, immediate reject |
| EAR blink threshold | `challenge_response.py` | `0.22` | Eye Aspect Ratio to count as a blink |
| Head turn threshold | `challenge_response.py` | `0.35` | Nose-tip displacement to count as a head turn |
| Challenge token TTL | `challenge_response.py` | `120s` | Tokens expire after 2 minutes |

---

## Running the System

All three services must be running simultaneously. Open three terminals:

**Terminal 1 — Python ML Service:**

```bash
cd rppg-ml-service
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The ML service will log `=== STARTUP OK ===` and print its SQLite DB path. Verify it's healthy:

```bash
curl http://localhost:8000/api/health
# {"status":"ok","db_path":"...","db_exists":true}
```

**Terminal 2 — Spring Boot Gateway:**

```bash
cd biometric/biometric
./mvnw spring-boot:run
# Starts on port 8080
```

**Terminal 3 — Next.js Frontend:**

```bash
# From project root
npm run dev
# Opens on http://localhost:3000
```

Open `http://localhost:3000` in your browser. Allow camera access when prompted.

---

## API Reference

### Python ML Service (port 8000)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns DB path and status |
| `GET` | `/api/auth/challenge-token` | Issue a one-time random challenge token |
| `POST` | `/api/auth/enroll` | Enroll a user (Layer 1 rPPG + face embedding) |
| `POST` | `/api/auth/enroll-video` | Same as above (direct frontend alias) |
| `POST` | `/api/auth/login-video` | Full 4-layer pipeline login |
| `POST` | `/api/ml/analyze` | Enrollment proxy called by Spring Boot |
| `POST` | `/api/ml/analyze-full` | Login proxy called by Spring Boot |
| `GET` | `/api/auth/check-user/{username}` | Check if a user exists |
| `GET` | `/api/debug/db-test` | Round-trip SQLite write/read test |
| `POST` | `/api/debug/save-frames` | Save extracted frames to disk for debugging |

**Challenge token response:**

```json
{
  "token": "a3f8c2...",
  "challenges": ["blink", "head_turn"],
  "expires_at": 1718000000.0
}
```

**Login success response:**

```json
{
  "success": true,
  "username": "alice",
  "is_real": true,
  "coherence_score": 0.72,
  "face_similarity": 0.84,
  "challenge_passed": true,
  "bcg_passed": true,
  "bcg_hr_bpm": 70.0,
  "rppg_hr_bpm": 72.0,
  "bcg_signal_power": 0.031,
  "bcg_freq_match": true
}
```

### Spring Boot Gateway (port 8080)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/enroll` | Enroll — forwards video to ML, saves embedding to PostgreSQL |
| `POST` | `/api/auth/login-video` | Login — forwards to ML, returns success/failure |
| `GET` | `/api/auth/check-user/{username}` | User existence check |

Both `POST` endpoints accept `multipart/form-data` with fields `username` (string) and `video` (WebM file, max 15 MB).

---

## The Four-Layer Anti-Spoofing Pipeline

### Layer 1 — rPPG Coherence

MediaPipe Face Mesh extracts three facial ROIs per frame (forehead landmarks `{10, 338, 297, 332, 284}`, left cheek `{118, 119, 100, 126}`, right cheek `{347, 348, 329, 355}`). The green channel mean of each ROI is recorded over time, then bandpass-filtered at 0.7–3.0 Hz (42–180 BPM) using a 3rd-order Butterworth filter applied via `filtfilt` for zero phase distortion. The mean Pearson correlation across the three ROI pairs is the coherence score.

- Score `< 0.15` → no physiological signal → **FAIL**
- Score `0.15–0.90` → plausible cardiac activity → **PASS**
- Score `> 0.90` → near-perfect cross-ROI correlation (screen replay) → **HARD-BLOCK**

### Layer 2 — Challenge-Response (Mandatory)

Before recording begins, the frontend calls `GET /api/auth/challenge-token`. The server randomly selects challenges from `["blink", "head_turn"]`, stores a single-use token in memory with a 120-second TTL, and returns the token alongside the challenge list. The frontend displays each challenge in sequence with a timed overlay. The token is sent back with the video.

- `consume_challenge_token()` validates and immediately invalidates the token — replay attacks using a previous video always fail.
- All challenges must pass (100% pass rate required). A failed challenge triggers an immediate HTTP 401 regardless of other layers.

**Blink detection** uses Eye Aspect Ratio (EAR): `(v1 + v2) / (2 * horizontal_distance)`. A blink is counted when EAR drops below `0.22` for at least 2 consecutive frames, with a minimum gap of 12 frames between blinks.

**Head-turn detection** tracks the normalized nose-tip X position relative to face width. A turn is confirmed when this ratio goes below `0.35` (right turn) or above `0.65` (left turn).

### Layer 3 — BCG Micro-Motion

Frame-to-frame optical flow magnitude in the face bounding box is computed across the video. The resulting micro-motion time series is filtered to the cardiac band and its dominant FFT peak is converted to BPM. The BCG layer passes if the BCG heart rate agrees with the rPPG heart rate within a harmonic-aliasing-aware tolerance window.

### Liveness Decision

```
if coherence_score > 0.90  →  HARD-BLOCK (screen replay)
if challenge failed         →  REJECT (mandatory layer)

layers_passed = 0
if 0.15 ≤ coherence_score ≤ 0.95  →  layers_passed++
if challenge_passed                →  layers_passed++
if bcg_passed                      →  layers_passed++

if layers_passed ≥ 2  →  proceed to face matching
else                  →  REJECT
```

### Face Identity Matching

Face detection uses OpenCV Haar Cascade (with MediaPipe Face Mesh as fallback). The detected face is cropped, converted to greyscale, resized to 64×64 px, normalized to `[0, 1]`, and flattened to a 4096-element `float32` vector. Cosine similarity is computed between the live embedding and the stored enrollment embedding. Authentication succeeds if similarity ≥ 0.75.

Cosine similarity is illumination-invariant (unlike Euclidean distance), making it robust to minor lighting differences between enrollment and login sessions.

---

## Troubleshooting

**"No face detected in video"**
- Ensure the face is well-lit, centered, and clearly visible throughout recording.
- Avoid strong backlighting (e.g. sitting in front of a window).
- Frames wider than 640 px are automatically downscaled; very low-resolution streams may cause detection failures.

**"Liveness check failed: challenge_failed"**
- The gestures must be visible and deliberate — a subtle blink may not register.
- For blink: close your eyes fully for at least 2–3 frames (~100 ms at 30 fps).
- For head turn: turn clearly left or right; the nose tip must cross the `0.35` threshold.
- Fetch a new challenge token each login attempt — tokens expire in 120 seconds.

**"Invalid or expired challenge token"**
- Tokens are single-use and expire in 2 minutes. Refresh the page and start a new login attempt.
- Submitting the same recorded video twice always fails — the token is consumed on first use.

**"Face mismatch (similarity=0.xx)"**
- Re-enroll under consistent lighting conditions.
- The current embedding is a flat 64×64 pixel vector; large changes in face angle, lighting, or facial hair will reduce similarity.

**ML service won't start / import errors**
```bash
# Confirm you're in the venv
which python  # Should point inside your venv folder

# Reinstall with exact versions
pip install -r requirements.txt --force-reinstall
```

**Spring Boot can't connect to PostgreSQL**
- Confirm the Neon DB URL in `application.yml` is correct and the instance is not paused (Neon free tier suspends after inactivity).
- For local Postgres, ensure the DB exists: `createdb biometric_db`

**Browser reports "camera not available"**
- Only one tab/app can access the webcam at a time. Close other tabs using the camera.
- The app must be served over `localhost` or HTTPS — camera access is blocked on plain HTTP from non-localhost origins.

**WebM video has 0 fps or 1000 fps in container metadata**
- This is a known browser `MediaRecorder` bug. The ML service automatically clamps FPS to `[5, 120]` and defaults to 30 fps when the container value is unreliable. No action needed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui |
| API Gateway | Spring Boot 3, Java 17, Spring Data JPA, Lombok |
| ML Service | Python 3.10, FastAPI, uvicorn |
| Computer Vision | OpenCV (headless), MediaPipe Face Mesh |
| Signal Processing | NumPy, SciPy (Butterworth filter, FFT) |
| Database (local cache) | SQLite with WAL journal mode |


-------

**References:**
- Li et al., "Remote Heart Rate Measurement from Face Videos," CVPR 2014
- Balakrishnan et al., "Detecting Pulse from Head Motions in Video," CVPR 2013
- Pan et al., "Eyeblink-based Anti-Spoofing in Face Recognition," ICCV 2007
- Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv 2019
