# 🛡️ rPPG Contactless Biometric Authentication & Anti-Spoofing

![Next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge&logo=next.js)
![FastAPI](https://img.shields.io/badge/ML_Engine-FastAPI_&_Python-009688?style=for-the-badge&logo=fastapi)
![Spring Boot](https://img.shields.io/badge/Gateway-Spring_Boot_3-6DB33F?style=for-the-badge&logo=springboot)
![PostgreSQL](https://img.shields.io/badge/Database-Neon_PostgreSQL-336791?style=for-the-badge&logo=postgresql)

An enterprise-grade, fully contactless biometric authentication system. This project uses **Remote Photoplethysmography (rPPG)** to detect the live human heartbeat from a standard webcam stream, ensuring high-security liveness detection and preventing deepfake, printed photo, and screen-replay attacks.

---

## ✨ Key Features
1. **rPPG Signal Extraction**: Uses MediaPipe and optical physics to extract the human pulse by measuring microscopic color changes in facial regions (Forehead, Cheeks).
2. **Spatiotemporal Anti-Spoofing**: Prevents screen-replay attacks. While a human heart creates slight phase shifts in blood flow across the face, a replayed video on an LCD screen modulates uniformly. The system calculates cross-ROI Spatial Coherence to block spoof attempts.
3. **FaceNet Biometric Embeddings**: Extracts 128-dimensional facial embeddings for highly accurate identity verification using Cosine Similarity.
4. **Microservice Architecture**: A clean separation of concerns using React (UI), Java Spring Boot (Security Gateway), and Python FastAPI (Heavy Math/ML).

---

## 🏛️ System Architecture

The project is divided into three parallel microservices:

```text
[ Browser / Webcam ]
        │  (Captures 5s WebM Video at 30fps)
        ▼
[ Next.js Frontend ] (Port 3000)
        │  (Sends via multipart/form-data)
        ▼
[ Spring Boot API Gateway ] (Port 8080)
        │  (Manages DB, Routes traffic, Verifies Match)
        ▼
[ Python ML Service ] (Port 8000)
           ├── MediaPipe (Face tracking & ROI extraction)
           ├── SciPy (Butterworth Bandpass Filtering 0.7-3.0 Hz)
           └── DeepFace (FaceNet 128D Embedding extraction)
🚀 Getting Started
Prerequisites

Node.js (v18+)

Python (3.9+)

Java (JDK 17)

Maven

Neon Serverless PostgreSQL (or local Postgres)

1. Start the Python ML Engine

This service handles the heavy mathematical lifting, signal filtering, and AI processing.

code
Bash
download
content_copy
expand_less
cd rppg-ml-service
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install fastapi uvicorn python-multipart opencv-python mediapipe numpy scipy deepface tf-keras
python main.py

Runs on http://localhost:8000

2. Start the Spring Boot Gateway

This acts as the orchestrator and talks to the PostgreSQL database.

Open src/main/resources/application.yml.

Update the spring.datasource.url, username, and password with your Neon PostgreSQL credentials.

Run the application:

code
Bash
download
content_copy
expand_less
cd biometric
mvn spring-boot:run

Runs on http://localhost:8080

3. Start the Next.js Frontend

The UI built with Tailwind CSS and Shadcn UI.

code
Bash
download
content_copy
expand_less
cd frontend
npm install
npm run dev

Runs on http://localhost:3000

🧪 How to Use the System

Enrollment:

Navigate to http://localhost:3000.

Click 1. Enroll Face.

Look directly at the camera and hold still for 5 seconds.

What happens: The system calculates your liveness. If you are a real human, it extracts your 128D facial embedding and securely stores it as a JSON string in the Neon PostgreSQL database.

Login / Authentication:

Click 2. Login.

Hold still for 5 seconds.

What happens: Python analyzes your pulse to ensure you aren't holding up an iPad with a recorded video. Spring Boot then calculates the Cosine Similarity between the live video and your stored database embedding. If it is > 0.75, access is granted.

Spoofing Attempt:

Try pointing your phone screen at the webcam playing a video of yourself.

The Python service will calculate a coherence_score > 0.95 (due to uniform LCD pixel illumination) and throw a 401: Spoof Detected error.

🛠️ Tech Stack details

Frontend: React, Next.js, Tailwind CSS, Shadcn UI, MediaRecorder API.

Backend: Java 17, Spring Boot 3, Spring Data JPA, Hibernate.

Machine Learning: Python, FastAPI, OpenCV, MediaPipe (Face Mesh), SciPy (Signal Processing), DeepFace.

Database: PostgreSQL (Neon Tech Serverless).

🔮 Future Enhancements

Challenge-Response: Prompting the user to blink twice or turn their head to dynamically verify liveness.

Micro-ballistocardiography (BCG): Tracking sub-pixel vertical head movements that couple with the human heartbeat.

JWT Implementation: Returning standard JSON Web Tokens upon successful Spring Boot authentication.
