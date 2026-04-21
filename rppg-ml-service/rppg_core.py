import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_roi_signals(video_path):
    logger.info(f"Opening video: {video_path}")

    # Lazy import — keeps MediaPipe out of the module-level import chain
    # which was crashing the uvicorn reloader worker process on startup
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
    except Exception as e:
        logger.error(f"Failed to import mediapipe: {e}")
        return None, None

    cap = None
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            logger.info(f"Opened video with backend {backend}")
            break

    if not cap or not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FIX 1: Guard against bad FPS metadata from browser WebM files.
    # MediaRecorder WebM often reports 0 or 1000 fps — clamp to a sane default.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        logger.warning(f"Unreliable FPS from container ({fps}) — defaulting to 30")
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {total_frames} frames, {fps} fps, {width}x{height}")

    # FIX 2: Use static_image_mode=False (tracking mode) for video input.
    # static_image_mode=True re-runs full detection on every frame independently,
    # causing many frames to fail detection in compressed browser WebM streams.
    # Tracking mode is far more robust for consecutive video frames.
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,        # FIXED: was True — must be False for video
        max_num_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    signals = {"forehead": [], "left_cheek": [], "right_cheek": []}
    first_frame    = None   # set to first frame where a face IS confirmed
    frame_count    = 0
    frames_with_face = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # FIX 3: Resize large frames before processing.
        # Browser webcams can send 1280×720 or larger which slows MediaPipe.
        # Downscale to max 640 wide while keeping aspect ratio.
        h_orig, w_orig = frame.shape[:2]
        if w_orig > 640:
            scale  = 640.0 / w_orig
            frame  = cv2.resize(frame, (640, int(h_orig * scale)))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            frames_with_face += 1
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            if first_frame is None:
                first_frame = frame.copy()
                logger.info(f"First confirmed face at frame {frame_count}")

            def get_roi_mean(indices):
                pts = np.array([[int(landmarks[i].x * w),
                                 int(landmarks[i].y * h)] for i in indices])
                x, y, wb, hb = cv2.boundingRect(pts)
                x,  y  = max(0, x),      max(0, y)
                wb, hb = min(wb, w - x), min(hb, h - y)
                if wb <= 0 or hb <= 0:
                    return [0, 0, 0]
                roi = frame[y:y + hb, x:x + wb]
                return cv2.mean(roi)[:3] if roi.size > 0 else [0, 0, 0]

            signals["forehead"].append(get_roi_mean([10, 338, 297, 332, 284]))
            signals["left_cheek"].append(get_roi_mean([118, 119, 100, 126]))
            signals["right_cheek"].append(get_roi_mean([347, 348, 329, 355]))

            if frame_count % 30 == 0:
                logger.info(f"Frames processed: {frame_count}, with face: {frames_with_face}")
        else:
            if frame_count <= 10:
                logger.warning(f"No face in frame {frame_count}")

    cap.release()
    face_mesh.close()
    logger.info(f"Done: {frame_count} frames total, {frames_with_face} with face")

    for roi in signals:
        signals[roi] = np.array(signals[roi]) if signals[roi] else np.array([])

    if frames_with_face == 0:
        logger.error("No faces detected in any frame")
        return None, None

    # FIX 4: Return the validated fps alongside signals so callers don't have
    # to re-read container metadata (which may again be wrong).
    # We attach fps as a top-level key in the signals dict for backward compat.
    signals["_fps"] = fps

    return signals, first_frame