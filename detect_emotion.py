import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent

# Accept either filename so the project is tolerant of .hdf5 / .h5.
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "emotion_model.hdf5",
    BASE_DIR / "model" / "emotion_model.h5",
]

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# Browser-facing theme colors. OpenCV itself uses BGR internally.
EMOTION_COLORS = {
    "Happy": "#39E58C",
    "Sad": "#5EA7FF",
    "Angry": "#FF5C70",
    "Surprise": "#FFC857",
    "Fear": "#B783FF",
    "Disgust": "#67D88B",
    "Neutral": "#B9C2D0",
}

# Use OpenCV's bundled cascade instead of requiring a manually copied XML.
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

_model = None
_model_error = None
IMG_SIZE = 48

try:
    MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

    if MODEL_PATH is None:
        _model_error = (
            "Model file not found. Put emotion_model.hdf5 "
            "inside the model/ folder."
        )
    else:
        _model = load_model(str(MODEL_PATH), compile=False)

        # Works with models shaped like (None, 48, 48, 1) or (None, 64, 64, 1).
        if len(_model.input_shape) >= 3 and _model.input_shape[1]:
            IMG_SIZE = int(_model.input_shape[1])

except Exception as exc:
    _model_error = f"Could not load the emotion model: {exc}"

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")


def get_model_status():
    return {
        "ready": _model is not None,
        "model": "CNN / HDF5" if _model is not None else "CNN / HDF5 unavailable",
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "emotions": len(EMOTION_LABELS),
        "face_detector": "OpenCV Haar Cascade",
        "error": _model_error,
    }


def _prepare_face(gray, x, y, w, h):
    # Add a small margin around the detected face.
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.10)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(gray.shape[1], x + w + margin_x)
    y2 = min(gray.shape[0], y + h + margin_y)

    roi = gray[y1:y2, x1:x2]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return roi


def predict_frame(stream: BinaryIO):
    if _model is None:
        return {
            "error": _model_error or "Emotion model is unavailable.",
            "code": "MODEL_MISSING",
        }

    raw = stream.read()
    if not raw:
        return {"error": "Empty frame received."}

    image = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image frame."}

    # Limit processing size for predictable server performance.
    max_width = 960
    if frame.shape[1] > max_width:
        ratio = max_width / frame.shape[1]
        frame = cv2.resize(
            frame,
            (max_width, int(frame.shape[0] * ratio)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect on a half-size image for better latency.
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    faces_small = face_cascade.detectMultiScale(
        small,
        scaleFactor=1.10,
        minNeighbors=4,
        minSize=(24, 24),
    )

    faces = []
    for x, y, w, h in faces_small:
        faces.append((int(x * 2), int(y * 2), int(w * 2), int(h * 2)))

    # Keep the UI predictable if a crowded scene is presented.
    faces = faces[:5]

    results = []

    for x, y, w, h in faces:
        roi = _prepare_face(gray, x, y, w, h)

        prediction = np.asarray(_model.predict(roi, verbose=0)[0], dtype=np.float32)

        # Normalize defensively if a custom model returns slightly unusual output.
        total = float(prediction.sum())
        if total > 0:
            prediction = prediction / total

        index = int(np.argmax(prediction))
        emotion = EMOTION_LABELS[index]
        confidence = float(np.clip(prediction[index], 0.0, 1.0))

        scores = {
            EMOTION_LABELS[i]: float(np.clip(prediction[i], 0.0, 1.0))
            for i in range(min(len(EMOTION_LABELS), len(prediction)))
        }

        results.append({
            "box": {"x": x, "y": y, "width": w, "height": h},
            "emotion": emotion,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 1),
            "color": EMOTION_COLORS.get(emotion, "#FFFFFF"),
            "scores": scores,
        })

    primary = results[0] if results else None

    return {
        "faces": results,
        "face_count": len(results),
        "primary": primary,
        "frame": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        },
    }
