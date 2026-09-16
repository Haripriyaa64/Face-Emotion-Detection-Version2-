import os

# Reduce TensorFlow console noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
from typing import BinaryIO
import threading

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================

MODEL_CANDIDATES = [
    BASE_DIR / "model" / "emotion_model.hdf5",
    BASE_DIR / "model" / "emotion_model.h5",
]


# ============================================================
# EMOTION LABELS
# IMPORTANT:
# This order MUST match the order used when training the model.
# ============================================================

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


# ============================================================
# COLORS
# ============================================================

EMOTION_COLORS = {
    "Happy": "#39E58C",
    "Sad": "#5EA7FF",
    "Angry": "#FF5C70",
    "Surprise": "#FFC857",
    "Fear": "#B783FF",
    "Disgust": "#67D88B",
    "Neutral": "#B9C2D0",
}


# ============================================================
# OPENCV HAAR CASCADE
# ============================================================

CASCADE_PATH = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

_model = None
_model_error = None
MODEL_PATH = None

IMG_SIZE = 48
MODEL_CHANNELS = 1


# Prevent multiple TensorFlow predictions from running
# simultaneously on the Render instance.
_model_lock = threading.Lock()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    MODEL_PATH = next(
        (
            path
            for path in MODEL_CANDIDATES
            if path.exists()
        ),
        None,
    )

    if MODEL_PATH is None:

        _model_error = (
            "Emotion model not found. "
            "Expected model/emotion_model.hdf5 "
            "or model/emotion_model.h5."
        )

    else:

        print(
            f"[NeuralFace] Loading model: {MODEL_PATH}",
            flush=True,
        )

        _model = load_model(
            str(MODEL_PATH),
            compile=False,
        )

        # ----------------------------------------------------
        # Determine input shape
        # ----------------------------------------------------

        input_shape = _model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        print(
            f"[NeuralFace] Model input shape: {input_shape}",
            flush=True,
        )

        # Expected:
        # (None, 48, 48, 1)
        # or
        # (None, 48, 48, 3)

        if len(input_shape) == 4:

            if input_shape[1] is not None:
                IMG_SIZE = int(input_shape[1])

            if input_shape[-1] in (1, 3):
                MODEL_CHANNELS = int(input_shape[-1])

        print(
            f"[NeuralFace] Image size: {IMG_SIZE}x{IMG_SIZE}",
            flush=True,
        )

        print(
            f"[NeuralFace] Channels: {MODEL_CHANNELS}",
            flush=True,
        )

        # ----------------------------------------------------
        # Validate output classes
        # ----------------------------------------------------

        output_shape = _model.output_shape

        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        print(
            f"[NeuralFace] Model output shape: {output_shape}",
            flush=True,
        )

        if (
            len(output_shape) < 2
            or output_shape[-1] != len(EMOTION_LABELS)
        ):

            _model_error = (
                f"Model output has "
                f"{output_shape[-1] if len(output_shape) else 'unknown'} "
                f"classes, but this application expects "
                f"{len(EMOTION_LABELS)}."
            )

            _model = None

        else:

            # ------------------------------------------------
            # Warm up TensorFlow model
            # ------------------------------------------------

            print(
                "[NeuralFace] Warming up TensorFlow...",
                flush=True,
            )

            dummy = np.zeros(
                (
                    1,
                    IMG_SIZE,
                    IMG_SIZE,
                    MODEL_CHANNELS,
                ),
                dtype=np.float32,
            )

            _model.predict(
                dummy,
                verbose=0,
            )

            print(
                "[NeuralFace] Model ready.",
                flush=True,
            )


except Exception as exc:

    _model = None

    _model_error = (
        f"Could not load the emotion model: "
        f"{type(exc).__name__}: {exc}"
    )

    print(
        f"[NeuralFace] MODEL ERROR: {_model_error}",
        flush=True,
    )


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    str(CASCADE_PATH)
)

if face_cascade.empty():

    raise RuntimeError(
        f"Could not load Haar cascade: {CASCADE_PATH}"
    )

print(
    "[NeuralFace] Haar face detector ready.",
    flush=True,
)


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status():

    return {
        "ready": _model is not None,
        "model": (
            "CNN / HDF5"
            if _model is not None
            else "CNN / HDF5 unavailable"
        ),
        "model_file": (
            MODEL_PATH.name
            if MODEL_PATH is not None
            else None
        ),
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "channels": MODEL_CHANNELS,
        "emotions": len(EMOTION_LABELS),
        "face_detector": "OpenCV Haar Cascade",
        "error": _model_error,
    }


# ============================================================
# PREPARE FACE
# ============================================================

def _prepare_face(
    gray,
    x,
    y,
    w,
    h,
):
    """
    Crop and preprocess the detected face.
    """

    # Add margin around face
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)

    x1 = max(
        0,
        x - margin_x,
    )

    y1 = max(
        0,
        y - margin_y,
    )

    x2 = min(
        gray.shape[1],
        x + w + margin_x,
    )

    y2 = min(
        gray.shape[0],
        y + h + margin_y,
    )

    face = gray[
        y1:y2,
        x1:x2,
    ]

    if face.size == 0:
        return None

    # Resize to model size
    face = cv2.resize(
        face,
        (
            IMG_SIZE,
            IMG_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Normalize 0-1
    face = face.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Model expects grayscale
    # --------------------------------------------------------

    if MODEL_CHANNELS == 1:

        roi = face.reshape(
            1,
            IMG_SIZE,
            IMG_SIZE,
            1,
        )

    # --------------------------------------------------------
    # Model expects RGB
    # --------------------------------------------------------

    elif MODEL_CHANNELS == 3:

        rgb = cv2.cvtColor(
            face,
            cv2.COLOR_GRAY2RGB,
        )

        roi = rgb.reshape(
            1,
            IMG_SIZE,
            IMG_SIZE,
            3,
        )

    else:

        raise ValueError(
            f"Unsupported model channel count: "
            f"{MODEL_CHANNELS}"
        )

    return roi


# ============================================================
# CONVERT MODEL OUTPUT TO PROBABILITIES
# ============================================================

def _normalize_prediction(prediction):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    ).flatten()

    if len(prediction) != len(EMOTION_LABELS):

        raise ValueError(
            f"Model returned {len(prediction)} values. "
            f"Expected {len(EMOTION_LABELS)}."
        )

    # Remove invalid numbers
    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # If values are already probabilities
    # between 0 and 1 and approximately sum to 1,
    # keep them.
    total = float(
        prediction.sum()
    )

    if (
        np.all(prediction >= 0)
        and total > 0
        and abs(total - 1.0) < 0.10
    ):

        prediction = (
            prediction / total
        )

    else:

        # Otherwise treat output as logits
        # and apply softmax.
        shifted = (
            prediction
            - np.max(prediction)
        )

        exp_values = np.exp(
            shifted
        )

        exp_total = float(
            exp_values.sum()
        )

        if exp_total <= 0:
            raise ValueError(
                "Invalid model prediction."
            )

        prediction = (
            exp_values
            / exp_total
        )

    return prediction


# ============================================================
# PREDICT FRAME
# ============================================================

def predict_frame(
    stream: BinaryIO,
):
    """
    Receive a JPEG frame from the browser.

    Pipeline:

        JPEG
          ↓
        OpenCV
          ↓
        Haar face detection
          ↓
        Face crop
          ↓
        CNN
          ↓
        Emotion
          ↓
        JSON
    """

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if _model is None:

        return {
            "error": (
                _model_error
                or "Emotion model is unavailable."
            ),
            "code": "MODEL_MISSING",
        }

    try:

        # ====================================================
        # 1. READ FRAME
        # ====================================================

        raw = stream.read()

        if not raw:

            return {
                "error": "Empty frame received.",
                "code": "EMPTY_FRAME",
            }

        # ====================================================
        # 2. DECODE JPEG
        # ====================================================

        image = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image,
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            return {
                "error": "Invalid camera image.",
                "code": "INVALID_FRAME",
            }

        # ====================================================
        # 3. RESIZE FOR RENDER
        # ====================================================

        max_width = 640

        if frame.shape[1] > max_width:

            scale = (
                max_width
                / frame.shape[1]
            )

            frame = cv2.resize(
                frame,
                (
                    max_width,
                    int(
                        frame.shape[0]
                        * scale
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # ====================================================
        # 4. GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ====================================================
        # 5. FACE DETECTION
        # ====================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )

        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            return {
                "faces": [],
                "face_count": 0,
                "primary": None,
                "frame": {
                    "width": int(
                        frame_width
                    ),
                    "height": int(
                        frame_height
                    ),
                },
            }

        # ====================================================
        # 6. SORT BY SIZE
        # ====================================================

        faces = sorted(
            faces,
            key=lambda face:
                face[2] * face[3],
            reverse=True,
        )

        # Process only largest face.
        # This keeps Render processing fast.
        x, y, w, h = faces[0]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # ====================================================
        # 7. PREPARE FACE
        # ====================================================

        roi = _prepare_face(
            gray,
            x,
            y,
            w,
            h,
        )

        if roi is None:

            return {
                "error": "Could not crop detected face.",
                "code": "EMPTY_FACE",
            }

        # ====================================================
        # 8. TENSORFLOW PREDICTION
        # ====================================================

        # Only one prediction at a time.
        # This prevents multiple browser requests
        # from hitting TensorFlow simultaneously.
        with _model_lock:

            prediction = _model.predict_on_batch(
                roi
            )

        # ====================================================
        # 9. NORMALIZE OUTPUT
        # ====================================================

        prediction = _normalize_prediction(
            prediction
        )

        # ====================================================
        # 10. GET EMOTION
        # ====================================================

        index = int(
            np.argmax(
                prediction
            )
        )

        emotion = EMOTION_LABELS[index]

        confidence = float(
            np.clip(
                prediction[index],
                0.0,
                1.0,
            )
        )

        # ====================================================
        # 11. ALL EMOTION SCORES
        # ====================================================

        scores = {}

        for i, label in enumerate(
            EMOTION_LABELS
        ):

            scores[label] = float(
                np.clip(
                    prediction[i],
                    0.0,
                    1.0,
                )
            )

        # ====================================================
        # 12. FACE BOX
        # ====================================================

        box = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        }

        # ====================================================
        # 13. PRIMARY RESULT
        # ====================================================

        primary = {
            "box": box,
            "emotion": emotion,
            "confidence": confidence,
            "confidence_percent": round(
                confidence * 100,
                1,
            ),
            "color": EMOTION_COLORS.get(
                emotion,
                "#FFFFFF",
            ),
            "scores": scores,
        }

        # ====================================================
        # 14. RETURN RESULT
        # ====================================================

        return {
            "faces": [
                primary
            ],
            "face_count": 1,
            "primary": primary,
            "frame": {
                "width": int(
                    frame_width
                ),
                "height": int(
                    frame_height
                ),
            },
        }

    # ========================================================
    # BACKEND ERROR
    # ========================================================

    except Exception as exc:

        print(
            "[PREDICT ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return {
            "error": str(exc),
            "code": "PREDICTION_ERROR",
        }