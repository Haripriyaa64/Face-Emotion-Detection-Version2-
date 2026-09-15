from flask import Flask, jsonify, render_template, request
from detect_emotion import get_model_status, predict_frame

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024  # 3 MB/frame


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    status = get_model_status()
    return jsonify({
        "status": "ok" if status["ready"] else "degraded",
        **status,
    })


@app.get("/api/model-info")
def model_info():
    return jsonify(get_model_status())


@app.post("/predict")
def predict():
    frame = request.files.get("frame")

    if frame is None:
        return jsonify({"error": "No frame was uploaded."}), 400

    result = predict_frame(frame.stream)

    if "error" in result:
        return jsonify(result), 503 if result.get("code") == "MODEL_MISSING" else 400

    return jsonify(result)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Frame is too large. Keep the camera image under 3 MB."}), 413


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "The server could not process this frame."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
