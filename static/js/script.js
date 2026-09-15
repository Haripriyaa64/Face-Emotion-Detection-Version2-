(() => {
    "use strict";

    const camera = document.getElementById("camera");
    const overlay = document.getElementById("overlay");
    const ctx = overlay.getContext("2d");

    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const flipButton = document.getElementById("flipButton");
    const snapshotButton = document.getElementById("snapshotButton");

    const placeholder = document.getElementById("cameraPlaceholder");
    const cameraStage = document.getElementById("cameraStage");
    const liveBadge = document.getElementById("liveBadge");

    const statusText = document.getElementById("statusText");
    const statusDot = document.querySelector(".status-dot");

    const emotionName = document.getElementById("emotionName");
    const emotionGlyph = document.getElementById("emotionGlyph");
    const emotionOrb = document.getElementById("emotionOrb");
    const confidenceText = document.getElementById("confidenceText");
    const confidenceFill = document.getElementById("confidenceFill");
    const faceCount = document.getElementById("faceCount");
    const latency = document.getElementById("latency");
    const analysisState = document.getElementById("analysisState");

    const fpsLabel = document.getElementById("fpsLabel");
    const resolutionLabel = document.getElementById("resolutionLabel");

    const modelChip = document.getElementById("modelChip");
    const modelArchitecture = document.getElementById("modelArchitecture");
    const modelInput = document.getElementById("modelInput");
    const modelClasses = document.getElementById("modelClasses");
    const modelDetector = document.getElementById("modelDetector");

    const captureCanvas = document.createElement("canvas");
    const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: false });

    let stream = null;
    let analyzing = false;
    let requestInFlight = false;
    let loopTimer = null;
    let lastResultTime = performance.now();
    let framesInWindow = 0;
    let fpsTimer = performance.now();
    let currentFacingMode = "user";

    const EMOTION_GLYPHS = {
        Angry: "!",
        Disgust: "×",
        Fear: "◌",
        Happy: "⌣",
        Sad: "⌢",
        Surprise: "!",
        Neutral: "—"
    };

    const BAR_COLORS = {
        Angry: "#FF5C70",
        Disgust: "#67D88B",
        Fear: "#B783FF",
        Happy: "#39E58C",
        Sad: "#5EA7FF",
        Surprise: "#FFC857",
        Neutral: "#B9C2D0"
    };

    function setStatus(text, kind = "ok") {
        statusText.textContent = text;
        statusDot.classList.remove("warning", "error");
        if (kind === "warning") statusDot.classList.add("warning");
        if (kind === "error") statusDot.classList.add("error");
    }

    function setLive(active) {
        liveBadge.classList.toggle("active", active);
        liveBadge.innerHTML = active
            ? "<span></span> LIVE"
            : "<span></span> STANDBY";
        cameraStage.classList.toggle("active", active);
    }

    function resizeOverlay() {
        const width = camera.videoWidth || 640;
        const height = camera.videoHeight || 480;
        overlay.width = width;
        overlay.height = height;
        resolutionLabel.textContent = `${width} × ${height}`;
    }

    function clearOverlay() {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function drawResults(data) {
        clearOverlay();

        if (!data || !Array.isArray(data.faces)) return;

        const sourceWidth = data.frame?.width || camera.videoWidth || 640;
        const sourceHeight = data.frame?.height || camera.videoHeight || 480;

        const scaleX = overlay.width / sourceWidth;
        const scaleY = overlay.height / sourceHeight;

        data.faces.forEach((face) => {
            const box = face.box;
            const x = box.x * scaleX;
            const y = box.y * scaleY;
            const w = box.width * scaleX;
            const h = box.height * scaleY;
            const color = face.color || "#9A8FFF";

            ctx.save();

            // Glow.
            ctx.shadowBlur = 18;
            ctx.shadowColor = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.2;
            ctx.strokeRect(x, y, w, h);

            // Corner accents.
            ctx.shadowBlur = 0;
            ctx.lineWidth = 4;
            const c = Math.min(20, w * 0.18, h * 0.18);

            ctx.beginPath();
            ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
            ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
            ctx.moveTo(x, y + h - c); ctx.lineTo(x, y + h); ctx.lineTo(x + c, y + h);
            ctx.moveTo(x + w - c, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - c);
            ctx.stroke();

            // Label.
            const label = `${face.emotion}  ${face.confidence_percent}%`;
            ctx.font = "600 13px Inter, system-ui, sans-serif";
            const textWidth = ctx.measureText(label).width;
            const labelHeight = 25;
            const labelY = Math.max(0, y - labelHeight - 6);

            ctx.fillStyle = "rgba(5, 8, 15, .88)";
            ctx.fillRect(x, labelY, textWidth + 20, labelHeight);
            ctx.fillStyle = color;
            ctx.fillRect(x, labelY, 3, labelHeight);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, x + 10, labelY + 17);

            ctx.restore();
        });
    }

    function updateBars(scores = {}) {
        document.querySelectorAll(".bar-row").forEach((row) => {
            const emotion = row.dataset.emotion;
            const value = Math.max(0, Math.min(1, Number(scores[emotion] || 0)));
            const fill = row.querySelector("i");
            const number = row.querySelector("b");

            fill.style.width = `${value * 100}%`;
            fill.style.background = BAR_COLORS[emotion] || "linear-gradient(90deg,#6f61f5,#34e4b3)";
            number.textContent = `${Math.round(value * 100)}%`;
        });
    }

    function updatePrimary(primary, count, requestMs) {
        faceCount.textContent = count;
        latency.textContent = requestMs ? `${Math.round(requestMs)}ms` : "—";

        if (!primary) {
            emotionName.textContent = "No face";
            emotionGlyph.textContent = "·";
            confidenceText.textContent = "0%";
            confidenceFill.style.width = "0%";
            analysisState.textContent = analyzing ? "Scanning" : "Idle";
            updateBars({});
            return;
        }

        const confidence = Math.round(primary.confidence * 100);
        emotionName.textContent = primary.emotion;
        emotionGlyph.textContent = EMOTION_GLYPHS[primary.emotion] || "•";
        confidenceText.textContent = `${confidence}%`;
        confidenceFill.style.width = `${confidence}%`;
        analysisState.textContent = "Tracking";

        emotionOrb.style.borderColor = primary.color || "rgba(255,255,255,.13)";
        emotionOrb.style.boxShadow =
            `inset 0 0 25px rgba(255,255,255,.025), 0 0 38px ${primary.color || "#7c6cff"}33`;

        emotionOrb.classList.remove("flash");
        requestAnimationFrame(() => emotionOrb.classList.add("flash"));

        updateBars(primary.scores || {});
    }

    function updateFps() {
        framesInWindow += 1;
        const now = performance.now();

        if (now - fpsTimer >= 1000) {
            fpsLabel.textContent = `${framesInWindow} FPS`;
            framesInWindow = 0;
            fpsTimer = now;
        }
    }

    async function predictCurrentFrame() {
        if (!analyzing || requestInFlight || !camera.videoWidth) return;

        requestInFlight = true;
        const started = performance.now();

        try {
            // Small JPEG keeps mobile -> Render traffic reasonable.
            const targetWidth = Math.min(640, camera.videoWidth);
            const targetHeight = Math.round(targetWidth * (camera.videoHeight / camera.videoWidth));

            captureCanvas.width = targetWidth;
            captureCanvas.height = targetHeight;
            captureCtx.drawImage(camera, 0, 0, targetWidth, targetHeight);

            const blob = await new Promise((resolve) =>
                captureCanvas.toBlob(resolve, "image/jpeg", 0.72)
            );

            if (!blob) throw new Error("Could not create camera frame.");

            const form = new FormData();
            form.append("frame", blob, "frame.jpg");

            const response = await fetch("/predict", {
                method: "POST",
                body: form,
                cache: "no-store"
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            drawResults(data);
            updatePrimary(
                data.primary,
                data.face_count || 0,
                performance.now() - started
            );

            updateFps();
            lastResultTime = performance.now();
        } catch (error) {
            console.error(error);
            analysisState.textContent = "Error";

            if (error.message.includes("Model file") || error.message.includes("model")) {
                setStatus("Model unavailable", "error");
            } else {
                setStatus("Inference error", "error");
            }
        } finally {
            requestInFlight = false;
        }
    }

    function schedulePrediction() {
        if (!analyzing) return;

        // Roughly 4–5 requests/sec. This is much lighter than streaming 30 FPS.
        loopTimer = window.setTimeout(async () => {
            await predictCurrentFrame();
            schedulePrediction();
        }, 220);
    }

    async function startCamera() {
    if (analyzing) return;

    try {
        setStatus("Requesting camera…", "warning");

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        camera.srcObject = stream;

        // Don't wait for camera.play() to finish.
        // Some browsers can keep the video visible while play()
        // takes too long to resolve.
        camera.play().catch(error => {
            console.warn("Video play warning:", error);
        });

        resizeOverlay();

        placeholder.classList.add("hidden");
        startButton.disabled = true;
        stopButton.disabled = false;
        flipButton.disabled = false;
        snapshotButton.disabled = false;

        analyzing = true;
        setLive(true);
        setStatus("Vision system online");
        analysisState.textContent = "Scanning";

        // Start AI prediction immediately.
        schedulePrediction();

    } catch (error) {
        console.error("Camera error:", error);

        setStatus("Camera permission required", "error");
        analysisState.textContent = "Blocked";

        if (
            !window.isSecureContext &&
            location.hostname !== "localhost" &&
            location.hostname !== "127.0.0.1"
        ) {
            alert("Camera access requires HTTPS.");
        } else {
            alert(
                "Camera access failed. Please allow camera permission and try again."
            );
        }
    }
}

    function stopCamera() {
        analyzing = false;

        if (loopTimer) {
            clearTimeout(loopTimer);
            loopTimer = null;
        }

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        camera.srcObject = null;
        clearOverlay();

        placeholder.classList.remove("hidden");
        startButton.disabled = false;
        stopButton.disabled = true;
        flipButton.disabled = true;
        snapshotButton.disabled = true;

        setLive(false);
        setStatus("System ready");
        updatePrimary(null, 0, null);
        fpsLabel.textContent = "0 FPS";
        resolutionLabel.textContent = "—";
    }

    async function flipCamera() {
        currentFacingMode = currentFacingMode === "user" ? "environment" : "user";

        if (!analyzing) return;

        const wasAnalyzing = analyzing;
        stopCamera();

        if (wasAnalyzing) {
            await startCamera();
        }
    }

    function snapshot() {
        if (!camera.videoWidth) return;

        const shot = document.createElement("canvas");
        shot.width = camera.videoWidth;
        shot.height = camera.videoHeight;

        const shotCtx = shot.getContext("2d");
        shotCtx.translate(shot.width, 0);
        shotCtx.scale(-1, 1);
        shotCtx.drawImage(camera, 0, 0, shot.width, shot.height);

        // Draw the current overlay as well.
        shotCtx.setTransform(1, 0, 0, 1, 0, 0);
        shotCtx.drawImage(overlay, 0, 0, shot.width, shot.height);

        const link = document.createElement("a");
        link.download = `neuralface-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
        link.href = shot.toDataURL("image/jpeg", 0.92);
        link.click();
    }

    async function checkModel() {
        try {
            const response = await fetch("/health", { cache: "no-store" });
            const data = await response.json();

            modelArchitecture.textContent = data.model || "CNN / HDF5";
            modelInput.textContent = data.input_size || "—";
            modelClasses.textContent = data.emotions ?? "7";
            modelDetector.textContent = data.face_detector || "OpenCV Haar";

            if (data.ready) {
                modelChip.textContent = "READY";
                modelChip.classList.remove("error");
                if (!analyzing) setStatus("System ready");
            } else {
                modelChip.textContent = "MISSING";
                modelChip.classList.add("error");
                setStatus("Model unavailable", "error");
                console.warn(data.error);
            }
        } catch (error) {
            modelChip.textContent = "OFFLINE";
            modelChip.classList.add("error");
            setStatus("Backend unavailable", "error");
        }
    }

    startButton.addEventListener("click", startCamera);
    stopButton.addEventListener("click", stopCamera);
    flipButton.addEventListener("click", flipCamera);
    snapshotButton.addEventListener("click", snapshot);

    document.getElementById("themeGlow").addEventListener("click", () => {
        document.body.classList.toggle("glow-off");
    });

    camera.addEventListener("loadedmetadata", resizeOverlay);
    window.addEventListener("resize", resizeOverlay);

    window.addEventListener("beforeunload", () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
    });

    if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("Browser camera unsupported", "error");
        startButton.disabled = true;
    }

    checkModel();
})();
