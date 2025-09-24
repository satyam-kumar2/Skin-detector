// static/main.js
// Handles upload, predict, gradcam and live camera tracking

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");

const resultImage = document.getElementById("result-image");
const detectionsList = document.getElementById("detections-list");
const gradcamControls = document.getElementById("gradcam-controls");
const gradcamImage = document.getElementById("gradcam-image");

// Camera elements
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const camFpsInput = document.getElementById("camFps");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay ? overlay.getContext("2d") : null;

// Skin score display and panel
const skinScoreValue = document.getElementById("skin-score-value");
const scorePanel = document.querySelector(".score-panel");

let lastFile = null;
let lastDetections = [];
let camStream = null;
let camInterval = null;
let tracker = {}; // {id: {box, lastSeen}}
let nextTrackId = 1;
const IOU_THRESHOLD_TRACK = 0.3;

// ---------- helpers ----------
function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtn.style.opacity = on ? 0.6 : 1;
}

function resetUI() {
  resultImage.src = "";
  gradcamImage.src = "";
  detectionsList.innerHTML = "";
  gradcamControls.innerHTML = "";
  lastDetections = [];
  lastFile = null;
  fileInput.value = "";
  stopCamera();
  clearOverlay();
  updateSkinScoreVisual(null);
}

function clearOverlay() {
  if (!overlay || !octx) return;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  tracker = {};
  nextTrackId = 1;
}

// update score panel visuals and text
function updateSkinScoreVisual(score) {
  if (!scorePanel || !skinScoreValue) return;

  // clean previous state
  scorePanel.classList.remove("good", "moderate", "poor");

  if (score === null || score === undefined || Number.isNaN(Number(score))) {
    skinScoreValue.textContent = "--";
    return;
  }

  // clamp and round to 0-100
  let s = Math.round(Number(score));
  if (s < 0) s = 0;
  if (s > 100) s = 100;

  skinScoreValue.textContent = s;

  // thresholds (tweakable)
  if (s >= 75) scorePanel.classList.add("good");
  else if (s >= 50) scorePanel.classList.add("moderate");
  else scorePanel.classList.add("poor");
}

// IoU helper
function iou(boxA, boxB) {
  const [x1,y1,x2,y2] = boxA;
  const [X1,Y1,X2,Y2] = boxB;
  const xx1 = Math.max(x1, X1);
  const yy1 = Math.max(y1, Y1);
  const xx2 = Math.min(x2, X2);
  const yy2 = Math.min(y2, Y2);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w*h;
  const areaA = Math.max(0, x2-x1) * Math.max(0, y2-y1);
  const areaB = Math.max(0, X2-X1) * Math.max(0, Y2-Y1);
  const union = areaA + areaB - inter + 1e-6;
  return inter / union;
}

// draw box with id and label
function drawBox(ctx, box, label, color="#0b74ff", lineWidth=2, id=null) {
  const [x1,y1,x2,y2] = box;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.fillStyle = color;
  ctx.font = "14px Inter, Arial";
  ctx.textBaseline = "top";
  const text = id ? `${label} #${id}` : `${label}`;
  const textWidth = ctx.measureText(text).width;
  const pad = 6;
  const rectW = textWidth + pad*2;
  const rectH = 20;
  ctx.globalAlpha = 0.9;
  ctx.fillRect(x1, Math.max(0,y1-rectH-6), rectW, rectH);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x1 + pad, Math.max(0,y1-rectH-6) + 2);
}

// ---------- Upload flow ----------
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (!f) return;
  lastFile = f;
  const reader = new FileReader();
  reader.onload = (e) => {
    resultImage.src = e.target.result;
  };
  reader.readAsDataURL(f);
});

resetBtn.addEventListener("click", (e) => {
  e.preventDefault();
  resetUI();
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files[0]) {
    alert("Please select an image first.");
    return;
  }
  setLoading(true);
  detectionsList.innerHTML = "";
  gradcamControls.innerHTML = "";
  gradcamImage.src = "";

  try {
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("conf_thresh", "0.25");

    const res = await fetch("/predict/", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({error:"Server returned error"}));
      alert("Error: " + (err.error || "Server returned error"));
      setLoading(false);
      return;
    }
    const data = await res.json();

    // show annotated image
    resultImage.src = "data:image/jpeg;base64," + data.result_image;

    // update skin score if provided
    if (data.skin_score !== undefined) updateSkinScoreVisual(data.skin_score);
    else updateSkinScoreVisual(null);

    // list detections
    lastDetections = data.detections || [];
    detectionsList.innerHTML = "";
    gradcamControls.innerHTML = "";

    if (lastDetections.length === 0) {
      const li = document.createElement("li"); li.textContent = "No detections found."; detectionsList.appendChild(li);
    } else {
      lastDetections.forEach((d, idx) => {
        const li = document.createElement("li");
        li.innerHTML = `<span>Detection ${idx+1} — ${d.label} (${(d.score*100).toFixed(1)}%)</span>`;
        li.addEventListener("click", () => requestGradcam(idx));
        detectionsList.appendChild(li);
        const btn = document.createElement("button");
        btn.className = "btn ghost"; btn.type = "button"; btn.textContent = `Explain #${idx+1}`;
        btn.addEventListener("click", () => requestGradcam(idx));
        gradcamControls.appendChild(btn);
      });
    }
  } catch (err) {
    console.error(err);
    alert("Error during request: " + err.message);
  } finally {
    setLoading(false);
  }
});

// ---------- Grad-CAM ----------
async function requestGradcam(index) {
  if (!lastFile && !fileInput.files[0]) {
    alert("Please upload an image first.");
    return;
  }
  setLoading(true);
  try {
    const fd = new FormData();
    fd.append("file", fileInput.files[0] || lastFile);
    fd.append("box_index", String(index));
    fd.append("conf_thresh", "0.25");

    const res = await fetch("/gradcam/", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({error:"server error"}));
      alert("Grad-CAM Error: " + (err.error || "server error"));
      setLoading(false);
      return;
    }
    const data = await res.json();
    gradcamImage.src = "data:image/jpeg;base64," + data.gradcam_image;
  } catch (err) {
    console.error(err);
    alert("Grad-CAM request failed: " + err.message);
  } finally {
    setLoading(false);
  }
}

// ---------- Live camera flow ----------
startCamBtn.addEventListener("click", startCamera);
stopCamBtn.addEventListener("click", stopCamera);

async function startCamera() {
  if (camStream) return;
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = camStream;
    await video.play();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    overlay.style.width = video.style.width || video.videoWidth + "px";
    overlay.style.height = video.style.height || video.videoHeight + "px";

    startCamBtn.disabled = true;
    stopCamBtn.disabled = false;

    const fps = Math.max(1, Math.min(15, parseInt(camFpsInput.value || "5")));
    const intervalMs = Math.round(1000 / fps);

    camInterval = setInterval(captureAndSendFrame, intervalMs);
  } catch (err) {
    console.error("Camera error:", err);
    alert("Could not start camera: " + err.message);
    camStream = null;
  }
}

function stopCamera() {
  if (camInterval) {
    clearInterval(camInterval);
    camInterval = null;
  }
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  startCamBtn.disabled = false;
  stopCamBtn.disabled = true;
  clearOverlay();
}

async function captureAndSendFrame() {
  if (!video || video.readyState < 2) return;

  const tmp = document.createElement("canvas");
  tmp.width = video.videoWidth;
  tmp.height = video.videoHeight;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(video, 0, 0, tmp.width, tmp.height);

  tmp.toBlob(async (blob) => {
    try {
      const fd = new FormData();
      fd.append("file", blob, "frame.jpg");
      fd.append("conf_thresh", String( parseFloat(document.getElementById("camFps").dataset.conf || 0.35) || 0.35 ));

      const res = await fetch("/predict_frame/", { method: "POST", body: fd });
      if (!res.ok) return;
      const data = await res.json();

      // update live skin score if provided
      if (data.skin_score !== undefined) updateSkinScoreVisual(data.skin_score);

      const dets = data.detections || [];
      drawLiveDetections(dets, tmp.width, tmp.height);
    } catch (err) {
      console.error("capture/send error", err);
    }
  }, "image/jpeg", 0.8);
}

function drawLiveDetections(detections, imgW, imgH) {
  if (!octx) return;
  overlay.width = imgW;
  overlay.height = imgH;
  octx.clearRect(0,0, overlay.width, overlay.height);

  const unmatchedNew = [];
  const newBoxes = detections.map(d => [d.x1, d.y1, d.x2, d.y2, d.score]);

  const assignedPrev = new Set();
  const assignedNew = new Set();

  for (let i=0; i<newBoxes.length; i++) {
    let bestId = null;
    let bestIou = IOU_THRESHOLD_TRACK;
    for (const [tid, t] of Object.entries(tracker)) {
      if (assignedPrev.has(tid)) continue;
      const iouVal = iou(newBoxes[i], t.box);
      if (iouVal > bestIou) {
        bestIou = iouVal;
        bestId = parseInt(tid);
      }
    }
    if (bestId !== null) {
      tracker[bestId] = { box: newBoxes[i], lastSeen: Date.now() };
      assignedPrev.add(String(bestId));
      assignedNew.add(i);
    } else {
      unmatchedNew.push(i);
    }
  }

  for (const idx of unmatchedNew) {
    const id = nextTrackId++;
    tracker[id] = { box: newBoxes[idx], lastSeen: Date.now() };
    assignedNew.add(idx);
  }

  const now = Date.now();
  for (const [tid, t] of Object.entries(tracker)) {
    if (now - t.lastSeen > 1500) {
      delete tracker[tid];
    }
  }

  for (const [tid, t] of Object.entries(tracker)) {
    const id = parseInt(tid);
    const [x1,y1,x2,y2,score] = t.box;
    drawBox(octx, [x1,y1,x2,y2], `Acne ${(score*100).toFixed(0)}%`, "#0b74ff", 2, id);
  }
}
