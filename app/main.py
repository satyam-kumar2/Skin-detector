# app/main.py
import io
import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np

from app.model_loader import yolo_model
from app.utils import draw_detections, image_to_base64
from app.gradcam import gradcam_full_image_from_box

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = str(BASE_DIR / "templates")
STATIC_DIR = str(BASE_DIR / "static")

app = FastAPI(title="Skin Detector")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...), conf_thresh: float = Form(0.25)):
    """
    Original predict endpoint: returns annotated image + detection list + skin score
    """
    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    detections = yolo_model.predict(img)

    # Normalize class id -> 0 (Acne) for frontend consistency
    detections_norm = []
    for d in detections:
        x1, y1, x2, y2, score, cls = d
        if score < conf_thresh:
            continue
        detections_norm.append([int(x1), int(y1), int(x2), int(y2), float(score), 0])

    annotated = draw_detections(img.copy(), detections_norm, class_names={0: "Acne"})
    img_b64 = image_to_base64(annotated)

    # --- compute skin score ---
    if detections_norm:
        weighted_sum = sum(d[4] for d in detections_norm)
        skin_score = max(1, min(100, int(100 - weighted_sum * 2)))
    else:
        skin_score = 100

    return {
        "detections": [
            {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3],
             "score": d[4], "class_id": d[5], "label": "Acne"}
            for d in detections_norm
        ],
        "result_image": img_b64,
        "skin_score": skin_score
    }


@app.post("/predict_frame/")
async def predict_frame(file: UploadFile = File(...), conf_thresh: float = Form(0.35)):
    """
    Lightweight frame endpoint for live camera:
    Accepts a single image frame (JPEG/PNG) and returns detections + skin score.
    """
    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid frame"}, status_code=400)

    detections = yolo_model.predict(img)

    detections_norm = []
    for d in detections:
        x1, y1, x2, y2, score, cls = d
        if score < conf_thresh:
            continue
        detections_norm.append([int(x1), int(y1), int(x2), int(y2), float(score), 0])

    # --- compute skin score ---
    if detections_norm:
        weighted_sum = sum(d[4] for d in detections_norm)
        skin_score = max(1, min(100, int(100 - weighted_sum * 2)))
    else:
        skin_score = 100

    return {
        "detections": [
            {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3],
             "score": d[4], "class_id": d[5], "label": "Acne"}
            for d in detections_norm
        ],
        "skin_score": skin_score
    }


@app.post("/gradcam/")
async def gradcam(file: UploadFile = File(...), box_index: int = Form(...), conf_thresh: float = Form(0.25)):
    """
    Compute Grad-CAM overlay for the chosen detection index and return base64 of full image with overlay.
    """
    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    detections = yolo_model.predict(img)
    if len(detections) == 0:
        return JSONResponse({"error": "No detections"}, status_code=400)
    if box_index < 0 or box_index >= len(detections):
        return JSONResponse({"error": "box_index out of range"}, status_code=400)

    chosen = detections[box_index]
    x1, y1, x2, y2, score, cls = chosen
    box = [int(x1), int(y1), int(x2), int(y2)]

    out_rgb = gradcam_full_image_from_box(img, box, pad_ratio=0.05)  # RGB
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    out_b64 = image_to_base64(out_bgr)

    return {"gradcam_image": out_b64, "box_index": box_index}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
