import cv2
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression to filter overlapping boxes."""
    idxs = np.argsort(scores)[::-1]  # sort by confidence descending
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break

        ious = compute_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]

    return keep


def compute_iou(box, boxes):
    """Compute IoU between a single box and multiple boxes."""
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h

    box_area = (x2 - x1) * (y2 - y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)


def preprocess_image(image: np.ndarray, img_size: int = 640):
    """Resize, normalize, and prepare input tensor for YOLO ONNX."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))  
    img_resized = np.expand_dims(img_resized, axis=0)  
    return img_resized


def postprocess(preds: np.ndarray, orig_shape, conf_threshold=0.5, iou_threshold=0.45, max_detections=50):
    h, w = orig_shape[:2]
    detections = []

    preds = preds[0]  
    preds = preds.transpose(1, 0)  

    boxes, scores, class_ids = [], [], []
    for pred in preds:
        cx, cy, bw, bh, *class_logits = pred

        class_probs = 1 / (1 + np.exp(-np.array(class_logits)))
        cls_id = int(np.argmax(class_probs))
        cls_conf = float(class_probs[cls_id])

        if cls_conf < conf_threshold:
            continue

        # xywh → xyxy
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)

        boxes.append([x1, y1, x2, y2])
        scores.append(cls_conf)
        class_ids.append(cls_id)

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = nms(boxes, scores, iou_threshold)

    keep = sorted(keep, key=lambda i: scores[i], reverse=True)[:max_detections]

    detections = [
        [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]),
         float(scores[i]), int(class_ids[i])]
        for i in keep
    ]

    print(f"[DEBUG] Final detections: {len(detections)}")
    return detections
