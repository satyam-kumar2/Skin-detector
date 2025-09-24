import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def draw_detections(image, detections, class_names=None, color=(0, 255, 0)):
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)

        # fallback if class id not in dictionary
        label_name = class_names.get(cls_id, "Acne") if class_names else str(cls_id)
        label = f"{label_name} {score:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output




def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV BGR image to base64 string.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV BGR image.
    """
    img_bytes = base64.b64decode(base64_str)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)



def save_image(image: np.ndarray, path: str):
    """
    Save OpenCV BGR image to disk.
    """
    cv2.imwrite(path, image)
