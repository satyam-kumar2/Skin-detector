import cv2
from app.model_loader import yolo_model
from app.utils import draw_detections

# Load test image
img = cv2.imread("tests/sample.jpeg")
if img is None:
    raise FileNotFoundError("tests/sample.jpg not found or unreadable!")

# Run inference
detections = yolo_model.predict(img)
print(f"Detections: {detections}")

# Draw boxes
out_img = draw_detections(img, detections, class_names={0: "Acne"})

# Save result
cv2.imwrite("tests/output.jpg", out_img)
print("✅ Inference complete. Results saved to tests/output.jpg")
