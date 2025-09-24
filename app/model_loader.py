import os
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from app.onnx_infer import preprocess_image, postprocess


load_dotenv()

MODEL_ONNX_PATH = os.getenv("MODEL_ONNX_PATH", "exported_models/model.onnx")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
IMG_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 640))


class YOLOOnnxModel:
    def __init__(self, model_path: str = MODEL_ONNX_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        print(f"[INFO] Loading ONNX model from {model_path} ...")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[INFO] Model loaded. Input: {self.input_name}, Output: {self.output_name}")

    def predict(self, image: np.ndarray):
        input_tensor = preprocess_image(image, IMG_SIZE)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        preds = np.array(outputs[0])
        
        
        results = postprocess(preds, image.shape,
                            conf_threshold=CONF_THRESHOLD,
                            iou_threshold=IOU_THRESHOLD)
        return results


yolo_model = YOLOOnnxModel(MODEL_ONNX_PATH)
