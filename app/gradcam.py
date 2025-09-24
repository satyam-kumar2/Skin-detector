# app/gradcam.py
import torch
import numpy as np
import cv2
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_resnet = models.resnet50(pretrained=True).to(DEVICE)
_resnet.eval()

_TARGET_LAYER = _resnet.layer4[-1].conv3


_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _tensor_from_crop(crop_rgb: np.ndarray) -> torch.Tensor:
    """
    crop_rgb : HxWx3 uint8 in RGB
    returns tensor (1,3,224,224) on DEVICE
    """
    t = _preprocess(crop_rgb)
    t = t.unsqueeze(0).to(DEVICE)
    return t


def gradcam_on_crop(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM heatmap overlay for a crop using ResNet50.
    crop_rgb : HxWx3 uint8 (RGB)
    returns: overlay RGB uint8 (same size as crop)
    """
    input_tensor = _tensor_from_crop(crop_rgb)
    with torch.no_grad():
        logits = _resnet(input_tensor)
    pred_cls = int(logits.argmax(dim=1).cpu().item())

    cam = GradCAM(model=_resnet, target_layers=[_TARGET_LAYER], use_cuda=(DEVICE != "cpu"))
    targets = [ClassifierOutputTarget(pred_cls)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H,W) numpy

    
    grayscale_cam_resized = cv2.resize(grayscale_cam, (crop_rgb.shape[1], crop_rgb.shape[0]))

    rgb_float = crop_rgb.astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam_resized, use_rgb=True)  
    return overlay



def gradcam_full_image_from_box(full_img_bgr: np.ndarray, box: list, pad_ratio: float = 0.05) -> np.ndarray:
    """
    Compute CAM on the crop defined by box and paste overlay back into full image.
    full_img_bgr: HxWx3 (BGR)
    box: [x1, y1, x2, y2]
    pad_ratio: fraction padding around box
    returns: full image RGB np.uint8 with overlay region replaced
    """
    h, w = full_img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    # pad
    pad = int(max(x2-x1, y2-y1) * pad_ratio)
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop_bgr = full_img_bgr[y1p:y2p, x1p:x2p]
    if crop_bgr.size == 0:
        rgb_full = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB)
        return rgb_full

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = gradcam_on_crop(crop_rgb) 

    if overlay_rgb.shape[:2] != crop_rgb.shape[:2]:
        overlay_rgb = cv2.resize(overlay_rgb, (crop_rgb.shape[1], crop_rgb.shape[0]))

    out_full = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB).copy()
    out_full[y1p:y2p, x1p:x2p] = overlay_rgb

    return out_full
