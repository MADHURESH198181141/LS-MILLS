"""
Enhancement Pipeline
Phase 1: Real-ESRGAN 2x (GPU) → CodeFormer (GPU, w=1.0, only_center_face)
Exports to ONNX INT8 for CPU inference
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
import logging
import os

log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# QUALITY CHECK
# ─────────────────────────────────────────

def laplacian_quality_score(image: np.ndarray) -> float:
    """Higher score = sharper image. < 100 = needs enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def needs_enhancement(image: np.ndarray, threshold: float = 100.0) -> bool:
    return laplacian_quality_score(image) < threshold


# ─────────────────────────────────────────
# REAL-ESRGAN 2x WRAPPER
# ─────────────────────────────────────────

class ESRGANEnhancer:
    """Real-ESRGAN 2x GPU enhancer."""

    def __init__(self, device: str = "cuda", scale: int = 2):
        self.device = device
        self.scale = scale
        self.model = None
        self._load()

    def _load(self):
        try:
            # Compatibility patch for newer torchvision
            import torchvision.transforms.functional as F_tv
            import torchvision.transforms.functional_tensor as F_t
        except ImportError:
            try:
                import torchvision.transforms.functional as F_tv
                import sys
                sys.modules["torchvision.transforms.functional_tensor"] = F_tv
            except Exception:
                pass

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            net = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=self.scale
            )
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
            weight_path = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")
            if not Path(weight_path).exists():
                log.warning(f"ESRGAN weights not found at {weight_path}. Downloading...")
                os.makedirs(weights_dir, exist_ok=True)
                import urllib.request
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                urllib.request.urlretrieve(url, weight_path)

            self.model = RealESRGANer(
                scale=self.scale,
                model_path=weight_path,
                model=net,
                half=(self.device == "cuda"),
                device=self.device
            )
            log.info("Real-ESRGAN 2x loaded")
        except Exception as e:
            log.warning(f"Real-ESRGAN load failed: {e}. Will skip ESRGAN step.")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            return image
        try:
            out, _ = self.model.enhance(image, outscale=self.scale)
            return out
        except Exception as e:
            log.warning(f"ESRGAN enhance failed: {e}")
            return image


# ─────────────────────────────────────────
# CODEFORMER WRAPPER
# ─────────────────────────────────────────

class CodeFormerEnhancer:
    """CodeFormer face restoration — w=1.0, only_center_face=True."""

    def __init__(self, device: str = "cuda", fidelity_weight: float = 1.0):
        self.device = device
        self.w = fidelity_weight
        self.model = None
        self.face_helper = None
        self._load()

    def _load(self):
        try:
            import sys
            sys.path.insert(0, "CodeFormer")
            from basicsr.utils.download_util import load_file_from_url
            from facelib.utils.face_restoration_helper import FaceRestoreHelper

            net = torch.hub.load(
                "sczhou/CodeFormer",
                "codeformer",
                pretrained=True,
                source="github"
            ).to(self.device).eval()

            if self.device == "cuda":
                net = net.half()

            self.model = net
            self.FaceRestoreHelper = FaceRestoreHelper
            log.info("✅ CodeFormer loaded")
        except Exception as e:
            log.warning(f"CodeFormer load failed: {e}. Will skip CodeFormer step.")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            return image
        try:
            helper = self.FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                device=self.device
            )
            helper.read_image(image)
            helper.get_face_landmarks_5(
                only_center_face=True,   # ← only center face
                resize=640,
                eye_dist_threshold=5
            )
            helper.align_warp_face()

            if not helper.cropped_faces:
                return image

            for face in helper.cropped_faces:
                face_t = T.ToTensor()(face).unsqueeze(0).to(self.device)
                if self.device == "cuda":
                    face_t = face_t.half()

                with torch.no_grad():
                    out = self.model(face_t, w=self.w, adain=True)[0]

                restored = out.squeeze(0).float().cpu()
                restored = restored.numpy().transpose(1, 2, 0)
                restored = (restored * 255).clip(0, 255).astype(np.uint8)
                helper.add_restored_face(restored)

            helper.get_inverse_affine(None)
            result = helper.paste_faces_to_input_image()
            return result

        except Exception as e:
            log.warning(f"CodeFormer enhance failed: {e}")
            return image


# ─────────────────────────────────────────
# COMBINED ENHANCEMENT PIPELINE
# ─────────────────────────────────────────

class EnhancementPipeline:
    """
    Full Phase 1 pipeline:
    Quality Check → Real-ESRGAN 2x → CodeFormer (w=1.0, center_face)
    """

    def __init__(self, device: str = "cuda", quality_threshold: float = 100.0):
        self.device = device
        self.threshold = quality_threshold
        self.esrgan = ESRGANEnhancer(device=device, scale=2)
        self.codeformer = CodeFormerEnhancer(device=device, fidelity_weight=1.0)

    def enhance(self, image: np.ndarray, apply_codeformer: bool = True) -> np.ndarray:
        """
        Enhance image through full pipeline.
        apply_codeformer=False for /match endpoint (faster).
        """
        # Quality Check
        if not needs_enhancement(image, self.threshold):
            return image

        # Stage 1: Real-ESRGAN 2x
        image = self.esrgan.enhance(image)

        # Stage 2: CodeFormer (enroll only)
        if apply_codeformer:
            image = self.codeformer.enhance(image)

        return image

    def enhance_batch(self, images: list, apply_codeformer: bool = True) -> list:
        """Enhance a batch of images."""
        return [self.enhance(img, apply_codeformer) for img in images]