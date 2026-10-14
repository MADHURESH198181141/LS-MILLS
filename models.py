"""
Face Recognition Models
- IResNet50 backbone with ArcFace loss (GPU training)
- SCRFD 10g face detection
- GlintR100 / Antelopev2 via InsightFace
- ONNX INT8 export for CPU inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import os

log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# IRESNET BACKBONE
# ─────────────────────────────────────────

class IBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.PReLU(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class IResNet50(nn.Module):
    """
    IResNet-50 backbone for face recognition.
    Input:  (B, 3, 112, 112)
    Output: (B, 512) L2-normalized embedding
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4, img_size: int = 112):
        super().__init__()
        layers = [3, 4, 14, 3]  # ResNet-50 config

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64,  layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * IBottleneck.expansion)
        self.dropout = nn.Dropout(dropout)

        # Dynamically compute flatten size
        # img_size=112 → after conv(stride1) + 4x stride2 layers → 112/16 = 7 → 7x7
        feat_map = img_size // 16
        flat_size = 512 * IBottleneck.expansion * feat_map * feat_map
        self.fc = nn.Linear(flat_size, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(IBottleneck(self.in_planes, planes, s))
            self.in_planes = planes * IBottleneck.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


# ─────────────────────────────────────────
# ARCFACE LOSS
# ─────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Face Recognition.
    Paper: https://arxiv.org/abs/1801.07698
    s=64, m=0.5 recommended for face recognition.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 1000,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Cosine similarity
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # phi = cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return self.criterion(output, labels)


# ─────────────────────────────────────────
# INSIGHTFACE WRAPPER (SCRFD + GlintR100)
# ─────────────────────────────────────────

class InsightFaceRecognizer:
    """
    InsightFace Antelopev2 wrapper.
    Uses SCRFD 10g for detection + GlintR100 for recognition.
    CPU inference via ONNX.
    """

    def __init__(self, model_name: str = "antelopev2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.app = None
        self._load()

    def _load(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )

            self.app = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            self.app.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=(640, 640)
            )
            log.info(f"✅ InsightFace {self.model_name} loaded on {self.device}")
        except Exception as e:
            log.error(f"InsightFace load failed: {e}")
            raise

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim GlintR100 embedding.
        Returns None if no face detected.
        """
        faces = self.app.get(image)
        if not faces:
            return None

        # Use the largest detected face
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face.embedding  # 512-dim float32

    def get_all_faces(self, image: np.ndarray):
        """Return all detected faces with embeddings."""
        return self.app.get(image)


# ─────────────────────────────────────────
# ONNX EXPORT + QUANTIZE
# ─────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    save_path: str,
    img_size: int = 112,
    device: str = "cuda"
):
    """Export trained IResNet50 to ONNX."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    torch.onnx.export(
        model, dummy, save_path,
        opset_version=11,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        }
    )
    log.info(f"✅ ONNX model saved → {save_path}")


def quantize_to_int8(onnx_path: str, int8_path: str):
    """Quantize ONNX model to INT8 for fast CPU inference."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        onnx_path,
        int8_path,
        weight_type=QuantType.QInt8
    )
    log.info(f"✅ INT8 model saved → {int8_path}")


# ─────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────

def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(e1, e2))