"""OSNet (Omni-Scale Network) for person re-identification.

The architecture below is a compact port of ``osnet_x1_0`` from
deep-person-reid / torchreid (K. Zhou et al., "Omni-Scale Feature Learning for
Person Re-Identification", ICCV 2019, MIT licence). Module and parameter names
match torchreid exactly so the official checkpoints load with strict=True.

Default weights: osnet_x1_0 trained on MSMT17 (combineall), published by the
OSNet author at https://huggingface.co/kaiyangzhou/osnet . They are downloaded
on first use to ``UC_OSNET_WEIGHTS`` (default ``weights/osnet_x1_0_msmt17.pth``).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from models.insightface_model import ModelInitializationError
from services.config import UniqueCountConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------------


class ConvLayer(nn.Module):
    """Convolution + BN (or IN) + ReLU."""

    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1, padding: int = 0, groups: int = 1, IN: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_c, affine=True) if IN else nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + BN, no activation."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3x3: 1x1 (linear) followed by depth-wise 3x3."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Unified aggregation gate that fuses multi-scale streams."""

    def __init__(self, in_c: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, in_c, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Omni-scale residual block with four receptive-field streams."""

    def __init__(self, in_c: int, out_c: int, IN: bool = False, bottleneck_reduction: int = 4):
        super().__init__()
        mid = out_c // bottleneck_reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None
        self.IN = nn.InstanceNorm2d(out_c, affine=True) if IN else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        out = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        if self.IN is not None:
            out = self.IN(out)
        return self.relu(out)


class OSNet(nn.Module):
    def __init__(self, layers: list[int], channels: list[int], feature_dim: int = 512, num_classes: int = 1000):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.feature_dim = feature_dim
        # Identity classifier of the training set; unused at inference.
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _make_layer(n: int, in_c: int, out_c: int, reduce_spatial_size: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_c, out_c)]
        layers += [OSBlock(out_c, out_c) for _ in range(1, n)]
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        v = self.global_avgpool(x).flatten(1)
        return self.fc(v)


def osnet_x1_0() -> OSNet:
    return OSNet(layers=[2, 2, 2], channels=[64, 256, 384, 512])


# ----------------------------------------------------------------------------
# Loader / inference wrapper
# ----------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_HW = (256, 128)  # torchreid default for person crops


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    # weights_only=True: never unpickle arbitrary objects from a checkpoint.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        k = k[7:] if k.startswith("module.") else k
        if k.startswith("classifier."):
            continue  # training-set specific head, size differs per dataset
        clean[k] = v
    return clean


class OSNetModel:
    """Produces L2-normalised 512-d body embeddings (lifecycle owned by ModelManager)."""

    def __init__(self, cfg: UniqueCountConfig):
        self.device = torch.device(f"cuda:{cfg.gpu_device_id}" if torch.cuda.is_available() else "cpu")
        if cfg.require_gpu and self.device.type != "cuda":
            raise ModelInitializationError("UC_REQUIRE_GPU is set but PyTorch has no CUDA device for OSNet")
        path = Path(cfg.osnet_weights)
        try:
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading OSNet weights to %s", path)
                tmp = path.with_suffix(path.suffix + ".part")
                torch.hub.download_url_to_file(cfg.osnet_weights_url, str(tmp), progress=False)
                tmp.replace(path)

            model = osnet_x1_0()
            state = _load_state_dict(path)
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing = [k for k in missing if not k.startswith("classifier.")]
            if missing or unexpected:
                raise ModelInitializationError(
                    f"OSNet checkpoint does not match osnet_x1_0 (missing={missing[:5]}, unexpected={unexpected[:5]})"
                )
        except ModelInitializationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelInitializationError(f"OSNet initialisation failed: {exc}") from exc

        self.model = model.eval().to(self.device)
        self.half = self.device.type == "cuda" and cfg.osnet_fp16
        if self.half:
            self.model.half()
        # One forward pass at a time on this module (BN buffers / CUDA stream).
        self._infer_lock = threading.Lock()
        logger.info("OSNet x1_0 ready on %s (fp16=%s, weights=%s)", self.device, self.half, path)

    def warmup(self) -> None:
        self.embed([np.zeros((256, 128, 3), np.uint8)] * 4)

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_bgr, (_INPUT_HW[1], _INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((img - _MEAN) / _STD).transpose(2, 0, 1)

    @torch.inference_mode()
    def embed(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Embed BGR person crops -> (N, 512) float32, L2-normalised."""
        if not crops:
            return np.zeros((0, self.model.feature_dim), dtype=np.float32)
        out = []
        with self._infer_lock:
            for i in range(0, len(crops), batch_size):
                batch = np.stack([self._preprocess(c) for c in crops[i:i + batch_size]])
                t = torch.from_numpy(batch).to(self.device, non_blocking=True)
                if self.half:
                    t = t.half()
                f = self.model(t).float()
                out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
