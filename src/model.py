"""
ForestWatch AI — model.py
Arsitektur CNN untuk klasifikasi penyebab deforestasi.

Tersedia:
  - ForestWatchCNN  : CNN kustom ringan (model utama)
  - ForestWatchResNet : transfer learning berbasis ResNet-18
  - build_model()   : factory function

Model menerima input (batch, channels, patch_size, patch_size)
dan menghasilkan logit (batch, num_classes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Blok konvolusi dasar
# ──────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Blok residual sederhana (2 konvolusi + skip connection)."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


# ──────────────────────────────────────────────────────────────
# ForestWatchCNN — Model Utama
# ──────────────────────────────────────────────────────────────

class ForestWatchCNN(nn.Module):
    """
    CNN kustom untuk klasifikasi penyebab deforestasi.

    Arsitektur:
      Stem  : Conv(3×3) → BN → ReLU
      Stage1: 2× ResidualBlock (64ch) + MaxPool
      Stage2: 2× ResidualBlock (128ch) + MaxPool
      Stage3: 2× ResidualBlock (256ch) + MaxPool
      Head  : GlobalAvgPool → Dropout → FC(num_classes)

    Parameter:
      in_channels  : jumlah kanal input (2=S1, 6=S2, 8=fused)
      num_classes  : jumlah kelas output
      base_filters : lebar jaringan (default 64)
      dropout      : dropout pada blok residual dan head
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 6,
        base_filters: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        f = base_filters

        # Stem
        self.stem = ConvBNReLU(in_channels, f, kernel=3)

        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(f, dropout=dropout),
            ResidualBlock(f, dropout=dropout),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(f, f * 2),
            ResidualBlock(f * 2, dropout=dropout),
            ResidualBlock(f * 2, dropout=dropout),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(f * 2, f * 4),
            ResidualBlock(f * 4, dropout=dropout),
            ResidualBlock(f * 4, dropout=dropout),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling + Head
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(f * 4, f * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(f * 2, num_classes),
        )

        self._init_weights()
        self.in_channels  = in_channels
        self.num_classes   = num_classes
        self.base_filters  = base_filters
        self.dropout_rate  = dropout

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mengekstrak fitur sebelum head klasifikasi."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return x.flatten(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan probabilitas per kelas (softmax)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# ForestWatchResNet — Transfer Learning
# ──────────────────────────────────────────────────────────────

class ForestWatchResNet(nn.Module):
    """
    Transfer learning berbasis ResNet-18 dengan modifikasi lapisan awal
    untuk mendukung jumlah kanal input yang bervariasi.

    Catatan: bobot ImageNet bawaan (3 kanal) disesuaikan dengan
    rata-rata kanal pertama untuk in_channels ≠ 3.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 6,
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:
            raise ImportError(
                "torchvision diperlukan untuk ForestWatchResNet. "
                "Gunakan ForestWatchCNN sebagai alternatif."
            )

        # Modifikasi lapisan pertama untuk multi-kanal
        orig_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained and in_channels != 3:
            # Rata-rata bobot ImageNet ke in_channels baru
            with torch.no_grad():
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight = nn.Parameter(
                    mean_weight.repeat(1, in_channels, 1, 1)
                )
        backbone.conv1 = new_conv

        # Ganti head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone     = backbone
        self.in_channels  = in_channels
        self.num_classes  = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────

def build_model(
    architecture: str = "cnn",
    in_channels: int = 8,
    num_classes: int = 6,
    **kwargs,
) -> nn.Module:
    """
    Factory function untuk membuat model.

    Parameters
    ----------
    architecture : str
        'cnn' → ForestWatchCNN
        'resnet' → ForestWatchResNet
    """
    arch = architecture.lower()
    if arch == "cnn":
        model = ForestWatchCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            **{k: v for k, v in kwargs.items()
               if k in ("base_filters", "dropout")},
        )
    elif arch == "resnet":
        model = ForestWatchResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            **{k: v for k, v in kwargs.items()
               if k in ("pretrained", "dropout")},
        )
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {architecture}. Pilih 'cnn' atau 'resnet'.")

    logger.info(
        "Model dibuat: %s | in_ch=%d | classes=%d | params=%s",
        architecture, in_channels, num_classes,
        f"{model.count_parameters():,}"
    )
    return model


# ──────────────────────────────────────────────────────────────
# Simpan / muat model
# ──────────────────────────────────────────────────────────────

def save_model(
    model: nn.Module,
    path: str | Path,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Menyimpan state_dict model beserta metadata konfigurasi.

    Format checkpoint:
      {
        "state_dict"  : model.state_dict(),
        "architecture": str,
        "in_channels" : int,
        "num_classes" : int,
        "metadata"    : dict,
      }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict"   : model.state_dict(),
        "architecture" : model.__class__.__name__,
        "in_channels"  : getattr(model, "in_channels", None),
        "num_classes"  : getattr(model, "num_classes", None),
        "base_filters" : getattr(model, "base_filters", 64),
        "dropout"      : getattr(model, "dropout_rate", 0.25),
        "metadata"     : metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info("Model disimpan: %s", path)


def load_model(
    path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Memuat model dari checkpoint.

    Returns
    -------
    model  : nn.Module (siap digunakan untuk inferensi)
    meta   : dict metadata checkpoint
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    arch         = checkpoint.get("architecture", "ForestWatchCNN")
    in_channels  = checkpoint.get("in_channels", 8)
    num_classes  = checkpoint.get("num_classes", 6)
    base_filters = checkpoint.get("base_filters", 64)
    dropout      = checkpoint.get("dropout", 0.25)
    metadata     = checkpoint.get("metadata", {})

    if "ResNet" in arch:
        model = ForestWatchResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        model = ForestWatchCNN(
            in_channels   = in_channels,
            num_classes   = num_classes,
            base_filters  = base_filters,
            dropout       = dropout,
        )

    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    model.to(device)
    model.eval()

    logger.info(
        "Model dimuat: %s | in_ch=%d | classes=%d | device=%s",
        arch, in_channels, num_classes, device
    )
    return model, metadata


def get_device(preference: str = "auto") -> torch.device:
    """
    Menentukan perangkat komputasi.

    Parameters
    ----------
    preference : str
        'auto' → CUDA jika tersedia, else CPU
        'cpu'  → paksa CPU
        'cuda' → paksa CUDA (error jika tidak tersedia)
    """
    if preference == "cpu":
        return torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak tersedia di sistem ini.")
        return torch.device("cuda")
    else:  # auto
        if torch.cuda.is_available():
            logger.info("CUDA tersedia — menggunakan GPU.")
            return torch.device("cuda")
        logger.info("CUDA tidak tersedia — menggunakan CPU.")
        return torch.device("cpu")
