"""
ForestWatch AI — train.py
Loop pelatihan model CNN untuk klasifikasi penyebab deforestasi.

Fitur:
  - Training loop dengan validation setiap epoch
  - Early stopping berdasarkan validation loss
  - LR scheduler (StepLR)
  - Penyimpanan model terbaik
  - Logging metrik per epoch ke CSV
  - Callback progress (untuk integrasi Streamlit)
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Objek hasil pelatihan
# ──────────────────────────────────────────────────────────────

class TrainingHistory:
    """Menyimpan riwayat metrik pelatihan per epoch."""

    def __init__(self):
        self.train_loss : List[float] = []
        self.val_loss   : List[float] = []
        self.train_acc  : List[float] = []
        self.val_acc    : List[float] = []
        self.lr_history : List[float] = []
        self.best_epoch : int = 0
        self.best_val_loss : float = float("inf")
        self.total_epochs  : int = 0
        self.training_time_s : float = 0.0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.lr_history.append(lr)
        self.total_epochs = epoch + 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = epoch

    def to_dict(self) -> Dict:
        return {
            "train_loss"      : self.train_loss,
            "val_loss"        : self.val_loss,
            "train_acc"       : self.train_acc,
            "val_acc"         : self.val_acc,
            "lr_history"      : self.lr_history,
            "best_epoch"      : self.best_epoch,
            "best_val_loss"   : self.best_val_loss,
            "total_epochs"    : self.total_epochs,
            "training_time_s" : self.training_time_s,
        }

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
            )
            for i in range(self.total_epochs):
                writer.writerow([
                    i + 1,
                    f"{self.train_loss[i]:.6f}",
                    f"{self.val_loss[i]:.6f}",
                    f"{self.train_acc[i]:.4f}",
                    f"{self.val_acc[i]:.4f}",
                    f"{self.lr_history[i]:.8f}",
                ])
        logger.info("Riwayat pelatihan disimpan: %s", path)


# ──────────────────────────────────────────────────────────────
# Satu epoch train / validate
# ──────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool = True,
) -> Tuple[float, float]:
    """
    Menjalankan satu epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float (0–1)
    """
    model.train(is_train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping untuk stabilitas
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────
# Fungsi pelatihan utama
# ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_step_size: int = 15,
    lr_gamma: float = 0.5,
    patience: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
    history_csv_path: Optional[str | Path] = None,
    class_weights: Optional[torch.Tensor] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Melatih model dengan early stopping dan menyimpan model terbaik.

    Parameters
    ----------
    progress_callback : callable | None
        Dipanggil setiap epoch dengan dict berisi metrik terkini.
        Berguna untuk memperbarui UI Streamlit secara real-time.

    Returns
    -------
    model   : nn.Module (model terbaik sudah di-load kembali)
    history : TrainingHistory
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Loss function (dengan class weights jika tersedia)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    history         = TrainingHistory()
    best_val_loss   = float("inf")
    epochs_no_improv = 0
    best_state_dict = None
    start_time      = time.time()

    logger.info(
        "Pelatihan dimulai: %d epoch | device=%s | lr=%.5f",
        num_epochs, device, learning_rate
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)

        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | "
            "lr=%.6f | %.1fs",
            epoch + 1, num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, epoch_time,
        )

        # Callback untuk UI
        if progress_callback is not None:
            progress_callback({
                "epoch"      : epoch + 1,
                "num_epochs" : num_epochs,
                "train_loss" : train_loss,
                "val_loss"   : val_loss,
                "train_acc"  : train_acc,
                "val_acc"    : val_acc,
                "lr"         : current_lr,
            })

        # Simpan model terbaik
        if val_loss < best_val_loss:
            best_val_loss      = val_loss
            best_state_dict    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improv   = 0

            if save_path is not None:
                from src.model import save_model
                save_model(
                    model,
                    save_path,
                    metadata={
                        "best_epoch"     : epoch + 1,
                        "best_val_loss"  : best_val_loss,
                        "best_val_acc"   : val_acc,
                        "num_epochs_run" : epoch + 1,
                    },
                )
        else:
            epochs_no_improv += 1

        # Early stopping
        if epochs_no_improv >= patience:
            logger.info(
                "Early stopping setelah %d epoch tanpa peningkatan.", patience
            )
            break

    history.training_time_s = time.time() - start_time

    # Muat kembali state terbaik
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            "Model terbaik dimuat (epoch %d, val_loss=%.4f).",
            history.best_epoch + 1, history.best_val_loss
        )

    model.eval()

    # Simpan riwayat CSV
    if history_csv_path is not None:
        history.save_csv(history_csv_path)

    logger.info(
        "Pelatihan selesai: %d epoch | %.1f detik",
        history.total_epochs, history.training_time_s
    )
    return model, history


# ──────────────────────────────────────────────────────────────
# Hitung class weights untuk dataset tidak seimbang
# ──────────────────────────────────────────────────────────────

def compute_class_weights(
    class_distribution: Dict[int, int],
    num_classes: int,
) -> torch.Tensor:
    """
    Menghitung bobot kelas invers-frekuensi untuk CrossEntropyLoss.

    Returns
    -------
    weights : torch.Tensor, shape (num_classes,)
    """
    total = sum(class_distribution.values())
    weights = []
    for cls in range(num_classes):
        count = class_distribution.get(cls, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)
