"""
ForestWatch AI — dataset.py
PyTorch Dataset dan DataLoader untuk patch citra satelit.

Mendukung tiga mode:
  - 's1'    : hanya Sentinel-1 (kanal 0..N_S1-1)
  - 's2'    : hanya Sentinel-2 (kanal N_S1..N_S1+N_S2-1)
  - 'fused' : gabungan S1 + S2

Dataset menerima list Patch (dari patch_generator) atau
path ke manifest CSV + direktori .npy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.patch_generator import Patch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dataset utama
# ──────────────────────────────────────────────────────────────

class DeforestationDataset(Dataset):
    """
    PyTorch Dataset untuk klasifikasi penyebab deforestasi.

    Parameters
    ----------
    patches : list[Patch]
        Daftar patch yang sudah berlabel.
    mode : str
        's1', 's2', atau 'fused'.
    num_channels_s1 : int
        Jumlah kanal Sentinel-1 (untuk slicing).
    augment : bool
        Aktifkan augmentasi sederhana (flip + rotasi).
    """

    def __init__(
        self,
        patches: List[Patch],
        mode: str = "fused",
        num_channels_s1: int = 2,
        augment: bool = False,
    ):
        self.mode = mode
        self.num_channels_s1 = num_channels_s1
        self.augment = augment

        # Filter hanya patch berlabel
        self.patches = [p for p in patches if p.label is not None]
        if len(self.patches) < len(patches):
            dropped = len(patches) - len(self.patches)
            logger.warning("%d patch tanpa label dibuang.", dropped)

        if not self.patches:
            raise ValueError("Dataset kosong — tidak ada patch berlabel.")

        self._validate_mode()
        logger.info(
            "Dataset diinisialisasi: %d patch | mode=%s | augment=%s",
            len(self.patches), mode, augment
        )

    def _validate_mode(self):
        modes = {"s1", "s2", "fused"}
        if self.mode not in modes:
            raise ValueError(f"mode harus salah satu dari {modes}.")

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        """Memilih kanal sesuai mode."""
        C = data.shape[0]
        if self.mode == "s1":
            return data[:self.num_channels_s1]
        elif self.mode == "s2":
            return data[self.num_channels_s1:]
        else:  # fused
            return data

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Augmentasi sederhana: flip horizontal/vertikal dan rotasi 90°."""
        if np.random.rand() < 0.5:
            data = np.flip(data, axis=2).copy()   # flip horizontal
        if np.random.rand() < 0.5:
            data = np.flip(data, axis=1).copy()   # flip vertikal
        k = np.random.randint(0, 4)
        if k > 0:
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
        return data

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        patch = self.patches[idx]
        data = self._select_channels(patch.data)

        if self.augment:
            data = self._augment(data)

        tensor = torch.from_numpy(data).float()
        label  = int(patch.label)
        return tensor, label

    def get_class_distribution(self) -> Dict[int, int]:
        """Mengembalikan jumlah sampel per kelas."""
        dist: Dict[int, int] = {}
        for p in self.patches:
            dist[p.label] = dist.get(p.label, 0) + 1
        return dist

    def get_num_channels(self) -> int:
        if not self.patches:
            return 0
        sample = self._select_channels(self.patches[0].data)
        return sample.shape[0]


# ──────────────────────────────────────────────────────────────
# Dataset dari manifest CSV
# ──────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """
    Dataset yang memuat patch langsung dari file .npy
    berdasarkan manifest CSV.

    CSV harus memiliki kolom: 'npy_path', 'class_idx'
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        mode: str = "fused",
        num_channels_s1: int = 2,
        augment: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.mode = mode
        self.num_channels_s1 = num_channels_s1
        self.augment = augment

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest tidak ditemukan: {manifest_path}"
            )

        self.df = pd.read_csv(manifest_path)
        required = {"npy_path", "class_idx"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Kolom wajib tidak ada di manifest: {missing}")

        # Filter baris dengan label valid
        self.df = self.df.dropna(subset=["class_idx"])
        self.df["class_idx"] = self.df["class_idx"].astype(int)

        logger.info(
            "ManifestDataset: %d sampel | mode=%s",
            len(self.df), mode
        )

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        if self.mode == "s1":
            return data[:self.num_channels_s1]
        elif self.mode == "s2":
            return data[self.num_channels_s1:]
        return data

    def _augment(self, data: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            data = np.flip(data, axis=2).copy()
        if np.random.rand() < 0.5:
            data = np.flip(data, axis=1).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
        return data

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        data = np.load(row["npy_path"]).astype(np.float32)
        data = self._select_channels(data)
        if self.augment:
            data = self._augment(data)
        tensor = torch.from_numpy(data).float()
        label  = int(row["class_idx"])
        return tensor, label

    def get_class_distribution(self) -> Dict[int, int]:
        return dict(self.df["class_idx"].value_counts().items())


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    balance_classes: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Membuat DataLoader untuk latih, validasi, dan uji.

    Parameters
    ----------
    balance_classes : bool
        Jika True, gunakan WeightedRandomSampler untuk menangani
        ketidakseimbangan kelas pada DataLoader latih.
    """
    torch.manual_seed(seed)

    # Sampler untuk kelas tidak seimbang
    train_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    if balance_classes and hasattr(train_dataset, "patches"):
        dist = train_dataset.get_class_distribution()
        total = sum(dist.values())
        class_weights = {
            cls: total / (len(dist) * count)
            for cls, count in dist.items()
        }
        sample_weights = [
            class_weights.get(p.label, 1.0)
            for p in train_dataset.patches
        ]
        sampler = WeightedRandomSampler(
            weights     = torch.DoubleTensor(sample_weights),
            num_samples = len(sample_weights),
            replacement = True,
        )
        train_loader_kwargs["sampler"] = sampler
    else:
        train_loader_kwargs["shuffle"] = True

    loaders = {
        "train": DataLoader(train_dataset, **train_loader_kwargs),
        "val"  : DataLoader(
            val_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
        ),
    }

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
        )

    return loaders


# ──────────────────────────────────────────────────────────────
# Utilitas
# ──────────────────────────────────────────────────────────────

def collate_with_metadata(batch):
    """
    Custom collate function yang juga menyertakan indeks patch.
    Berguna untuk melacak prediksi ke patch aslinya.
    """
    tensors = torch.stack([item[0] for item in batch])
    labels  = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return tensors, labels


def save_manifest(manifest: List[Dict], path: Union[str, Path]) -> None:
    """Menyimpan manifest patch ke CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(manifest)
    df.to_csv(path, index=False)
    logger.info("Manifest disimpan: %s (%d baris)", path, len(df))


def load_manifest(path: Union[str, Path]) -> pd.DataFrame:
    """Memuat manifest CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {path}")
    return pd.read_csv(path)
