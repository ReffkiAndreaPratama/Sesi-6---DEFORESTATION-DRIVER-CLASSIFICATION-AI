"""
ForestWatch AI — patch_generator.py
Membuat patch (potongan citra) berukuran seragam dari raster besar.

Pipeline:
  1. Sliding window dengan overlap yang dapat dikonfigurasi.
  2. Validasi setiap patch (minimum piksel valid).
  3. Penghubungan patch dengan label (opsional).
  4. Penyimpanan patch ke file .npy atau penyediaan langsung ke memori.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dataclass patch
# ──────────────────────────────────────────────────────────────

@dataclass
class Patch:
    """Representasi satu patch citra satelit."""
    patch_id    : str
    data        : np.ndarray          # shape (C, patch_size, patch_size)
    row_start   : int
    col_start   : int
    patch_size  : int
    valid_ratio : float               # proporsi piksel valid
    label       : Optional[int] = None
    label_name  : Optional[str] = None
    source_path : Optional[str] = None
    geo_bounds  : Optional[Tuple] = None  # (left, bottom, right, top)
    metadata    : Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Fungsi inti pembuatan patch
# ──────────────────────────────────────────────────────────────

def generate_patches(
    data: np.ndarray,
    patch_size: int = 64,
    overlap: float = 0.25,
    nodata_value: float = 0.0,
    min_valid_ratio: float = 0.8,
    source_path: Optional[str] = None,
    transform=None,                   # rasterio.transform.Affine | None
) -> List[Patch]:
    """
    Membuat patch dari array raster dengan sliding window.

    Parameters
    ----------
    data : np.ndarray, shape (C, H, W)
    patch_size : int
        Ukuran patch (lebar = tinggi).
    overlap : float
        Fraksi overlap [0, 1). 0 = tidak ada overlap.
    nodata_value : float
        Nilai piksel yang dianggap tidak valid.
    min_valid_ratio : float
        Patch dibuang jika rasio piksel valid < nilai ini.
    transform : rasterio.Affine | None
        Transformasi spasial untuk menghitung geo_bounds.

    Returns
    -------
    list[Patch]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]   # tambah dimensi band

    C, H, W = data.shape

    if patch_size > H or patch_size > W:
        logger.warning(
            "patch_size (%d) lebih besar dari dimensi raster (%dx%d). "
            "Mengembalikan satu patch saja.", patch_size, H, W
        )
        patch_size = min(H, W)

    stride = max(1, int(patch_size * (1 - overlap)))
    patches: List[Patch] = []
    patch_idx = 0

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            crop = data[:, r:r + patch_size, c:c + patch_size]

            # Validasi piksel
            valid_mask = (
                (crop != nodata_value) &
                ~np.isnan(crop) &
                ~np.isinf(crop)
            )
            valid_ratio = float(np.mean(valid_mask))

            if valid_ratio < min_valid_ratio:
                continue

            # Hitung geo_bounds jika transform tersedia
            geo_bounds = None
            if transform is not None:
                try:
                    left, top = transform * (c, r)
                    right, bottom = transform * (c + patch_size, r + patch_size)
                    geo_bounds = (left, bottom, right, top)
                except Exception:
                    pass

            patch = Patch(
                patch_id    = f"patch_{patch_idx:06d}",
                data        = crop.astype(np.float32),
                row_start   = r,
                col_start   = c,
                patch_size  = patch_size,
                valid_ratio = valid_ratio,
                source_path = source_path,
                geo_bounds  = geo_bounds,
            )
            patches.append(patch)
            patch_idx += 1

    logger.debug(
        "Patch dibuat: %d (dari raster %dx%d, stride=%d)",
        len(patches), H, W, stride
    )
    return patches


def generate_patches_generator(
    data: np.ndarray,
    patch_size: int = 64,
    overlap: float = 0.25,
    nodata_value: float = 0.0,
    min_valid_ratio: float = 0.8,
) -> Generator[Patch, None, None]:
    """Versi generator untuk raster besar (hemat memori)."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    C, H, W = data.shape
    stride = max(1, int(patch_size * (1 - overlap)))
    patch_idx = 0

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            crop = data[:, r:r + patch_size, c:c + patch_size]
            valid_mask = (
                (crop != nodata_value) &
                ~np.isnan(crop) &
                ~np.isinf(crop)
            )
            valid_ratio = float(np.mean(valid_mask))
            if valid_ratio < min_valid_ratio:
                continue
            yield Patch(
                patch_id    = f"patch_{patch_idx:06d}",
                data        = crop.astype(np.float32),
                row_start   = r,
                col_start   = c,
                patch_size  = patch_size,
                valid_ratio = valid_ratio,
            )
            patch_idx += 1


# ──────────────────────────────────────────────────────────────
# Penghubungan patch dengan label
# ──────────────────────────────────────────────────────────────

def assign_labels_from_raster(
    patches: List[Patch],
    label_data: np.ndarray,
    class_map: Optional[Dict[int, str]] = None,
) -> List[Patch]:
    """
    Menghubungkan setiap patch dengan label dari raster label (mask).
    Label ditentukan berdasarkan nilai modus (majority vote) dalam patch.

    Parameters
    ----------
    label_data : np.ndarray, shape (H, W)
        Raster label dengan nilai integer kelas.
    class_map : dict | None
        Pemetaan integer → nama kelas.
    """
    H, W = label_data.shape
    labeled: List[Patch] = []

    for patch in patches:
        r = patch.row_start
        c = patch.col_start
        ps = patch.patch_size

        # Pastikan tidak keluar batas
        r_end = min(r + ps, H)
        c_end = min(c + ps, W)
        label_crop = label_data[r:r_end, c:c_end]

        # Buang piksel background (0 dianggap unlabeled/no-data)
        valid_labels = label_crop[label_crop > 0]
        if valid_labels.size == 0:
            continue

        # Modus
        values, counts = np.unique(valid_labels, return_counts=True)
        label_idx = int(values[np.argmax(counts)])
        patch.label = label_idx
        if class_map:
            patch.label_name = class_map.get(label_idx, f"Kelas {label_idx}")
        labeled.append(patch)

    logger.debug(
        "Patch berlabel: %d / %d", len(labeled), len(patches)
    )
    return labeled


def assign_labels_from_dataframe(
    patches: List[Patch],
    label_df,                         # pd.DataFrame
    id_col: str = "patch_id",
    label_col: str = "class_idx",
    name_col: Optional[str] = "class_name",
) -> List[Patch]:
    """
    Menghubungkan patch dengan label dari DataFrame CSV.
    """
    try:
        id_to_row = {row[id_col]: row for _, row in label_df.iterrows()}
    except Exception as exc:
        logger.error("Gagal memproses DataFrame label: %s", exc)
        return patches

    labeled: List[Patch] = []
    for patch in patches:
        row = id_to_row.get(patch.patch_id)
        if row is None:
            continue
        patch.label = int(row[label_col])
        if name_col and name_col in row:
            patch.label_name = str(row[name_col])
        labeled.append(patch)

    return labeled


# ──────────────────────────────────────────────────────────────
# Penyimpanan dan pemuatan patch
# ──────────────────────────────────────────────────────────────

def save_patches(
    patches: List[Patch],
    output_dir: str | Path,
    split: str = "train",
) -> List[Dict]:
    """
    Menyimpan setiap patch ke file .npy dan metadata ke list dict.

    Returns
    -------
    manifest : list[dict]  — metadata semua patch (untuk CSV)
    """
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for patch in patches:
        npy_path = output_dir / f"{patch.patch_id}.npy"
        np.save(npy_path, patch.data)
        manifest.append({
            "patch_id"   : patch.patch_id,
            "npy_path"   : str(npy_path),
            "split"      : split,
            "row_start"  : patch.row_start,
            "col_start"  : patch.col_start,
            "patch_size" : patch.patch_size,
            "valid_ratio": patch.valid_ratio,
            "class_idx"  : patch.label,
            "class_name" : patch.label_name,
            "source_path": patch.source_path,
        })

    logger.info("Tersimpan %d patch ke %s", len(patches), output_dir)
    return manifest


def load_patch(npy_path: str | Path) -> np.ndarray:
    """Memuat satu patch dari file .npy."""
    return np.load(npy_path).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Pembagian data latih / validasi / uji
# ──────────────────────────────────────────────────────────────

def split_patches(
    patches: List[Patch],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[Patch], List[Patch], List[Patch]]:
    """
    Membagi patch ke set latih, validasi, dan uji.

    Parameters
    ----------
    stratify : bool
        Jika True, pembagian mempertahankan distribusi kelas (per kelas).
    """
    rng = np.random.default_rng(seed)
    labeled = [p for p in patches if p.label is not None]

    if not labeled:
        logger.warning(
            "Tidak ada patch berlabel. Membagi berdasarkan urutan."
        )
        n = len(patches)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        idx    = list(range(n))
        rng.shuffle(idx)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return (
            [patches[i] for i in train_idx],
            [patches[i] for i in val_idx],
            [patches[i] for i in test_idx],
        )

    if stratify:
        train_list, val_list, test_list = [], [], []
        classes = sorted(set(p.label for p in labeled))
        for cls in classes:
            cls_patches = [p for p in labeled if p.label == cls]
            rng.shuffle(cls_patches)
            n = len(cls_patches)
            n_test = max(1, int(n * test_ratio))
            n_val  = max(1, int(n * val_ratio))
            test_list  += cls_patches[:n_test]
            val_list   += cls_patches[n_test:n_test + n_val]
            train_list += cls_patches[n_test + n_val:]
        return train_list, val_list, test_list
    else:
        n = len(labeled)
        idx = list(range(n))
        rng.shuffle(idx)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        return (
            [labeled[i] for i in idx[n_test + n_val:]],
            [labeled[i] for i in idx[n_test:n_test + n_val]],
            [labeled[i] for i in idx[:n_test]],
        )


# ──────────────────────────────────────────────────────────────
# Generator patch sintetis (untuk demo & testing)
# ──────────────────────────────────────────────────────────────

def generate_synthetic_patches(
    num_patches: int = 100,
    num_classes: int = 6,
    channels: int = 8,
    patch_size: int = 64,
    seed: int = 42,
) -> List[Patch]:
    """
    Membuat patch sintetis berlabel untuk demo dan pengujian unit.
    Data tidak mewakili kondisi deforestasi nyata.
    """
    rng = np.random.default_rng(seed)
    patches: List[Patch] = []

    for i in range(num_patches):
        cls = i % num_classes
        # Setiap kelas punya pola statistik berbeda untuk membedakan
        mean_val = 0.1 + cls * 0.1
        data = rng.normal(mean_val, 0.1, (channels, patch_size, patch_size))
        data = np.clip(data, 0.0, 1.0).astype(np.float32)

        patches.append(Patch(
            patch_id    = f"synth_{i:05d}",
            data        = data,
            row_start   = 0,
            col_start   = 0,
            patch_size  = patch_size,
            valid_ratio = 1.0,
            label       = cls,
            label_name  = f"Kelas_{cls}",
            source_path = "synthetic",
        ))

    logger.info("Patch sintetis dibuat: %d", num_patches)
    return patches
