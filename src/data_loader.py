"""
ForestWatch AI — data_loader.py
Memuat citra Sentinel-1 dan Sentinel-2 dari file raster (GeoTIFF),
memvalidasi metadata, dan menyediakan utilitas inventori dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Rasterio diimpor dengan penanganan error agar modul tetap bisa
# di-import meskipun rasterio belum terinstal (berguna saat testing).
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio tidak tersedia — fungsi baca raster dinonaktifkan.")


# ──────────────────────────────────────────────────────────────
# Struktur metadata raster
# ──────────────────────────────────────────────────────────────

def _empty_meta() -> Dict:
    return {
        "path"       : None,
        "driver"     : None,
        "dtype"      : None,
        "nodata"     : None,
        "width"      : None,
        "height"     : None,
        "count"      : None,
        "crs"        : None,
        "transform"  : None,
        "bounds"     : None,
        "res"        : None,
        "valid"      : False,
        "error"      : None,
    }


# ──────────────────────────────────────────────────────────────
# Pembacaan raster tunggal
# ──────────────────────────────────────────────────────────────

def read_raster(
    path: str | Path,
    band_indices: Optional[List[int]] = None,
    dtype: str = "float32",
) -> Tuple[np.ndarray, Dict]:
    """
    Membaca file raster dan mengembalikan array NumPy serta metadata.

    Parameters
    ----------
    path : str | Path
        Jalur file GeoTIFF.
    band_indices : list[int] | None
        Indeks band berbasis-1 yang akan dimuat. None = semua band.
    dtype : str
        Tipe data output array (default float32).

    Returns
    -------
    data : np.ndarray, shape (bands, height, width)
    meta : dict
        Metadata raster lengkap.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan untuk membaca file raster.")

    path = Path(path)
    meta = _empty_meta()
    meta["path"] = str(path)

    if not path.exists():
        meta["error"] = f"File tidak ditemukan: {path}"
        logger.error(meta["error"])
        raise FileNotFoundError(meta["error"])

    try:
        with rasterio.open(path) as src:
            meta["driver"]    = src.driver
            meta["dtype"]     = src.dtypes[0]
            meta["nodata"]    = src.nodata
            meta["width"]     = src.width
            meta["height"]    = src.height
            meta["count"]     = src.count
            meta["crs"]       = str(src.crs) if src.crs else None
            meta["transform"] = src.transform
            meta["bounds"]    = src.bounds
            meta["res"]       = src.res
            meta["valid"]     = True

            if band_indices is None:
                band_indices = list(range(1, src.count + 1))

            # Validasi indeks band
            invalid = [b for b in band_indices if b < 1 or b > src.count]
            if invalid:
                raise ValueError(
                    f"Indeks band tidak valid {invalid}. "
                    f"File memiliki {src.count} band."
                )

            data = src.read(band_indices).astype(dtype)

        logger.debug(
            "Raster dimuat: %s | shape=%s | crs=%s",
            path.name, data.shape, meta["crs"]
        )
        return data, meta

    except Exception as exc:
        meta["error"] = str(exc)
        meta["valid"] = False
        logger.error("Gagal membaca raster %s: %s", path, exc)
        raise


def get_raster_metadata(path: str | Path) -> Dict:
    """Mengambil metadata tanpa memuat seluruh array piksel."""
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan.")

    path = Path(path)
    meta = _empty_meta()
    meta["path"] = str(path)

    if not path.exists():
        meta["error"] = f"File tidak ditemukan: {path}"
        return meta

    try:
        with rasterio.open(path) as src:
            meta["driver"]    = src.driver
            meta["dtype"]     = src.dtypes[0]
            meta["nodata"]    = src.nodata
            meta["width"]     = src.width
            meta["height"]    = src.height
            meta["count"]     = src.count
            meta["crs"]       = str(src.crs) if src.crs else None
            meta["transform"] = src.transform
            meta["bounds"]    = src.bounds
            meta["res"]       = src.res
            meta["valid"]     = True
    except Exception as exc:
        meta["error"] = str(exc)

    return meta


# ──────────────────────────────────────────────────────────────
# Inventori dataset
# ──────────────────────────────────────────────────────────────

RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".vrt"}


def scan_directory(directory: str | Path, recursive: bool = True) -> List[Path]:
    """Memindai direktori dan mengembalikan daftar file raster."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        files = [
            p for ext in RASTER_EXTENSIONS
            for p in directory.rglob(f"*{ext}")
        ]
    else:
        files = [
            p for ext in RASTER_EXTENSIONS
            for p in directory.glob(f"*{ext}")
        ]

    return sorted(set(files))


def build_dataset_inventory(
    sentinel1_dir: str | Path,
    sentinel2_dir: str | Path,
    labels_dir: str | Path,
) -> Dict:
    """
    Membangun inventori file dataset yang tersedia.

    Returns
    -------
    dict berisi:
        s1_files      : list path Sentinel-1
        s2_files      : list path Sentinel-2
        label_files   : list path label (GeoJSON/Shapefile/GeoTIFF)
        summary       : ringkasan jumlah file
        has_data      : bool — apakah minimal ada satu sumber data
    """
    s1_files    = scan_directory(sentinel1_dir)
    s2_files    = scan_directory(sentinel2_dir)

    labels_dir  = Path(labels_dir)
    label_files = []
    if labels_dir.exists():
        for ext in [".geojson", ".shp", ".tif", ".tiff", ".json", ".csv"]:
            label_files.extend(labels_dir.rglob(f"*{ext}"))
        label_files = sorted(set(label_files))

    summary = {
        "sentinel1_count" : len(s1_files),
        "sentinel2_count" : len(s2_files),
        "label_count"     : len(label_files),
    }

    has_data = any([s1_files, s2_files, label_files])

    return {
        "s1_files"    : [str(p) for p in s1_files],
        "s2_files"    : [str(p) for p in s2_files],
        "label_files" : [str(p) for p in label_files],
        "summary"     : summary,
        "has_data"    : has_data,
    }


def validate_raster_pair(
    s1_path: str | Path,
    s2_path: str | Path,
    tolerance_m: float = 30.0,
) -> Dict:
    """
    Memvalidasi kesesuaian spasial antara citra S1 dan S2.
    Memeriksa: CRS, resolusi, dan overlap bounding box.

    Returns
    -------
    dict berisi status validasi dan pesan peringatan.
    """
    result = {"valid": False, "warnings": [], "errors": []}

    if not RASTERIO_AVAILABLE:
        result["errors"].append("rasterio tidak tersedia.")
        return result

    try:
        meta_s1 = get_raster_metadata(s1_path)
        meta_s2 = get_raster_metadata(s2_path)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    if not meta_s1["valid"]:
        result["errors"].append(f"S1 tidak valid: {meta_s1['error']}")
    if not meta_s2["valid"]:
        result["errors"].append(f"S2 tidak valid: {meta_s2['error']}")
    if result["errors"]:
        return result

    # CRS
    if meta_s1["crs"] != meta_s2["crs"]:
        result["warnings"].append(
            f"CRS berbeda: S1={meta_s1['crs']}, S2={meta_s2['crs']}. "
            "Reproyeksi diperlukan."
        )

    # Resolusi
    res_diff = max(
        abs(meta_s1["res"][0] - meta_s2["res"][0]),
        abs(meta_s1["res"][1] - meta_s2["res"][1]),
    )
    if res_diff > tolerance_m:
        result["warnings"].append(
            f"Perbedaan resolusi {res_diff:.1f} m melebihi toleransi "
            f"{tolerance_m} m. Resampling diperlukan."
        )

    # Bounding box overlap
    if meta_s1["bounds"] and meta_s2["bounds"]:
        b1 = meta_s1["bounds"]
        b2 = meta_s2["bounds"]
        overlap = (
            max(b1.left, b2.left) < min(b1.right, b2.right) and
            max(b1.bottom, b2.bottom) < min(b1.top, b2.top)
        )
        if not overlap:
            result["errors"].append(
                "Tidak ada overlap spasial antara S1 dan S2."
            )
            return result

    result["valid"] = len(result["errors"]) == 0
    return result


# ──────────────────────────────────────────────────────────────
# Generator citra sintetis (untuk demo & testing)
# ──────────────────────────────────────────────────────────────

def generate_synthetic_raster(
    path: str | Path,
    num_bands: int = 2,
    height: int = 256,
    width: int = 256,
    crs: str = "EPSG:32748",
    dtype: str = "float32",
    nodata: float = -9999.0,
    value_range: Tuple[float, float] = (-20.0, 0.0),
    seed: int = 42,
) -> Path:
    """
    Membuat file GeoTIFF sintetis untuk keperluan demonstrasi dan pengujian.
    Data tidak mewakili kondisi nyata deforestasi.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan untuk menulis GeoTIFF.")

    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = rng.uniform(
        value_range[0], value_range[1], (num_bands, height, width)
    ).astype(dtype)

    # Simulasi blok NoData kecil (5% piksel)
    nodata_mask = rng.random((height, width)) < 0.05
    for b in range(num_bands):
        data[b][nodata_mask] = nodata

    from rasterio.transform import from_bounds
    # Koordinat sekitar Bengkulu, Sumatera (west, south, east, north)
    transform = from_bounds(102.0, -4.0, 103.0, -3.0, width, height)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)

    logger.info("Raster sintetis dibuat: %s | shape=(%d,%d,%d)", path.name, num_bands, height, width)
    return path


def load_label_csv(path: str | Path) -> Optional[object]:
    """
    Memuat file CSV label patch deforestasi.
    Format yang diharapkan: kolom 'patch_id', 'class_name', 'class_idx'
    dan opsional koordinat geospasial.

    Returns
    -------
    pd.DataFrame | None
    """
    try:
        import pandas as pd
        path = Path(path)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        required = {"patch_id", "class_idx"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "Kolom wajib tidak ditemukan di CSV label: %s", missing
            )
        return df
    except Exception as exc:
        logger.error("Gagal memuat CSV label: %s", exc)
        return None


def save_inventory_json(inventory: Dict, path: str | Path) -> None:
    """Menyimpan inventori dataset ke file JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Inventori disimpan: %s", path)
