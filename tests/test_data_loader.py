"""
ForestWatch AI — tests/test_data_loader.py
Pengujian fungsi pembacaan raster dan utilitas inventori.
"""

import numpy as np
import pytest
from pathlib import Path


def test_read_raster_basic(tmp_geotiff):
    """Membaca GeoTIFF sintetis dan memeriksa shape/metadata."""
    from src.data_loader import read_raster
    data, meta = read_raster(tmp_geotiff)

    assert data.ndim == 3, "Output harus 3D (bands, H, W)"
    assert data.shape[0] == 2, "Harus 2 band"
    assert data.shape[1] == 128
    assert data.shape[2] == 128
    assert data.dtype == np.float32
    assert meta["valid"] is True
    assert meta["count"] == 2
    assert meta["crs"] is not None


def test_read_raster_band_selection(tmp_geotiff):
    """Memilih subset band saat membaca raster."""
    from src.data_loader import read_raster
    data, meta = read_raster(tmp_geotiff, band_indices=[1])
    assert data.shape[0] == 1


def test_read_raster_invalid_band(tmp_geotiff):
    """Indeks band tidak valid harus memunculkan ValueError."""
    from src.data_loader import read_raster
    with pytest.raises(ValueError):
        read_raster(tmp_geotiff, band_indices=[99])


def test_read_raster_file_not_found():
    """File tidak ada harus memunculkan FileNotFoundError."""
    from src.data_loader import read_raster
    with pytest.raises(FileNotFoundError):
        read_raster("nonexistent_file.tif")


def test_get_raster_metadata(tmp_geotiff):
    """Metadata raster dapat diambil tanpa memuat piksel."""
    from src.data_loader import get_raster_metadata
    meta = get_raster_metadata(tmp_geotiff)
    assert meta["valid"] is True
    assert meta["width"] == 128
    assert meta["height"] == 128
    assert meta["count"] == 2


def test_scan_directory(tmp_path):
    """scan_directory menemukan file .tif."""
    from src.data_loader import scan_directory
    # Buat file .tif dummy
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    found = scan_directory(tmp_path, recursive=False)
    assert len(found) == 1
    assert found[0].suffix == ".tif"


def test_scan_directory_empty(tmp_path):
    """Direktori kosong mengembalikan list kosong."""
    from src.data_loader import scan_directory
    assert scan_directory(tmp_path) == []


def test_scan_directory_not_exist():
    """Direktori tidak ada mengembalikan list kosong."""
    from src.data_loader import scan_directory
    assert scan_directory("/path/does/not/exist") == []


def test_build_inventory(tmp_path):
    """Inventori dataset mengembalikan struktur yang benar."""
    from src.data_loader import build_dataset_inventory
    s1_dir = tmp_path / "s1"
    s2_dir = tmp_path / "s2"
    lb_dir = tmp_path / "labels"
    s1_dir.mkdir(); s2_dir.mkdir(); lb_dir.mkdir()
    (s1_dir / "scene1.tif").write_bytes(b"")
    (s2_dir / "scene1.tif").write_bytes(b"")

    inv = build_dataset_inventory(s1_dir, s2_dir, lb_dir)
    assert "s1_files" in inv
    assert "s2_files" in inv
    assert inv["summary"]["sentinel1_count"] == 1
    assert inv["summary"]["sentinel2_count"] == 1
    assert inv["has_data"] is True


def test_generate_synthetic_raster(tmp_path):
    """Raster sintetis dapat dibuat dan dibaca kembali."""
    from src.data_loader import generate_synthetic_raster, read_raster
    path = tmp_path / "synth.tif"
    generate_synthetic_raster(path, num_bands=4, height=64, width=64)
    data, meta = read_raster(path)
    assert data.shape == (4, 64, 64)
    assert meta["crs"] is not None


def test_load_label_csv(tmp_path):
    """CSV label dapat dimuat dan divalidasi."""
    import pandas as pd
    from src.data_loader import load_label_csv
    csv_path = tmp_path / "labels.csv"
    df = pd.DataFrame({
        "patch_id"  : ["patch_000001", "patch_000002"],
        "class_idx" : [0, 3],
        "class_name": ["Pertanian Skala Kecil", "Penebangan Selektif"],
    })
    df.to_csv(csv_path, index=False)
    loaded = load_label_csv(csv_path)
    assert loaded is not None
    assert len(loaded) == 2
