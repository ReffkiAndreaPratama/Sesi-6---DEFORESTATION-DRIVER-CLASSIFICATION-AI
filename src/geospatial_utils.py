"""
ForestWatch AI — geospatial_utils.py
Utilitas geospasial untuk manipulasi raster, reproyeksi,
pembuatan GeoDataFrame prediksi, dan ekspor ke format vektor/raster.

Semua fungsi menggunakan koordinat yang berasal dari metadata
atau data pengguna — tidak ada koordinat fiktif yang dibuat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds, from_origin
    from rasterio.warp import (
        calculate_default_transform,
        reproject as rio_reproject,
    )
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import box, mapping
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Informasi spasial raster
# ──────────────────────────────────────────────────────────────

def get_spatial_info(path: str | Path) -> Dict:
    """
    Mengambil informasi spasial lengkap dari file raster.

    Returns
    -------
    dict:
        crs, crs_epsg, bounds, center_lon, center_lat,
        res_x, res_y, width, height, num_bands, is_geographic
    """
    if not RASTERIO_AVAILABLE:
        return {"error": "rasterio tidak tersedia."}

    path = Path(path)
    if not path.exists():
        return {"error": f"File tidak ditemukan: {path}"}

    try:
        with rasterio.open(path) as src:
            crs    = src.crs
            bounds = src.bounds
            res    = src.res
            info   = {
                "crs"          : str(crs) if crs else None,
                "crs_epsg"     : crs.to_epsg() if crs else None,
                "bounds"       : {
                    "left"   : bounds.left,
                    "bottom" : bounds.bottom,
                    "right"  : bounds.right,
                    "top"    : bounds.top,
                },
                "center_lon"   : (bounds.left + bounds.right) / 2,
                "center_lat"   : (bounds.bottom + bounds.top) / 2,
                "res_x"        : res[0],
                "res_y"        : res[1],
                "width"        : src.width,
                "height"       : src.height,
                "num_bands"    : src.count,
                "is_geographic": crs.is_geographic if crs else None,
                "nodata"       : src.nodata,
                "dtype"        : src.dtypes[0],
                "transform"    : src.transform,
            }
        return info
    except Exception as exc:
        return {"error": str(exc)}


def check_crs_compatibility(path1: str | Path, path2: str | Path) -> Dict:
    """Memeriksa apakah dua raster memiliki CRS yang kompatibel."""
    info1 = get_spatial_info(path1)
    info2 = get_spatial_info(path2)

    if "error" in info1 or "error" in info2:
        return {
            "compatible": False,
            "error": info1.get("error") or info2.get("error"),
        }

    same_crs = info1["crs"] == info2["crs"]
    return {
        "compatible"   : same_crs,
        "crs1"         : info1["crs"],
        "crs2"         : info2["crs"],
        "needs_reproject": not same_crs,
    }


# ──────────────────────────────────────────────────────────────
# Reproyeksi raster
# ──────────────────────────────────────────────────────────────

def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    target_crs: str = "EPSG:4326",
    resampling: str = "bilinear",
) -> Path:
    """
    Memproyeksikan ulang raster ke CRS target.

    Parameters
    ----------
    target_crs : str  — kode EPSG, contoh 'EPSG:4326' atau 'EPSG:32748'
    resampling : str  — 'bilinear', 'nearest', 'cubic'
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan untuk reproyeksi.")

    resamp_map = {
        "bilinear": Resampling.bilinear,
        "nearest" : Resampling.nearest,
        "cubic"   : Resampling.cubic,
    }
    resamp = resamp_map.get(resampling, Resampling.bilinear)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update({
            "crs"       : target_crs,
            "transform" : transform,
            "width"     : width,
            "height"    : height,
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                rio_reproject(
                    source      = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = target_crs,
                    resampling    = resamp,
                )

    logger.info("Reproyeksi selesai: %s → %s", Path(src_path).name, dst_path.name)
    return dst_path


# ──────────────────────────────────────────────────────────────
# Crop raster ke bounding box
# ──────────────────────────────────────────────────────────────

def crop_raster_to_bbox(
    src_path: str | Path,
    dst_path: str | Path,
    bbox: Tuple[float, float, float, float],
) -> Path:
    """
    Memotong raster sesuai bounding box (left, bottom, right, top).
    Koordinat bbox harus dalam CRS yang sama dengan raster.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan.")

    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    geom = [mapping(shapely_box(*bbox))]

    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, geom, crop=True)
        out_profile = src.profile.copy()
        out_profile.update({
            "height"    : out_image.shape[1],
            "width"     : out_image.shape[2],
            "transform" : out_transform,
        })
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(out_image)

    logger.info("Crop selesai: %s", dst_path.name)
    return dst_path


# ──────────────────────────────────────────────────────────────
# Prediksi → GeoDataFrame
# ──────────────────────────────────────────────────────────────

def predictions_to_geodataframe(
    patch_results: List[Dict],
    patches,
    crs: str = "EPSG:4326",
    class_names: Optional[List[str]] = None,
    class_colors: Optional[Dict[str, str]] = None,
) -> Optional[object]:
    """
    Mengubah hasil prediksi patch menjadi GeoDataFrame untuk pemetaan.

    Hanya patch dengan geo_bounds yang valid yang dimasukkan.

    Returns
    -------
    geopandas.GeoDataFrame | None
    """
    if not GEOPANDAS_AVAILABLE:
        logger.warning("geopandas tidak tersedia — GeoDataFrame tidak dibuat.")
        return None

    rows = []
    for patch, res in zip(patches, patch_results):
        if patch.geo_bounds is None:
            continue
        left, bottom, right, top = patch.geo_bounds
        geom = box(left, bottom, right, top)

        cls_idx  = res.get("class_idx", -1)
        cls_name = res.get("class_name") or (
            class_names[cls_idx]
            if class_names and 0 <= cls_idx < len(class_names)
            else str(cls_idx)
        )
        color = (
            class_colors.get(cls_name, "#808080")
            if class_colors else "#808080"
        )

        rows.append({
            "geometry"   : geom,
            "patch_id"   : patch.patch_id,
            "class_idx"  : cls_idx,
            "class_name" : cls_name,
            "confidence" : round(res.get("confidence", 0.0), 4),
            "color"      : color,
        })

    if not rows:
        logger.warning("Tidak ada patch dengan koordinat geospasial valid.")
        return None

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    logger.info("GeoDataFrame dibuat: %d fitur | CRS=%s", len(gdf), crs)
    return gdf


def save_geodataframe(
    gdf,
    path: str | Path,
    driver: str = "GeoJSON",
) -> Path:
    """
    Menyimpan GeoDataFrame ke file.

    Parameters
    ----------
    driver : str
        'GeoJSON', 'GPKG', 'ESRI Shapefile'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver=driver)
    logger.info("GeoDataFrame disimpan: %s (%d fitur)", path, len(gdf))
    return path


# ──────────────────────────────────────────────────────────────
# Peta prediksi → GeoTIFF
# ──────────────────────────────────────────────────────────────

def prediction_map_to_geotiff(
    prediction_map: np.ndarray,
    output_path: str | Path,
    transform,
    crs: str = "EPSG:4326",
    nodata: int = -1,
) -> Path:
    """
    Menyimpan peta prediksi integer ke GeoTIFF dengan georeferensi.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path, "w",
        driver    = "GTiff",
        height    = prediction_map.shape[0],
        width     = prediction_map.shape[1],
        count     = 1,
        dtype     = "int16",
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = "lzw",
    ) as dst:
        dst.write(prediction_map.astype("int16"), 1)

    logger.info("GeoTIFF prediksi disimpan: %s", output_path)
    return output_path


# ──────────────────────────────────────────────────────────────
# Komputasi statistik zonal
# ──────────────────────────────────────────────────────────────

def compute_class_area(
    prediction_map: np.ndarray,
    transform,
    class_names: Optional[List[str]] = None,
    nodata: int = -1,
) -> Dict:
    """
    Menghitung luas area per kelas dari peta prediksi.

    Returns
    -------
    dict:
        per_class: {class_name: {pixel_count, area_m2, area_ha}}
        total_classified_pixels: int
    """
    if not RASTERIO_AVAILABLE or transform is None:
        # Hitung hanya dalam piksel jika transform tidak tersedia
        unique, counts = np.unique(prediction_map, return_counts=True)
        result = {}
        for u, c in zip(unique, counts):
            if u == nodata:
                continue
            name = class_names[u] if class_names and 0 <= u < len(class_names) else str(u)
            result[name] = {"pixel_count": int(c)}
        return {"per_class": result, "unit": "pixels"}

    # Luas piksel dalam meter persegi
    # transform.a = ukuran piksel sumbu X, transform.e = sumbu Y (negatif)
    pixel_area_m2 = abs(transform.a * transform.e)

    unique, counts = np.unique(prediction_map, return_counts=True)
    per_class = {}
    total_pixels = 0

    for u, c in zip(unique, counts):
        if u == nodata:
            continue
        name = class_names[int(u)] if class_names and 0 <= int(u) < len(class_names) else str(u)
        area_m2 = float(c) * pixel_area_m2
        per_class[name] = {
            "pixel_count" : int(c),
            "area_m2"     : round(area_m2, 2),
            "area_ha"     : round(area_m2 / 10000, 4),
        }
        total_pixels += int(c)

    return {
        "per_class"               : per_class,
        "total_classified_pixels" : total_pixels,
        "pixel_area_m2"           : pixel_area_m2,
        "unit"                    : "meters",
    }


# ──────────────────────────────────────────────────────────────
# Utilitas koordinat
# ──────────────────────────────────────────────────────────────

def pixel_to_coord(row: int, col: int, transform) -> Tuple[float, float]:
    """Mengubah indeks piksel (row, col) ke koordinat (lon, lat)."""
    lon, lat = transform * (col, row)
    return float(lon), float(lat)


def coord_to_pixel(lon: float, lat: float, transform) -> Tuple[int, int]:
    """Mengubah koordinat (lon, lat) ke indeks piksel (row, col)."""
    inv = ~transform
    col, row = inv * (lon, lat)
    return int(row), int(col)


def bounds_to_bbox_string(bounds) -> str:
    """Mengubah rasterio Bounds ke string WGS84 bbox."""
    return (
        f"Barat: {bounds.left:.4f}°, "
        f"Selatan: {bounds.bottom:.4f}°, "
        f"Timur: {bounds.right:.4f}°, "
        f"Utara: {bounds.top:.4f}°"
    )


def epsg_to_proj_name(epsg: Optional[int]) -> str:
    """Mengembalikan nama sistem proyeksi berdasarkan kode EPSG."""
    if epsg is None:
        return "Tidak diketahui"
    known = {
        4326 : "WGS 84 (Geographic)",
        32747: "WGS 84 / UTM zone 47S",
        32748: "WGS 84 / UTM zone 48S",
        32749: "WGS 84 / UTM zone 49S",
        23837: "DGN95 / Indonesia TM-3 zone 46.2",
        4755 : "DGN95",
    }
    return known.get(epsg, f"EPSG:{epsg}")
