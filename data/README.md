# Data — ForestWatch AI

Direktori ini menyimpan file dataset citra satelit dan label deforestasi.

## Struktur

```
data/
├── sentinel1/     ← Citra SAR Sentinel-1 (GeoTIFF)
├── sentinel2/     ← Citra Optik Sentinel-2 (GeoTIFF)
└── labels/        ← File label penyebab deforestasi
```

## Sentinel-1

**Kanal:** VV (polarisasi vertikal-vertikal) dan VH (vertikal-horizontal)  
**Format:** GeoTIFF multi-band, nilai dalam dB  
**Rentang valid:** -50 dB hingga +10 dB  
**Resolusi spasial:** 10–20 meter (tergantung mode akuisisi)

**Sumber yang direkomendasikan:**
- [ESA Copernicus Open Access Hub](https://scihub.copernicus.eu) — gratis setelah registrasi
- [Google Earth Engine](https://earthengine.google.com) — ekspor sebagai GeoTIFF

## Sentinel-2

**Kanal yang digunakan:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)  
**Format:** GeoTIFF Level-2A (Bottom-of-Atmosphere reflectance)  
**Skala nilai:** 0–10000 (integer) → dibagi 10000 menjadi 0–1  
**Resolusi spasial:** 10–20 meter

**Sumber yang direkomendasikan:**
- [ESA Copernicus Open Access Hub](https://scihub.copernicus.eu)
- [Google Earth Engine](https://earthengine.google.com)
- [Sentinel Hub](https://www.sentinel-hub.com)

## Label

**Format yang didukung:**
- GeoJSON (`.geojson`) — poligon dengan atribut kelas
- Shapefile (`.shp`) — vektor poligon
- Raster label (`.tif`) — nilai integer per piksel
- CSV patch (`.csv`) — kolom `patch_id`, `class_idx`, `class_name`

**Kelas penyebab deforestasi:**

| Indeks | Kelas |
|--------|-------|
| 0 | Pertanian Skala Kecil |
| 1 | Pertanian Skala Besar |
| 2 | Pembangunan Jalan |
| 3 | Penebangan Selektif |
| 4 | Pertambangan |
| 5 | Kebakaran / Gangguan Alami |

## Catatan Penting

- Periksa **lisensi** setiap dataset sebelum digunakan
- Pastikan CRS Sentinel-1 dan Sentinel-2 **konsisten** (atau reproyeksi terlebih dahulu)
- Tangani citra **berawan** sebelum melatih model
- Pisahkan data latih dan uji berdasarkan **wilayah atau kejadian** untuk menghindari data leakage spasial
- File raster besar **tidak disertakan** di repositori GitHub

## Mode Demo

Jika belum memiliki dataset, gunakan mode demo:

```bash
python scripts/prepare_data.py --demo
```

Mode demo menggunakan data sintetis yang **tidak mewakili** kondisi deforestasi nyata,
hanya untuk menguji fungsi aplikasi.
