"""
ForestWatch AI — app.py
Aplikasi Streamlit untuk klasifikasi penyebab deforestasi
menggunakan citra satelit Sentinel-1 dan Sentinel-2.
"""

import sys
import logging
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

# ── Konfigurasi halaman ────────────────────────────────────────
st.set_page_config(
    page_title = "ForestWatch AI — Bengkulu",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

from config.settings import (
    SENTINEL1_DIR, SENTINEL2_DIR, LABELS_DIR,
    MODELS_DIR, OUTPUTS_DIR,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES,
    NUM_CHANNELS_S1, NUM_CHANNELS_S2, NUM_CHANNELS_FUSED,
    PATCH_SIZE, PATCH_OVERLAP, MIN_VALID_RATIO,
    MODEL_FILENAME_S1, MODEL_FILENAME_S2, MODEL_FILENAME_FUSED,
    DEVICE, DEMO_MODE,
)

logging.basicConfig(level=logging.WARNING)

# ── Gaya CSS global ────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
  [data-testid="stSidebar"]          { background: #1e293b; }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  h1, h2, h3 { color: #38bdf8 !important; }
  .stMetric label { color: #94a3b8 !important; }
  .stMetric [data-testid="stMetricValue"] { color: #38bdf8 !important; }
  div[data-testid="stDataFrame"] { background: #1e293b; }
  .stAlert { border-radius: 10px; }
  hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str, mode: str, device: str):
    """Memuat model sekali dan menyimpannya di cache Streamlit."""
    from src.inference import DeforestationPredictor
    prep_config = OUTPUTS_DIR / "prep_config.json"
    return DeforestationPredictor.from_checkpoint(
        model_path       = model_path,
        prep_config_path = str(prep_config) if prep_config.exists() else None,
        class_names      = CLASS_NAMES,
        device           = device,
        mode             = mode,
        num_channels_s1  = NUM_CHANNELS_S1,
    )


def _model_path(mode: str) -> Path:
    fn = {
        "s1"   : MODEL_FILENAME_S1,
        "s2"   : MODEL_FILENAME_S2,
        "fused": MODEL_FILENAME_FUSED,
    }
    return MODELS_DIR / fn[mode]


def _available_models():
    modes = []
    for m in ["s1", "s2", "fused"]:
        if _model_path(m).exists():
            modes.append(m)
    return modes


def _get_device_str():
    import torch
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def _section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# Halaman 1 — Dashboard
# ═══════════════════════════════════════════════════════════════

def page_dashboard():
    _section_header(
        "🌿 ForestWatch AI — Dashboard",
        "Sistem Klasifikasi Penyebab Deforestasi | Bengkulu, Sumatera"
    )

    from src.data_loader import build_dataset_inventory

    # Status cards
    inv = build_dataset_inventory(SENTINEL1_DIR, SENTINEL2_DIR, LABELS_DIR)
    avail_models = _available_models()
    manifest_path = OUTPUTS_DIR / "manifest.csv"
    n_patches = 0
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        n_patches = len(df)

    import torch
    gpu_info = "CUDA tersedia ✅" if torch.cuda.is_available() else "CPU saja"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛰️ File Sentinel-1", inv["summary"]["sentinel1_count"])
    c2.metric("🛰️ File Sentinel-2", inv["summary"]["sentinel2_count"])
    c3.metric("📦 Patch Tersedia", n_patches)
    c4.metric("🤖 Model Siap", len(avail_models))

    st.markdown("#### Status Sistem")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dataset**")
        if inv["has_data"]:
            st.success(f"Dataset ditemukan — S1: {inv['summary']['sentinel1_count']}, S2: {inv['summary']['sentinel2_count']}")
        else:
            st.warning("Dataset belum ditemukan. Jalankan `python scripts/prepare_data.py --demo`")

        st.markdown("**Patch**")
        if n_patches > 0:
            st.success(f"{n_patches} patch tersedia di manifest")
        else:
            st.info("Belum ada patch. Siapkan data terlebih dahulu.")

    with col2:
        st.markdown("**Model**")
        if avail_models:
            for m in avail_models:
                p = _model_path(m)
                size_mb = p.stat().st_size / 1024 / 1024
                st.success(f"✅ Model `{m.upper()}` — {size_mb:.1f} MB")
        else:
            st.warning("Belum ada model terlatih. Jalankan `python scripts/train_model.py`")

        st.markdown("**Komputasi**")
        st.info(f"🖥️ {gpu_info}")

    # Kelas yang didukung
    st.markdown("#### Kelas Penyebab Deforestasi yang Didukung")
    cols = st.columns(3)
    for i, (name, color) in enumerate(CLASS_COLORS.items()):
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:{color}22;border:1px solid {color}66;'
                f'border-radius:8px;padding:10px;margin:4px 0;">'
                f'<span style="color:{color};font-weight:bold;">■</span> {name}</div>',
                unsafe_allow_html=True
            )

    # Hasil evaluasi terbaru (jika ada)
    eval_dir = OUTPUTS_DIR / "evaluation"
    comp_path = eval_dir / "comparison_table.json"
    if comp_path.exists():
        import json
        st.markdown("#### Ringkasan Evaluasi Model Terakhir")
        with open(comp_path) as f:
            table = json.load(f)
        import pandas as pd
        df_table = pd.DataFrame(table)
        st.dataframe(df_table, use_container_width=True)

    st.markdown("---")
    st.caption(
        "ForestWatch AI | Penelitian: Klasifikasi Penyebab Deforestasi "
        "Menggunakan Sentinel-1 & Sentinel-2 Berbasis Deep Learning | Bengkulu"
    )


# ═══════════════════════════════════════════════════════════════
# Halaman 2 — Dataset
# ═══════════════════════════════════════════════════════════════

def page_dataset():
    _section_header(
        "📁 Dataset",
        "Inventori dan informasi file dataset satelit"
    )

    from src.data_loader import build_dataset_inventory, save_inventory_json

    tab1, tab2, tab3 = st.tabs(["📦 Inventori", "🏷️ Label & Kelas", "📖 Panduan Impor"])

    with tab1:
        inv = build_dataset_inventory(SENTINEL1_DIR, SENTINEL2_DIR, LABELS_DIR)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentinel-1", inv["summary"]["sentinel1_count"], "file raster")
        col2.metric("Sentinel-2", inv["summary"]["sentinel2_count"], "file raster")
        col3.metric("Label",      inv["summary"]["label_count"],     "file label")

        if inv["s1_files"]:
            with st.expander("🛰️ File Sentinel-1", expanded=False):
                for f in inv["s1_files"][:20]:
                    st.code(f, language=None)
                if len(inv["s1_files"]) > 20:
                    st.caption(f"... dan {len(inv['s1_files'])-20} file lainnya")

        if inv["s2_files"]:
            with st.expander("🛰️ File Sentinel-2", expanded=False):
                for f in inv["s2_files"][:20]:
                    st.code(f, language=None)

        if inv["label_files"]:
            with st.expander("🏷️ File Label", expanded=False):
                for f in inv["label_files"][:20]:
                    st.code(f, language=None)

        if not inv["has_data"]:
            st.warning(
                "Dataset belum tersedia. Letakkan file raster ke:\n"
                f"- `{SENTINEL1_DIR}`\n"
                f"- `{SENTINEL2_DIR}`\n"
                f"- `{LABELS_DIR}`\n\n"
                "Atau gunakan mode demo:\n"
                "```\npython scripts/prepare_data.py --demo\n```"
            )

        # Manifest
        manifest_path = OUTPUTS_DIR / "manifest.csv"
        if manifest_path.exists():
            import pandas as pd
            st.markdown("#### Manifest Patch")
            df = pd.read_csv(manifest_path)
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(f"Total: {len(df)} patch | Menampilkan 50 baris pertama")

            from src.visualization import plot_class_distribution
            if "class_idx" in df.columns:
                labeled = df.dropna(subset=["class_idx"])
                if not labeled.empty:
                    dist = labeled.groupby("class_idx").size().to_dict()
                    dist_int = {int(k): int(v) for k, v in dist.items()}
                    fig = plot_class_distribution(
                        dist_int, CLASS_NAMES, CLASS_COLORS,
                        "Distribusi Kelas pada Manifest"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Konfigurasi Kelas Deforestasi")
        import pandas as pd
        cls_df = pd.DataFrame([
            {"Indeks": idx, "Nama Kelas": name, "Warna": color}
            for name, idx in zip(
                CLASS_NAMES,
                range(NUM_CLASSES)
            )
            for color in [CLASS_COLORS.get(name, "#808080")]
        ])
        st.dataframe(cls_df, use_container_width=True)

        st.info(
            "Kelas dikonfigurasi di `config/settings.py`. "
            "Sesuaikan dengan label dataset Anda sebelum melatih model."
        )

    with tab3:
        st.markdown("#### Panduan Persiapan Dataset")
        st.markdown("""
**Sumber Dataset yang Direkomendasikan:**
- [Copernicus Open Access Hub](https://scihub.copernicus.eu) — Sentinel-1 & Sentinel-2
- [Google Earth Engine](https://earthengine.google.com) — ekspor citra ke GeoTIFF
- [Global Forest Watch](https://www.globalforestwatch.org) — data kehilangan tutupan hutan

**Format yang Didukung:**
- Citra raster: `.tif`, `.tiff` (GeoTIFF dengan georeferensi)
- Label: `.geojson`, `.shp`, `.csv`, atau raster label `.tif`

**Struktur Direktori:**
```
data/
├── sentinel1/   ← file raster SAR (.tif)
├── sentinel2/   ← file raster optik (.tif)
└── labels/      ← file label (.geojson/.shp/.csv)
```

**Langkah Persiapan:**
```bash
# 1. Letakkan file raster ke direktori data/
# 2. Jalankan persiapan data:
python scripts/prepare_data.py

# Atau mode demo (tanpa data nyata):
python scripts/prepare_data.py --demo
```

**Catatan Penting:**
- Periksa lisensi dataset sebelum digunakan
- Pastikan CRS S1 dan S2 konsisten
- Tangani citra berawan sebelum melatih model
- Pisahkan data latih/uji berdasarkan wilayah untuk menghindari data leakage
        """)


# ═══════════════════════════════════════════════════════════════
# Halaman 3 — Analisis Citra Satelit
# ═══════════════════════════════════════════════════════════════

def page_analisis_citra():
    _section_header(
        "🔬 Analisis Citra Satelit",
        "Pemuatan, inspeksi, dan preprocessing citra Sentinel"
    )

    tab1, tab2, tab3 = st.tabs(["📂 Muat Citra", "📊 Metadata", "🎨 Visualisasi"])

    with tab1:
        st.markdown("#### Pilih Sumber Citra")
        source = st.radio(
            "Sumber",
            ["Unggah File", "Gunakan Raster Sintetis (Demo)"],
            horizontal=True
        )

        raster_data = None
        raster_meta = None

        if source == "Unggah File":
            uploaded = st.file_uploader(
                "Unggah file GeoTIFF",
                type=["tif", "tiff"],
                help="File raster Sentinel-1 atau Sentinel-2 dalam format GeoTIFF"
            )
            if uploaded:
                try:
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(
                        suffix=".tif", delete=False
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    from src.data_loader import read_raster
                    raster_data, raster_meta = read_raster(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state["raster_data"] = raster_data
                    st.session_state["raster_meta"] = raster_meta
                    st.success(
                        f"✅ Raster dimuat: {uploaded.name} | "
                        f"Shape: {raster_data.shape}"
                    )
                except Exception as exc:
                    st.error(f"Gagal memuat raster: {exc}")

        else:
            n_bands = st.selectbox(
                "Jumlah band sintetis",
                [2, 6, 8],
                index=2,
                help="2=S1, 6=S2, 8=Fused"
            )
            if st.button("🔄 Buat Raster Sintetis"):
                try:
                    import tempfile, os
                    from src.data_loader import generate_synthetic_raster, read_raster
                    with tempfile.NamedTemporaryFile(
                        suffix=".tif", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                    generate_synthetic_raster(
                        tmp_path, num_bands=n_bands,
                        height=256, width=256, seed=42
                    )
                    raster_data, raster_meta = read_raster(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state["raster_data"] = raster_data
                    st.session_state["raster_meta"] = raster_meta
                    st.success(
                        f"✅ Raster sintetis dibuat: shape={raster_data.shape} "
                        "(data tidak mewakili kondisi nyata)"
                    )
                except Exception as exc:
                    st.error(f"Gagal membuat raster sintetis: {exc}")

    with tab2:
        meta = st.session_state.get("raster_meta")
        if meta is None:
            st.info("Muat citra terlebih dahulu di tab 'Muat Citra'.")
        else:
            import pandas as pd
            meta_display = {k: str(v) for k, v in meta.items()
                           if k not in ("transform",)}
            df_meta = pd.DataFrame(
                list(meta_display.items()),
                columns=["Properti", "Nilai"]
            )
            st.dataframe(df_meta, use_container_width=True)

            if meta.get("bounds"):
                b = meta["bounds"]
                st.markdown("**Bounding Box:**")
                st.code(
                    f"Barat: {b.left:.6f}°  |  Timur: {b.right:.6f}°\n"
                    f"Selatan: {b.bottom:.6f}°  |  Utara: {b.top:.6f}°"
                )

    with tab3:
        data = st.session_state.get("raster_data")
        if data is None:
            st.info("Muat citra terlebih dahulu di tab 'Muat Citra'.")
        else:
            C, H, W = data.shape
            st.markdown(f"**Dimensi citra:** {C} kanal × {H} × {W} piksel")

            viz_type = st.selectbox(
                "Jenis Visualisasi",
                ["Komposit RGB", "False Color (NIR-Red-Green)",
                 "Kanal Tunggal", "Histogram Piksel"]
            )

            from src.visualization import (
                make_rgb_composite, make_false_color,
                plot_pixel_histogram,
            )

            if viz_type == "Komposit RGB":
                r_max = C - 1
                c1, c2, c3 = st.columns(3)
                r_idx = c1.slider("Band Merah",  0, r_max, min(2, r_max))
                g_idx = c2.slider("Band Hijau",  0, r_max, min(1, r_max))
                b_idx = c3.slider("Band Biru",   0, r_max, 0)
                rgb   = make_rgb_composite(data, r_idx, g_idx, b_idx)
                st.image(rgb, caption="Komposit RGB", use_column_width=True)

            elif viz_type == "False Color (NIR-Red-Green)":
                if C >= 4:
                    fc = make_false_color(data)
                    st.image(fc, caption="False Color (NIR-Red-Green)",
                             use_column_width=True)
                else:
                    st.warning(f"Butuh ≥4 band untuk false color. Citra ini punya {C} band.")

            elif viz_type == "Kanal Tunggal":
                band_idx = st.slider("Pilih Kanal", 0, C - 1, 0)
                band_data = data[band_idx]
                # Normalisasi untuk display
                vmin, vmax = np.nanpercentile(band_data, [2, 98])
                display = np.clip((band_data - vmin) / max(vmax - vmin, 1e-6), 0, 1)
                st.image(display.astype(np.float32),
                         caption=f"Kanal {band_idx+1}",
                         use_column_width=True,
                         clamp=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Min", f"{float(np.nanmin(band_data)):.4f}")
                col2.metric("Maks", f"{float(np.nanmax(band_data)):.4f}")
                col3.metric("Mean", f"{float(np.nanmean(band_data)):.4f}")

            elif viz_type == "Histogram Piksel":
                band_names = [f"Kanal {i+1}" for i in range(C)]
                fig = plot_pixel_histogram(data, band_names)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Halaman 4 — Klasifikasi Deforestasi
# ═══════════════════════════════════════════════════════════════

def page_klasifikasi():
    _section_header(
        "🤖 Klasifikasi Deforestasi",
        "Inferensi model deep learning pada citra satelit"
    )

    avail = _available_models()

    if not avail:
        st.error(
            "❌ Belum ada model terlatih.\n\n"
            "Latih model terlebih dahulu:\n"
            "```\npython scripts/train_model.py\n```"
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        mode_label = {"s1": "Sentinel-1", "s2": "Sentinel-2", "fused": "S1 + S2 (Gabungan)"}
        selected_mode = st.selectbox(
            "Pilih Mode Model",
            avail,
            format_func=lambda m: f"{mode_label.get(m, m)} ({m.upper()})"
        )
    with col2:
        device_str = _get_device_str()
        st.metric("Perangkat", device_str.upper())

    st.markdown("---")

    tab1, tab2 = st.tabs(["🖼️ Inferensi Patch/Raster", "📊 Hasil Prediksi"])

    with tab1:
        source = st.radio(
            "Sumber Input",
            ["Raster yang Sudah Dimuat", "Raster Sintetis (Demo)"],
            horizontal=True
        )

        run_inference = st.button("▶️ Jalankan Inferensi", type="primary")

        if run_inference:
            with st.spinner("Memuat model..."):
                try:
                    predictor = _load_model_cached(
                        str(_model_path(selected_mode)),
                        selected_mode,
                        device_str
                    )
                except Exception as exc:
                    st.error(f"Gagal memuat model: {exc}")
                    return

            # Ambil atau buat raster
            if source == "Raster yang Sudah Dimuat":
                data = st.session_state.get("raster_data")
                meta = st.session_state.get("raster_meta")
                if data is None:
                    st.warning("Muat raster dulu di halaman 'Analisis Citra Satelit'.")
                    return
                transform = meta.get("transform") if meta else None
            else:
                import tempfile, os
                from src.data_loader import generate_synthetic_raster, read_raster
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    generate_synthetic_raster(
                        tmp_path, num_bands=NUM_CHANNELS_FUSED,
                        height=256, width=256, seed=7
                    )
                    data, meta = read_raster(tmp_path)
                    transform = meta.get("transform")
                finally:
                    os.unlink(tmp_path)
                st.info("⚠️ Menggunakan raster sintetis — hasil bukan prediksi nyata.")

            with st.spinner("Menjalankan inferensi..."):
                result = predictor.predict_raster(
                    data            = data,
                    patch_size      = PATCH_SIZE,
                    overlap         = PATCH_OVERLAP,
                    nodata_value    = 0.0,
                    min_valid_ratio = MIN_VALID_RATIO,
                    transform       = transform,
                )

            st.session_state["pred_result"]    = result
            st.session_state["pred_transform"] = transform
            st.session_state["pred_mode"]      = selected_mode

            n_patches = result["num_patches"]
            preds = [r["class_idx"] for r in result["patch_results"] if r.get("class_idx", -1) >= 0]
            st.success(f"✅ Inferensi selesai — {n_patches} patch diproses")

            if preds:
                unique, counts = np.unique(preds, return_counts=True)
                cols = st.columns(min(len(unique), 3))
                for i, (u, c) in enumerate(zip(unique, counts)):
                    name  = CLASS_NAMES[u] if u < len(CLASS_NAMES) else str(u)
                    color = CLASS_COLORS.get(name, "#808080")
                    with cols[i % 3]:
                        pct = 100 * c / len(preds)
                        st.markdown(
                            f'<div style="background:{color}22;border:1px solid {color}66;'
                            f'border-radius:8px;padding:8px;margin:4px;">'
                            f'<b style="color:{color};">{name}</b><br>'
                            f'{c} patch ({pct:.1f}%)</div>',
                            unsafe_allow_html=True
                        )

    with tab2:
        result = st.session_state.get("pred_result")
        if result is None:
            st.info("Jalankan inferensi terlebih dahulu.")
            return

        patch_results = result["patch_results"]
        if not patch_results:
            st.warning("Tidak ada hasil prediksi.")
            return

        import pandas as pd
        df_pred = pd.DataFrame([
            {
                "Patch"      : i,
                "Kelas"      : r.get("class_name", CLASS_NAMES[r["class_idx"]]
                                     if r.get("class_idx", -1) >= 0 else "—"),
                "Kepercayaan": f"{r.get('confidence', 0):.3f}",
                "Waktu (ms)" : r.get("inference_ms", "—"),
            }
            for i, r in enumerate(patch_results[:200])
        ])
        st.dataframe(df_pred, use_container_width=True)
        st.caption(f"Menampilkan {min(200, len(patch_results))} dari {len(patch_results)} patch")

        # Ekspor
        col1, col2 = st.columns(2)
        with col1:
            csv_str = df_pred.to_csv(index=False)
            st.download_button(
                "⬇️ Unduh CSV Prediksi",
                data     = csv_str,
                file_name= "forestwatch_predictions.csv",
                mime     = "text/csv",
            )

        # Peta prediksi sebagai gambar
        pred_map = result.get("prediction_map")
        if pred_map is not None:
            from src.visualization import render_prediction_map
            img = render_prediction_map(pred_map, CLASS_NAMES, CLASS_COLORS)
            if img is not None:
                st.markdown("#### Peta Prediksi Spasial")
                st.image(img, caption="Peta Klasifikasi Penyebab Deforestasi",
                         use_column_width=True)


# ═══════════════════════════════════════════════════════════════
# Halaman 5 — Evaluasi Model
# ═══════════════════════════════════════════════════════════════

def page_evaluasi():
    _section_header(
        "📈 Evaluasi Model",
        "Metrik performa, confusion matrix, dan perbandingan eksperimen"
    )

    eval_dir  = OUTPUTS_DIR / "evaluation"
    comp_path = eval_dir / "comparison_table.json"
    avail     = _available_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrik",
        "🔲 Confusion Matrix",
        "📉 Riwayat Pelatihan",
        "🏆 Perbandingan Model",
    ])

    with tab1:
        if not avail:
            st.warning("Belum ada model. Latih model terlebih dahulu.")
            return

        selected = st.selectbox(
            "Pilih Model", avail,
            format_func=lambda m: m.upper()
        )
        eval_path = eval_dir / f"eval_{selected}.json"

        if not eval_path.exists():
            st.info(
                f"Hasil evaluasi belum ada untuk model `{selected}`.\n"
                f"Jalankan:\n```\npython scripts/evaluate_model.py --mode {selected}\n```"
            )
        else:
            import json, pandas as pd
            with open(eval_path) as f:
                ev = json.load(f)
            metrics = ev.get("metrics", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",        f"{metrics.get('accuracy', 0):.4f}")
            c2.metric("Precision Macro", f"{metrics.get('precision_macro', 0):.4f}")
            c3.metric("Recall Macro",    f"{metrics.get('recall_macro', 0):.4f}")
            c4.metric("F1 Macro",        f"{metrics.get('f1_macro', 0):.4f}")

            if metrics.get("avg_inference_ms"):
                st.metric("Rata-rata Inferensi",
                          f"{metrics['avg_inference_ms']:.2f} ms/sampel")

            from src.visualization import plot_per_class_metrics
            fig = plot_per_class_metrics(metrics, CLASS_NAMES,
                                         f"Metrik Per Kelas — {selected.upper()}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Error analysis
            err = metrics.get("error_analysis", {})
            if err:
                with st.expander("🔍 Analisis Kesalahan Klasifikasi"):
                    st.metric("Total Error", err.get("total_errors", 0))
                    st.metric("Error Rate",  f"{err.get('error_rate', 0):.4f}")
                    confused = err.get("most_confused_pairs", [])
                    if confused:
                        st.markdown("**Pasangan kelas paling sering salah:**")
                        df_c = pd.DataFrame(confused)
                        st.dataframe(df_c, use_container_width=True)

    with tab2:
        selected2 = st.selectbox(
            "Model untuk Confusion Matrix", avail,
            format_func=lambda m: m.upper(), key="cm_select"
        ) if avail else None

        if selected2:
            eval_path2 = eval_dir / f"eval_{selected2}.json"
            if eval_path2.exists():
                import json
                with open(eval_path2) as f:
                    ev2 = json.load(f)
                cm = ev2.get("metrics", {}).get("confusion_matrix")
                if cm:
                    from src.visualization import plot_confusion_matrix
                    normalize = st.toggle("Normalisasi", value=True)
                    fig_cm = plot_confusion_matrix(
                        cm, CLASS_NAMES, normalize=normalize,
                        title=f"Confusion Matrix — {selected2.upper()}"
                    )
                    if fig_cm:
                        st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Jalankan evaluasi terlebih dahulu.")

    with tab3:
        plots_dir = OUTPUTS_DIR / "plots"
        history_files = list(plots_dir.glob("history_*.csv")) if plots_dir.exists() else []

        if not history_files:
            st.info(
                "Riwayat pelatihan belum tersedia.\n"
                "Jalankan pelatihan terlebih dahulu: "
                "`python scripts/train_model.py`"
            )
        else:
            import pandas as pd
            sel_hist = st.selectbox(
                "Pilih riwayat", history_files,
                format_func=lambda p: p.stem
            )
            df_h = pd.read_csv(sel_hist)
            history_dict = {
                "train_loss"  : df_h["train_loss"].tolist(),
                "val_loss"    : df_h["val_loss"].tolist(),
                "train_acc"   : df_h["train_acc"].tolist(),
                "val_acc"     : df_h["val_acc"].tolist(),
                "best_epoch"  : int(df_h["val_loss"].idxmin()),
            }
            from src.visualization import plot_training_history
            fig_hist = plot_training_history(history_dict)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab4:
        if not comp_path.exists():
            st.info(
                "Tabel perbandingan belum ada.\n"
                "Jalankan: `python scripts/evaluate_model.py --all`"
            )
        else:
            import json, pandas as pd
            with open(comp_path) as f:
                table = json.load(f)
            df_comp = pd.DataFrame(table)
            st.dataframe(df_comp, use_container_width=True)

            from src.visualization import plot_model_comparison
            fig_comp = plot_model_comparison(table, "Radar Perbandingan Model")
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)

            st.info(
                "Tabel di atas menunjukkan hasil eksperimen yang benar-benar "
                "dilakukan. Jangan bandingkan model yang dilatih pada data berbeda."
            )


# ═══════════════════════════════════════════════════════════════
# Halaman 6 — Peta Hasil
# ═══════════════════════════════════════════════════════════════

def page_peta():
    _section_header(
        "🗺️ Peta Hasil Klasifikasi",
        "Visualisasi geospasial hasil prediksi pada peta interaktif"
    )

    result    = st.session_state.get("pred_result")
    transform = st.session_state.get("pred_transform")
    meta      = st.session_state.get("raster_meta")

    if result is None:
        st.info(
            "Belum ada hasil prediksi. "
            "Jalankan inferensi di halaman 'Klasifikasi Deforestasi' terlebih dahulu."
        )
        return

    # Pusat peta dari metadata raster
    center_lat, center_lon = -3.5, 102.5
    if meta and meta.get("bounds"):
        b = meta["bounds"]
        center_lon = (b.left + b.right) / 2
        center_lat = (b.bottom + b.top) / 2

    tab1, tab2, tab3 = st.tabs(["🗺️ Peta Interaktif", "🖼️ Peta Raster", "📐 Statistik Area"])

    with tab1:
        from src.visualization import build_folium_map

        # Coba buat GeoDataFrame dari patch results
        gdf = None
        patch_results = result.get("patch_results", [])

        try:
            from src.patch_generator import generate_patches
            from src.data_loader import generate_synthetic_raster, read_raster
            from src.geospatial_utils import predictions_to_geodataframe
            import tempfile, os

            # Gunakan patches yang ada di session (jika ada transform spasial)
            if transform is not None and patch_results:
                # Buat dummy patches untuk GDF hanya jika ada geo_bounds
                dummy_patches = []
                data_shape = result["prediction_map"].shape
                H, W = data_shape

                stride = max(1, int(PATCH_SIZE * (1 - PATCH_OVERLAP)))
                idx = 0
                from src.patch_generator import Patch
                for r in range(0, H - PATCH_SIZE + 1, stride):
                    for c_idx in range(0, W - PATCH_SIZE + 1, stride):
                        if idx >= len(patch_results):
                            break
                        try:
                            left, top   = transform * (c_idx, r)
                            right, bottom = transform * (c_idx + PATCH_SIZE, r + PATCH_SIZE)
                        except Exception:
                            idx += 1
                            continue
                        dummy_patches.append(Patch(
                            patch_id    = f"p_{idx:06d}",
                            data        = np.zeros((1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32),
                            row_start   = r, col_start=c_idx,
                            patch_size  = PATCH_SIZE, valid_ratio=1.0,
                            geo_bounds  = (left, bottom, right, top),
                        ))
                        idx += 1

                if dummy_patches:
                    gdf = predictions_to_geodataframe(
                        patch_results[:len(dummy_patches)],
                        dummy_patches,
                        crs          = meta.get("crs") or "EPSG:4326",
                        class_names  = CLASS_NAMES,
                        class_colors = CLASS_COLORS,
                    )
        except Exception as exc:
            st.warning(f"Tidak bisa membuat layer peta: {exc}")

        folium_map = build_folium_map(
            gdf         = gdf,
            center_lat  = center_lat,
            center_lon  = center_lon,
            zoom_start  = 10,
            class_colors= CLASS_COLORS,
        )

        if folium_map:
            try:
                from streamlit_folium import st_folium
                st_folium(folium_map, width=None, height=520)
            except ImportError:
                # Fallback: simpan HTML
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, mode="w"
                ) as f:
                    folium_map.save(f.name)
                    html_path = f.name
                with open(html_path, "r", encoding="utf-8") as f:
                    html_str = f.read()
                st.components.v1.html(html_str, height=520)
        else:
            st.warning("folium tidak tersedia. Install: `pip install folium streamlit-folium`")

    with tab2:
        pred_map = result.get("prediction_map")
        if pred_map is not None:
            from src.visualization import render_prediction_map
            img = render_prediction_map(pred_map, CLASS_NAMES, CLASS_COLORS)
            if img is not None:
                st.image(img, caption="Peta Klasifikasi", use_column_width=True)

            # Ekspor GeoTIFF
            if transform is not None:
                if st.button("⬇️ Ekspor Peta ke GeoTIFF"):
                    import tempfile
                    from src.inference import save_prediction_raster
                    with tempfile.NamedTemporaryFile(
                        suffix=".tif", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                    try:
                        save_prediction_raster(
                            pred_map, tmp_path, transform,
                            crs=meta.get("crs") or "EPSG:4326"
                        )
                        with open(tmp_path, "rb") as f:
                            tif_bytes = f.read()
                        st.download_button(
                            "⬇️ Unduh GeoTIFF",
                            data=tif_bytes,
                            file_name="prediction_map.tif",
                            mime="image/tiff",
                        )
                    except Exception as exc:
                        st.error(f"Gagal ekspor: {exc}")

    with tab3:
        pred_map = result.get("prediction_map")
        if pred_map is not None:
            from src.geospatial_utils import compute_class_area
            area_stats = compute_class_area(pred_map, transform, CLASS_NAMES)
            per_class  = area_stats.get("per_class", {})

            if per_class:
                import pandas as pd
                rows = []
                for cls_name, stats in per_class.items():
                    row = {"Kelas": cls_name}
                    row.update(stats)
                    rows.append(row)
                df_area = pd.DataFrame(rows)
                st.dataframe(df_area, use_container_width=True)
                st.caption(f"Unit: {area_stats.get('unit', 'pixels')}")

                from src.visualization import plot_class_distribution
                dist = {i: stats["pixel_count"]
                        for i, (_, stats) in enumerate(per_class.items())}
                fig = plot_class_distribution(
                    dist,
                    list(per_class.keys()),
                    CLASS_COLORS,
                    "Distribusi Area per Kelas"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Halaman 7 — Tentang Proyek
# ═══════════════════════════════════════════════════════════════

def page_tentang():
    _section_header("ℹ️ Tentang Proyek")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
## ForestWatch AI — Bengkulu

**Judul Penelitian:**
> Klasifikasi Penyebab Deforestasi Menggunakan Citra Satelit Sentinel-1 dan Sentinel-2
> Berbasis Deep Learning

### Latar Belakang
Hutan tropis Bengkulu mengalami tekanan signifikan akibat aktivitas manusia.
Citra satelit dapat mendeteksi kehilangan tutupan hutan, namun tidak secara
langsung mengidentifikasi penyebabnya. Sistem ini mengembangkan pipeline deep
learning untuk mengklasifikasikan **penyebab langsung** deforestasi berdasarkan
karakteristik spektral dan radar dari citra Sentinel.

### Tujuan Penelitian
1. Mengolah citra Sentinel-1 (SAR) dan Sentinel-2 (optik)
2. Mengembangkan CNN untuk klasifikasi 6 kelas penyebab deforestasi
3. Membandingkan kontribusi S1, S2, dan gabungan keduanya
4. Mengevaluasi kinerja model secara ilmiah
5. Menyajikan hasil dalam bentuk peta geospasial

### Arsitektur Model
- **ForestWatchCNN**: CNN kustom dengan residual blocks
  - Stem → Stage1 → Stage2 → Stage3 → GlobalAvgPool → FC
  - Mendukung kanal input: 2 (S1), 6 (S2), 8 (Fused)
- **Transfer Learning**: ResNet-18 yang dimodifikasi (opsional)

### Kelas yang Didukung
        """)

        for name, color in CLASS_COLORS.items():
            st.markdown(
                f'<span style="color:{color};font-weight:bold;">■</span> {name}   ',
                unsafe_allow_html=True
            )

        st.markdown("""
### Teknologi
| Komponen | Teknologi |
|----------|-----------|
| Deep Learning | PyTorch |
| Raster GIS | Rasterio |
| Data Geospasial | GeoPandas, Shapely |
| Analitik | NumPy, Pandas, scikit-learn |
| Antarmuka | Streamlit |
| Visualisasi | Plotly, Matplotlib, Folium |
| Pengujian | pytest |

### Keterbatasan Sistem
- Model yang belum dilatih pada data berlabel tidak dapat memberikan prediksi yang valid
- Akurasi model bergantung pada kualitas dan kelengkapan label dataset
- Data sintetis yang digunakan dalam mode demo tidak mewakili kondisi deforestasi nyata
- Hasil prediksi perlu divalidasi dengan survei lapangan
- Klasifikasi penyebab deforestasi memerlukan label yang terverifikasi secara ilmiah

### Sumber Data yang Direkomendasikan
- ESA Copernicus Open Access Hub (Sentinel-1, Sentinel-2)
- Global Forest Watch — Hansen et al. (2013)
- GLAD Forest Alert — University of Maryland
- JRC Global Surface Water Dataset

### Referensi
- Hansen et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*.
- Torres et al. (2012). GMES Sentinel-1 mission. *Remote Sensing of Environment*.
- Drusch et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission. *Remote Sensing of Environment*.
        """)

    with col2:
        st.markdown("### Struktur Proyek")
        st.code("""
forestwatch-ai/
├── app.py
├── requirements.txt
├── config/
│   └── settings.py
├── src/
│   ├── data_loader.py
│   ├── raster_preprocessing.py
│   ├── patch_generator.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── geospatial_utils.py
│   └── visualization.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
├── data/
├── models/
└── outputs/
        """, language="text")

        st.markdown("### Cara Menjalankan")
        st.code("""
# 1. Instal dependensi
pip install -r requirements.txt

# 2. Siapkan data (mode demo)
python scripts/prepare_data.py --demo

# 3. Latih model
python scripts/train_model.py

# 4. Evaluasi
python scripts/evaluate_model.py

# 5. Jalankan aplikasi
streamlit run app.py
        """, language="bash")

        st.markdown("### Lisensi")
        st.info("MIT License — Bebas digunakan untuk keperluan penelitian dan pendidikan.")


# ═══════════════════════════════════════════════════════════════
# Navigasi utama
# ═══════════════════════════════════════════════════════════════

PAGES = {
    "🏠 Dashboard"               : page_dashboard,
    "📁 Dataset"                 : page_dataset,
    "🔬 Analisis Citra Satelit"  : page_analisis_citra,
    "🤖 Klasifikasi Deforestasi" : page_klasifikasi,
    "📈 Evaluasi Model"          : page_evaluasi,
    "🗺️ Peta Hasil"              : page_peta,
    "ℹ️ Tentang Proyek"          : page_tentang,
}

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:16px 0;">'
        '<span style="font-size:2rem;">🌿</span><br>'
        '<b style="font-size:1.1rem;color:#38bdf8;">ForestWatch AI</b><br>'
        '<span style="font-size:0.75rem;color:#64748b;">Bengkulu — Deep Learning</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    selected_page = st.radio(
        "Navigasi",
        list(PAGES.keys()),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("v1.0.0 | Python + PyTorch + Streamlit")

# Render halaman yang dipilih
PAGES[selected_page]()
