"""
ForestWatch AI — visualization.py
Fungsi visualisasi untuk Streamlit menggunakan Plotly dan Matplotlib.

Meliputi:
  - Tampilan kanal citra satelit (RGB composite, false color)
  - Grafik training/validation loss dan accuracy
  - Confusion matrix interaktif (Plotly heatmap)
  - Bar chart distribusi kelas
  - Tabel perbandingan model
  - Peta Folium hasil klasifikasi
  - Histogram distribusi piksel
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Plotly diimpor secara lazy untuk menghindari error saat tidak terinstal
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly tidak tersedia — grafik interaktif dinonaktifkan.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Komposit RGB dari array raster
# ──────────────────────────────────────────────────────────────

def make_rgb_composite(
    data: np.ndarray,
    r_band: int = 2,
    g_band: int = 1,
    b_band: int = 0,
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Membuat komposit RGB dari array multi-kanal.

    Parameters
    ----------
    data : np.ndarray, shape (C, H, W)
    r_band, g_band, b_band : indeks kanal (berbasis-0)
    percentile_clip : (low, high) — persen kliping kontras

    Returns
    -------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    """
    C, H, W = data.shape
    max_band = max(r_band, g_band, b_band)
    if max_band >= C:
        # Fallback: gunakan kanal yang tersedia
        r_band = min(r_band, C - 1)
        g_band = min(g_band, C - 1)
        b_band = min(b_band, C - 1)

    def stretch(arr: np.ndarray) -> np.ndarray:
        valid = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if valid.size == 0:
            return np.zeros_like(arr)
        lo = np.percentile(valid, percentile_clip[0])
        hi = np.percentile(valid, percentile_clip[1])
        if hi == lo:
            return np.zeros_like(arr)
        stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
        return (stretched * 255).astype(np.uint8)

    r = stretch(data[r_band])
    g = stretch(data[g_band])
    b = stretch(data[b_band])
    return np.stack([r, g, b], axis=-1)


def make_false_color(
    data: np.ndarray,
    nir_band: int = 3,
    red_band: int = 2,
    green_band: int = 1,
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    False color composite NIR-Red-Green untuk visualisasi vegetasi.
    """
    return make_rgb_composite(
        data, r_band=nir_band, g_band=red_band, b_band=green_band,
        percentile_clip=percentile_clip,
    )


# ──────────────────────────────────────────────────────────────
# Grafik pelatihan (Plotly)
# ──────────────────────────────────────────────────────────────

def plot_training_history(history: Dict) -> Optional[object]:
    """
    Grafik training/validation loss dan accuracy per epoch.

    Returns
    -------
    plotly.graph_objects.Figure | None
    """
    if not PLOTLY_AVAILABLE:
        return None

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])
    train_acc  = history.get("train_acc", [])
    val_acc    = history.get("val_acc", [])

    if not train_loss:
        return None

    epochs = list(range(1, len(train_loss) + 1))
    best_ep = history.get("best_epoch", 0) + 1

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss per Epoch", "Accuracy per Epoch"),
    )

    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, mode="lines", name="Train Loss",
        line=dict(color="#3b82f6", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss, mode="lines", name="Val Loss",
        line=dict(color="#ef4444", width=2),
    ), row=1, col=1)
    fig.add_vline(
        x=best_ep, line_dash="dash", line_color="#22c55e",
        annotation_text=f"Best (ep {best_ep})",
        row=1, col=1,
    )

    # Accuracy
    if train_acc:
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in train_acc], mode="lines",
            name="Train Acc", line=dict(color="#3b82f6", width=2, dash="dot"),
        ), row=1, col=2)
    if val_acc:
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in val_acc], mode="lines",
            name="Val Acc", line=dict(color="#ef4444", width=2, dash="dot"),
        ), row=1, col=2)

    fig.update_layout(
        title="Riwayat Pelatihan ForestWatch AI",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")

    return fig


# ──────────────────────────────────────────────────────────────
# Confusion matrix (Plotly heatmap)
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Optional[object]:
    """
    Confusion matrix interaktif sebagai heatmap Plotly.
    """
    if not PLOTLY_AVAILABLE:
        return None

    cm_arr = np.array(cm)
    n = cm_arr.shape[0]
    labels = class_names if class_names and len(class_names) == n else [
        f"Kelas {i}" for i in range(n)
    ]

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_arr.astype(float) / row_sums
        fmt_vals   = [[f"{v:.2f}" for v in row] for row in cm_display]
        colorscale = "Blues"
    else:
        cm_display = cm_arr.astype(float)
        fmt_vals   = [[str(int(v)) for v in row] for row in cm_display]
        colorscale = "Blues"

    fig = go.Figure(go.Heatmap(
        z             = cm_display.tolist(),
        x             = labels,
        y             = labels,
        text          = fmt_vals,
        texttemplate  = "%{text}",
        colorscale    = colorscale,
        showscale     = True,
        hovertemplate = "True: %{y}<br>Pred: %{x}<br>Nilai: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title      = title,
        xaxis_title = "Prediksi",
        yaxis_title = "Aktual",
        template   = "plotly_dark",
        height     = 500,
        paper_bgcolor = "#0f172a",
        plot_bgcolor  = "#1e293b",
        font = dict(color="#e2e8f0"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Distribusi kelas
# ──────────────────────────────────────────────────────────────

def plot_class_distribution(
    distribution: Dict[int, int],
    class_names: Optional[List[str]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    title: str = "Distribusi Kelas",
) -> Optional[object]:
    """
    Bar chart distribusi sampel per kelas.
    """
    if not PLOTLY_AVAILABLE:
        return None

    sorted_keys = sorted(distribution.keys())
    names  = []
    counts = []
    colors = []

    for k in sorted_keys:
        name = (
            class_names[k]
            if class_names and k < len(class_names)
            else f"Kelas {k}"
        )
        names.append(name)
        counts.append(distribution[k])
        if class_colors:
            colors.append(class_colors.get(name, "#64748b"))
        else:
            colors.append("#3b82f6")

    fig = go.Figure(go.Bar(
        x=names, y=counts,
        marker_color=colors,
        text=counts,
        textposition="outside",
        hovertemplate="%{x}: %{y} sampel<extra></extra>",
    ))
    fig.update_layout(
        title         = title,
        xaxis_title   = "Kelas",
        yaxis_title   = "Jumlah Sampel",
        template      = "plotly_dark",
        height        = 400,
        paper_bgcolor = "#0f172a",
        plot_bgcolor  = "#1e293b",
        font          = dict(color="#e2e8f0"),
        xaxis         = dict(tickangle=-20),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Metrik per kelas (grouped bar)
# ──────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    metrics: Dict,
    class_names: Optional[List[str]] = None,
    title: str = "Metrik Per Kelas",
) -> Optional[object]:
    """
    Bar chart Precision, Recall, F1 per kelas.
    """
    if not PLOTLY_AVAILABLE:
        return None

    precision = metrics.get("precision_per_class", [])
    recall    = metrics.get("recall_per_class", [])
    f1        = metrics.get("f1_per_class", [])
    present   = metrics.get("present_classes", list(range(len(precision))))

    if not precision:
        return None

    names = [
        class_names[i] if class_names and i < len(class_names) else f"Kelas {i}"
        for i in present
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Precision", x=names,
        y=[round(v, 4) for v in precision],
        marker_color="#3b82f6",
    ))
    fig.add_trace(go.Bar(
        name="Recall", x=names,
        y=[round(v, 4) for v in recall],
        marker_color="#22c55e",
    ))
    fig.add_trace(go.Bar(
        name="F1-Score", x=names,
        y=[round(v, 4) for v in f1],
        marker_color="#f59e0b",
    ))

    fig.update_layout(
        title         = title,
        barmode       = "group",
        yaxis         = dict(range=[0, 1.05], title="Skor"),
        xaxis         = dict(title="Kelas", tickangle=-20),
        template      = "plotly_dark",
        height        = 450,
        paper_bgcolor = "#0f172a",
        plot_bgcolor  = "#1e293b",
        font          = dict(color="#e2e8f0"),
        legend        = dict(orientation="h", y=-0.2),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Perbandingan model (radar chart)
# ──────────────────────────────────────────────────────────────

def plot_model_comparison(
    comparison_rows: List[Dict],
    title: str = "Perbandingan Model",
) -> Optional[object]:
    """
    Radar chart perbandingan metrik antar eksperimen.
    """
    if not PLOTLY_AVAILABLE or not comparison_rows:
        return None

    metric_keys = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
    valid_rows = [r for r in comparison_rows if "Accuracy" in r]
    if not valid_rows:
        return None

    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]
    fig    = go.Figure()

    for i, row in enumerate(valid_rows):
        values = [
            float(row.get(k, 0)) for k in metric_keys
        ]
        values_closed = values + [values[0]]
        cats_closed   = metric_keys + [metric_keys[0]]

        fig.add_trace(go.Scatterpolar(
            r     = values_closed,
            theta = cats_closed,
            fill  = "toself",
            name  = row["Eksperimen"],
            line  = dict(color=colors[i % len(colors)]),
            opacity = 0.7,
        ))

    fig.update_layout(
        title          = title,
        polar          = dict(radialaxis=dict(visible=True, range=[0, 1])),
        template       = "plotly_dark",
        height         = 420,
        paper_bgcolor  = "#0f172a",
        font           = dict(color="#e2e8f0"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Histogram distribusi nilai piksel
# ──────────────────────────────────────────────────────────────

def plot_pixel_histogram(
    data: np.ndarray,
    band_names: Optional[List[str]] = None,
    title: str = "Distribusi Nilai Piksel",
    bins: int = 80,
) -> Optional[object]:
    """
    Histogram distribusi nilai piksel per kanal.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    C = data.shape[0]
    colors_palette = [
        "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
        "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    ]
    fig = go.Figure()

    for b in range(C):
        band_data = data[b].ravel()
        valid     = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
        if valid.size == 0:
            continue

        name = band_names[b] if band_names and b < len(band_names) else f"Kanal {b+1}"
        counts, bin_edges = np.histogram(valid, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig.add_trace(go.Bar(
            x=bin_centers.tolist(), y=counts.tolist(),
            name=name,
            marker_color=colors_palette[b % len(colors_palette)],
            opacity=0.7,
        ))

    fig.update_layout(
        title         = title,
        xaxis_title   = "Nilai Piksel",
        yaxis_title   = "Frekuensi",
        barmode       = "overlay",
        template      = "plotly_dark",
        height        = 380,
        paper_bgcolor = "#0f172a",
        plot_bgcolor  = "#1e293b",
        font          = dict(color="#e2e8f0"),
        legend        = dict(orientation="h", y=-0.2),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Peta Folium hasil klasifikasi
# ──────────────────────────────────────────────────────────────

def build_folium_map(
    gdf=None,
    center_lat: float = -3.5,
    center_lon: float = 102.5,
    zoom_start: int = 10,
    class_colors: Optional[Dict[str, str]] = None,
    basemap: str = "OpenStreetMap",
) -> Optional[object]:
    """
    Membangun peta Folium dengan hasil klasifikasi sebagai layer GeoJSON.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame | None
        Hasil prediksi dengan geometri. Jika None, tampilkan peta kosong.
    center_lat, center_lon : float
        Pusat peta. Diambil dari metadata raster, bukan nilai hardcoded.

    Returns
    -------
    folium.Map | None
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium tidak tersedia — peta interaktif dinonaktifkan.")
        return None

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=basemap,
    )

    if gdf is not None and len(gdf) > 0:
        try:
            geojson_data = gdf.__geo_interface__

            def style_fn(feature):
                cls_name = feature["properties"].get("class_name", "")
                color    = (
                    class_colors.get(cls_name, "#808080")
                    if class_colors else "#3b82f6"
                )
                conf = feature["properties"].get("confidence", 0.5)
                return {
                    "fillColor"   : color,
                    "color"       : "#1a1a1a",
                    "weight"      : 0.5,
                    "fillOpacity" : max(0.3, conf),
                }

            folium.GeoJson(
                geojson_data,
                name        = "Hasil Klasifikasi",
                style_function = style_fn,
                tooltip     = folium.GeoJsonTooltip(
                    fields    = ["class_name", "confidence", "patch_id"],
                    aliases   = ["Kelas:", "Kepercayaan:", "Patch ID:"],
                    localize  = True,
                ),
            ).add_to(m)
        except Exception as exc:
            logger.error("Gagal menambahkan GeoJSON ke peta: %s", exc)

    # Legenda (jika class_colors tersedia)
    if class_colors:
        legend_html = _build_legend_html(class_colors)
        m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def _build_legend_html(class_colors: Dict[str, str]) -> str:
    items = "".join([
        f'<li style="display:flex;align-items:center;margin:4px 0;">'
        f'<span style="background:{color};width:16px;height:16px;'
        f'border-radius:3px;display:inline-block;margin-right:8px;"></span>'
        f'<span style="font-size:12px;">{name}</span></li>'
        for name, color in class_colors.items()
    ])
    return f"""
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 999;
        background: rgba(15,23,42,0.92); padding: 12px 16px;
        border-radius: 10px; border: 1px solid #334155;
        font-family: sans-serif; color: #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    ">
        <div style="font-weight:bold;margin-bottom:8px;font-size:13px;">
            Legenda Kelas
        </div>
        <ul style="list-style:none;padding:0;margin:0;">{items}</ul>
    </div>
    """


# ──────────────────────────────────────────────────────────────
# Visualisasi peta prediksi (Matplotlib → numpy untuk Streamlit)
# ──────────────────────────────────────────────────────────────

def render_prediction_map(
    prediction_map: np.ndarray,
    class_names: Optional[List[str]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    nodata: int = -1,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[np.ndarray]:
    """
    Merender peta prediksi sebagai gambar numpy (H, W, 3) RGBA.
    Cocok untuk ditampilkan dengan st.image().

    Returns
    -------
    np.ndarray shape (H, W, 4) uint8 | None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    unique_classes = [v for v in np.unique(prediction_map) if v != nodata]
    n_classes      = len(unique_classes)

    if n_classes == 0:
        return None

    # Buat colormap diskrit
    if class_colors and class_names:
        color_list = []
        for cls in sorted(unique_classes):
            idx  = int(cls)
            name = class_names[idx] if idx < len(class_names) else str(idx)
            color_list.append(class_colors.get(name, "#808080"))
    else:
        cmap_base   = plt.cm.get_cmap("tab10", n_classes)
        color_list  = [
            mcolors.to_hex(cmap_base(i)) for i in range(n_classes)
        ]

    cmap = mcolors.ListedColormap(color_list)
    bounds_cm = [c - 0.5 for c in sorted(unique_classes)] + [sorted(unique_classes)[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds_cm, cmap.N)

    # Mask nodata
    masked = np.ma.masked_where(prediction_map == nodata, prediction_map)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks(sorted(unique_classes))
    cbar.set_ticklabels([
        class_names[int(c)] if class_names and int(c) < len(class_names) else str(c)
        for c in sorted(unique_classes)
    ])
    cbar.ax.tick_params(labelsize=9, colors="white")
    cbar.set_label("Kelas Deforestasi", color="white")

    ax.set_title("Peta Klasifikasi Penyebab Deforestasi", color="white", fontsize=13)
    ax.set_axis_off()
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Konversi ke array numpy
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf  = buf.reshape(h_px, w_px, 4)
    # ARGB → RGBA
    rgba = np.concatenate([buf[:, :, 1:], buf[:, :, :1]], axis=2)
    plt.close(fig)
    return rgba


# ──────────────────────────────────────────────────────────────
# Utilitas Streamlit helper
# ──────────────────────────────────────────────────────────────

def metric_card_html(label: str, value: str, icon: str = "📊", color: str = "#3b82f6") -> str:
    """
    HTML komponen kartu metrik kecil untuk Streamlit st.markdown().
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 1px solid {color}55;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 1.8rem;">{icon}</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: {color}; margin: 4px 0;">
            {value}
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8;">{label}</div>
    </div>
    """


def status_badge(label: str, status: str) -> str:
    """
    HTML badge status untuk Streamlit.
    status: 'ok', 'warn', 'error', 'info'
    """
    colors = {
        "ok"    : ("#22c55e", "#14532d"),
        "warn"  : ("#f59e0b", "#451a03"),
        "error" : ("#ef4444", "#450a0a"),
        "info"  : ("#3b82f6", "#1e3a5f"),
    }
    fg, bg = colors.get(status, colors["info"])
    icons  = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️"}
    icon   = icons.get(status, "ℹ️")
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:999px;font-size:0.8rem;border:1px solid {fg}44;">'
        f'{icon} {label}</span>'
    )
