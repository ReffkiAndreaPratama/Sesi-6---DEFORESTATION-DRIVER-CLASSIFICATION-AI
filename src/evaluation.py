"""
ForestWatch AI — evaluation.py
Evaluasi komprehensif model klasifikasi deforestasi.

Metrik:
  - Accuracy, Precision, Recall, F1-score (macro & per-kelas)
  - Confusion matrix
  - Classification report
  - Waktu inferensi rata-rata
  - Perbandingan antar model (tabel)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Evaluasi dari DataLoader
# ──────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    return_predictions: bool = True,
) -> Dict:
    """
    Mengevaluasi model pada DataLoader.

    Returns
    -------
    dict berisi:
        accuracy, precision_macro, recall_macro, f1_macro,
        confusion_matrix, classification_report (dict),
        y_true, y_pred, y_prob (jika return_predictions=True),
        avg_inference_ms
    """
    model.eval()
    all_labels  : List[int] = []
    all_preds   : List[int] = []
    all_probs   : List[List[float]] = []
    total_time  = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_np = labels.numpy()

            t0     = time.perf_counter()
            logits = model(inputs)
            total_time += time.perf_counter() - t0

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            total_samples += inputs.size(0)

    avg_inference_ms = (total_time / max(total_samples, 1)) * 1000

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred, class_names=class_names)
    metrics["avg_inference_ms"] = round(avg_inference_ms, 3)
    metrics["num_samples"]      = total_samples

    if return_predictions:
        metrics["y_true"]  = y_true.tolist()
        metrics["y_pred"]  = y_pred.tolist()
        metrics["y_prob"]  = all_probs

    return metrics


# ──────────────────────────────────────────────────────────────
# Metrik dari array
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Menghitung metrik dari array label dan prediksi.

    Returns
    -------
    dict berisi metrik lengkap.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    if len(y_true) == 0:
        return {"error": "Tidak ada sampel untuk dievaluasi."}

    present_classes = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    labels_arg      = present_classes if len(present_classes) > 1 else None

    accuracy = float(accuracy_score(y_true, y_pred))

    # Macro metrics (tidak terpengaruh ketidakseimbangan kelas)
    kw = dict(labels=labels_arg, average="macro", zero_division=0)
    precision_macro = float(precision_score(y_true, y_pred, **kw))
    recall_macro    = float(recall_score(y_true, y_pred, **kw))
    f1_macro        = float(f1_score(y_true, y_pred, **kw))

    # Per-kelas
    kw_pc = dict(labels=labels_arg, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, **kw_pc).tolist()
    recall_per_class    = recall_score(y_true, y_pred, **kw_pc).tolist()
    f1_per_class        = f1_score(y_true, y_pred, **kw_pc).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=labels_arg).tolist()

    # Classification report sebagai dict
    target_names = (
        [class_names[i] for i in present_classes]
        if class_names else None
    )
    report_dict = classification_report(
        y_true, y_pred,
        labels       = labels_arg,
        target_names = target_names,
        zero_division = 0,
        output_dict  = True,
    )

    return {
        "accuracy"            : accuracy,
        "precision_macro"     : precision_macro,
        "recall_macro"        : recall_macro,
        "f1_macro"            : f1_macro,
        "precision_per_class" : precision_per_class,
        "recall_per_class"    : recall_per_class,
        "f1_per_class"        : f1_per_class,
        "confusion_matrix"    : cm,
        "classification_report": report_dict,
        "present_classes"     : present_classes,
        "class_names"         : class_names,
    }


# ──────────────────────────────────────────────────────────────
# Tabel perbandingan model
# ──────────────────────────────────────────────────────────────

def build_comparison_table(
    results: Dict[str, Dict],
) -> List[Dict]:
    """
    Membangun tabel perbandingan dari beberapa hasil evaluasi.

    Parameters
    ----------
    results : dict
        {nama_eksperimen: dict_metrics}
        Contoh: {"S1 Only": {...}, "S2 Only": {...}, "Fused": {...}}

    Returns
    -------
    list[dict]  — setiap baris adalah satu eksperimen
    """
    rows = []
    for name, metrics in results.items():
        if "error" in metrics:
            rows.append({"Eksperimen": name, "Status": metrics["error"]})
            continue
        rows.append({
            "Eksperimen"        : name,
            "Accuracy"          : round(metrics.get("accuracy", 0), 4),
            "Precision (Macro)" : round(metrics.get("precision_macro", 0), 4),
            "Recall (Macro)"    : round(metrics.get("recall_macro", 0), 4),
            "F1 (Macro)"        : round(metrics.get("f1_macro", 0), 4),
            "Sampel"            : metrics.get("num_samples", "—"),
            "Inferensi (ms)"    : metrics.get("avg_inference_ms", "—"),
        })
    return rows


# ──────────────────────────────────────────────────────────────
# Analisis kesalahan klasifikasi
# ──────────────────────────────────────────────────────────────

def error_analysis(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    top_n: int = 10,
) -> Dict:
    """
    Merangkum pola kesalahan klasifikasi.

    Returns
    -------
    dict berisi:
        total_errors, error_rate,
        most_confused_pairs (pasangan kelas yang paling sering salah),
        per_class_error_rate
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    wrong_mask  = y_true_arr != y_pred_arr
    total       = len(y_true_arr)
    total_errors = int(wrong_mask.sum())
    error_rate   = total_errors / max(total, 1)

    # Pasangan (true, pred) yang paling sering salah
    wrong_pairs  = list(zip(y_true_arr[wrong_mask], y_pred_arr[wrong_mask]))
    pair_counts: Dict = {}
    for t, p in wrong_pairs:
        key = (int(t), int(p))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:top_n]
    confused = []
    for (t, p), count in top_pairs:
        true_name = class_names[t] if class_names and t < len(class_names) else str(t)
        pred_name = class_names[p] if class_names and p < len(class_names) else str(p)
        confused.append({
            "true_class" : true_name,
            "pred_class" : pred_name,
            "count"      : count,
        })

    # Error rate per kelas
    classes = sorted(set(y_true_arr.tolist()))
    per_class_error: List[Dict] = []
    for cls in classes:
        mask    = y_true_arr == cls
        n_cls   = int(mask.sum())
        n_err   = int((y_pred_arr[mask] != cls).sum())
        cname   = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        per_class_error.append({
            "class"      : cname,
            "n_samples"  : n_cls,
            "n_errors"   : n_err,
            "error_rate" : round(n_err / max(n_cls, 1), 4),
        })

    return {
        "total_errors"       : total_errors,
        "total_samples"      : total,
        "error_rate"         : round(error_rate, 4),
        "most_confused_pairs": confused,
        "per_class_error"    : per_class_error,
    }


# ──────────────────────────────────────────────────────────────
# Simpan / muat hasil evaluasi
# ──────────────────────────────────────────────────────────────

def save_evaluation_results(
    metrics: Dict,
    path: str | Path,
    experiment_name: str = "experiment",
) -> None:
    """Menyimpan hasil evaluasi ke JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment"  : experiment_name,
        "metrics"     : {
            k: v for k, v in metrics.items()
            if k not in ("y_true", "y_pred", "y_prob")
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Hasil evaluasi disimpan: %s", path)


def load_evaluation_results(path: str | Path) -> Dict:
    """Memuat hasil evaluasi dari JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File evaluasi tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
