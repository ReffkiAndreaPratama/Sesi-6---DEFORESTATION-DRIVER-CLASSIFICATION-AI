"""
ForestWatch AI — tests/test_evaluation.py
Pengujian perhitungan metrik evaluasi model.
"""

import numpy as np
import pytest


NUM_CLASSES = 6
CLASS_NAMES = [
    "Pertanian Skala Kecil", "Pertanian Skala Besar",
    "Pembangunan Jalan", "Penebangan Selektif",
    "Pertambangan", "Kebakaran / Gangguan Alami",
]


# ── compute_metrics ───────────────────────────────────────────

def test_perfect_prediction():
    """Prediksi sempurna: accuracy=1, f1=1."""
    from src.evaluation import compute_metrics
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    m = compute_metrics(y, y, CLASS_NAMES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)


def test_random_prediction():
    """Metrik dihitung tanpa error meskipun prediksi acak."""
    from src.evaluation import compute_metrics
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, NUM_CLASSES, 200)
    y_pred = rng.integers(0, NUM_CLASSES, 200)
    m = compute_metrics(y_true, y_pred, CLASS_NAMES)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert "confusion_matrix" in m


def test_single_class_prediction():
    """Hanya satu kelas dalam y_true tidak crash."""
    from src.evaluation import compute_metrics
    y_true = np.zeros(50, dtype=int)
    y_pred = np.zeros(50, dtype=int)
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)


def test_empty_data():
    """Array kosong mengembalikan pesan error, tidak crash."""
    from src.evaluation import compute_metrics
    m = compute_metrics(np.array([]), np.array([]))
    assert "error" in m


def test_confusion_matrix_shape():
    """Confusion matrix berukuran n×n di mana n = jumlah kelas hadir."""
    from src.evaluation import compute_metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 1, 1, 0])
    m = compute_metrics(y_true, y_pred)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape[0] == cm.shape[1]
    assert cm.shape[0] == 3   # 3 kelas hadir


def test_per_class_metrics_length():
    """Jumlah metrik per kelas sesuai jumlah kelas hadir."""
    from src.evaluation import compute_metrics
    y_true = np.array([0, 1, 2, 3, 4, 5] * 10)
    y_pred = np.array([0, 1, 2, 3, 4, 5] * 10)
    m = compute_metrics(y_true, y_pred, CLASS_NAMES)
    assert len(m["precision_per_class"]) == NUM_CLASSES
    assert len(m["f1_per_class"]) == NUM_CLASSES


# ── build_comparison_table ────────────────────────────────────

def test_comparison_table():
    """Tabel perbandingan berisi semua eksperimen."""
    from src.evaluation import build_comparison_table
    results = {
        "S1 Only" : {"accuracy": 0.70, "precision_macro": 0.68,
                     "recall_macro": 0.69, "f1_macro": 0.685,
                     "num_samples": 50, "avg_inference_ms": 2.1},
        "S2 Only" : {"accuracy": 0.78, "precision_macro": 0.76,
                     "recall_macro": 0.77, "f1_macro": 0.765,
                     "num_samples": 50, "avg_inference_ms": 3.4},
        "Fused"   : {"accuracy": 0.82, "precision_macro": 0.81,
                     "recall_macro": 0.80, "f1_macro": 0.805,
                     "num_samples": 50, "avg_inference_ms": 4.0},
    }
    table = build_comparison_table(results)
    assert len(table) == 3
    names = [r["Eksperimen"] for r in table]
    assert "S1 Only" in names
    assert "Fused" in names


def test_comparison_table_with_error():
    """Eksperimen dengan error tetap muncul di tabel."""
    from src.evaluation import build_comparison_table
    results = {
        "Model A": {"accuracy": 0.75, "f1_macro": 0.73,
                    "precision_macro": 0.72, "recall_macro": 0.74,
                    "num_samples": 30, "avg_inference_ms": 2.0},
        "Model B": {"error": "Model tidak ditemukan."},
    }
    table = build_comparison_table(results)
    assert len(table) == 2


# ── error_analysis ────────────────────────────────────────────

def test_error_analysis_basic():
    """error_analysis mengembalikan struktur yang benar."""
    from src.evaluation import error_analysis
    y_true = [0, 1, 2, 0, 1, 2, 3, 3]
    y_pred = [0, 0, 2, 1, 1, 2, 3, 0]
    result = error_analysis(y_true, y_pred, CLASS_NAMES)
    assert "total_errors" in result
    assert "error_rate"   in result
    assert "most_confused_pairs" in result
    assert result["total_errors"] == 3  # 3 salah prediksi
    assert 0.0 <= result["error_rate"] <= 1.0


def test_error_analysis_perfect():
    """Tidak ada kesalahan: error_rate = 0."""
    from src.evaluation import error_analysis
    y = [0, 1, 2, 3]
    result = error_analysis(y, y, CLASS_NAMES)
    assert result["total_errors"] == 0
    assert result["error_rate"] == 0.0


# ── save/load evaluation results ─────────────────────────────

def test_save_load_eval_results(tmp_path):
    """Hasil evaluasi dapat disimpan ke JSON dan dimuat kembali."""
    from src.evaluation import save_evaluation_results, load_evaluation_results, compute_metrics
    y_true = np.array([0, 1, 2, 3, 4, 5] * 5)
    y_pred = np.array([0, 1, 2, 3, 4, 5] * 5)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    path = tmp_path / "eval_test.json"
    save_evaluation_results(metrics, path, experiment_name="TestExp")
    loaded = load_evaluation_results(path)

    assert loaded["experiment"] == "TestExp"
    assert "metrics" in loaded
    assert "accuracy" in loaded["metrics"]


def test_load_nonexistent_eval():
    from src.evaluation import load_evaluation_results
    with pytest.raises(FileNotFoundError):
        load_evaluation_results("/nonexistent/eval.json")
