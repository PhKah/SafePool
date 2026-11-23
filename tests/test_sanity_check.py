"""
tests/test_sanity_check.py
==========================
WP-2.2 — Kiểm thử tự động Gate 2 (Overfit 1 batch nhỏ cho cả Stage 2 và Stage 1).
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch = pytest.importorskip("torch")

from scripts.sanity_check import run_stage1_sanity_check, run_stage2_sanity_check


def test_stage2_sanity_check_passes_gate2():
    """S2-00: 1 batch cân bằng 12 ROI thật -> Loss < 0.05, Recall 3 lớp & Macro F1 = 100%, FAR = 0."""
    if not Path("data/processed/train").is_dir():
        pytest.skip("Chưa sinh data/processed/train (chạy scripts/prepare_roi_dataset.py trước)")

    res = run_stage2_sanity_check(epochs=50, lr=3e-3, seed=42, device_pref="auto")
    m = res["metrics"]

    assert res["final_loss"] < 0.05, f"Stage 2 loss chưa hội tụ: {res['final_loss']}"
    assert res["final_loss"] < res["initial_loss"]
    assert m["drowning_recall"] == pytest.approx(1.0)
    assert all(r == pytest.approx(1.0) for r in m["recall"])
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["far"] == pytest.approx(0.0)


def test_stage1_sanity_check_passes_gate2():
    """S1-00: 1 batch 2 ảnh 416x416 thật -> Loss < 0.10, Person Recall >= 95%, mAP@50 >= 95%."""
    manifest = Path("data/stage1/manifest.json")
    if not manifest.is_file():
        pytest.skip("Chưa sinh data/stage1/manifest.json (chạy scripts/datasets/stage1/prepare_stage1_dataset.py trước)")

    res = run_stage1_sanity_check(manifest_path=manifest, epochs=80, lr=3e-3, seed=42, device_pref="auto")
    m = res["metrics"]

    assert res["final_loss"] < 0.10, f"Stage 1 loss chưa hội tụ: {res['final_loss']}"
    assert res["final_loss"] < res["initial_loss"]
    assert m["person_recall"] >= 0.95, f"Person Recall thấp: {m['person_recall']}"
    assert m["map50"] >= 0.95, f"mAP@50 thấp: {m['map50']}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
