"""Unit test WP-2.1. Chạy từ thư mục gốc repo:  python -m pytest tests/test_metrics_and_trainer.py -v

Phần metric chỉ cần NumPy; phần trainer tự skip nếu máy chưa cài PyTorch.
"""
import gc
import math

import numpy as np
import pytest

from src.metrics import (
    box_iou_xyxy,
    compute_classification_metrics,
    confusion_matrix,
    detection_metrics,
    metrics_from_confusion_matrix,
    safe_div,
)


# ----------------------------------------------------------------------------- metrics (Stage 2)
def test_safe_div_zero_denominator():
    assert safe_div(3, 0) == 0.0
    assert safe_div(3, 0, default=1.0) == 1.0
    out = safe_div(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert out.tolist() == [0.0, 0.5]


def test_confusion_matrix_matches_hand_calculation():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 2, 2, 2, 0, 2]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[3, 1, 0], [0, 1, 1], [1, 0, 3]]


def test_classification_metrics_match_hand_calculation():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 2, 2, 2, 0, 2]
    m = compute_classification_metrics(y_true, y_pred)
    assert m["recall"] == pytest.approx([0.75, 0.5, 0.75])
    assert m["precision"] == pytest.approx([0.75, 0.5, 0.75])
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["drowning_recall"] == pytest.approx(0.75)
    assert m["far"] == pytest.approx(1 / 6)  # FP(Drowning)=1 / 6 mẫu thật không phải Drowning
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["per_class"]["Person out of water"]["support"] == 2


def test_missing_class_in_batch_has_no_nan_or_inf():
    # Batch không có lớp 2 (Swimming) ở nhãn thật lẫn dự đoán
    m = compute_classification_metrics([0, 0, 1], [0, 1, 1])
    assert m["recall"][2] == 0.0 and m["precision"][2] == 0.0 and m["f1"][2] == 0.0
    assert m["macro_f1"] == pytest.approx(2 / 3)  # chỉ macro trên 2 lớp có mặt
    flat = [m["macro_f1"], m["far"], m["accuracy"], *m["recall"], *m["precision"], *m["f1"]]
    assert all(math.isfinite(v) for v in flat)


def test_batch_without_any_drowning_gives_zero_far_and_recall():
    m = compute_classification_metrics([1, 2, 2], [1, 2, 2])
    assert m["drowning_recall"] == 0.0 and m["far"] == 0.0
    # Toàn bộ dự đoán sai thành Drowning -> FAR = 1
    assert compute_classification_metrics([1, 2], [0, 0])["far"] == 1.0


def test_empty_input_does_not_crash():
    m = metrics_from_confusion_matrix(np.zeros((3, 3), dtype=int))
    assert m["macro_f1"] == 0.0 and m["accuracy"] == 0.0 and m["far"] == 0.0


def test_out_of_range_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 1], 3)


# ----------------------------------------------------------------------------- metrics (Stage 1)
def test_box_iou():
    iou = box_iou_xyxy(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert iou[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_detection_metrics_hand_calculation():
    gt = [{"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60]])}]
    pred = [{"boxes": np.array([[0, 0, 10, 10], [100, 100, 110, 110]]), "scores": np.array([0.9, 0.8])}]
    m = detection_metrics(pred, gt, iou_thr=0.5, score_thr=0.25)
    assert m["person_recall"] == pytest.approx(0.5)  # tìm được 1/2 người
    assert m["map50"] == pytest.approx(0.5)  # TP(0.9), FP(0.8): AP = 0.5 * 1.0
    assert (m["num_gt"], m["num_pred"]) == (2, 2)


def test_detection_metrics_perfect_and_empty():
    gt = [{"boxes": np.array([[0, 0, 10, 10]])}]
    perfect = [{"boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.7])}]
    assert detection_metrics(perfect, gt)["map50"] == pytest.approx(1.0)
    none = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}]
    m = detection_metrics(none, gt)
    assert m["person_recall"] == 0.0 and m["map50"] == 0.0
    # Không có GT nào: không chia cho 0
    m0 = detection_metrics(perfect, [{"boxes": np.zeros((0, 4))}])
    assert m0["person_recall"] == 0.0 and m0["map50"] == 0.0


def test_detection_score_threshold_affects_recall_not_ap():
    gt = [{"boxes": np.array([[0, 0, 10, 10]])}]
    pred = [{"boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.1])}]
    m = detection_metrics(pred, gt, score_thr=0.25)
    assert m["person_recall"] == 0.0 and m["map50"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- trainer (cần PyTorch)
def _torch_stack():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.engine.trainer import evaluate, get_device, set_seed, train_one_epoch

    return torch, nn, DataLoader, TensorDataset, evaluate, get_device, set_seed, train_one_epoch


def _make(torch, nn, DataLoader, TensorDataset, n=24, bs=8, shuffle=False):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = torch.arange(n) % 3  # đủ 3 lớp
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 3))
    return model, DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle)


def _live_tensors():
    torch = __import__("torch")
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))


def test_overfit_small_batch_no_nan():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    set_seed(42)
    device = get_device("cpu")
    model, loader = _make(torch, nn, DL, TD, n=12, bs=12)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    first = train_one_epoch(model, loader, crit, opt, device)["loss"]
    for _ in range(80):
        last = train_one_epoch(model, loader, crit, opt, device)["loss"]
    assert math.isfinite(last) and last < first and last < 0.05
    res = evaluate(model, loader, crit, device)
    assert res["drowning_recall"] == 1.0 and res["macro_f1"] == pytest.approx(1.0)
    assert math.isfinite(res["loss"])


def test_same_seed_gives_same_loss():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()

    def run():
        set_seed(123)
        model, loader = _make(torch, nn, DL, TD, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return [train_one_epoch(model, loader, nn.CrossEntropyLoss(), opt, torch.device("cpu"))["loss"] for _ in range(3)]

    assert run() == run()


def test_no_memory_growth_over_many_epochs():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    set_seed(0)
    device = get_device("auto")
    model, loader = _make(torch, nn, DL, TD)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_one_epoch(model, loader, crit, opt, device)  # warm-up (tạo state của optimizer)
    evaluate(model, loader, crit, device)
    tensors_before = _live_tensors()
    cuda_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0
    for _ in range(40):
        out = train_one_epoch(model, loader, crit, opt, device)
        assert isinstance(out["loss"], float)
        evaluate(model, loader, crit, device)
    assert _live_tensors() <= tensors_before + 2
    if device.type == "cuda":
        assert torch.cuda.memory_allocated() <= cuda_before + 1024 * 1024


def test_nan_loss_is_detected():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    model, loader = _make(torch, nn, DL, TD)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    nan_crit = lambda out, y: out.sum() * float("nan")  # noqa: E731
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, loader, nan_crit, opt, torch.device("cpu"))


def test_evaluate_accumulates_across_batches_with_missing_class_batches():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    # Dữ liệu sắp theo lớp -> mỗi batch (bs=8) chỉ chứa 1 lớp; kết quả cộng dồn vẫn phải đúng
    x = torch.randn(24, 3, 8, 8)
    y = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 3))
    res = evaluate(model, DL(TD(x, y), batch_size=8, shuffle=False), None, torch.device("cpu"))
    assert sum(res["support"]) == 24 and res["support"] == [8, 8, 8]
    assert res["loss"] is None
    assert all(math.isfinite(v) for v in res["recall"] + res["precision"] + res["f1"])

def test_move_to_device_handles_variable_length_box_targets():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    from src.engine.trainer import move_to_device

    # Giống Stage1Dataset: tuple các tensor [N_i, 5] với N_i khác nhau
    targets = (torch.zeros(2, 5), torch.ones(5, 5), torch.zeros(0, 5))
    moved = move_to_device(targets, torch.device("cpu"))
    assert isinstance(moved, tuple) and [t.shape[0] for t in moved] == [2, 5, 0]
    assert move_to_device({"a": [torch.ones(1)], "b": "x"}, torch.device("cpu"))["b"] == "x"


def test_train_one_epoch_with_tuple_targets_stage1_style():
    torch, nn, DL, TD, evaluate, get_device, set_seed, train_one_epoch = _torch_stack()
    set_seed(0)
    imgs = torch.randn(6, 3, 8, 8)
    boxes = [torch.rand(n, 5) for n in (1, 3, 2, 5, 1, 4)]  # số box mỗi ảnh khác nhau

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return imgs[i], boxes[i]

    loader = DL(DS(), batch_size=3, collate_fn=lambda b: (torch.stack([x[0] for x in b]), tuple(x[1] for x in b)))
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    seen = []

    def criterion(out, targets):  # loss giả: kiểm tra targets đã là tuple tensor trên đúng device
        assert isinstance(targets, tuple) and all(t.device == out.device for t in targets)
        seen.append(len(targets))
        return out.pow(2).mean()

    res = train_one_epoch(model, loader, criterion, opt, torch.device("cpu"))
    assert math.isfinite(res["loss"]) and res["num_samples"] == 6 and seen == [3, 3]