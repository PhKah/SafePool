"""
tests/test_splits.py
====================
Bộ kiểm thử tự động (Data Contract & Post-Execution Verification)
đối soát kết quả phân chia tập dữ liệu sạch (Clean Group Split)
so với đặc tả trong Specs/dataset_specs.md và PM/plan.md.
"""

import os
import json
from collections import Counter
import pytest
from PIL import Image


MANIFEST_PATH = "data/splits/clean_group_split.json"
TRAIN_TXT = "data/splits/clean_train.txt"
VALID_TXT = "data/splits/clean_valid.txt"
TEST_TXT = "data/splits/clean_test.txt"


@pytest.fixture(scope="module")
def manifest_data():
    assert os.path.exists(MANIFEST_PATH), f"Không tìm thấy manifest tại {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def split_file_lists():
    res = {}
    for name, p in [("train", TRAIN_TXT), ("valid", VALID_TXT), ("test", TEST_TXT)]:
        assert os.path.exists(p), f"Không tìm thấy file danh sách {p}"
        with open(p, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        res[name] = lines
    return res


def test_conservation(manifest_data, split_file_lists):
    """
    Tiêu chí 1: Bảo toàn tổng thể (Conservation Check).
    - Tổng số ảnh = 12,365.
    - Tổng số box = 23,021.
    - Số lượng từng lớp: Class 0 = 13,231; Class 1 = 785; Class 2 = 9,005.
    """
    splits = manifest_data["splits"]
    
    total_imgs = sum(s["images_count"] for s in splits.values())
    assert total_imgs == 12365, f"Tổng số ảnh trong manifest ({total_imgs}) != 12,365"
    
    total_txt_imgs = sum(len(lines) for lines in split_file_lists.values())
    assert total_txt_imgs == 12365, f"Tổng số ảnh trong các file .txt ({total_txt_imgs}) != 12,365"
    
    total_boxes = sum(s["boxes_count"] for s in splits.values())
    assert total_boxes == 23021, f"Tổng số box ({total_boxes}) != 23,021"
    
    c0 = sum(s["class_distribution"]["0"] for s in splits.values())
    c1 = sum(s["class_distribution"]["1"] for s in splits.values())
    c2 = sum(s["class_distribution"]["2"] for s in splits.values())
    
    assert c0 == 13231, f"Class 0 count ({c0}) != 13,231"
    assert c1 == 785, f"Class 1 count ({c1}) != 785"
    assert c2 == 9005, f"Class 2 count ({c2}) != 9,005"


def test_zero_leakage_group_level(manifest_data):
    """
    Tiêu chí 2a: Triệt tiêu rò rỉ cấp độ nhóm (Level 2 Video Roots).
    Giao tập giữa các split phải bằng rỗng.
    """
    train_groups = set(manifest_data["splits"]["train"]["groups"])
    valid_groups = set(manifest_data["splits"]["valid"]["groups"])
    test_groups = set(manifest_data["splits"]["test"]["groups"])
    
    inter_tv = train_groups & valid_groups
    inter_tt = train_groups & test_groups
    inter_vt = valid_groups & test_groups
    
    assert len(inter_tv) == 0, f"Rò rỉ nhóm giữa Train và Valid: {inter_tv}"
    assert len(inter_tt) == 0, f"Rò rỉ nhóm giữa Train và Test: {inter_tt}"
    assert len(inter_vt) == 0, f"Rò rỉ nhóm giữa Valid và Test: {inter_vt}"


def test_zero_leakage_frame_prefix_level(split_file_lists):
    """
    Tiêu chí 2b: Triệt tiêu rò rỉ cấp độ tiền tố frame trước .rf. (Level 1).
    Toàn bộ 1,932 frame prefixes không được xuất hiện ở 2 split khác nhau.
    """
    prefixes_by_split = {}
    for s_name, paths in split_file_lists.items():
        pref = set(os.path.basename(p).split(".rf.")[0].lower() for p in paths)
        prefixes_by_split[s_name] = pref
        
    train_pref = prefixes_by_split["train"]
    valid_pref = prefixes_by_split["valid"]
    test_pref = prefixes_by_split["test"]
    
    assert len(train_pref & valid_pref) == 0, f"Rò rỉ frame prefix giữa Train và Valid: {train_pref & valid_pref}"
    assert len(train_pref & test_pref) == 0, f"Rò rỉ frame prefix giữa Train và Test: {train_pref & test_pref}"
    assert len(valid_pref & test_pref) == 0, f"Rò rỉ frame prefix giữa Valid và Test: {valid_pref & test_pref}"
    
    all_prefixes = train_pref | valid_pref | test_pref
    assert len(all_prefixes) == 1932, f"Tổng số tiền tố frame duy nhất ({len(all_prefixes)}) != 1,932"


def test_large_video_assignment(split_file_lists):
    """
    Tiêu chí 3: Video clip siêu lớn drowning1_mp4 (1,048 ảnh) phải nằm 100% trong tập Train.
    """
    d1_train = [p for p in split_file_lists["train"] if "drowning1_mp4" in os.path.basename(p).lower()]
    d1_valid = [p for p in split_file_lists["valid"] if "drowning1_mp4" in os.path.basename(p).lower()]
    d1_test = [p for p in split_file_lists["test"] if "drowning1_mp4" in os.path.basename(p).lower()]
    
    assert len(d1_train) == 1048, f"drowning1_mp4 trong Train ({len(d1_train)}) != 1,048"
    assert len(d1_valid) == 0, f"Phát hiện drowning1_mp4 rò rỉ vào Valid: {len(d1_valid)}"
    assert len(d1_test) == 0, f"Phát hiện drowning1_mp4 rò rỉ vào Test: {len(d1_test)}"


def test_minority_class_gate(manifest_data):
    """
    Tiêu chí 4: Ngưỡng bảo vệ lớp thiểu số cực đoan Class 1 (Person out of water).
    Cả Valid và Test phải có ít nhất >= 70 boxes.
    """
    val_c1 = manifest_data["splits"]["valid"]["class_distribution"]["1"]
    test_c1 = manifest_data["splits"]["test"]["class_distribution"]["1"]
    
    assert val_c1 >= 70, f"Lớp 1 trong Valid ({val_c1}) < 70"
    assert test_c1 >= 70, f"Lớp 1 trong Test ({test_c1}) < 70"


def test_split_ratios_tolerance(manifest_data):
    """
    Tiêu chí 5: Tỷ lệ chia tập nằm trong khoảng dung sai cho phép:
    Train: [67%, 73%]; Valid: [12%, 18%]; Test: [12%, 18%].
    """
    train_pct = manifest_data["splits"]["train"]["images_pct"]
    val_pct = manifest_data["splits"]["valid"]["images_pct"]
    test_pct = manifest_data["splits"]["test"]["images_pct"]
    
    assert 67.0 <= train_pct <= 73.0, f"Tỷ lệ Train ({train_pct}%) ngoài dải [67%, 73%]"
    assert 12.0 <= val_pct <= 18.0, f"Tỷ lệ Valid ({val_pct}%) ngoài dải [12%, 18%]"
    assert 12.0 <= test_pct <= 18.0, f"Tỷ lệ Test ({test_pct}%) ngoài dải [12%, 18%]"


def test_edge_cases_leading_dash(split_file_lists):
    """
    Tiêu chí 6: Các file bắt đầu bằng dấu '-' được xử lý an toàn và phân bổ đầy đủ.
    """
    all_paths = []
    for paths in split_file_lists.values():
        all_paths.extend(paths)
        
    dash_files = [p for p in all_paths if os.path.basename(p).startswith("-")]
    assert len(dash_files) == 116, f"Số file bắt đầu bằng '-' ({len(dash_files)}) != 116"
    
    # Kiểm tra tất cả các file này đều tồn tại trên đĩa
    for p in dash_files:
        assert os.path.exists(p), f"File có dấu '-' không tồn tại trên đĩa: {p}"


def test_physical_file_existence_and_resolution(split_file_lists):
    """
    Tiêu chí 7: Tồn tại vật lý & Kích thước ảnh 416x416.
    Kiểm tra 100% file tồn tại và kiểm tra ngẫu nhiên mẫu ảnh có đúng 416x416.
    """
    for s_name, paths in split_file_lists.items():
        for p in paths:
            assert os.path.exists(p), f"Tệp ảnh không tồn tại: {p}"
            lbl_p = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            assert os.path.exists(lbl_p), f"Tệp nhãn tương ứng không tồn tại: {lbl_p}"
            
        # Kiểm tra mẫu ảnh đầu tiên của từng split xem có đúng 416x416
        with Image.open(paths[0]) as img:
            assert img.size == (416, 416), f"Kích thước ảnh {paths[0]} ({img.size}) != (416, 416)"

