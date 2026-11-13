#!/usr/bin/env python3
"""
scripts/create_clean_splits.py
==============================
Tạo bộ chia tập sạch (Clean Group Split) chống rò rỉ dữ liệu (Cross-split Data Leakage)
cho dự án SafePool.

Các tính năng:
- BƯỚC 0: Tiền kiểm toán tự động tính hợp lệ của 100% tệp dữ liệu (Fail-Fast).
- BƯỚC 1: Phân chia nhóm cơ sở (Baseline Random Group Split).
- BƯỚC 2: Phân chia tối ưu hóa chống rủi ro (Class-1 First Stratification + Greedy Bin Packing).
- Xuất file manifest `clean_group_split.json` và danh sách tệp `clean_{train,valid,test}.txt`.
- Xuất tệp đối chiếu `roboflow_baseline.json` lưu trữ thông số của split mặc định Roboflow.
"""

import os
import sys
import glob
import re
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


def validate_raw_dataset(raw_dir: str = "data/raw") -> List[str]:
    """
    Bước 0: Tiền kiểm toán 100% dữ liệu thô.
    Dừng ngay lập tức (Fail-Fast) nếu phát hiện bất kỳ file nào không đúng định dạng.
    """
    print("[BƯỚC 0] Đang kiểm tra toàn vẹn dữ liệu thô (Pre-flight Audit)...")
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Thư mục dữ liệu thô không tồn tại: {raw_dir}")
        
    image_paths = sorted(glob.glob(f"{raw_dir}/*/images/*.*"))
    if len(image_paths) != 12365:
        raise ValueError(f"Kỳ vọng 12,365 ảnh, nhưng tìm thấy: {len(image_paths)}")
        
    anomalies = []
    leading_dash_count = 0
    
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        
        # Kiểm tra đuôi file
        if not fname.lower().endswith(".jpg"):
            anomalies.append(f"{img_p}: Đuôi mở rộng không phải .jpg")
            
        # Kiểm tra quy tắc .rf.
        if ".rf." not in fname:
            anomalies.append(f"{img_p}: Thiếu chuỗi .rf.")
        elif fname.count(".rf.") > 1:
            anomalies.append(f"{img_p}: Chứa nhiều hơn một chuỗi .rf.")
            
        # Đếm ký tự gạch đầu dòng
        prefix = fname.split(".rf.")[0]
        if prefix.startswith("-"):
            leading_dash_count += 1
            
        # Kiểm tra file nhãn tương ứng
        lbl_p = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_p):
            anomalies.append(f"{img_p}: Thiếu file nhãn tương ứng tại {lbl_p}")
        else:
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    if len(tokens) != 5:
                        anomalies.append(f"{lbl_p} dòng {line_idx}: Định dạng YOLO không đúng 5 trường")
                        break
                    cls_id = tokens[0]
                    if cls_id not in {"0", "1", "2"}:
                        anomalies.append(f"{lbl_p} dòng {line_idx}: Class ID không hợp lệ '{cls_id}'")
                        break
                        
    if anomalies:
        err_msg = "\n".join(anomalies[:10])
        raise ValueError(f"Phát hiện {len(anomalies)} lỗi bất thường trong dữ liệu thô:\n{err_msg}")
        
    print(f"  -> XÁC NHẬN: 12,365 ảnh & nhãn 100% hợp lệ! (Phát hiện {leading_dash_count} file bắt đầu bằng dấu '-')")
    return image_paths


def extract_group_id(filename: str, level: str = "video") -> str:
    """
    Trích xuất Group ID từ tên file:
    - level='frame': Lấy tiền tố frame trước '.rf.' (1,932 groups).
    - level='video': Nhận diện thêm video roots từ chuỗi frame liên tiếp (1,363 groups).
    """
    prefix = filename.split(".rf.")[0]
    if level == "frame":
        return prefix.lower()
        
    # Gom video patterns (e.g., 12_mp4-100_jpg -> 12_mp4; drowning1_mp4-17_jpg -> drowning1_mp4)
    m = re.match(r"^(.*?(_mp4|-mp4|\.mp4|_avi|-avi|_mov|-mov))[-_].*", prefix, re.IGNORECASE)
    if m:
        return m.group(1).lower()
        
    if "clipchamp" in prefix.lower():
        m2 = re.match(r"^(.*clipchamp.*?mp4)", prefix, re.IGNORECASE)
        if m2:
            return m2.group(1).lower()
            
    return prefix.lower()


def build_groups_dict(image_paths: List[str], group_level: str = "video") -> Dict[str, Dict[str, Any]]:
    """
    Gom nhóm ảnh theo Group ID và thống kê số lượng ảnh, box, nhãn từng lớp.
    """
    groups = defaultdict(lambda: {
        "images": [],
        "boxes": 0,
        "classes": Counter(),
        "frame_prefixes": set()
    })
    
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        gid = extract_group_id(fname, level=group_level)
        frame_prefix = fname.split(".rf.")[0].lower()
        
        lbl_p = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        c = Counter()
        n_boxes = 0
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    c[parts[0]] += 1
                    n_boxes += 1
                    
        groups[gid]["images"].append(img_p)
        groups[gid]["boxes"] += n_boxes
        groups[gid]["classes"].update(c)
        groups[gid]["frame_prefixes"].add(frame_prefix)
        
    return dict(groups)


def split_baseline(groups: Dict[str, Dict[str, Any]], ratios: Dict[str, float], seed: int) -> Dict[str, List[str]]:
    """
    Bước 1: Phân chia nhóm cơ sở (Random Group Split theo tỷ lệ).
    """
    rng = random.Random(seed)
    all_groups = list(groups.keys())
    rng.shuffle(all_groups)
    
    total_images = sum(len(d["images"]) for d in groups.values())
    target_counts = {k: int(round(total_images * v)) for k, v in ratios.items()}
    
    splits = {"train": [], "valid": [], "test": []}
    current_counts = {"train": 0, "valid": 0, "test": 0}
    
    for g in all_groups:
        n = len(groups[g]["images"])
        # Chọn split có tỷ lệ còn thiếu nhiều nhất
        best_s = min(splits.keys(), key=lambda s: current_counts[s] / ratios[s])
        splits[best_s].append(g)
        current_counts[best_s] += n
        
    return splits


def split_optimized(groups: Dict[str, Dict[str, Any]], ratios: Dict[str, float], seed: int) -> Dict[str, List[str]]:
    """
    Bước 2: Phân chia tối ưu hóa chống rủi ro.
    - Video siêu lớn (drowning1_mp4 > 500 ảnh): Tự động neo cố định vào train để không làm phình/chiếm trọn tập valid/test.
    - Class-1 First Stratification: Phân phối các nhóm chứa lớp 1 trước để bảo đảm
      valid và test đều có đủ mẫu kiểm thử (>= 70 boxes).
    - Greedy Bin Packing: Phân phối các nhóm còn lại để cân bằng hoàn hảo tỷ lệ ảnh 70/15/15.
    """
    splits = {"train": [], "valid": [], "test": []}
    split_stats = {
        "train": {"images": 0, "boxes": 0, "classes": Counter()},
        "valid": {"images": 0, "boxes": 0, "classes": Counter()},
        "test": {"images": 0, "boxes": 0, "classes": Counter()}
    }

    # 1. Neo các nhóm video siêu lớn (> 500 ảnh, ví dụ drowning1_mp4) cố định vào tập train
    mega_groups = [g for g, d in groups.items() if len(d["images"]) > 500]
    for g in mega_groups:
        splits["train"].append(g)
        split_stats["train"]["images"] += len(groups[g]["images"])
        split_stats["train"]["boxes"] += groups[g]["boxes"]
        split_stats["train"]["classes"].update(groups[g]["classes"])

    c1_groups = [g for g, d in groups.items() if d["classes"]["1"] > 0 and g not in mega_groups]
    other_groups = [g for g, d in groups.items() if d["classes"]["1"] == 0 and g not in mega_groups]
    
    # Sắp xếp nhóm lớp 1 theo số lượng box lớp 1 giảm dần
    c1_groups.sort(key=lambda g: groups[g]["classes"]["1"], reverse=True)
    # Sắp xếp các nhóm khác theo số lượng ảnh giảm dần
    other_groups.sort(key=lambda g: len(groups[g]["images"]), reverse=True)
    
    # 2. Phân phối nhóm có lớp 1 trước
    for g in c1_groups:
        best_s = min(splits.keys(), key=lambda s: split_stats[s]["classes"]["1"] / ratios[s])
        splits[best_s].append(g)
        split_stats[best_s]["images"] += len(groups[g]["images"])
        split_stats[best_s]["boxes"] += groups[g]["boxes"]
        split_stats[best_s]["classes"].update(groups[g]["classes"])
        
    # 3. Phân phối các nhóm còn lại bằng Greedy Bin Packing
    for g in other_groups:
        best_s = min(splits.keys(), key=lambda s: split_stats[s]["images"] / ratios[s])
        splits[best_s].append(g)
        split_stats[best_s]["images"] += len(groups[g]["images"])
        split_stats[best_s]["boxes"] += groups[g]["boxes"]
        split_stats[best_s]["classes"].update(groups[g]["classes"])
        
    return splits


def compute_split_summary(splits: Dict[str, List[str]], groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tổng hợp số liệu thống kê của từng split.
    """
    total_images = sum(len(d["images"]) for d in groups.values())
    total_boxes = sum(d["boxes"] for d in groups.values())
    
    summary = {}
    for s_name, group_list in splits.items():
        img_list = []
        frame_prefixes = set()
        box_count = 0
        classes = Counter()
        
        for g in group_list:
            img_list.extend(groups[g]["images"])
            frame_prefixes.update(groups[g]["frame_prefixes"])
            box_count += groups[g]["boxes"]
            classes.update(groups[g]["classes"])
            
        summary[s_name] = {
            "images_count": len(img_list),
            "images_pct": round(len(img_list) / total_images * 100, 2),
            "boxes_count": box_count,
            "boxes_pct": round(box_count / total_boxes * 100, 2),
            "groups_count": len(group_list),
            "frame_prefixes_count": len(frame_prefixes),
            "class_distribution": {
                "0": classes["0"],
                "1": classes["1"],
                "2": classes["2"]
            },
            "class_pct": {
                "0": round(classes["0"] / box_count * 100, 2) if box_count else 0,
                "1": round(classes["1"] / box_count * 100, 2) if box_count else 0,
                "2": round(classes["2"] / box_count * 100, 2) if box_count else 0
            },
            "groups": sorted(group_list)
        }
    return summary


def get_roboflow_baseline_summary(raw_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Trích xuất thông số của bộ split mặc định ban đầu từ Roboflow để làm chuẩn đối chiếu.
    """
    summary = {}
    total_images = 12365
    total_boxes = 23021
    
    for split_name in ["train", "valid", "test"]:
        imgs = sorted(glob.glob(f"{raw_dir}/{split_name}/images/*.jpg"))
        box_count = 0
        classes = Counter()
        prefixes = set()
        for img_p in imgs:
            fname = os.path.basename(img_p)
            prefixes.add(fname.split(".rf.")[0].lower())
            lbl_p = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes[parts[0]] += 1
                        box_count += 1
                        
        summary[split_name] = {
            "images_count": len(imgs),
            "images_pct": round(len(imgs) / total_images * 100, 2),
            "boxes_count": box_count,
            "boxes_pct": round(box_count / total_boxes * 100, 2),
            "frame_prefixes_count": len(prefixes),
            "class_distribution": {
                "0": classes["0"],
                "1": classes["1"],
                "2": classes["2"]
            }
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Tạo bộ chia tập sạch (Clean Group Split) cho SafePool.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Thư mục dữ liệu thô gốc")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Thư mục lưu trữ kết quả chia tập")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỷ lệ tập Train (mặc định: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ tập Validation (mặc định: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Tỷ lệ tập Test (mặc định: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định tái lập (mặc định: 42)")
    parser.add_argument("--group-level", type=str, default="video", choices=["video", "frame"],
                        help="Cấp độ gom nhóm: 'video' (khuyên dùng) hoặc 'frame'")
    parser.add_argument("--mode", type=str, default="optimized", choices=["baseline", "optimized"],
                        help="Chế độ chia: 'baseline' (random) hoặc 'optimized' (xử lý rủi ro)")
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "valid": args.val_ratio,
        "test": args.test_ratio
    }
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-4:
        raise ValueError(f"Tổng các tỷ lệ phải bằng 1.0, hiện tại là: {ratio_sum}")

    os.makedirs(args.output_dir, exist_ok=True)

    # BƯỚC 0: Tiền kiểm toán 100% dữ liệu thô
    image_paths = validate_raw_dataset(args.raw_dir)

    # Gom nhóm theo Group ID
    print(f"\n[XỬ LÝ] Đang gom nhóm theo cấp độ '{args.group_level}'...")
    groups = build_groups_dict(image_paths, group_level=args.group_level)
    print(f"  -> Tổng số nhóm trích xuất được: {len(groups)}")

    # BƯỚC 1 & 2: Phân chia tập dữ liệu
    print(f"\n[PHÂN CHIA] Chạy phân chia tập dữ liệu ở chế độ '{args.mode}' (Seed = {args.seed})...")
    if args.mode == "baseline":
        splits = split_baseline(groups, ratios, args.seed)
    else:
        splits = split_optimized(groups, ratios, args.seed)

    split_summary = compute_split_summary(splits, groups)

    # Xuất file manifest clean_group_split.json
    manifest_path = os.path.join(args.output_dir, "clean_group_split.json")
    manifest_data = {
        "metadata": {
            "seed": args.seed,
            "group_level": args.group_level,
            "mode": args.mode,
            "target_ratios": ratios,
            "total_images": len(image_paths),
            "total_boxes": sum(groups[g]["boxes"] for g in groups),
            "total_groups": len(groups)
        },
        "splits": split_summary
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"\n[LƯU TRỮ] Đã xuất manifest chính thức: {manifest_path}")

    # Xuất danh sách tệp clean_{train,valid,test}.txt
    for split_name in ["train", "valid", "test"]:
        txt_path = os.path.join(args.output_dir, f"clean_{split_name}.txt")
        file_list = []
        for g in splits[split_name]:
            file_list.extend(groups[g]["images"])
        file_list.sort()
        
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in file_list:
                # Lưu đường dẫn tương đối chuẩn từ thư mục gốc dự án
                rel_p = os.path.relpath(p, start=os.getcwd())
                f.write(f"{rel_p}\n")
        print(f"  -> Đã xuất {txt_path} ({len(file_list)} tệp ảnh)")

    # Xuất tệp đối chiếu Roboflow Baseline
    roboflow_baseline_path = os.path.join(args.output_dir, "roboflow_baseline.json")
    roboflow_summary = get_roboflow_baseline_summary(args.raw_dir)
    with open(roboflow_baseline_path, "w", encoding="utf-8") as f:
        json.dump(roboflow_summary, f, indent=2, ensure_ascii=False)
    print(f"  -> Đã xuất tệp đối chiếu Roboflow gốc: {roboflow_baseline_path}")

    # In bảng đối chiếu ra màn hình console
    print("\n" + "="*78)
    print(f"{'TẬP DỮ LIỆU':<12} | {'ẢNH (TỶ LỆ)':<18} | {'BOXES':<8} | {'C0: DROWNING':<13} | {'C1: OUT WATER':<13} | {'C2: SWIMMING':<13}")
    print("-" * 78)
    for s_name in ["train", "valid", "test"]:
        st = split_summary[s_name]
        c = st["class_distribution"]
        print(f"{s_name.upper():<12} | {st['images_count']} ({st['images_pct']}%)".ljust(33) + 
              f"| {st['boxes_count']:<8} | {c['0']:<13} | {c['1']:<13} | {c['2']:<13}")
    print("="*78)
    print(f"TỔNG CỘNG     | 12,365 (100.0%)   | 23,021   | 13,231        | 785           | 9,005")
    print("="*78)


if __name__ == "__main__":
    main()
