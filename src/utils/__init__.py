"""Utilities for profiling Edge AI constraints and auxiliary detection diagnostics."""

from .edge_profiler import (
    STAGE1_BUDGET,
    STAGE2_BUDGET,
    benchmark_latency_fps,
    compute_flops,
    compute_small_box_recall,
    compute_stage1_slicing_metrics,
    count_parameters,
    measure_model_size_mb,
    profile_edge_model,
    targets_yolo_to_xyxy,
)

__all__ = [
    "STAGE1_BUDGET",
    "STAGE2_BUDGET",
    "benchmark_latency_fps",
    "compute_flops",
    "compute_small_box_recall",
    "compute_stage1_slicing_metrics",
    "count_parameters",
    "measure_model_size_mb",
    "profile_edge_model",
    "targets_yolo_to_xyxy",
]
