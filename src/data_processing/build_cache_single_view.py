"""
========================================================================================
TẬP TIN ĐỘC LẬP TẠO BỘ NHỚ ĐỆM RAM 450D (BUILD CACHE SINGLE-VIEW)
HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU VIỆT NAM (VSL400)
========================================================================================
Mục tiêu:
- Độc lập hoàn toàn khỏi Notebook, tránh chiếm dụng bộ nhớ hay gây lỗi trong giao diện.
- Nạp 3 tập CSV (train_front.csv, val_front.csv, test_front.csv).
- Thực thi pipeline tiền xử lý chuẩn hóa 450D:
    1. Điền khuyết zero-frame (impute_zero_frames) trên 3 kênh độc lập (RH, LH, Pose).
    2. Nội suy thời gian (resample_sequence) về cố định T=64 frames.
    3. Chuẩn hóa không gian (normalize_spatial): Wrist-centering bàn tay + Chest-centering vai.
    4. Đạo hàm vận tốc động học (compute_velocity_features): ghép [P || Delta_P] -> 450D.
    5. Khử triệt để 100% NaN (Zero-NaN Guarantee).
- Lưu thành các tệp nhị phân siêu tốc .npy trong data/processed/cache/:
    - X_train_front_450D.npy, y_train_front.npy
    - X_val_front_450D.npy,   y_val_front.npy
    - X_test_front_450D.npy,  y_test_front.npy
========================================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Tự động định vị thư mục gốc dự án
def get_project_root() -> Path:
    cur = Path.cwd()
    if (cur / "data" / "processed").exists():
        return cur
    for parent in cur.parents:
        if (parent / "data" / "processed").exists():
            return parent
    # Fallback
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent

PROJECT_ROOT = get_project_root()
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 64
FEATURE_DIM = 450

def impute_zero_frames(sequence: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính điền khuyết zero-frame trên 3 kênh độc lập."""
    T, D = sequence.shape
    out = sequence.copy()
    for start, end in [(0, 63), (63, 126), (126, 225)]:
        sub = out[:, start:end]
        zero_mask = np.all(sub == 0, axis=1)
        if np.any(zero_mask) and not np.all(zero_mask):
            valid = np.where(~zero_mask)[0]
            all_idx = np.arange(T)
            for d in range(end - start):
                y = sub[valid, d]
                f = interp1d(valid, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
                out[:, start + d] = f(all_idx)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)

def resample_sequence(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Nội suy đưa chuỗi về cố định T=64 khung hình."""
    T, D = sequence.shape
    if T == target_frames:
        return sequence.astype(np.float32)
    orig_idx = np.linspace(0, 1, T)
    target_idx = np.linspace(0, 1, target_frames)
    f = interp1d(orig_idx, sequence, axis=0, kind="linear")
    return np.nan_to_num(f(target_idx), nan=0.0).astype(np.float32)

def normalize_spatial(sequence: np.ndarray) -> np.ndarray:
    """Chuẩn hóa Wrist-Centering cho 2 bàn tay và Chest-Centering vai (Khử triệt để NaN)."""
    out = sequence.copy()
    T = out.shape[0]
    
    # 1. Bàn tay phải: Trừ tọa độ cổ tay (khớp 0)
    rh = out[:, :63].reshape(T, 21, 3)
    rh[:, 1:, :] -= rh[:, 0:1, :]
    out[:, :63] = rh.reshape(T, 63)
    
    # 2. Bàn tay trái: Trừ tọa độ cổ tay (khớp 0)
    lh = out[:, 63:126].reshape(T, 21, 3)
    lh[:, 1:, :] -= lh[:, 0:1, :]
    out[:, 63:126] = lh.reshape(T, 63)
    
    # 3. Pose: Chest-centering & co giãn theo độ rộng hai vai (Chống chia 0)
    pose = out[:, 126:225].reshape(T, 33, 3)
    l_sh = pose[:, 11:12, :]
    r_sh = pose[:, 12:13, :]
    chest = (l_sh + r_sh) / 2.0
    width = np.linalg.norm(l_sh - r_sh, axis=-1, keepdims=True)
    width = np.where(np.isnan(width) | (width < 1e-4), 1.0, width)
    pose = (pose - chest) / width
    out[:, 126:225] = pose.reshape(T, 99)
    
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def compute_velocity_features(sequence: np.ndarray) -> np.ndarray:
    """Ghép nối vector tọa độ và vận tốc đạo hàm [P || Delta_P] -> 450D."""
    vel = np.zeros_like(sequence)
    vel[1:] = sequence[1:] - sequence[:-1]
    res = np.concatenate([sequence, vel], axis=-1)
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def preprocess_single(rel_path: str) -> np.ndarray:
    """Xử lý đơn lẻ 1 tệp landmark .npy thành tensor [64, 450]."""
    full_path = PROJECT_ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
    seq = np.load(full_path)
    seq = impute_zero_frames(seq)
    seq = resample_sequence(seq, target_frames=TARGET_FRAMES)
    seq = normalize_spatial(seq)
    seq_450 = compute_velocity_features(seq)
    return seq_450

def process_and_save_split(df: pd.DataFrame, split_name: str):
    """Xử lý toàn bộ phân vùng và ghi tệp cache .npy xuống đĩa."""
    N = len(df)
    out_X_path = CACHE_DIR / f"X_{split_name}_front_450D.npy"
    out_y_path = CACHE_DIR / f"y_{split_name}_front.npy"
    
    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU TẠO CACHE CHO TẬP: {split_name.upper()} ({N:,} mẫu)")
    print(f"Đích lưu: {out_X_path.name}")
    print("=" * 70)
    
    # Cấp phát trước mảng NumPy để tối ưu hóa bộ nhớ
    X_arr = np.zeros((N, TARGET_FRAMES, FEATURE_DIM), dtype=np.float32)
    y_arr = np.array(df["label"].values, dtype=np.int32)
    
    t0 = time.time()
    t_last = t0
    
    for i, (_, row) in enumerate(df.iterrows()):
        X_arr[i] = preprocess_single(row["holistic_npy_path"])
        
        # Báo cáo tiến độ mỗi 2.000 mẫu
        if (i + 1) % 2000 == 0 or (i + 1) == N:
            now = time.time()
            elapsed = now - t0
            speed = (i + 1) / elapsed
            remaining = (N - (i + 1)) / speed if speed > 0 else 0
            print(f"  [{split_name.upper()}] {i + 1:6,d} / {N:,d} ({((i + 1)/N)*100:5.1f}%) | Tốc độ: {speed:5.1f} mẫu/s | Còn lại: {remaining:4.1f}s")
            
    # Kiểm toán rà soát NaN
    nan_count = int(np.isnan(X_arr).sum())
    print(f"  Kiểm toán dữ liệu: {nan_count} giá trị NaN phát hiện.")
    assert nan_count == 0, f"LỖI NGHIÊM TRỌNG: Còn NaN trong {split_name}!"
    
    # Lưu tệp cache
    np.save(out_X_path, X_arr)
    np.save(out_y_path, y_arr)
    
    size_mb = out_X_path.stat().st_size / (1024 ** 2)
    print(f"  THÀNH CÔNG: Đã lưu {out_X_path.name} ({size_mb:.1f} MB) và {out_y_path.name} trong {time.time() - t0:.1f} giây.")

def main():
    print("Khởi chạy quy trình tạo bộ nhớ đệm RAM độc lập cho VSL400 Front-View...")
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Splits Dir   : {SPLITS_DIR}")
    print(f"Cache Dir    : {CACHE_DIR}")
    
    train_df = pd.read_csv(SPLITS_DIR / "train_front.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_front.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_front.csv")
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"Tổng số mẫu cần xử lý: {total_samples:,} mẫu (Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,})")
    
    t_global = time.time()
    
    process_and_save_split(val_df, "val")
    process_and_save_split(test_df, "test")
    process_and_save_split(train_df, "train")
    
    total_elapsed = time.time() - t_global
    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH 100% QUÁ TRÌNH TẠO CACHE CHO CẢ 3 TẬP TRAIN, VAL, TEST!")
    print(f"Tổng thời gian: {total_elapsed / 60:.2f} phút.")
    print(f"Thư mục chứa cache: {CACHE_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

