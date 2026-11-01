"""
====================================================================
Script: 02_extract_landmarks.py
Mục đích: Trích xuất tọa độ bàn tay (MediaPipe Hands) từ video VSL400,
          được tổ chức và xử lý phân cấp theo từng Part (Part_1 -> Part_7).

Đầu vào:
    data/raw/Part_{X}/split_{X}/{view}/*.mp4
    Metadata: data/raw/Part_{X}/split_{X}/{view}.json

Đầu ra:
    - File mảng landmark:
      data/processed/landmarks/Part_{X}/{view}/{video_id}_{signer_id}.npy
      shape: [num_frames, 126] (2 tay x 21 khớp x 3 tọa độ x, y, z)
    - File báo cáo tiến độ theo từng Part:
      reports/extraction_report_Part_{X}.csv
    - File báo cáo tổng hợp toàn bộ:
      reports/extraction_report_all.csv

Cơ chế Resume:
    - Xử lý độc lập theo từng Part.
    - Video được xem là đã hoàn thành khi đồng thời có file .npy hợp lệ
      và bản ghi trong file CSV của Part đó có status là 'ok' hoặc 'no_hand'.
    - Khi khởi động chế độ thường (không dry-run), script rollback số lượng
      video bằng số NUM_WORKERS của Part đó để đảm bảo không sót frame dở dang.

Cách sử dụng:
    # 1. Chạy riêng một Part cụ thể (ví dụ Part 1)
    python src/data_processing/02_extract_landmarks.py --part 1

    # 2. Chạy một Part với góc nhìn cụ thể (ví dụ front_view)
    python src/data_processing/02_extract_landmarks.py --part 1 --view front_view

    # 3. Chạy nhiều Part chỉ định (ví dụ Part 1, Part 2)
    python src/data_processing/02_extract_landmarks.py --part 1,2

    # 4. Chạy toàn bộ tất cả Part (tuần tự từ Part 1 đến Part 7)
    python src/data_processing/02_extract_landmarks.py

    # 5. Chạy thử nghiệm dry-run 20 video của Part 1
    python src/data_processing/02_extract_landmarks.py --part 1 --dry_run

    # 6. Demo trích xuất nhanh 1 video
    python src/data_processing/02_extract_landmarks.py --demo path/to/video.mp4
====================================================================
"""

import os
import sys
import warnings
from contextlib import contextmanager

# 1. Tắt toàn bộ Python UserWarning (như protobuf SymbolDatabase deprecation)
warnings.filterwarnings("ignore")

# 2. Cấu hình biến môi trường chặn triệt để log C++ của TensorFlow & MediaPipe/glog
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Tắt TF INFO, WARNING, ERROR
os.environ["GLOG_minloglevel"] = "3"          # 3 = FATAL only (tắt INFO, WARNING)
os.environ["GLOG_logtostderr"] = "0"          # KHÔNG xuất glog ra stderr
os.environ["GLOG_stderrthreshold"] = "3"      # Ngưỡng xuất stderr là FATAL
os.environ["ABSL_LOG_PREFIX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Tắt oneDNN floating-point warning


@contextmanager
def suppress_c_stderr():
    """
    Tạm thời chuyển hướng C-level stderr (file descriptor 2) sang os.devnull
    để dập tắt các log C++ (TFLite XNNPACK, MediaPipe absl, protobuf).
    """
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        old_stderr_fd = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
    except Exception:
        yield


# Import thư viện với c_stderr được dập tắt để tránh log khởi tạo ban đầu
with suppress_c_stderr():
    import cv2
    import mediapipe as mp
    import numpy as np

import json
import csv
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Fix encoding UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ==================================================================
# CẤU HÌNH HỆ THỐNG
# ==================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "landmarks"
REPORT_DIR   = PROJECT_ROOT / "reports"

VIEWS = ["front_view", "left_view", "right_view"]

MAX_HANDS     = 2
MIN_DETECTION = 0.5
MIN_TRACKING  = 0.5

NUM_WORKERS = 4
NUM_PARTS   = 7

# Các trạng thái được xem là đã hoàn thành
COMPLETED_STATUSES = {"ok", "no_hand"}

CSV_FIELDS = [
    "video_id", "signer_id", "gloss", "view", "part",
    "num_frames", "hand_det_rate", "status", "error_msg"
]
# ==================================================================


def get_part_report_path(part_idx: int) -> Path:
    """Trả về đường dẫn file CSV báo cáo của một Part cụ thể."""
    return REPORT_DIR / f"extraction_report_Part_{part_idx}.csv"


def get_all_report_path() -> Path:
    """Trả về đường dẫn file CSV báo cáo tổng hợp toàn bộ các Part."""
    return REPORT_DIR / "extraction_report_all.csv"


def task_key(data: dict) -> tuple:
    """Khóa định danh video để đối chiếu bản ghi CSV và file NPY."""
    return (
        str(data.get("part", "")),
        str(data.get("view", "")),
        str(data.get("video_id", "")),
        str(data.get("signer_id", "")),
    )


def read_part_report(part_idx: int) -> list:
    """Đọc file CSV báo cáo của một Part cụ thể."""
    report_file = get_part_report_path(part_idx)
    if not report_file.exists():
        return []

    try:
        with open(report_file, "r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        print(f"[WARN] Không đọc được CSV báo cáo Part_{part_idx}: {exc}")
        return []


def write_part_report(part_idx: int, rows: list) -> None:
    """Ghi lại file CSV báo cáo của Part thông qua file tạm an toàn."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = get_part_report_path(part_idx)
    temp_path = report_file.with_suffix(".csv.tmp")

    with open(temp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, report_file)


def rollback_last_results(tasks: list, report_rows: list, count: int) -> list:
    """
    Xóa N kết quả hoàn thành cuối cùng thuộc tập task của Part đang chạy:
    - Xóa file .npy tương ứng.
    - Xóa bản ghi tương ứng khỏi danh sách report_rows.
    """
    if count <= 0 or not report_rows:
        return report_rows

    task_by_key = {task_key(task): task for task in tasks}
    rollback_keys = []
    seen = set()

    for row in reversed(report_rows):
        key = task_key(row)
        if key in task_by_key and key not in seen:
            if row.get("status") in COMPLETED_STATUSES:
                rollback_keys.append(key)
                seen.add(key)
            if len(rollback_keys) >= count:
                break

    if not rollback_keys:
        return report_rows

    rollback_set = set(rollback_keys)

    for key in rollback_keys:
        task = task_by_key[key]
        output_path = Path(task["output_path"])
        try:
            if output_path.exists():
                output_path.unlink()
                print(f"[RESUME] Đã rollback file NPY: {output_path.name}")
        except OSError as exc:
            print(f"[WARN] Không xóa được {output_path}: {exc}")

    updated_rows = [
        row for row in report_rows
        if task_key(row) not in rollback_set
    ]

    print(f"[RESUME] Đã rollback {len(rollback_keys)}/{count} video cuối cùng để xử lý lại an toàn.")
    return updated_rows


def get_tasks_for_part(part_idx: int, views_to_process: list = None, dry_run: bool = False) -> list:
    """
    Duyệt dữ liệu của một Part cụ thể và tạo danh sách task.
    Cấu trúc đầu ra: data/processed/landmarks/Part_{part_idx}/{view}/
    """
    if views_to_process is None:
        views_to_process = VIEWS

    split_dir = RAW_DATA_DIR / f"Part_{part_idx}" / f"split_{part_idx}"
    if not split_dir.exists():
        print(f"[WARN] Không tìm thấy thư mục: {split_dir}")
        return []

    tasks = []

    for view in views_to_process:
        video_dir = split_dir / view
        json_path = split_dir / f"{view}.json"

        # Thư mục đầu ra được phân cấp rõ ràng theo Part và View
        output_dir = OUTPUT_DIR / f"Part_{part_idx}" / view
        output_dir.mkdir(parents=True, exist_ok=True)

        if not json_path.exists():
            print(f"[WARN] Không tìm thấy JSON: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Không đọc được JSON {json_path}: {exc}")
            continue

        for entry in metadata_list:
            video_id  = str(entry["video_id"])
            signer_id = str(entry["signer_id"])
            gloss     = entry.get("gloss", "")

            video_path  = video_dir / f"{video_id}.mp4"
            output_path = output_dir / f"{video_id}_{signer_id}.npy"

            if not video_path.exists():
                continue

            tasks.append({
                "video_path":  str(video_path),
                "output_path": str(output_path),
                "video_id":    video_id,
                "signer_id":   signer_id,
                "gloss":       gloss,
                "view":        view,
                "part":        part_idx,
            })

    if dry_run:
        tasks = tasks[:20]
        print(f"[DRY RUN] Giới hạn {len(tasks)} video đầu tiên của Part_{part_idx}.")

    return tasks


def get_pending_tasks(tasks: list, report_rows: list) -> list:
    """
    Chỉ bỏ qua video khi:
      1. File .npy tồn tại trên đĩa.
      2. File CSV có bản ghi thành công ('ok' hoặc 'no_hand') cho video đó.
    """
    successful_csv_keys = {
        task_key(row)
        for row in report_rows
        if row.get("status") in COMPLETED_STATUSES
    }

    pending = []
    skipped = 0

    for task in tasks:
        key = task_key(task)
        npy_exists = Path(task["output_path"]).is_file()
        csv_exists = key in successful_csv_keys

        if npy_exists and csv_exists:
            skipped += 1
        else:
            pending.append(task)

    print(f"[RESUME] Đã hoàn thành trước đó: {skipped} video")
    print(f"[RESUME] Cần xử lý tiếp / xử lý mới: {len(pending)} video")

    return pending


def remove_old_rows_for_pending(report_rows: list, pending_tasks: list) -> list:
    """Xóa các bản ghi cũ của những task sắp chạy lại để tránh trùng lặp dòng trong CSV."""
    pending_keys = {task_key(task) for task in pending_tasks}
    return [row for row in report_rows if task_key(row) not in pending_keys]


def extract_landmarks_from_video(task: dict) -> dict:
    """
    Xử lý 1 video bằng MediaPipe Hands và lưu mảng landmark shape [num_frames, 126].
    126 = 2 tay x 21 khớp x 3 tọa độ (x, y, z).
    Toàn bộ quá trình gọi MediaPipe được bọc trong suppress_c_stderr để tránh log rác.
    """
    video_path  = task["video_path"]
    output_path = task["output_path"]

    result = {
        "video_id":      task["video_id"],
        "signer_id":     task["signer_id"],
        "gloss":         task["gloss"],
        "view":          task["view"],
        "part":          task["part"],
        "num_frames":    0,
        "hand_det_rate": 0.0,
        "status":        "error",
        "error_msg":     "",
    }

    cap = None
    hands_detector = None

    try:
        with suppress_c_stderr():
            mp_hands = mp.solutions.hands
            hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                min_detection_confidence=MIN_DETECTION,
                min_tracking_confidence=MIN_TRACKING,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result["error_msg"] = "Không mở được video"
                return result

            all_frames_landmarks = []
            frame_count = 0
            hand_detected = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_result = hands_detector.process(frame_rgb)

                frame_landmarks = np.zeros(126, dtype=np.float32)

                if detection_result.multi_hand_landmarks:
                    hand_detected += 1
                    handedness_list = detection_result.multi_handedness or []

                    for hand_lm, hand_info in zip(
                        detection_result.multi_hand_landmarks,
                        handedness_list
                    ):
                        label = hand_info.classification[0].label
                        offset = 0 if label == "Left" else 63

                        for kp_idx, lm in enumerate(hand_lm.landmark):
                            base = offset + kp_idx * 3
                            frame_landmarks[base]     = lm.x
                            frame_landmarks[base + 1] = lm.y
                            frame_landmarks[base + 2] = lm.z

                all_frames_landmarks.append(frame_landmarks)
                frame_count += 1

        if frame_count == 0:
            result["error_msg"] = "Video rỗng (0 frame)"
            return result

        landmarks_array = np.asarray(all_frames_landmarks, dtype=np.float32)

        # Ghi ra file tạm rồi đổi tên để tránh file hỏng khi tiến trình bị ngắt đột ngột
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output_file.with_name(output_file.stem + ".tmp.npy")
        np.save(temp_output, landmarks_array)
        os.replace(temp_output, output_file)

        result["num_frames"]    = frame_count
        result["hand_det_rate"] = round(hand_detected / frame_count * 100, 2)
        result["status"]        = "ok" if hand_detected > 0 else "no_hand"

        if hand_detected == 0:
            result["error_msg"] = "Không phát hiện tay trong bất kỳ frame nào"

    except Exception as exc:
        result["error_msg"] = f"{type(exc).__name__}: {exc}"

    finally:
        with suppress_c_stderr():
            if cap is not None:
                cap.release()
            if hands_detector is not None:
                try:
                    hands_detector.close()
                except Exception:
                    pass

    return result


def process_single_part(part_idx: int, views: list = None, dry_run: bool = False, workers: int = NUM_WORKERS) -> dict:
    """
    Thực hiện trích xuất toàn bộ dữ liệu của một Part cụ thể.
    """
    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU XỬ LÝ: Part_{part_idx}")
    print(f"Thư mục lưu landmark: data/processed/landmarks/Part_{part_idx}/")
    print(f"File log tiến độ    : reports/extraction_report_Part_{part_idx}.csv")
    print(f"Số workers song song: {workers}")
    print("=" * 65)

    tasks = get_tasks_for_part(part_idx=part_idx, views_to_process=views, dry_run=dry_run)
    total_part_tasks = len(tasks)
    print(f"Tổng video của Part_{part_idx}: {total_part_tasks}")

    if total_part_tasks == 0:
        print(f"[INFO] Không tìm thấy video nào cho Part_{part_idx}.")
        return {"total": 0, "ok": 0, "no_hand": 0, "error": 0}

    report_rows = read_part_report(part_idx)

    # Rollback các kết quả cuối cùng nếu là chạy thường (không áp dụng với dry-run)
    if not dry_run:
        report_rows = rollback_last_results(
            tasks=tasks,
            report_rows=report_rows,
            count=workers,
        )

    pending_tasks = get_pending_tasks(tasks, report_rows)

    if not pending_tasks:
        write_part_report(part_idx, report_rows)
        print(f"[INFO] Toàn bộ video trong Part_{part_idx} đã được xử lý xong.")
        return {"total": total_part_tasks, "ok": total_part_tasks, "no_hand": 0, "error": 0}

    # Loại bỏ bản ghi cũ của các task sắp chạy lại để tránh trùng dòng
    report_rows = remove_old_rows_for_pending(report_rows, pending_tasks)
    write_part_report(part_idx, report_rows)

    stats = {"ok": 0, "no_hand": 0, "error": 0}
    start_time = time.time()

    part_report_file = get_part_report_path(part_idx)

    with open(part_report_file, "a", newline="", encoding="utf-8") as log_file:
        csv_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(extract_landmarks_from_video, task): task
                for task in pending_tasks
            }

            with tqdm(
                total=len(pending_tasks),
                desc=f"Part_{part_idx}",
                unit="video"
            ) as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "video_id":      task["video_id"],
                            "signer_id":     task["signer_id"],
                            "gloss":         task["gloss"],
                            "view":          task["view"],
                            "part":          task["part"],
                            "num_frames":    0,
                            "hand_det_rate": 0.0,
                            "status":        "error",
                            "error_msg":     f"Worker error: {type(exc).__name__}: {exc}",
                        }

                    csv_writer.writerow(result)
                    log_file.flush()

                    stats[result["status"]] += 1
                    pbar.update(1)

                    if result["status"] == "error":
                        pbar.write(
                            f"[ERROR] Part_{part_idx}/{result['view']}/{result['video_id']}: {result['error_msg']}"
                        )

    elapsed = time.time() - start_time
    total = sum(stats.values())

    print("\n" + "-" * 65)
    print(f"KẾT QUẢ PART_{part_idx}:")
    print(f"  - Đã xử lý mới     : {total}")
    print(f"  - Thành công (ok)  : {stats['ok']}")
    print(f"  - Không có tay     : {stats['no_hand']}")
    print(f"  - Lỗi              : {stats['error']}")
    if elapsed > 0:
        print(f"  - Thời gian chạy   : {elapsed / 60:.1f} phút")
        print(f"  - Tốc độ trung bình: {total / elapsed:.2f} video/giây")
    print(f"  - Báo cáo chi tiết : {part_report_file}")
    print("-" * 65)

    return stats


def merge_all_reports() -> None:
    """Gộp các file CSV báo cáo của từng Part thành reports/extraction_report_all.csv."""
    all_rows = []
    seen_keys = set()

    for p in range(1, NUM_PARTS + 1):
        rows = read_part_report(p)
        for r in rows:
            k = task_key(r)
            if k not in seen_keys:
                all_rows.append(r)
                seen_keys.add(k)

    if not all_rows:
        return

    all_report_file = get_all_report_path()
    temp_path = all_report_file.with_suffix(".csv.tmp")

    with open(temp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    os.replace(temp_path, all_report_file)
    print(f"[INFO] Đã cập nhật file báo cáo tổng hợp: {all_report_file} ({len(all_rows)} bản ghi)")


def demo_single_video(video_path: str) -> None:
    """Chế độ demo trích xuất nhanh 1 video mẫu để kiểm tra."""
    print(f"\nDemo trích xuất landmark từ: {video_path}")

    task = {
        "video_path":  str(video_path),
        "output_path": str(PROJECT_ROOT / "demo_output.npy"),
        "video_id":    "demo",
        "signer_id":   "000",
        "gloss":       "Unknown",
        "view":        "front_view",
        "part":        0,
    }

    result = extract_landmarks_from_video(task)

    print(f"Trạng thái         : {result['status']}")
    print(f"Số frame xử lý     : {result['num_frames']}")
    print(f"Tỉ lệ phát hiện tay: {result['hand_det_rate']}%")

    if result["status"] in COMPLETED_STATUSES:
        landmarks = np.load(task["output_path"])
        print(f"Shape mảng landmark: {landmarks.shape}")
        print(f"Giá trị min/max    : {landmarks.min():.4f} / {landmarks.max():.4f}")
        print(f"Số frame zero      : {(landmarks.sum(axis=1) == 0).sum()}")
    else:
        print(f"Lỗi: {result['error_msg']}")


def parse_parts_arg(part_arg: str) -> list:
    """
    Phân tích chuỗi tham số --part thành danh sách các số nguyên.
    Chấp nhận: '1', 'Part_1', '1,2,3', '1 2 3'.
    """
    if not part_arg:
        return list(range(1, NUM_PARTS + 1))

    cleaned = part_arg.replace("Part_", "").replace("part_", "")
    parts = []
    for item in cleaned.replace(" ", ",").split(","):
        item = item.strip()
        if item.isdigit():
            val = int(item)
            if 1 <= val <= NUM_PARTS:
                parts.append(val)
            else:
                print(f"[WARN] Part {val} không hợp lệ (hệ thống có từ Part 1 đến Part {NUM_PARTS}).")

    return sorted(list(set(parts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất landmark MediaPipe Hands theo từng Part cho bộ dữ liệu VSL400"
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Chỉ định Part cần xử lý (ví dụ: '1', 'Part_1', hoặc '1,2,3'). Mặc định: lần lượt Part 1 -> Part 7"
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        choices=["front_view", "left_view", "right_view"],
        help="Chỉ xử lý một góc nhìn cụ thể (mặc định: cả 3 view)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Số worker song song (mặc định: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Chỉ xử lý thử 20 video đầu tiên của mỗi Part để kiểm tra"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Đường dẫn đến 1 video cụ thể để chạy demo"
    )

    args = parser.parse_args()

    if args.demo:
        demo_single_video(args.demo)
    else:
        parts_to_run = parse_parts_arg(args.part)
        views_to_run = [args.view] if args.view else None

        print("=" * 65)
        print("HỆ THỐNG TRÍCH XUẤT LANDMARK VSL400 (THEO TỪNG PART)")
        print(f"Danh sách Part sẽ chạy: {parts_to_run}")
        print(f"Góc nhìn (views)       : {views_to_run if views_to_run else 'Tất cả 3 views'}")
        print(f"Số workers             : {args.workers}")
        print(f"Chế độ                 : {'DRY RUN (20 video/part)' if args.dry_run else 'CHẠY ĐẦY ĐỦ'}")
        print("=" * 65)

        for p in parts_to_run:
            process_single_part(
                part_idx=p,
                views=views_to_run,
                dry_run=args.dry_run,
                workers=args.workers
            )

        # Gộp báo cáo tổng hợp sau khi xử lý xong các Part
        merge_all_reports()
