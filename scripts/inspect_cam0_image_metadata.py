#!/usr/bin/env python3
"""检查 mav0 数据集中 cam0 图像存储格式 (灰度 / 彩色) 的元数据脚本。

逐帧读取 cam0/data.csv 中的文件名，用 Pillow 惰性打开图像文件头（不解码像素），
提取 format / mode / 尺寸 / 位深度 / 文件大小，汇总 (format, mode, 尺寸) 组合
分布，并给出 "全部灰度 / 全部彩色 / 混合" 的结论。

用法:
    python3 inspect_cam0_image_metadata.py [MAV0_DIR] [--max-detail N]
默认 MAV0_DIR 为 ~/vio_ws/mav0
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Pillow mode 到总位深度的映射 (PNG 每通道 8/16-bit)
BIT_DEPTH_BY_MODE = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "I": 8,
    "I;16": 16,
    "I;16B": 16,
    "I;16N": 16,
    "RGB": 24,
    "RGBA": 32,
    "RGBX": 32,
    "YCbCr": 24,
    "LAB": 24,
    "HSV": 24,
    "CMYK": 32,
    "F": 32,
}

GRAYSCALE_MODES = frozenset({"1", "L", "LA", "I", "I;16", "I;16B", "I;16N"})
COLOR_MODES = frozenset({"RGB", "RGBA", "RGBX", "YCbCr", "LAB", "HSV", "CMYK"})

IMAGE_KIND_GRAYSCALE = "灰度"
IMAGE_KIND_COLOR = "彩色"
IMAGE_KIND_GRAYSCALE_PALETTE = "灰度调色板"
IMAGE_KIND_COLOR_PALETTE = "彩色调色板"
IMAGE_KIND_OTHER = "其他"

DEFAULT_MAV0_DIR = Path.home() / "vio_ws" / "mav0"
DEFAULT_MAX_DETAIL = 5


@dataclass(frozen=True)
class ImageMeta:
    timestamp_ns: int
    path: Path
    format: str
    mode: str
    width: int
    height: int
    bit_depth: int
    kind: str
    file_bytes: int


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mav0_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_MAV0_DIR,
        help="mav0 数据集根目录 (默认 ~/vio_ws/mav0)",
    )
    parser.add_argument(
        "--max-detail",
        type=int,
        default=DEFAULT_MAX_DETAIL,
        help="逐图详情最大打印帧数, 汇总统计始终覆盖全部图像",
    )
    return parser.parse_args()


def load_image_records(mav0_dir: Path):
    """解析 cam0/data.csv, 返回 (timestamp_ns, 图像路径) 列表; 文件缺失时抛异常。"""
    csv_path = mav0_dir / "cam0" / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到 cam0/data.csv: {csv_path}")
    records = []
    for raw_line in csv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        if len(fields) < 2:
            continue
        try:
            timestamp_ns = int(fields[0])
        except ValueError:
            continue
        records.append((timestamp_ns, mav0_dir / "cam0" / "data" / fields[1]))
    return records


def classify_kind(mode: str, image: Image.Image) -> str:
    """根据 Pillow mode 判断灰度/彩色; 调色板模式需逐条检查调色板是否 r==g==b。"""
    if mode in GRAYSCALE_MODES:
        return IMAGE_KIND_GRAYSCALE
    if mode in COLOR_MODES:
        return IMAGE_KIND_COLOR
    if mode == "P":
        palette = image.getpalette()
        if palette is None:
            return IMAGE_KIND_OTHER
        for index in range(0, len(palette), 3):
            red, green, blue = palette[index : index + 3]
            if red != green or green != blue:
                return IMAGE_KIND_COLOR_PALETTE
        return IMAGE_KIND_GRAYSCALE_PALETTE
    return IMAGE_KIND_OTHER


def inspect_image(timestamp_ns: int, path: Path) -> ImageMeta:
    """惰性打开图像头提取元数据; 不解码像素, 开销与文件大小无关。"""
    with Image.open(path) as image:
        format_ = image.format
        mode = image.mode
        width, height = image.size
        bit_depth = BIT_DEPTH_BY_MODE.get(mode, 0)
        kind = classify_kind(mode, image)
    return ImageMeta(
        timestamp_ns=timestamp_ns,
        path=path,
        format=format_,
        mode=mode,
        width=width,
        height=height,
        bit_depth=bit_depth,
        kind=kind,
        file_bytes=path.stat().st_size,
    )


def print_summary(metas, missing_paths, corrupt_entries, max_detail):
    total = len(metas) + len(missing_paths) + len(corrupt_entries)
    print(f"[INFO] 数据集: {metas[0].path.parents[2] if metas else DEFAULT_MAV0_DIR}")
    print(f"[INFO] 图像总数: {total}, 正常: {len(metas)}, "
          f"缺失: {len(missing_paths)}, 损坏: {len(corrupt_entries)}")
    if corrupt_entries:
        for timestamp_ns, path in corrupt_entries:
            print(f"[ERROR] 无法解码: {timestamp_ns} {path}")

    combination_counts = Counter(
        (meta.format, meta.mode, meta.width, meta.height) for meta in metas
    )
    print(f"\n(format, mode, 宽, 高) 组合统计 (共 {len(combination_counts)} 种):")
    for (format_, mode, width, height), count in combination_counts.most_common():
        bit_depth = BIT_DEPTH_BY_MODE.get(mode, 0)
        print(f"  {format_:<5} mode={mode:<6} {width}x{height}  总位深 {bit_depth:>2}  "
              f"帧数 {count}")

    mode_counts = Counter(meta.mode for meta in metas)
    print("\n各 mode 帧数占比:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode:<6} {count} 帧 ({count / total * 100:.1f}%)")

    print(f"\n前 {max_detail} 帧逐图详情:")
    for meta in metas[:max_detail]:
        print(
            f"  ts={meta.timestamp_ns}  {meta.path.name}  format={meta.format}  "
            f"mode={meta.mode}  {meta.width}x{meta.height}  位深 {meta.bit_depth}  "
            f"{meta.kind}  {meta.file_bytes} B"
        )
    if max_detail < len(metas):
        print(f"  ... (其余 {len(metas) - max_detail} 帧从略)")


def print_verdict(metas):
    kind_counts = Counter(meta.kind for meta in metas)
    grayscale_only = all(meta.kind in (IMAGE_KIND_GRAYSCALE, IMAGE_KIND_GRAYSCALE_PALETTE) for meta in metas)
    color_only = all(meta.kind in (IMAGE_KIND_COLOR, IMAGE_KIND_COLOR_PALETTE) for meta in metas)
    if grayscale_only:
        verdict = "全部为灰度图像"
    elif color_only:
        verdict = "全部为彩色图像"
    else:
        verdict = "灰度/彩色混合"
    print(f"\n结论: cam0 图像存储格式为 {verdict}")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count} 帧")


def main():
    args = parse_arguments()
    if not args.mav0_dir.is_dir():
        print(f"[ERROR] mav0 目录不存在: {args.mav0_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_detail < 0:
        print("[ERROR] --max-detail 不能为负数", file=sys.stderr)
        sys.exit(1)

    records = load_image_records(args.mav0_dir)
    if not records:
        print(f"[ERROR] cam0/data.csv 中没有有效记录: {args.mav0_dir}", file=sys.stderr)
        sys.exit(1)

    metas = []
    missing_paths = []
    corrupt_entries = []
    for timestamp_ns, path in records:
        if not path.is_file():
            missing_paths.append((timestamp_ns, path))
            continue
        try:
            metas.append(inspect_image(timestamp_ns, path))
        except (UnidentifiedImageError, OSError) as error:
            corrupt_entries.append((timestamp_ns, path, str(error)))

    print_summary(metas, missing_paths, corrupt_entries, args.max_detail)
    print_verdict(metas)


if __name__ == "__main__":
    main()
