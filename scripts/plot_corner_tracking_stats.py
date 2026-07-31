#!/usr/bin/env python3
"""绘制多次实验中 msckf 逐帧视觉任务合计耗时随帧数变化的对比曲线。

数据来源: msckf 在 ENABLE_TIMER=1 且 ENABLE_TIME_LOGGER=1 时输出的
~/vio_ws/CornerTrackingStats.csv (后续实验自动命名为
CornerTrackingStats1.csv, CornerTrackingStats2.csv, ...)。

用法:
  plot_corner_tracking_stats.py [数据目录] [-o 输出图片路径]
  数据目录默认为 ~/vio_ws; 不指定 -o 时弹窗显示图像。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

STATS_FILE_STEM = "CornerTrackingStats"
STATS_FILE_SUFFIX = ".csv"

# 经过色觉缺陷安全性验证的分类色板 (固定顺序分配, 不循环复用);
# 超出 8 条曲线的实验以灰色绘制并在图例中标注
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 蓝
    "#008300",  # 绿
    "#e87ba4",  # 洋红
    "#eda100",  # 黄
    "#1baf7a",  # 青绿
    "#eb6834",  # 橙
    "#4a3aa7",  # 紫
    "#e34948",  # 红
]
OVERFLOW_COLOR = "#898781"

SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

LINE_WIDTH = 2.0


def experiment_number_of(path: Path) -> int:
    """CornerTrackingStats.csv → 1, CornerTrackingStats1.csv → 2, 依此类推。"""
    match = re.fullmatch(
        re.escape(STATS_FILE_STEM) + r"(\d*)", path.stem
    )
    if match is None:
        raise ValueError(f"文件名不符合约定: {path.name}")
    return 1 if match.group(1) == "" else int(match.group(1)) + 1


def find_stats_files(data_directory: Path) -> list[Path]:
    files = [
        path
        for path in data_directory.glob(f"{STATS_FILE_STEM}*{STATS_FILE_SUFFIX}")
        if re.fullmatch(re.escape(STATS_FILE_STEM) + r"\d*", path.stem)
    ]
    return sorted(files, key=experiment_number_of)


def load_series(path: Path) -> tuple[list[int], list[float]]:
    frame_ids: list[int] = []
    elapsed_ms: list[float] = []
    with path.open(newline="") as stats_file:
        for row in csv.DictReader(stats_file):
            frame_ids.append(int(row["frame_id"]))
            elapsed_ms.append(float(row["visual_total_ms"]))
    return frame_ids, elapsed_ms


def plot_experiments(
    stats_files: list[Path], output_path: Path | None, use_chinese: bool
) -> None:
    labels = (
        {
            "title": "逐帧视觉任务合计耗时对比",
            "xlabel": "帧编号",
            "ylabel": "视觉任务合计耗时 (ms)",
            "legend": "实验 {number} ({name})",
        }
        if use_chinese
        else {
            "title": "Per-frame visual task total time",
            "xlabel": "Frame index",
            "ylabel": "Visual task total time (ms)",
            "legend": "Experiment {number} ({name})",
        }
    )

    figure, axes = plt.subplots(figsize=(10, 5), dpi=120)
    figure.set_facecolor(SURFACE_COLOR)
    axes.set_facecolor(SURFACE_COLOR)

    for index, path in enumerate(stats_files):
        frame_ids, elapsed_ms = load_series(path)
        color = (
            CATEGORICAL_PALETTE[index]
            if index < len(CATEGORICAL_PALETTE)
            else OVERFLOW_COLOR
        )
        axes.plot(
            frame_ids,
            elapsed_ms,
            color=color,
            linewidth=LINE_WIDTH,
            label=labels["legend"].format(
                number=experiment_number_of(path), name=path.name
            ),
        )

    axes.set_title(labels["title"], color=TEXT_PRIMARY)
    axes.set_xlabel(labels["xlabel"], color=TEXT_PRIMARY)
    axes.set_ylabel(labels["ylabel"], color=TEXT_PRIMARY)
    axes.grid(True, color=GRID_COLOR, linewidth=0.8)
    axes.tick_params(colors=TEXT_MUTED)
    for spine in axes.spines.values():
        spine.set_color(AXIS_COLOR)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(
        loc="upper right",
        framealpha=0.9,
        edgecolor=AXIS_COLOR,
        labelcolor=TEXT_PRIMARY,
    )
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, facecolor=SURFACE_COLOR)
        print(f"图像已保存到 {output_path.resolve()}")
    else:
        plt.show()


def configure_chinese_font() -> bool:
    """优先选用系统里可用的中文字体; 找不到时返回 False, 改用英文标签。"""
    from matplotlib import font_manager

    preferred_fonts = [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "SimHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in installed:
            plt.rcParams["font.family"] = font_name
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="绘制多次实验的逐帧视觉任务耗时曲线"
    )
    parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=Path.home() / "vio_ws",
        help="存放 CornerTrackingStats*.csv 的目录 (默认: ~/vio_ws)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出图片路径 (如 stats.png); 不指定则弹窗显示",
    )
    arguments = parser.parse_args()

    stats_files = find_stats_files(arguments.data_directory)
    if not stats_files:
        print(
            f"在 {arguments.data_directory} 中未找到 "
            f"{STATS_FILE_STEM}*{STATS_FILE_SUFFIX} 文件",
            file=sys.stderr,
        )
        return 1
    print(f"找到 {len(stats_files)} 次实验数据:")
    for path in stats_files:
        print(f"  实验 {experiment_number_of(path)}: {path}")
    if len(stats_files) > len(CATEGORICAL_PALETTE):
        print(
            f"注意: 超过 {len(CATEGORICAL_PALETTE)} 次实验, "
            "多余曲线以灰色绘制, 建议分批查看",
            file=sys.stderr,
        )

    use_chinese = configure_chinese_font()
    plot_experiments(stats_files, arguments.output, use_chinese)
    return 0


if __name__ == "__main__":
    sys.exit(main())
