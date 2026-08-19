#!/usr/bin/env python3
import locale
import platform
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager


def pick_cjk_font():
    """根据操作系统与语言环境自动选择已安装的 CJK 字体"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    language, _ = locale.getlocale() or (None, None)
    language = (language or "").lower()
    system = platform.system()

    if language.startswith("ja"):
        candidates = ["Noto Sans CJK JP", "Source Han Sans JP", "IPAexGothic",
                      "Yu Gothic"]
    elif language.startswith("ko"):
        candidates = ["Noto Sans CJK KR", "Source Han Sans KR", "NanumGothic",
                      "Malgun Gothic"]
    elif system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    elif system == "Darwin":
        candidates = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti"]
    elif language.startswith("zh"):
        candidates = ["Noto Sans CJK SC", "Source Han Sans SC",
                      "Source Han Sans CN", "WenQuanYi Zen Hei",
                      "WenQuanYi Micro Hei", "Droid Sans Fallback"]
    else:
        # 非 CJK 语言环境亦兜底尝试常见中文字体 (图中标签含中文)
        candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                      "Source Han Sans SC", "WenQuanYi Zen Hei",
                      "WenQuanYi Micro Hei", "Droid Sans Fallback"]

    for name in candidates:
        if name in installed:
            return name
    return None


# 1. 读取数据
try:
    df = pd.read_csv("evaluation_history.csv")
except FileNotFoundError:
    print("错误：未找到 evaluation_history.csv 文件，请检查路径。")
    sys.exit(1)

# 2. 绘图风格与字体自动适配 (操作系统字体库 + 语言环境)
sns.set_theme(style="whitegrid")
cjk_font = pick_cjk_font()
if cjk_font:
    plt.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans']
    print(f"已选择字体: {cjk_font}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("警告: 未在系统中找到 CJK 字体, 图中中文可能显示为方块 "
          "(建议安装 fonts-noto-cjk)")
plt.rcParams['axes.unicode_minus'] = False

# 3. 获取超参数名称: 默认取第一个非误差指标列 (兼容 ESKF/MSCKF 历史 CSV)
metric_columns = {"ATE", "RPE", "combined_error", "status"}
param_columns = [c for c in df.columns if c not in metric_columns]
hp_name = param_columns[0] if param_columns else "confidence_angular_displacement"
error_name = 'combined_error'
print(f"超参数列: {hp_name}, 误差列: {error_name}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- 图 1：收敛历史曲线 (Optimization History) ---
# 绘制所有尝试的点
axes[0].scatter(df.index, df[error_name], color='gray', alpha=0.5, label='单次迭代误差')
# 绘制截止到当前步的全局最优历史线
running_min = df[error_name].cummin()
axes[0].plot(df.index, running_min, color='red', linewidth=2, marker='o', label='历史最优误差曲线')

axes[0].set_title("优化收敛历史 (Optimization History)", fontsize=14)
axes[0].set_xlabel("迭代次数 (Trial)", fontsize=12)
axes[0].set_ylabel("综合误差 (Combined Error)", fontsize=12)
axes[0].legend()

# --- 图 2：参数敏感度/落点分析 (Slice Plot) ---
# 因为参数启用了 log=True，在对数坐标轴下看落点更直观
axes[1].scatter(df[hp_name], df[error_name], c=df.index, cmap='viridis', s=50, alpha=0.8)
axes[1].set_xscale('log')  # 启用对数轴

# 标记出最优的点
best_idx = df[error_name].idxmin()
best_hp = df.loc[best_idx, hp_name]
best_err = df.loc[best_idx, error_name]
axes[1].scatter(best_hp, best_err, color='red', marker='*', s=200, edgecolors='black',
                label=f'最优值点 ({best_hp:.2e})')

axes[1].set_title("参数分布与误差映射 (Slice Plot)", fontsize=14)
axes[1].set_xlabel(f"超参数: {hp_name} (对数轴)", fontsize=12)
axes[1].set_ylabel("综合误差 (Combined Error)", fontsize=12)
axes[1].legend()

# 侧边添加颜色条，代表迭代的时序（颜色越深越靠前，越黄越靠后）
cbar = fig.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('迭代顺序 (Trial Index)', fontsize=11)

plt.tight_layout()
plt.savefig("optimization_analysis.png", dpi=300)
print("分析图表已成功保存至本地: optimization_analysis.png")
plt.show()
