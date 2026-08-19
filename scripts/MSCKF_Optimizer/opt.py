#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MSCKF 超参数自动调优工具 (MHAT) 实现代码。
依据 scripts/MSCKF_Optimizer/README.md (软件需求规约 v1.0) 实现。

用法:
    python opt.py --config config.yaml
"""

import argparse
import csv
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# 第三方依赖库，需提前安装：pip install optuna evo pandas pyyaml
import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)
from optuna.trial import TrialState

# 引入 evo 核心模块，通过内存直接计算以提升效率（避免 subprocess 调用 CLI 的开销）
from evo.tools import file_interface as evo_file
from evo.core import sync as evo_sync
from evo.core import metrics as evo_metrics
from evo.core.metrics import PoseRelation, Unit

# ==============================================================================
# 常量定义 (FR-6, NFR-2)
# ==============================================================================

# 命令模板中由系统自动填充、不属于超参数的特殊占位符
_SPECIAL_PLACEHOLDERS = ("output", "pointcloud", "trial_dir")
# 失败评估的适应度 (FR-5.5, NFR-2)
_FAILED_COMBINED_ERROR = float("inf")
# 轨迹时间戳同步允许的最大时间差 (秒)
_TIMESTAMP_SYNC_MAX_DIFF = 0.01
# 有效匹配位姿数下限
_MIN_MATCHED_POSES = 10
# 早停判定中的最小改进量
_EARLY_STOP_TOL = 1e-5


# ==============================================================================
# 配置与日志初始化
# ==============================================================================


def setup_logger(log_file: str = "optimization.log") -> logging.Logger:
    """配置运行时日志记录器"""
    logger = logging.getLogger("MHAT")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # 控制台输出 (INFO 级别以上)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出 (DEBUG 级别以上)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class EarlyStoppingCallback:
    """
    Optuna 回调函数：实现早停机制 (U4, 4.6)。
    1. 当目标值小于设定阈值时停止。
    2. 当连续多次迭代没有产生优于当前的解时 (Patience) 停止。
    """

    def __init__(self, patience: int = 10, threshold: float = 0.01,
                 tol: float = _EARLY_STOP_TOL):
        self.patience = patience
        self.threshold = threshold
        self.tol = tol
        self.best_value = float("inf")
        self.no_improvement_count = 0

    def __call__(self, study: optuna.study.Study,
                 trial: optuna.trial.FrozenTrial) -> None:
        current_best = study.best_value

        if current_best < self.threshold:
            study.stop()
            logging.getLogger("MHAT").info(
                f"达到目标阈值 {self.threshold}，提前终止。"
            )
            return

        if current_best < self.best_value - self.tol:
            self.best_value = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
            logging.getLogger("MHAT").info(
                f"连续 {self.patience} 次迭代未见显著改善，触发早停。"
            )


# ==============================================================================
# 核心组件定义
# ==============================================================================


class MhatTuner:
    """MSCKF 超参数自动调优主类"""

    def __init__(self, config_path: str, mav0: Optional[str] = None,
                 ground_truth: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.cli_mav0 = mav0
        self.cli_ground_truth = ground_truth

        # 解析配置
        self.data_cfg = self.config.get("data", {})
        self.opt_cfg = self.config.get("optimization", {})
        self.err_cfg = self.config.get("error", {})
        self.out_cfg = self.config.get("output", {})
        self.hyperparams = self.config.get("hyperparameters", [])

        self.logger = setup_logger(self.out_cfg.get("log_file",
                                                    "optimization.log"))
        self.logger.info("=== MSCKF 超参数自动调优工具 (MHAT) 初始化 ===")

        self._validate_and_normalize_config()

        # 加载真值轨迹并在内存中缓存，避免每次迭代重新读取 (FR-6)
        self.logger.info("加载真值轨迹...")
        self.gt_traj = self._load_trajectory(
            self.data_cfg["ground_truth"],
            self.data_cfg.get("traj_format", "tum"),
        )
        self.logger.info(f"真值轨迹加载完成，位姿数量: {self.gt_traj.num_poses}")

        # 并发安全状态 (FR-7): 历史 CSV 追加与最优值更新共用同一把锁
        self._state_lock = threading.Lock()
        self._best_combined = _FAILED_COMBINED_ERROR
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_metrics: Dict[str, float] = {}

        # 断点续跑: 载入历史 CSV 中已评估的超参数组合 (FR-8)
        self.resume_rows = self._load_history_rows() if self.resume else []

    # --------------------------------------------------------------------------
    # 配置读取与校验 (FR-1, FR-2)
    # --------------------------------------------------------------------------

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """读取 YAML/JSON 格式的配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".json"):
                return json.load(f)
            return yaml.safe_load(f)

    def _validate_and_normalize_config(self):
        """校验参数合法性，并进行归一化或设置默认值"""
        # 误差权重校验与归一化 (C3)
        aw = float(self.err_cfg.get("ate_weight", 0.8))
        rw = float(self.err_cfg.get("rpe_weight", 0.2))
        total_w = aw + rw
        if total_w <= 0:
            raise ValueError("ATE 与 RPE 权重之和必须大于 0。")
        self.aw = aw / total_w
        self.rw = rw / total_w
        self.logger.info(
            f"归一化后的误差权重 - ATE: {self.aw:.3f}, RPE: {self.rw:.3f}"
        )

        # 命令模板校验 (FR-2)
        self.cmd_template = str(self.data_cfg.get("estimator_command", ""))
        if not self.cmd_template:
            raise ValueError("配置文件中未提供 estimator_command。")
        self.evaluator_timeout = float(
            self.data_cfg.get("evaluator_timeout_seconds", 600)
        )
        if self.evaluator_timeout <= 0:
            raise ValueError("evaluator_timeout_seconds 必须为正数。")

        # 路径占位符: 命令行选项优先, 其次配置 data 段, 最后默认值。
        # 一律转为绝对路径: 评估子进程的工作目录是独立的 trial 目录,
        # 相对路径会在那里失效
        if self.cli_ground_truth:
            self.data_cfg["ground_truth"] = self.cli_ground_truth
        self.data_cfg["ground_truth"] = os.path.abspath(
            str(self.data_cfg.get("ground_truth", ""))
        )
        self.mav0 = os.path.abspath(
            str(
                self.cli_mav0
                or self.data_cfg.get("mav0")
                or self.data_cfg.get("dataset_root")
                or "./mav0"
            )
        )
        self.path_placeholders = {
            "mav0": self.mav0,
            "ground_truth": str(self.data_cfg.get("ground_truth", "")),
        }
        self.logger.info(f"数据集 mav0 根目录: {self.mav0}")

        # 超参数定义校验
        if not self.hyperparams:
            raise ValueError("配置文件中未定义 hyperparameters。")
        self.param_names: List[str] = []
        for hp in self.hyperparams:
            name = hp.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"超参数缺少合法的 name 字段: {hp}")
            self.param_names.append(name)
            hp_type = str(hp.get("type", "float")).lower()
            if hp_type == "float":
                lo, hi = self._parse_range(hp, "range", float)
                hp["_lo"], hp["_hi"] = lo, hi
                hp["_log"] = bool(hp.get("log", False))
                if hp["_log"] and lo <= 0:
                    raise ValueError(
                        f"超参数 {name} 使用对数尺度时下界必须大于 0。"
                    )
            elif hp_type == "int":
                lo, hi = self._parse_range(hp, "range", int)
                hp["_lo"], hp["_hi"] = lo, hi
            elif hp_type == "categorical":
                choices = hp.get("choices")
                if not choices or not isinstance(choices, list):
                    raise ValueError(
                        f"超参数 {name} 为 categorical 类型时必须提供 choices 列表。"
                    )
                hp["_choices"] = [str(c) for c in choices]
            else:
                raise ValueError(f"超参数 {name} 的类型不受支持: {hp_type}")
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError("超参数 name 存在重复定义。")

        # 占位符与超参数名一一对应校验 (FR-2)
        used = set(re.findall(r"\{([A-Za-z0-9_]+)\}", self.cmd_template))
        allowed = (set(self.param_names) | set(_SPECIAL_PLACEHOLDERS)
                   | set(self.path_placeholders))
        unknown = used - allowed
        if unknown:
            raise ValueError(
                f"命令模板包含未定义的占位符: {sorted(unknown)} "
                f"(可用: {sorted(allowed)})"
            )
        unused = set(self.param_names) - used
        if unused:
            raise ValueError(
                f"超参数未出现在命令模板中, 无法生效: {sorted(unused)}"
            )

        # 优化算法与运行配置
        self.algorithm = str(self.opt_cfg.get("algorithm", "bayesian")).lower()
        if self.algorithm not in ("bayesian", "random", "grid"):
            raise ValueError(
                f"不支持的优化算法: {self.algorithm} (可选: bayesian | random | grid)"
            )
        self.max_iterations = int(self.opt_cfg.get("max_iterations", 50))
        self.n_initial_points = int(self.opt_cfg.get("n_initial_points", 10))
        self.parallel_workers = int(self.opt_cfg.get("parallel_workers", 1))
        self.patience = int(self.opt_cfg.get("patience", 10))
        self.stop_threshold = float(
            self.opt_cfg.get("stop_on_error_threshold", 0.01)
        )
        self.resume = bool(self.opt_cfg.get("resume", True))
        if self.max_iterations <= 0 or self.parallel_workers <= 0:
            raise ValueError("max_iterations 与 parallel_workers 必须为正整数。")

        # 输出目录
        self.work_dir = str(self.out_cfg.get("work_dir", "/tmp/mhat_runs"))
        os.makedirs(self.work_dir, exist_ok=True)
        self.history_csv = str(
            self.out_cfg.get("history_csv", "evaluation_history.csv")
        )
        self.best_result_file = str(
            self.out_cfg.get("best_result", "best_result.json")
        )

        # 误差指标配置
        self.rpe_delta = float(self.err_cfg.get("rpe_delta", 1.0))
        self.rpe_delta_unit = str(
            self.err_cfg.get("rpe_delta_unit", "s")
        ).lower()

    @staticmethod
    def _parse_range(hp: Dict[str, Any], key: str, cast) -> Tuple[Any, Any]:
        value = hp.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"超参数 {hp.get('name')} 缺少合法的 {key} 字段 (长度为 2 的列表)。"
            )
        lo, hi = cast(value[0]), cast(value[1])
        if lo > hi:
            raise ValueError(f"超参数 {hp.get('name')} 的 {key} 下界大于上界。")
        return lo, hi

    # --------------------------------------------------------------------------
    # 轨迹与误差计算 (FR-6)
    # --------------------------------------------------------------------------

    def _load_trajectory(self, path: str, fmt: str):
        """根据格式加载轨迹文件 (FR-1, A1)"""
        if fmt.lower() == "tum":
            return evo_file.read_tum_trajectory_file(path)
        if fmt.lower() == "euroc":
            return evo_file.read_euroc_csv_trajectory(path)
        raise ValueError(f"不支持的轨迹格式: {fmt}")

    def _calculate_metrics(self,
                           est_traj_path: str) -> Tuple[float, float, float]:
        """
        利用 evo 的 Python API 内存直算 ATE 与 RPE (FR-6)。
        估计轨迹固定为 msckf 输出的 TUM 格式; ATE 强制 SE(3) 对齐 (C1)。
        """
        try:
            est_traj = self._load_trajectory(est_traj_path, "tum")

            # 时间戳同步匹配
            gt_sync, est_sync = evo_sync.associate_trajectories(
                self.gt_traj, est_traj, max_diff=_TIMESTAMP_SYNC_MAX_DIFF
            )

            if gt_sync.num_poses < _MIN_MATCHED_POSES:
                self.logger.warning(
                    f"有效匹配的位姿数量过少 ({gt_sync.num_poses})。"
                )
                return (float("inf"), float("inf"), float("inf"))

            # 严格遵循 C1 约束：执行 SE(3) 轨迹对齐（禁止尺度缩放）
            est_sync.align(gt_sync, correct_scale=False)

            # 仅使用平移分量 (C2)
            pose_relation = PoseRelation.translation_part

            # 1. 计算 ATE (RMSE)
            ape_metric = evo_metrics.APE(pose_relation)
            ape_metric.process_data((gt_sync, est_sync))
            ate_rmse = ape_metric.get_statistic(evo_metrics.StatisticsType.rmse)

            # 2. 计算 RPE (RMSE)
            unit_map = {
                "s": Unit.seconds,
                "sec": Unit.seconds,
                "seconds": Unit.seconds,
                "m": Unit.meters,
                "metres": Unit.meters,
                "meters": Unit.meters,
                "f": Unit.frames,
                "frames": Unit.frames,
            }
            if self.rpe_delta_unit not in unit_map:
                raise ValueError(
                    f"不支持的 rpe_delta_unit: {self.rpe_delta_unit}"
                )
            rpe_metric = evo_metrics.RPE(
                pose_relation,
                delta=self.rpe_delta,
                delta_unit=unit_map[self.rpe_delta_unit],
                all_pairs=False,
            )
            rpe_metric.process_data((gt_sync, est_sync))
            rpe_rmse = rpe_metric.get_statistic(evo_metrics.StatisticsType.rmse)

            # 3. 计算综合误差
            combined_error = self.aw * ate_rmse + self.rw * rpe_rmse
            return ate_rmse, rpe_rmse, combined_error

        except Exception as e:
            self.logger.error(f"误差计算失败: {str(e)}")
            return (float("inf"), float("inf"), float("inf"))

    # --------------------------------------------------------------------------
    # 单次评估 (FR-5)
    # --------------------------------------------------------------------------

    @staticmethod
    def _format_value(value: Any) -> str:
        """将超参数值格式化为命令模板/CSV 使用的十进制字符串 (4.5)"""
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)

    def _evaluate_params(self, params: Dict[str, Any],
                         trial_id: int) -> Dict[str, float]:
        """
        执行一次完整评估: 生成命令 → 运行 msckf → 计算误差 (FR-5)。
        返回 {'ATE': .., 'RPE': .., 'combined_error': .., 'status': ..}。
        """
        # 每次评估使用独立的输出目录, 保证并行安全 (FR-7)
        trial_dir = os.path.join(self.work_dir, f"trial_{trial_id:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        out_path = os.path.join(trial_dir, "trajectory.tum")
        cloud_path = os.path.join(trial_dir, "pointcloud.ply")

        placeholders = {k: self._format_value(v) for k, v in params.items()}
        placeholders.update(self.path_placeholders)
        placeholders["output"] = out_path
        placeholders["pointcloud"] = cloud_path
        placeholders["trial_dir"] = trial_dir
        cmd = self.cmd_template.format(**placeholders)
        self.logger.debug(f"[Trial {trial_id:04d}] 执行命令: {cmd}")

        failed = {
            "ATE": float("inf"),
            "RPE": float("inf"),
            "combined_error": float("inf"),
            "status": "failed",
        }

        # 运行估算器 (NFR-2: 超时记为失败)
        started = time.time()
        try:
            process = subprocess.run(
                cmd,
                shell=True,
                cwd=trial_dir,
                capture_output=True,
                text=True,
                timeout=self.evaluator_timeout,
            )
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"[Trial {trial_id:04d}] 评估超时 (>{self.evaluator_timeout:.0f} s)。"
            )
            return failed
        elapsed = time.time() - started

        if process.returncode != 0:
            self.logger.warning(
                f"[Trial {trial_id:04d}] msckf 非零退出码: {process.returncode} "
                f"(耗时 {elapsed:.1f} s)"
            )
            self.logger.warning(f"STDERR 尾部: {process.stderr[-2000:]}")
            return failed

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            self.logger.warning(
                f"[Trial {trial_id:04d}] 轨迹文件未生成或为空。"
            )
            return failed

        ate_rmse, rpe_rmse, combined_error = self._calculate_metrics(out_path)
        status = "ok" if math.isfinite(combined_error) else "failed"
        self.logger.debug(f"[Trial {trial_id:04d}] 评估耗时 {elapsed:.1f} s")
        return {
            "ATE": ate_rmse,
            "RPE": rpe_rmse,
            "combined_error": combined_error,
            "status": status,
        }

    def _evaluate_and_record(self, params: Dict[str, Any],
                             trial_id: int) -> Dict[str, float]:
        """评估一组超参数, 即时写入历史 CSV 并更新当前最优 (FR-5.4, FR-9)"""
        result = self._evaluate_params(params, trial_id)
        self._append_history_row(params, result)
        with self._state_lock:
            if result["combined_error"] < self._best_combined:
                self._best_combined = result["combined_error"]
                self._best_params = dict(params)
                self._best_metrics = {
                    "ATE": result["ATE"],
                    "RPE": result["RPE"],
                    "combined_error": result["combined_error"],
                }
                self.logger.info(
                    f"[Trial {trial_id:04d}] 新最优: 综合误差 "
                    f"{self._best_combined:.6f} 参数: {self._best_params}"
                )
        self.logger.info(
            f"[Trial {trial_id:04d}] "
            f"参数: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()} } | "
            f"ATE: {result['ATE']:.4f} | RPE: {result['RPE']:.4f} | "
            f"综合: {result['combined_error']:.4f}"
        )
        return result

    # --------------------------------------------------------------------------
    # 历史 CSV (FR-9) 与断点续跑 (FR-8)
    # --------------------------------------------------------------------------

    def _append_history_row(self, params: Dict[str, Any],
                            result: Dict[str, float]):
        """逐行即时写入历史 CSV, 崩溃不丢已完成评估 (NFR-2)"""
        fieldnames = self.param_names + ["ATE", "RPE", "combined_error",
                                         "status"]
        row = [self._format_value(params[name]) for name in self.param_names]
        row += [f"{result['ATE']:.6f}", f"{result['RPE']:.6f}",
                f"{result['combined_error']:.6f}", result["status"]]
        with self._state_lock:
            new_file = (not os.path.exists(self.history_csv)
                        or os.path.getsize(self.history_csv) == 0)
            with open(self.history_csv, "a", newline="",
                      encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(fieldnames)
                writer.writerow(row)

    def _load_history_rows(self) -> List[Dict[str, Any]]:
        """从历史 CSV 恢复已评估的 (参数, 结果) 组合 (FR-8);
        文件缺失/损坏/属于其他调优系统时, 归档旧文件并从零开始 (不中断优化)"""
        if not os.path.exists(self.history_csv):
            self.logger.info("历史 CSV 不存在, 从零开始优化。")
            return []
        expected = self.param_names + ["ATE", "RPE", "combined_error", "status"]
        try:
            df = pd.read_csv(self.history_csv)
            if not all(col in df.columns for col in expected):
                raise ValueError(
                    f"表头不匹配, 期望列: {expected}, 实际列: {list(df.columns)}"
                )
        except Exception as e:
            archived = f"{self.history_csv}.mismatched"
            try:
                os.replace(self.history_csv, archived)
            except OSError:
                archived = self.history_csv
            self.logger.warning(
                f"历史 CSV 不可用于断点续跑 ({e}), 已将旧文件归档为 "
                f"'{archived}', 从零开始记录。"
            )
            return []

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            entry = {"_params": {}, "_combined": float("inf")}
            try:
                combined = float(row["combined_error"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(combined):
                continue
            ok = True
            for hp in self.hyperparams:
                name = hp["name"]
                raw = row[name]
                try:
                    hp_type = str(hp.get("type", "float")).lower()
                    if hp_type == "float":
                        entry["_params"][name] = float(raw)
                    elif hp_type == "int":
                        entry["_params"][name] = int(float(raw))
                    else:
                        entry["_params"][name] = str(raw)
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                entry["_combined"] = combined
                entry["_ate"] = float(row["ATE"]) if "ATE" in df.columns \
                    else float("inf")
                entry["_rpe"] = float(row["RPE"]) if "RPE" in df.columns \
                    else float("inf")
                rows.append(entry)
        self.logger.info(
            f"断点续跑: 从历史 CSV 恢复 {len(rows)} 组已评估超参数。"
        )
        return rows

    def _history_param_keys(self) -> set:
        """已评估参数组合的键集合 (用于 random/grid 去重)"""
        keys = set()
        for entry in self.resume_rows:
            keys.add(
                tuple(entry["_params"][name] for name in self.param_names)
            )
        return keys

    # --------------------------------------------------------------------------
    # 搜索空间与采样
    # --------------------------------------------------------------------------

    def _build_distributions(self) -> Dict[str, Any]:
        distributions: Dict[str, Any] = {}
        for hp in self.hyperparams:
            name = hp["name"]
            hp_type = str(hp.get("type", "float")).lower()
            if hp_type == "float":
                distributions[name] = FloatDistribution(
                    hp["_lo"], hp["_hi"], log=hp["_log"]
                )
            elif hp_type == "int":
                distributions[name] = IntDistribution(hp["_lo"], hp["_hi"])
            else:
                distributions[name] = CategoricalDistribution(hp["_choices"])
        return distributions

    def _sample_uniform(self, hp: Dict[str, Any],
                        rng: np.random.Generator) -> Any:
        hp_type = str(hp.get("type", "float")).lower()
        if hp_type == "float":
            if hp["_log"]:
                return math.exp(
                    rng.uniform(math.log(hp["_lo"]), math.log(hp["_hi"]))
                )
            return rng.uniform(hp["_lo"], hp["_hi"])
        if hp_type == "int":
            return int(rng.integers(hp["_lo"], hp["_hi"] + 1))
        return hp["_choices"][int(rng.integers(0, len(hp["_choices"])))]

    def _grid_values(self, hp: Dict[str, Any]) -> List[Any]:
        count = int(hp.get("grid_points", 5))
        hp_type = str(hp.get("type", "float")).lower()
        if hp_type == "float":
            if hp["_log"]:
                return list(np.geomspace(hp["_lo"], hp["_hi"], count))
            return list(np.linspace(hp["_lo"], hp["_hi"], count))
        if hp_type == "int":
            values = sorted({int(round(x)) for x in np.linspace(
                hp["_lo"], hp["_hi"], min(count, hp["_hi"] - hp["_lo"] + 1)
            )})
            return [v for v in values if hp["_lo"] <= v <= hp["_hi"]]
        return list(hp["_choices"])

    # --------------------------------------------------------------------------
    # 优化主流程 (FR-3, FR-4, FR-7, FR-8)
    # --------------------------------------------------------------------------

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna 优化的目标函数"""
        params: Dict[str, Any] = {}
        for hp in self.hyperparams:
            name = hp["name"]
            hp_type = str(hp.get("type", "float")).lower()
            if hp_type == "float":
                params[name] = trial.suggest_float(
                    name, hp["_lo"], hp["_hi"], log=hp["_log"]
                )
            elif hp_type == "int":
                params[name] = trial.suggest_int(name, hp["_lo"], hp["_hi"])
            else:
                params[name] = trial.suggest_categorical(
                    name, hp["_choices"]
                )

        result = self._evaluate_and_record(params, trial.number)
        trial.set_user_attr("ATE", result["ATE"])
        trial.set_user_attr("RPE", result["RPE"])
        trial.set_user_attr("combined_error", result["combined_error"])
        return result["combined_error"]

    def _run_bayesian(self):
        """贝叶斯优化 (TPE), 支持并行与断点续跑 (FR-3, FR-7, FR-8)"""
        sampler = TPESampler(
            n_startup_trials=self.n_initial_points, multivariate=True
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # 断点续跑: 将历史 CSV 中成功的评估注入研究 (FR-8)
        distributions = self._build_distributions()
        for entry in self.resume_rows:
            trial = optuna.trial.create_trial(
                params=entry["_params"],
                distributions=distributions,
                value=entry["_combined"],
                user_attrs={
                    "ATE": entry["_ate"],
                    "RPE": entry["_rpe"],
                    "combined_error": entry["_combined"],
                },
            )
            study.add_trial(trial)

        callback = EarlyStoppingCallback(
            patience=self.patience, threshold=self.stop_threshold
        )
        self.logger.info(
            f"开始贝叶斯优化: 最大迭代 {self.max_iterations}, "
            f"初始探索点 {self.n_initial_points}, 并行 {self.parallel_workers}"
        )
        study.optimize(
            self._objective,
            n_trials=self.max_iterations,
            n_jobs=self.parallel_workers,
            callbacks=[callback],
            catch=(Exception,),
        )

        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        self.logger.info(
            f"贝叶斯优化结束, 共完成 {len(completed)} 次评估, "
            f"最优综合误差: {study.best_value:.6f}"
        )

    def _run_random(self):
        """随机搜索 (FR-4), 跳过历史已评估组合 (FR-8)"""
        rng = np.random.default_rng()
        seen = self._history_param_keys()
        evaluated = 0
        attempts = 0
        max_attempts = max(self.max_iterations * 10, 100)
        with ThreadPoolExecutor(
            max_workers=self.parallel_workers
        ) as executor:
            futures = []
            self.logger.info(
                f"开始随机搜索: 最多 {self.max_iterations} 次评估, "
                f"并行 {self.parallel_workers}"
            )
            while evaluated < self.max_iterations and attempts < max_attempts:
                attempts += 1
                params = {hp["name"]: self._sample_uniform(hp, rng)
                          for hp in self.hyperparams}
                key = tuple(params[name] for name in self.param_names)
                if key in seen:
                    continue
                seen.add(key)
                futures.append(executor.submit(
                    self._evaluate_and_record, params, evaluated
                ))
                evaluated += 1
            for future in futures:
                future.result()
        self.logger.info(f"随机搜索结束, 共评估 {evaluated} 组参数。")

    def _run_grid(self):
        """网格搜索 (FR-4), 按配置顺序评估前 max_iterations 个组合 (FR-8)"""
        grids = [self._grid_values(hp) for hp in self.hyperparams]
        seen = self._history_param_keys()
        total_combos = 1
        for grid in grids:
            total_combos *= len(grid)
        self.logger.info(
            f"网格搜索空间共 {total_combos} 个组合, "
            f"最多评估 {self.max_iterations} 个 (跳过历史已评估)"
        )

        jobs: List[Dict[str, Any]] = []
        for values in itertools.product(*grids):
            params = dict(zip(self.param_names, values))
            key = tuple(values)
            if key in seen:
                continue
            jobs.append(params)
            if len(jobs) >= self.max_iterations:
                break

        with ThreadPoolExecutor(
            max_workers=self.parallel_workers
        ) as executor:
            futures = [
                executor.submit(self._evaluate_and_record, params, trial_id)
                for trial_id, params in enumerate(jobs)
            ]
            for future in futures:
                future.result()
        self.logger.info(f"网格搜索结束, 共评估 {len(jobs)} 组参数。")

    # --------------------------------------------------------------------------
    # 优化前预检 (及时暴露缺失的必需输入, 避免全部评估因缺文件失败)
    # --------------------------------------------------------------------------

    def _preflight_check_required_inputs(self):
        """优化前检查 msckf 必需输入文件, 缺失时给出明确警告"""
        required = [
            f"{self.mav0}/imu0/data.csv",
            f"{self.mav0}/imu0/sensor.yaml",
            f"{self.mav0}/cam0/sensor.yaml",
            f"{self.mav0}/cam0/data.csv",
        ]
        if os.path.isdir(f"{self.mav0}/cam1"):
            required += [
                f"{self.mav0}/cam1/sensor.yaml",
                f"{self.mav0}/cam1/data.csv",
            ]
        if "--init groundtruth" in self.cmd_template:
            required.append(
                f"{self.mav0}/state_groundtruth_estimate0/data.csv"
            )
        missing = [p for p in required if not os.path.isfile(p)]
        if missing:
            self.logger.warning(
                f"msckf 必需输入文件缺失 (mav0: {self.mav0}), "
                f"每次评估都会非零退出导致全部失败: {', '.join(missing)}; "
                f"请用 --mav0 指定正确的数据集根目录。"
            )
        else:
            self.logger.info(
                f"msckf 必需输入文件齐全 (mav0: {self.mav0}, "
                f"共 {len(required)} 项)。"
            )
        # 图像列表必须有数据行, 否则 msckf 加载到 0 帧图像直接退出
        for data_csv in (f"{self.mav0}/cam0/data.csv",
                         f"{self.mav0}/cam1/data.csv"):
            if not os.path.isfile(data_csv):
                continue
            try:
                with open(data_csv, "r", encoding="utf-8") as f:
                    data_lines = [line for line in f
                                  if line.strip()
                                  and not line.lstrip().startswith("#")]
            except OSError:
                continue
            if not data_lines:
                self.logger.warning(
                    f"{data_csv} 中没有任何图像数据行, msckf 将加载到 0 帧图像。"
                )

    def run(self):
        """启动优化主流程"""
        # 优化前预检: 及时发出缺失必需输入文件的警告
        self._preflight_check_required_inputs()

        if self.algorithm == "bayesian":
            self._run_bayesian()
        elif self.algorithm == "random":
            self._run_random()
        else:
            self._run_grid()
        self.logger.info("=== 优化结束 ===")
        self._save_best_result()

    # --------------------------------------------------------------------------
    # 结果输出 (FR-9)
    # --------------------------------------------------------------------------

    def _save_best_result(self):
        """保存最优超参数与误差指标至 JSON (FR-9)"""
        best_data = {
            "best_hyperparameters": self._best_params,
            "metrics": self._best_metrics,
        }
        with open(self.best_result_file, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=4, ensure_ascii=False)
        self.logger.info(f"已保存最优结果 JSON 到: {self.best_result_file}")
        if self._best_params is not None:
            self.logger.info(
                f"--> 全局最优综合误差: {self._best_combined:.6f}"
            )
            self.logger.info(f"--> 最佳参数组合: {self._best_params}")
        else:
            self.logger.error("没有任何成功的参数评估, 未产生有效最优解。")


# ==============================================================================
# 命令行入口 (3.3.2)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSCKF Hyperparameter Auto-Tuner (MHAT)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件的路径 (YAML/JSON)"
    )
    parser.add_argument(
        "--mav0", type=str, default=None,
        help="数据集 mav0 根目录 (覆盖配置 data.mav0, 默认 ./mav0)",
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="真值轨迹 CSV 路径 (覆盖配置 data.ground_truth)",
    )
    args = parser.parse_args()

    try:
        tuner = MhatTuner(args.config, mav0=args.mav0,
                          ground_truth=args.ground_truth)
        tuner.run()
    except Exception as e:
        logging.getLogger("MHAT").exception(f"程序执行发生致命错误: {e}")
        sys.exit(1)
