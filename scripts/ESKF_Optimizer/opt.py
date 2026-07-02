#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESKF 超参数自动调优工具 (EHAT) 实现代码。
"""

import os
import sys
import yaml
import json
import logging
import argparse
import tempfile
import subprocess
import pandas as pd
from typing import Dict, Any, Tuple

# 第三方依赖库，需提前安装：pip install optuna evo pandas pyyaml
import optuna
from optuna.samplers import TPESampler

# 引入 evo 核心模块，通过内存直接计算以达到最高效率（避免 subprocess 调用 CLI 的开销）
from evo.tools import file_interface as evo_file
from evo.core import sync as evo_sync
from evo.core import metrics as evo_metrics
from evo.core.metrics import PoseRelation, Unit

# ==============================================================================
# 配置与日志初始化
# ==============================================================================


def setup_logger(log_file: str = "optimization.log") -> logging.Logger:
    """配置运行时日志记录器"""
    logger = logging.getLogger("EHAT")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # 控制台输出 (INFO级别以上)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出 (DEBUG级别以上)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ==============================================================================
# 核心组件定义
# ==============================================================================


class EarlyStoppingCallback:
    """
    Optuna 回调函数：实现早停机制。
    1. 当目标值小于设定的阈值时停止。
    2. 当连续多次迭代没有产生优于当前的解时（Patience）停止。
    """

    def __init__(self, patience: int = 5, threshold: float = 1e-4, tol: float = 1e-5):
        self.patience = patience
        self.threshold = threshold
        self.tol = tol
        self.best_value = float("inf")
        self.no_improvement_count = 0

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        current_best = study.best_value

        # 满足误差绝对阈值，提前终止
        if current_best < self.threshold:
            study.stop()
            logging.getLogger("EHAT").info(f"达到目标阈值 {self.threshold}，提前终止。")
            return

        # 记录是否改进
        if current_best < self.best_value - self.tol:
            self.best_value = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
            logging.getLogger("EHAT").info(
                f"连续 {self.patience} 次迭代未见显著改善，触发早停。"
            )


class EHATuner:
    """ESKF 超参数自动调优主类"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

        # 解析配置
        self.data_cfg = self.config.get("data", {})
        self.opt_cfg = self.config.get("optimization", {})
        self.err_cfg = self.config.get("error", {})
        self.out_cfg = self.config.get("output", {})
        self.hyperparams = self.config.get("hyperparameters", [])

        self.logger = setup_logger(self.out_cfg.get("log_file", "optimization.log"))
        self.logger.info("=== ESKF 超参数自动调优工具 (EHAT) 初始化 ===")

        self._validate_and_normalize_config()

        # 获取真值轨迹并在内存中缓存，避免每次迭代重新读取，极大提高效率
        self.logger.info("加载真值轨迹...")
        self.gt_traj = self._load_trajectory(
            self.data_cfg["ground_truth"], self.data_cfg.get("traj_format", "tum")
        )
        self.logger.info(f"真值轨迹加载完成，位姿数量: {self.gt_traj.num_poses}")

    def _load_config(self) -> Dict[str, Any]:
        """读取 YAML/JSON 格式的配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"未找到配置文件: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            if self.config_path.endswith(".json"):
                return json.load(f)
            else:
                return yaml.safe_load(f)

    def _validate_and_normalize_config(self):
        """校验参数合法性，并进行归一化或设置默认值"""
        # 权重校验与归一化 (FR-1, FR-2, C3)
        aw = self.err_cfg.get("ate_weight", 0.5)
        rw = self.err_cfg.get("rpe_weight", 0.5)
        total_w = aw + rw
        if total_w <= 0:
            raise ValueError("ATE 与 RPE 权重之和必须大于 0。")
        self.aw = aw / total_w
        self.rw = rw / total_w
        self.logger.info(f"归一化后的误差权重 - ATE: {self.aw:.3f}, RPE: {self.rw:.3f}")

        # 命令模板校验
        self.cmd_template = self.data_cfg.get("estimator_command", "")
        if not self.cmd_template:
            raise ValueError("配置文件中未提供 estimator_command。")

        # 超参数校验
        if not self.hyperparams:
            raise ValueError("配置文件中未定义 hyperparameters。")

    def _load_trajectory(self, path: str, fmt: str):
        """根据格式加载轨迹文件 (FR-1, A1)"""
        if fmt.lower() == "tum":
            return evo_file.read_tum_trajectory_file(path)
        elif fmt.lower() == "euroc":
            return evo_file.read_euroc_csv_trajectory(path)
        else:
            raise ValueError(f"不支持的轨迹格式: {fmt}")

    def _calculate_metrics(self, est_traj_path: str) -> Tuple[float, float, float]:
        """
        利用 evo 的 Python API 内存直算 ATE 和 RPE 提升运行效率 (FR-6)
        """
        try:
            # 读取估计轨迹
            est_traj = self._load_trajectory(
                est_traj_path, self.data_cfg.get("traj_format", "tum")
            )

            # 时间戳同步匹配
            max_diff = 0.01  # 默认允许 10ms 的时间差
            gt_sync, est_sync = evo_sync.associate_trajectories(
                self.gt_traj, est_traj, max_diff=max_diff
            )

            if gt_sync.num_poses < 10:
                self.logger.warning(f"有效匹配的位姿数量过少 ({gt_sync.num_poses})。")
                return float("inf"), float("inf"), float("inf")

            # 严格遵循 C1 约束：执行 SE(3) 轨迹对齐（禁止尺度缩放）
            est_sync.align(gt_sync, correct_scale=False)

            # 配置评测关系：仅使用平移分量 (C2)
            pose_relation = PoseRelation.translation_part

            # 1. 计算 ATE (RMSE)
            ape_metric = evo_metrics.APE(pose_relation)
            ape_metric.process_data((gt_sync, est_sync))
            ate_rmse = ape_metric.get_statistic(evo_metrics.StatisticsType.rmse)

            # 2. 计算 RPE (RMSE)
            delta = self.err_cfg.get("rpe_delta", 1.0)
            unit_str = self.err_cfg.get("rpe_delta_unit", "s")
            unit = (
                Unit.frames  # 修改这里：将 Unit.seconds 改为 Unit.frames
                if unit_str.lower() in ["s", "sec", "seconds"]
                else Unit.meters
            )

            rpe_metric = evo_metrics.RPE(
                pose_relation, delta=delta, delta_unit=unit, all_pairs=False
            )
            rpe_metric.process_data((gt_sync, est_sync))
            rpe_rmse = rpe_metric.get_statistic(evo_metrics.StatisticsType.rmse)

            # 3. 计算综合误差
            combined_error = self.aw * ate_rmse + self.rw * rpe_rmse

            return ate_rmse, rpe_rmse, combined_error

        except Exception as e:
            self.logger.error(f"误差计算失败: {str(e)}")
            return float("inf"), float("inf"), float("inf")

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna 优化的目标函数"""
        # 1. 在搜索空间中采样参数 (FR-3, C4)
        params = {}
        for hp in self.hyperparams:
            name = hp["name"]
            range_lo = hp["range"][0]
            range_hi = hp["range"][1]
            hp_type = hp.get("type", "float").lower()

            if hp_type == "float":
                range_lo = float(range_lo)
                range_hi = float(range_hi)
                # 如果用户明确配置了 log 尺度
                log_scale = hp.get("log", False)
                params[name] = trial.suggest_float(
                    name, range_lo, range_hi, log=log_scale
                )
            elif hp_type == "int":
                range_lo = int(range_lo)
                range_hi = int(range_hi)
                params[name] = trial.suggest_int(name, range_lo, range_hi)
            elif hp_type == "categorical":
                params[name] = trial.suggest_categorical(name, hp["choices"])
            else:
                raise ValueError(f"未知参数类型: {hp_type}")

        # 2. 生成唯一的输出轨迹文件路径（支持并行安全）
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            out_path = tmp_file.name

        params["output"] = out_path

        # 3. 拼接并执行仿真评估指令 (FR-5)
        try:
            cmd = self.cmd_template.format(**params)
        except KeyError as e:
            self.logger.error(f"指令模板参数映射失败，缺少参数: {e}")
            return float("inf")

        self.logger.debug(f"[Trial {trial.number}] 执行命令: {cmd}")

        try:
            # 执行估算器 (NFR-1 要求秒级完成，可通过 timeout 限制)
            process = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            if process.returncode != 0:
                self.logger.warning(
                    f"[Trial {trial.number}] 仿真进程异常退出, 错误码: {process.returncode}"
                )
                self.logger.debug(f"STDERR: {process.stderr}")
                return float("inf")
        except subprocess.TimeoutExpired:
            self.logger.warning(f"[Trial {trial.number}] 仿真执行超时。")
            return float("inf")

        # 4. 读取产生的轨迹文件计算误差
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            self.logger.warning(f"[Trial {trial.number}] 轨迹文件未生成或为空。")
            if os.path.exists(out_path):
                os.remove(out_path)
            return float("inf")

        ate_rmse, rpe_rmse, combined_error = self._calculate_metrics(out_path)

        # 记录附加信息以备后用
        trial.set_user_attr("ATE", ate_rmse)
        trial.set_user_attr("RPE", rpe_rmse)
        trial.set_user_attr("combined_error", combined_error)

        self.logger.info(
            f"[Trial {trial.number:03d}] "
            f"Params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items() if k != 'output'} } | "
            f"ATE: {ate_rmse:.4f} | RPE: {rpe_rmse:.4f} | Combined: {combined_error:.4f}"
        )

        # 5. 清理临时轨迹文件以防占用过多磁盘空间
        os.remove(out_path)

        return combined_error

    def run(self):
        """启动优化主流程"""
        # 设置贝叶斯优化的具体实现 (FR-3)
        # Optuna 使用 TPE 采样器，自带初始随机探索 (n_startup_trials 即 n_initial_points)
        n_startup = self.opt_cfg.get("n_initial_points", 10)
        sampler = TPESampler(n_startup_trials=n_startup, multivariate=True)

        study = optuna.create_study(direction="minimize", sampler=sampler)

        # 设置早停回调机制
        callbacks = []
        patience = self.opt_cfg.get("patience", 5)
        stop_threshold = self.opt_cfg.get("stop_on_error_threshold", 1e-4)
        callbacks.append(
            EarlyStoppingCallback(patience=patience, threshold=stop_threshold)
        )

        n_trials = self.opt_cfg.get("max_iterations", 50)
        n_jobs = self.opt_cfg.get("parallel_workers", 1)  # (FR-7) 支持并行评估

        self.logger.info(
            f"开始执行贝叶斯优化，最大迭代次数: {n_trials}，并行进程数: {n_jobs}"
        )

        # 开始优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=callbacks,
            catch=(Exception,),
        )

        self.logger.info("=== 优化结束 ===")
        self._save_results(study)

    def _save_results(self, study: optuna.study.Study):
        """保存运行结果至指定路径 (FR-8)"""
        # 1. 提取所有已完成的 Trial 并输出 CSV
        history_file = self.out_cfg.get("history_csv", "evaluation_history.csv")

        trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value != float("inf")
        ]

        if not trials:
            self.logger.error("没有任何成功的参数评估，无法生成有效输出。")
            return

        data_rows = []
        for t in trials:
            row = t.params.copy()
            row["ATE"] = t.user_attrs.get("ATE", float("inf"))
            row["RPE"] = t.user_attrs.get("RPE", float("inf"))
            row["combined_error"] = t.user_attrs.get("combined_error", float("inf"))
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        # 根据配置文件中指定的顺序排列列头
        param_cols = [hp["name"] for hp in self.hyperparams]
        col_order = param_cols + ["ATE", "RPE", "combined_error"]
        df = df[col_order]

        df.to_csv(history_file, index=False)
        self.logger.info(f"已保存评估历史 CSV 到: {history_file}")

        # 2. 导出最优超参数至 JSON
        best_file = self.out_cfg.get("best_result", "best_result.json")
        best_data = {
            "best_hyperparameters": study.best_params,
            "metrics": {
                "ATE": study.best_trial.user_attrs.get("ATE"),
                "RPE": study.best_trial.user_attrs.get("RPE"),
                "combined_error": study.best_value,
            },
        }
        with open(best_file, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=4)
        self.logger.info(f"已保存最优结果 JSON 到: {best_file}")
        self.logger.info(f"--> 全局最优综合误差: {study.best_value:.6f}")
        self.logger.info(f"--> 最佳参数组合: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ESKF Hyperparameter Auto-Tuner (EHAT)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件的路径 (YAML/JSON)"
    )
    args = parser.parse_args()

    try:
        tuner = EHATuner(args.config)
        tuner.run()
    except Exception as e:
        logging.getLogger("EHAT").exception(f"程序执行发生致命错误: {e}")
        sys.exit(1)
