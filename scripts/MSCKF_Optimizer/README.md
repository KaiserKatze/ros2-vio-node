# 软件需求规约（SRS）
## 用于 MSCKF 超参数自动调优系统

---

### 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0  | 2026-08-18 | AI Assistant | 初始版本 |

---

## 1. 引言

### 1.1 项目标识

- **项目名称**：MSCKF超参数自动调优工具（MSCKF Hyperparameter Auto-Tuner, MHAT）
- **项目编号**：MHAT-2026-001

### 1.2 背景

euroc_vio 的 MSCKF 前端（`src/MSCKF/msckf.cpp`，可执行程序 `msckf`）在部分数据集上对超参数十分敏感：量测噪声（像素噪声、ZUPT 噪声、单目先验噪声）、量测门限（χ² 门限相关噪声、三角化视差角/深度/重投影质量门限、Huber 核阈值）、滑窗结构（克隆数、轨迹长度、更新行数、冗余克隆判定）共同决定滤波器能否收敛。例如在 MinecraftSim 数据集上，默认超参数组合曾导致视觉更新几乎全部被门限拒绝（794 帧仅 29 个地图点），轨迹误差高达数千米。

手动调整这些超参数效率低、难以覆盖组合空间，且部分参数存在对数尺度的"良好区域"。需要一套自动化搜索系统，在给定优化目标（ATE/RPE 综合误差）下找到最优超参数组合。

### 1.3 项目目标

开发一个软件系统，该系统能够：

- 以命令行方式调用 `msckf` 可执行程序，将超参数经 `--name=value` 选项注入每次评估运行。
- 在用户指定的超参数搜索空间内，利用**贝叶斯优化**算法自动搜索最优超参数配置。
- 优化目标为"综合误差"，即绝对轨迹误差（ATE）与相对位姿误差（RPE）的加权平均，其中 ATE 计算强制使用 SE(3) 变换（禁止 SIM(3) 变换）。
- 记录每次评估的超参数组合及对应的 ATE、RPE 和综合误差，输出为带表头的 CSV 文件，便于可视化分析。
- 输出最优超参数及其对应的综合误差值。
- 支持断点续跑：从历史 CSV 恢复已评估样本，避免中断后重复评估（单次评估耗时约分钟级）。

### 1.4 适用范围

本软件适用于 euroc_vio MSCKF 的离线超参数调优场景。待调优超参数为 `msckf` 已通过命令行暴露的 13 个标量（见附录 6.1 超参数目录），用户在配置文件中自由选择其中 1 至 13 个组成搜索空间。软件以命令行工具形式运行，不依赖图形用户界面（GUI）。

### 1.5 术语与定义

| 术语 | 定义 |
|------|------|
| **MSCKF** | 多状态约束卡尔曼滤波（Multi-State Constraint Kalman Filter），滑窗相机克隆 + 特征量测更新的视觉惯性里程计。 |
| **可调超参数** | `msckf` 中影响估计精度与鲁棒性的 13 个标量，已从 `static constexpr` 常量改为 `Msckf` 公有成员变量，并经 `--name=value` 命令行选项注入。 |
| **真实轨迹** | 数据集 `mav0/state_groundtruth_estimate0/data.csv`（EuRoC 格式，含时间戳、位置、四元数、速度、零偏等列）。 |
| **估计轨迹** | `msckf` 经 `--output` 写入的 TUM 格式轨迹文件。 |
| **ATE** | 绝对轨迹误差（Absolute Trajectory Error），基于 SE(3) 对齐后估计位姿与真实位姿之间的平移部分 RMSE。 |
| **RPE** | 相对位姿误差（Relative Pose Error），固定时间间隔位姿增量误差的平移部分 RMSE。 |
| **Umeyama 算法** | 估计两个点集之间最优变换的最小二乘算法；本系统仅允许 SE(3)（禁止尺度）。 |
| **综合误差** | `W_ATE * ATE + W_RPE * RPE`，其中 `W_ATE + W_RPE = 1`，默认 `W_ATE=0.8, W_RPE=0.2`。 |
| **断点续跑** | 从 `history_csv` 中已评估的行恢复优化状态，仅评估尚未评估的超参数组合。 |

### 1.6 参考文献

- Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation.
- Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns.
- Sturm, J. et al. (2012). TUM RGB-D benchmark for evaluating SLAM systems.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms.
- EVO documentation: https://github.com/MichaelGrupp/evo

---

## 2. 总体描述

### 2.1 用户角色

| 角色 | 描述 |
|------|------|
| **算法调优工程师** | 负责配置搜索空间、运行调优、分析历史 CSV 与最优结果。 |
| **系统集成者** | 将本工具集成到 euroc_vio 的评估流程中。 |

### 2.2 用户需求与特性

- **需求 U1**：用户应能够通过配置文件指定 MSCKF 超参数的搜索范围。每个超参数需指定名称（与 CLI 选项一致）、类型（连续或离散）、取值范围；搜索空间维度（1 至 13 维）由列表长度决定。
- **需求 U2**：用户应能够指定 ATE 与 RPE 的权重系数（浮点数，非负，和为 1）。
- **需求 U3**：系统默认使用**贝叶斯优化**算法；同时应支持扩展其他算法（如随机搜索、网格搜索）。
- **需求 U4**：用户应能够设置优化停止条件（最大迭代次数、连续无改进轮数、目标误差阈值）。
- **需求 U5**：系统应输出最优超参数及其综合误差，同时**必须**输出一个带表头的 CSV 文件，记录每次评估的超参数组合及其对应的 ATE、RPE、综合误差。
- **需求 U6**：禁止使用 SIM(3) 变换进行 ATE 计算，强制执行 SE(3) 对齐。
- **需求 U7**：单次评估失败（msckf 非零退出、输出轨迹缺失或位姿数不足、误差计算异常）时，该组适应度记为无穷大，优化过程不应中断，失败原因记入日志。

### 2.3 假设与依赖

- **假设 A1**：数据集按 EuRoC 目录结构组织（`mav0/cam0`、`mav0/imu0`、`mav0/state_groundtruth_estimate0`），且 `msckf` 可直接运行。
- **假设 A2**：`msckf` 单次评估结果**确定**（同输入同输出，前端与滤波器无随机源），故每组合仅需评估一次，无需重复采样取均值。
- **假设 A3**：单次评估耗时约 40~90 秒（与图像帧数、特征数量、是否启用计时日志相关）；优化预算（迭代数、并行度）由用户据此设定。
- **假设 A4**：真值轨迹与估计轨迹时间戳无需严格对齐（evo 按时间戳同步/裁剪），但估计轨迹时长可能短于真值（如数据集图像仅覆盖部分时段），evo 会自动处理。
- **依赖 D1**：系统依赖于 Python（≥3.8）及第三方库：`numpy`、`scipy`、`pandas`、`scikit-optimize`（或 `optuna`）与 `evo`；用户需预先安装。
- **依赖 D2**：评估端需要已编译安装的 `euroc_vio` 包（`ros2 run euroc_vio msckf` 可用）。

### 2.4 约束条件

- **约束 C1**：ATE 计算中 Umeyama 算法仅允许 SE(3) 变换，禁止尺度估计（`evo_ape -a`，不使用 `-as`）。
- **约束 C2**：综合误差中各分量均使用平移部分 RMSE（单位：米）。
- **约束 C3**：若用户未指定权重，默认 `W_ATE = 0.8, W_RPE = 0.2`。
- **约束 C4**：搜索空间维度建议 1~8 维；更高维度可用但优化效率下降。
- **约束 C5**：超参数仅通过 `msckf` 已暴露的 `--name=value` 选项注入（13 个标量，见附录 6.1）；未暴露的滤波器内部参数不在本系统范围内（见附录 6.5 扩展计划）。
- **约束 C6**：本系统不修改数据集与生成器（如 `Config.java` 的 GRAVITY）；如需对 IMU 注入噪声，用户在评估前自行运行 `ImuNoiseModel` 工具预处理。

---

## 3. 系统需求

### 3.1 功能需求

#### 3.1.1 输入处理

| ID | 需求描述 |
|----|----------|
| FR-1 | 系统应能够读取配置文件（YAML），包含以下内容：<br> - 数据集 `mav0` 路径与真值轨迹文件路径<br> - 估计器命令模板（含 `{参数名}` 与 `{output}`、`{pointcloud}` 占位符，见 3.3.1）<br> - 超参数列表：每个超参数的名称（与 CLI 选项一致）、类型（float/int/categorical）、取值范围（或离散值列表）<br> - ATE 与 RPE 权重（自动归一化）<br> - 优化算法选择（默认贝叶斯优化）与算法特定参数（采集函数、核、初始点数）<br> - 停止条件（最大迭代次数、patience、目标阈值）<br> - 输出路径（历史 CSV、最优结果 JSON、日志文件） |
| FR-2 | 系统应验证输入合法性：文件与路径存在、权重和为 1、取值范围有效、命令模板占位符与超参数名一一对应；不合法时给出明确错误信息并退出。 |

#### 3.1.2 超参数搜索与优化

| ID | 需求描述 |
|----|----------|
| FR-3 | 系统必须实现**贝叶斯优化**作为默认优化引擎：<br> - 代理模型：高斯过程（核可选 Matérn 5/2 或 RBF）<br> - 采集函数：默认期望改进（EI），可配置 UCB/PI<br> - 初始设计：拉丁超立方采样 `n_initial_points` 个点（默认 10）<br> - 支持连续、整数与类别型混合搜索空间。 |
| FR-4 | 系统应提供优化算法扩展接口（`algorithm` 字段切换随机搜索/网格搜索）。 |
| FR-5 | 对每一组候选超参数，系统应：<br> 1. 将超参数值代入命令模板生成完整命令（每个 worker 使用独立的 `{output}`/`{pointcloud}` 路径，避免并发写同一文件）<br> 2. 以子进程方式运行 `msckf`<br> 3. 读取输出的 TUM 轨迹，与真值计算 ATE/RPE/综合误差（见 FR-6）<br> 4. 将超参数组合、ATE、RPE、综合误差写入历史 CSV（逐行即时写入，崩溃不丢数据）。 |
| FR-6 | 综合误差计算流程：<br> 1. 读取真值轨迹（EuRoC CSV 或已转换的 TUM）与估计轨迹（TUM）<br> 2. 用 `evo_ape`（`-a`，仅 SE(3) 对齐）计算平移部分 RMSE 作为 ATE<br> 3. 用 `evo_rpe`（`--delta 1 --delta_unit s`，平移部分）计算 RPE<br> 4. 综合误差 = `W_ATE * ATE + W_RPE * RPE`<br> 5. 任一步异常（文件缺失、位姿数过少、对齐失败、非零退出码）→ 适应度 = +∞，并在日志中记录原因。 |
| FR-7 | 系统应支持多进程并行评估候选超参数（`parallel_workers`），每个 worker 拥有独立的输出目录与日志。 |
| FR-8 | 系统应支持**断点续跑**：启动时若历史 CSV 存在且表头匹配，则将其中的已评估组合载入优化器初始数据集，跳过重复评估。 |

#### 3.1.3 输出与日志

| ID | 需求描述 |
|----|----------|
| FR-9 | 优化结束后，系统应输出：<br> - 最优超参数组合及其综合误差、ATE、RPE 分量<br> - 带表头的历史 CSV（列顺序：`参数名..., ATE, RPE, combined_error`，另含 `status` 列标记失败评估）<br> - 最优结果 JSON 文件。 |
| FR-10 | 系统应提供 INFO 级运行日志：每次评估的命令行、超参数值、各误差分量、耗时、失败原因。 |

### 3.2 非功能需求

| ID | 需求描述 |
|----|----------|
| NFR-1 | **性能**：单次评估（一次 `msckf` 运行 + 误差计算）耗时以实际运行为准（40~90 秒量级），误差计算本身应小于 1 秒；优化过程的总耗时由 `max_iterations` 与 `parallel_workers` 共同控制，系统应在每轮迭代打印预计剩余时间。 |
| NFR-2 | **可靠性**：子进程超时（用户可配置，默认 600 秒）应记为该评估失败（+∞）并继续；历史 CSV 逐行写盘，进程被中断不丢失已完成评估。 |
| NFR-3 | **可扩展性**：模块化设计，允许替换优化算法与误差计算后端。 |
| NFR-4 | **易用性**：提供示例配置文件、`--help` 与参数说明文档。 |
| NFR-5 | **平台独立性**：Linux/macOS/Windows（WSL）可运行；`msckf` 子进程调用需在已配置 ROS2 环境的主机上执行。 |
| NFR-6 | **依赖透明性**：明确列出 Python 第三方库及版本建议（见附录 6.4）。 |

### 3.3 接口需求

#### 3.3.1 估计器接口（msckf 命令行）

`msckf` 已为调优暴露如下接口（本 SRS 配套修改，见 `src/MSCKF/msckf.cpp`）：

- 超参数经 `--name=value` 注入，共 13 个（附录 6.1）；
- 固定口径选项：`--init groundtruth`、`--init-gravity imu|fixed|数值`、`--time-offset`、`--output <tum>`、`--pointcloud <ply>`；
- 返回码：0 成功，非 0 失败。

命令模板示例：

```yaml
estimator_command: >
  ros2 run euroc_vio msckf /home/ros/vio_ws/mav0
  --init groundtruth
  --init-gravity imu
  --output {output}
  --pointcloud {pointcloud}
  --pixel-noise-sigma={pixel_noise_sigma}
  --min-parallax={min_parallax_radians}
  --huber-threshold={huber_loss_threshold}
  --max-clones={max_clone_count}
```

#### 3.3.2 用户界面

命令行：`python mhat.py --config config.yaml`，支持 `--help`。

### 3.4 配置文件示例

```yaml
# config.yaml (MSCKF 超参数自动调优)
data:
  dataset_root: "/home/ros/vio_ws/mav0"
  ground_truth: "/home/ros/vio_ws/mav0/state_groundtruth_estimate0/data.csv"
  traj_format: "euroc"          # 真值轨迹格式: tum | euroc
  estimator_command: >
    ros2 run euroc_vio msckf /home/ros/vio_ws/mav0
    --init groundtruth --init-gravity imu
    --output {output} --pointcloud {pointcloud}
    --pixel-noise-sigma={pixel_noise_sigma}
    --min-parallax={min_parallax_radians}
    --huber-threshold={huber_loss_threshold}
  evaluator_timeout_seconds: 600

hyperparameters:
  - name: "pixel_noise_sigma"          # 对应 CLI --pixel-noise-sigma
    type: "float"
    range: [0.1, 10.0]
    log: true
  - name: "min_parallax_radians"       # 对应 CLI --min-parallax
    type: "float"
    range: [1e-4, 0.1]
    log: true
  - name: "huber_loss_threshold"       # 对应 CLI --huber-threshold
    type: "float"
    range: [1e-4, 0.1]
    log: true

optimization:
  algorithm: "bayesian"
  max_iterations: 50
  n_initial_points: 10
  acquisition: "EI"
  kernel: "Matern52"
  patience: 10
  stop_on_error_threshold: 0.01
  parallel_workers: 1
  resume: true                        # 断点续跑

error:
  ate_weight: 0.8
  rpe_weight: 0.2
  rpe_delta: 1.0
  rpe_delta_unit: "s"

output:
  work_dir: "/tmp/mhat_runs"          # 每次评估的独立输出子目录
  log_file: "optimization.log"
  history_csv: "evaluation_history.csv"
  best_result: "best_result.json"
```

**CSV 输出示例表头**：

```csv
pixel_noise_sigma,min_parallax_radians,huber_loss_threshold,ATE,RPE,combined_error,status
1.500000,1.000000e-02,1.000000e-02,5190.187,316.220,4215.394,ok
0.820000,3.200000e-03,4.500000e-03,2.134,0.872,1.882,ok
...
```

---

## 4. 优化算法详细设计（贝叶斯优化）

### 4.1 算法选择理由

每次评估需运行一次完整 MSCKF（分钟级、确定性、黑盒、误差曲面非凸且可能存在多个局部极小与"悬崖"区域——门限类参数越过临界值会使误差突变）。贝叶斯优化通过高斯过程代理与采集函数权衡，能以较少的评估次数定位较优区域，显著优于网格/随机搜索。

### 4.2 实现框架

使用 `scikit-optimize`（skopt）或 `optuna`（TPE），要求支持混合变量（连续+整数+类别）与并行评估。本规约不强制指定具体库。

### 4.3 流程

1. **恢复/初始设计**：若启用断点续跑，先载入历史 CSV 的已评估点；不足 `n_initial_points` 时用拉丁超立方采样补足，逐一评估。
2. **代理模型拟合**：基于全部已评估点（超参数→综合误差，失败点记为 +∞ 并排除出 GP 训练集）。
3. **采集函数最大化**：选择下一个候选组合（EI 默认）。
4. **评估**：子进程运行 `msckf`，计算综合误差，即时写入历史 CSV。
5. **终止判断**：达到 `max_iterations`、连续 `patience` 轮最优值无改进、或综合误差低于阈值。
6. **输出最优组合**。

### 4.4 适应度函数

同 FR-6；失败评估记 +∞。

### 4.5 约束处理

- 连续/整数参数通过搜索空间定义直接约束边界；
- 类别型参数（如 `--init-gravity` 的取值集合）由 Categorical 空间支持；
- 超参数值格式化为十进制科学计数法字符串注入命令模板（`{:.10g}` 风格），保证 `msckf` 解析成功。

### 4.6 终止条件

满足任一即终止：
- 达到 `max_iterations`（含初始点）；
- 最优综合误差连续 `patience` 轮（默认 10）未下降超过 `tol=1e-5`；
- 最优综合误差低于 `stop_on_error_threshold`。

---

## 5. 验证与确认

### 5.1 测试用例

| 测试ID | 描述 | 预期结果 |
|--------|------|----------|
| TC-1 | 对单一超参数（如 `pixel_noise_sigma`）运行贝叶斯优化 | 系统正常完成，历史 CSV 表头正确，最优值优于默认值。 |
| TC-2 | 设置 `ate_weight=1.0, rpe_weight=0.0` | 系统仅优化 ATE。 |
| TC-3 | 故意传入一个非法超参数值（如 `--min-parallax=-1`） | msckf 非零退出，该评估记为 +∞，优化继续。 |
| TC-4 | 运行中途终止进程后重新启动（`resume: true`） | 已评估组合不重复运行，优化从断点继续。 |
| TC-5 | 并行评估（`parallel_workers: 2`） | 各 worker 输出文件互不冲突，结果与串行一致。 |
| TC-6 | CSV 输出完整性 | 每次评估（含失败）均写入一行，表头正确，无缺失值。 |

### 5.2 验收标准

- 能对 1~8 维超参数空间进行有效搜索；贝叶斯优化相比随机搜索以更少评估次数达到相似或更优误差。
- 在目标数据集上（如 MinecraftSim `mav0` 或 EuRoC MAV），优化后的综合误差相比默认超参数组合显著下降（以实际运行日志为准，目标至少一个数量级）。
- 提供完整的文档、示例配置与历史 CSV 样例。
- 代码通过单元测试（配置解析、命令模板替换、适应度计算、断点续跑）。

---

## 6. 附录

### 6.1 超参数目录（msckf 已暴露的 13 个标量）

| 超参数名（= CLI 选项名 = Msckf 公有成员名） | CLI 选项 | 类型 | 默认值 | 建议搜索范围 | 对数 |
|---|---|---|---|---|---|
| pixel_noise_sigma | `--pixel-noise-sigma` | float | 1.5 px | [0.1, 10] | ✓ |
| max_clone_count | `--max-clones` | int | 11 | [3, 30] | ✗ |
| min_track_length | `--min-track-length` | int | 3 | [2, 10] | ✗ |
| max_update_rows | `--max-update-rows` | int | 600 | [100, 3000] | ✗ |
| initial_time_offset_variance | `--time-offset-variance` | float | 2.5e-5 s² | [1e-8, 1e-2] | ✓ |
| zupt_velocity_sigma | `--zupt-sigma` | float | 0.02 m/s | [1e-3, 1.0] | ✓ |
| redundant_rotation_threshold | `--redundant-rot-threshold` | float | 0.2618 rad | [0.01, 1.0] | ✓ |
| redundant_translation_threshold | `--redundant-trans-threshold` | float | 0.4 m | [0.01, 2.0] | ✓ |
| monocular_rotation_sigma | `--mono-rot-sigma` | float | 0.005 rad | [1e-4, 0.1] | ✓ |
| monocular_direction_sigma | `--mono-dir-sigma` | float | 0.1 rad | [1e-3, 1.0] | ✓ |
| min_baseline_for_direction | `--mono-baseline-min` | float | 0.01 m | [1e-4, 0.1] | ✓ |
| huber_loss_threshold | `--huber-threshold` | float | 0.01 | [1e-4, 0.1] | ✓ |
| min_parallax_radians | `--min-parallax` | float | 0.01 rad | [1e-4, 0.1] | ✓ |

**实现说明**（配套 `src/MSCKF/msckf.cpp` 修改）：
- 上述参数由 `Msckf` 类的 `static constexpr` 常量改为**公有成员变量**（默认值不变）；
- `pixel_noise_sigma_px` 经焦距换算为归一化平面噪声（`PixelNoiseNormalized()`），量测门限与增益随其更新；
- 命令行经 `--name=value` 覆盖（`TryParseTuningOption`），并在姿态初始化**之前**应用到滤波器（`initial_time_offset_variance` 参与初始协方差构造）；
- 结构常量（状态维数、重力下标等）保持 `static constexpr`，不可调。

### 6.2 evo 命令（SE(3) 强制）

```bash
evo_ape tum gt.tum est.tum -a --pose_relation trans_part --save_results ape.zip
evo_rpe tum gt.tum est.tum --delta 1 --delta_unit s --pose_relation trans_part
```

其中 `-a` 为 SE(3) 对齐（禁止 `-as` 尺度校正）。真值为 EuRoC CSV 时使用 `evo_ape euroc ...` 或先转换为 TUM。

### 6.3 评估口径建议（固定部分，不参与调优）

- `--init groundtruth --init-gravity imu`（数据集重力与 9.81 不一致时尤其必要）；
- 每次评估使用独立 `--output`/`--pointcloud` 路径，避免并发写冲突；
- 数据集噪声口径（是否先运行 `ImuNoiseModel`）在调优前由用户确定并保持不变。

### 6.4 依赖库列表（建议版本）

- Python ≥ 3.8
- numpy ≥ 1.21
- scipy ≥ 1.7
- pandas ≥ 1.3
- scikit-optimize ≥ 0.9.0 或 optuna ≥ 3.0
- evo ≥ 1.12.0
- matplotlib（可选，收敛曲线绘图）

### 6.5 扩展计划（不在 v1.0 范围内）

| 扩展项 | 说明 |
|--------|------|
| 初始协方差对角块 | 姿态/位置/速度/零偏/重力/时间偏移的初始标准差，当前硬编码于两种初始化函数中，拟拆分为公有成员 + CLI 选项 |
| χ² 门限置信度 | 当前固定 95%（Wilson-Hilferty 近似），拟暴露 `gate_confidence`（0.8~0.999） |
| 前端参数 | 特征数上限、FAST 阈值、最小特征距离、金字塔层数、LK 窗口等（FeatureTracker 常量） |
| 静止检测器参数 | ZUPT 触发窗口与陀螺/加速度阈值（StationaryDetector 常量） |
| 类别型口径参数 | `--init`、`--init-gravity` 等取值集合纳入搜索空间（作为 Categorical 维度） |
