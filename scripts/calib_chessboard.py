#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinecraftSim EuRoC 相机标定
==========================
利用 cam0/data 的棋盘格图像与 state_groundtruth_estimate0/data.csv 的位姿数据，
估计相机内参矩阵 (fx, fy, cx, cy) 与畸变系数（径向 k1, k2, k3 + 切向 p1, p2）。

坐标系说明
----------
state_groundtruth_estimate0/data.csv 记录的是 MinecraftSim 的 body 位姿:
    - body 系约定:  x = 前向(forward), y = 左(left), z = 上(up)
    - 参考系(RS):    数据集 t=0 时刻的 body 系（位姿已由 t=0 锚定）
OpenCV 相机系约定:    x = 右, y = 下, z = 前
两者之间是固定的正交变换::

    C0 = [[0,-1,0], [0,0,-1], [1,0,0]]      # body -> OpenCV camera

世界点 X_R 到相机坐标:  X_cam = C0 @ R_RS^T @ (X_R - p_RS)

方法
----
1. 自动检测棋盘格尺寸（候选列表含 9x16 / 8x15 / 9x6 等，按"检出图像数 + PnP
   重投影质量"选择；本数据集实际检出 9x6 内角点，62/67 幅图像有效）。
2. 用 GT 位姿固定相机轨迹，每幅图像独立估计棋盘格位姿（PnP），外层
   Levenberg-Marquardt 联合优化内参 + 畸变，最小化全部角点重投影误差。
3. 静态棋盘格诊断：若棋盘格相对 GT 参考系静止（共享位姿模型重投影 RMS 很小），
   则改用"GT 固定相机 + 共享棋盘格位姿"的联合优化结果（更充分利用位姿数据）；
   否则说明目标棋盘格在运动，报告逐幅图像位姿模型的结果。
4. 用 cv2.calibrateCamera（自由相机位姿的经典标定）交叉验证。

运行:  python calib.py [mav0目录]   （默认取本脚本所在目录）
"""

import os
import sys
import re
import argparse

import numpy as np
import cv2
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SQUARE_SIZE = 1.0                 # 棋盘格单元边长 [米]（Minecraft: 1 block = 1 m）
                                  # 注意：只影响棋盘格位姿的尺度，不影响内参估计
PATTERN_CANDIDATES = [(9, 6), (6, 9)]
                                  # 候选内角点数 (列, 行)，按优先级排列：
                                  # 9x16 棋盘格对应 (9,16)（内角）或 (8,15)（方格）
MIN_IMAGES = 3                    # 最少有效图像数
MAX_PNP_INIT_ERR = 5.0            # 初始 PnP 验证阈值 [px]
STATIC_BOARD_MAX_RMS = 3.0        # 共享棋盘格位姿诊断阈值 [px]

# body(x=forward, y=left, z=up) -> OpenCV camera(x=right, y=down, z=forward)
C_BODY2CAM = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def quat_to_R(w, x, y, z):
    """四元数 (w,x,y,z) -> 旋转矩阵 (将 body 系向量旋转到参考系)。"""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_gt(gt_csv):
    """读取 state_groundtruth_estimate0/data.csv -> (ts, p[N,3], q[N,4])。"""
    ts, p, q = [], [], []
    with open(gt_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts.append(int(parts[0]))
            p.append([float(x) for x in parts[1:4]])
            q.append([float(x) for x in parts[4:8]])
    return np.array(ts), np.array(p), np.array(q)


def load_images(cam_csv):
    """读取 cam0/data.csv -> (ts[N], 文件名[N])。"""
    ts, names = [], []
    with open(cam_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts.append(int(parts[0]))
            names.append(parts[1].strip())
    return np.array(ts), names


def interpolate_pose(gt_ts, gt_p, gt_q, t):
    """在 GT 序列上插值位姿（位置线性插值，姿态 slerp）。"""
    if t <= gt_ts[0]:
        return gt_p[0], gt_q[0]
    if t >= gt_ts[-1]:
        return gt_p[-1], gt_q[-1]
    j = np.searchsorted(gt_ts, t)
    t0, t1 = gt_ts[j - 1], gt_ts[j]
    a = (t - t0) / (t1 - t0)
    p = (1 - a) * gt_p[j - 1] + a * gt_p[j]
    q0, q1 = gt_q[j - 1], gt_q[j]
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, dot)
    th = np.arccos(dot)
    if th < 1e-9:
        q = q0.copy()
    else:
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    return p, q / np.linalg.norm(q)


def camera_pose_from_gt(p_rs, q_rs):
    """GT 位姿 -> OpenCV 相机位姿 (R_camR, t_camR)。GT 参考系即相机所在系。"""
    R_rs = quat_to_R(*q_rs)
    return C_BODY2CAM @ R_rs.T, np.asarray(p_rs, dtype=np.float64)


def build_objp(pattern):
    """生成棋盘格三维角点坐标（Z=0，行优先，与 SB 检测顺序一致）。"""
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def detect_corners(gray, pattern):
    """棋盘格检测（SB 优先，经典方法回退）+ 亚像素细化。返回 (N,2) 或 None。"""
    for fn, flags in ((cv2.findChessboardCornersSB, cv2.CALIB_CB_NORMALIZE_IMAGE),
                      (cv2.findChessboardCorners, None)):
        try:
            ret, corners = fn(gray, pattern, flags)
        except cv2.error:
            ret, corners = False, None
        if ret and corners is not None and len(corners) == pattern[0] * pattern[1]:
            c = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            return c.reshape(-1, 2).astype(np.float64)
    return None


def align_corners(corners, objp, K, dist):
    """将检测角点按 0/90/180/270 度旋转对齐到 objp 的角点顺序，
    返回 (对齐后角点, rvec, tvec, 重投影RMS)。无法对齐时返回 None。"""
    cols = int(np.max(objp[:, 0]) / SQUARE_SIZE) + 1
    rows = int(np.max(objp[:, 1]) / SQUARE_SIZE) + 1
    grid = corners.reshape(rows, cols, 2)
    best = None
    for k in range(4):
        c = np.rot90(grid, k).reshape(-1, 2).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(objp, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        pred, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        err = float(np.sqrt(np.mean((pred.reshape(-1, 2) - c) ** 2)))
        if best is None or err < best[0]:
            best = (err, c, rv, tv)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def read_sensor_yaml(path):
    """从 cam0/sensor.yaml 读取 intrinsics 与 resolution（作为初始猜测）。"""
    fx = fy = cx = cy = None
    w = h = None
    try:
        with open(path) as f:
            text = f.read()
        m = re.search(r'intrinsics:\s*\[([^\]]*)\]', text)
        if m:
            vals = [float(x) for x in re.findall(r'[-+0-9.eE]+', m.group(1))]
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
        m = re.search(r'resolution:\s*\[([^\]]*)\]', text)
        if m:
            vals = [int(float(x)) for x in re.findall(r'[-+0-9.eE]+', m.group(1))]
            if len(vals) >= 2:
                w, h = vals[:2]
    except OSError:
        pass
    return fx, fy, cx, cy, w, h


# ---------------------------------------------------------------------------
# 标定主流程
# ---------------------------------------------------------------------------
def main(mav0_dir):
    cam_dir = os.path.join(mav0_dir, 'cam0', 'data')
    cam_csv = os.path.join(mav0_dir, 'cam0', 'data.csv')
    gt_csv = os.path.join(mav0_dir, 'state_groundtruth_estimate0', 'data.csv')
    sensor_yaml = os.path.join(mav0_dir, 'cam0', 'sensor.yaml')

    for p in (cam_dir, cam_csv, gt_csv):
        if not os.path.exists(p):
            print(f"[错误] 缺少文件/目录: {p}")
            return

    # 1) 载入 GT 与图像列表 ------------------------------------------------
    gt_ts, gt_p, gt_q = load_gt(gt_csv)
    img_ts, img_names = load_images(cam_csv)
    print(f"GT 位姿样本: {len(gt_ts)}   图像: {len(img_ts)}")

    # 2) 初始内参猜测 ------------------------------------------------------
    fx0, fy0, cx0, cy0, w, h = read_sensor_yaml(sensor_yaml)
    if w is None:
        first = cv2.imread(os.path.join(cam_dir, img_names[0]), cv2.IMREAD_GRAYSCALE)
        if first is not None:
            h, w = first.shape
        else:
            h, w = 480, 752
    if fx0 is None:
        fx0 = fy0 = float(w)
        cx0, cy0 = w / 2.0, h / 2.0
    K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]], np.float64)
    dist0 = np.zeros(5)
    print(f"初始内参猜测 (sensor.yaml): fx={fx0:.3f} fy={fy0:.3f} cx={cx0:.3f} cy={cy0:.3f}")

    # 3) 自动选择棋盘格尺寸 -------------------------------------------------
    files = [os.path.join(cam_dir, n) for n in img_names]
    grays = []
    for fn in files:
        g = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        grays.append(g)

    chosen = None
    for pat in PATTERN_CANDIDATES:
        hits = 0
        for g in grays:
            if g is not None and detect_corners(g, pat) is not None:
                hits += 1
        print(f"  候选棋盘格 {pat}: 检出 {hits}/{len(files)} 幅")
        if hits >= MIN_IMAGES:
            chosen = pat
            break
    if chosen is None:
        print("[错误] 所有候选棋盘格尺寸的检出图像数均不足，无法标定。")
        return

    # 4) 收集有效观测（检测 + 角点对齐 + PnP 验证） -------------------------
    objp = build_objp(chosen)
    all_corners = []      # 每幅图像: (N,2) 对齐后的角点
    all_poses_R = []      # 每幅图像: 相机位姿 R_camR (3,3)
    all_poses_p = []      # 每幅图像: 相机位置 p_RS  (3,)
    used_idx = []
    for i, g in enumerate(grays):
        if g is None:
            continue
        corners = detect_corners(g, chosen)
        if corners is None:
            continue
        res = align_corners(corners, objp, K0, dist0)
        if res is None or res[3] > MAX_PNP_INIT_ERR:
            continue
        R_camR, p_rs = camera_pose_from_gt(*interpolate_pose(gt_ts, gt_p, gt_q, img_ts[i]))
        all_corners.append(res[0])
        all_poses_R.append(R_camR)
        all_poses_p.append(p_rs)
        used_idx.append(i)
    n_imgs = len(all_corners)
    print(f"使用棋盘格 {chosen} (内角点 列x行), 有效图像 {n_imgs}/{len(files)}")
    if n_imgs < MIN_IMAGES:
        print("[错误] 有效图像不足，无法标定。")
        return

    # 5) 外层 LM 优化内参 + 畸变（相机位姿固定为 GT，棋盘格位姿逐幅 PnP） ----
    cache = {}   # 每幅图像的 PnP 位姿缓存（热启动，保证残差函数平滑）

    def init_cache(K, dist):
        cache.clear()
        for m, (obs, _) in enumerate(zip(all_corners, all_poses_R)):
            ok, rv, tv = cv2.solvePnP(objp, obs, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            cache[m] = (rv, tv) if ok else (None, None)

    def residual(params):
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        dist = np.array([k1, k2, p1, p2, k3], np.float64)
        errs = []
        for m, obs in enumerate(all_corners):
            # 去畸变到归一化坐标（迭代法）
            xu = cv2.undistortPoints(obs.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            rv, tv = cache.get(m, (None, None))
            if rv is not None:
                ok, rv, tv = cv2.solvePnP(objp, xu, np.eye(3), None, rv, tv, True,
                                          cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rv, tv = cv2.solvePnP(objp, xu, np.eye(3), None)
            if not ok:
                errs.append(np.full(2 * len(obs), 100.0))
                continue
            cache[m] = (rv, tv)
            pred, _ = cv2.projectPoints(objp, rv, tv, K, dist)
            errs.append((pred.reshape(-1, 2) - obs).flatten())
        return np.hstack(errs)

    init_cache(K0, dist0)
    x0 = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], 0, 0, 0, 0, 0]
    bounds = ([1.0, 1.0, -2 * w, -2 * h, -0.5, -0.5, -0.5, -0.5, -0.5],
              [1e6, 1e6, 3 * w, 3 * h, 0.5, 0.5, 0.5, 0.5, 0.5])

    # 计算初始重投影误差
    init_errs = []
    for m, obs in enumerate(all_corners):
        rv, tv = cache.get(m, (None, None))
        if rv is not None:
            pred, _ = cv2.projectPoints(objp, rv, tv, K0, dist0)
            init_errs.append((pred.reshape(-1, 2) - obs).flatten())
        else:
            init_errs.append(np.full(2 * len(obs), 100.0))
    init_rms = float(np.sqrt(np.mean(np.hstack(init_errs) ** 2)))
    print(f"初始内参下的重投影误差 RMS (使用初始 PnP 位姿): {init_rms:.4f} px")

    print("优化内参 + 畸变 (相机轨迹固定为 GT 位姿)...")
    result = least_squares(residual, x0, method='trf', bounds=bounds,
                           max_nfev=100, ftol=1e-10, xtol=1e-10, verbose=0)

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = result.x
    K_est = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dist_est = np.array([k1, k2, p1, p2, k3], np.float64)
    rms = float(np.sqrt(np.mean(result.fun ** 2)))

    # 6) 静态棋盘格诊断 -----------------------------------------------------
    # 用第 1 幅图像的棋盘格位姿作为共享位姿，投影到其余图像评估一致性。
    rv0, tv0 = cache[0]
    R_cb, _ = cv2.Rodrigues(rv0)
    R_bR = all_poses_R[0].T @ R_cb
    t_bR = all_poses_R[0].T @ tv0.reshape(3) + all_poses_p[0]
    errs_static = []
    for m, obs in enumerate(all_corners):
        X_R = (R_bR @ objp.T).T + t_bR
        X_c = (all_poses_R[m] @ (X_R - all_poses_p[m]).T).T
        u = fx * X_c[:, 0] / X_c[:, 2] + cx
        v = fy * X_c[:, 1] / X_c[:, 2] + cy
        errs_static.append(np.hstack([(u - obs[:, 0]), (v - obs[:, 1])]))
    rms_static = float(np.sqrt(np.mean(np.hstack(errs_static) ** 2)))

    # 7) 交叉验证: cv2.calibrateCamera (自由相机位姿) ------------------------
    obj_points = [objp.astype(np.float32) for _ in all_corners]
    img_points = [c.astype(np.float32) for c in all_corners]
    ret_cv, mtx_cv, dist_cv, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    # 8) 棋盘格在 GT 系中的运动统计（解释静态诊断） --------------------------
    centers, cam_pos = [], []
    for m in range(n_imgs):
        rv, tv = cache[m]
        R_cb, _ = cv2.Rodrigues(rv)
        R_bR = all_poses_R[m].T @ R_cb
        t_bR = all_poses_R[m].T @ tv.reshape(3) + all_poses_p[m]
        centers.append(R_bR @ np.array([(chosen[0] - 1) / 2, (chosen[1] - 1) / 2, 0]) + t_bR)
        cam_pos.append(all_poses_p[m])
    centers = np.array(centers)
    cam_pos = np.array(cam_pos)
    A_couple = np.sum(centers * cam_pos, axis=0) / np.maximum(np.sum(cam_pos ** 2, axis=0), 1e-12)

    # 9) 输出 ----------------------------------------------------------------
    print("\n" + "=" * 68)
    print("标定结果")
    print("=" * 68)
    print(f"棋盘格内角点: {chosen[0]}x{chosen[1]}   单元边长: {SQUARE_SIZE} m")
    print(f"有效图像: {n_imgs}/{len(files)}   角点观测总数: {n_imgs * chosen[0] * chosen[1]}")
    print(f"重投影误差 RMS: {rms:.4f} px")
    print(f"\n内参矩阵 K:")
    print(f"  fx={fx:.4f}  fy={fy:.4f}")
    print(f"  cx={cx:.4f}  cy={cy:.4f}")
    print(f"  [[{fx:.4f}, 0, {cx:.4f}], [0, {fy:.4f}, {cy:.4f}], [0, 0, 1]]")
    print(f"\n径向畸变:  k1={k1:+.6f}  k2={k2:+.6f}  k3={k3:+.6f}")
    print(f"切向畸变:  p1={p1:+.6f}  p2={p2:+.6f}")
    print(f"\ncv2.calibrateCamera 交叉验证 (自由相机位姿):")
    print(f"  K = [[{mtx_cv[0,0]:.4f}, 0, {mtx_cv[0,2]:.4f}], "
          f"[0, {mtx_cv[1,1]:.4f}, {mtx_cv[1,2]:.4f}], [0, 0, 1]]")
    print(f"  dist = {np.round(dist_cv.ravel(), 6)}")
    print(f"  RMS = {ret_cv:.4f} px")
    print(f"\n静态棋盘格诊断 (GT 固定相机 + 共享棋盘格位姿): RMS = {rms_static:.2f} px")
    if rms_static <= STATIC_BOARD_MAX_RMS:
        print("  -> 棋盘格相对 GT 参考系静止，上述内参即为 GT 约束标定结果。")
    else:
        print("  -> 棋盘格相对 GT 参考系存在运动（共享位姿模型不成立），"
              "内参取自逐幅图像位姿模型；GT 位姿用于固定相机轨迹。")
        print(f"     棋盘格中心与相机位置线性耦合系数 A = "
              f"({A_couple[0]:.3f}, {A_couple[1]:.3f}, {A_couple[2]:.3f})，"
              f"中心标准差 = ({np.std(centers[:,0]):.2f}, {np.std(centers[:,1]):.2f}, "
              f"{np.std(centers[:,2]):.2f}) m")
    print("=" * 68)

    # 10) 结果落盘 -----------------------------------------------------------
    out_path = os.path.join(mav0_dir, 'calib_result.txt')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"pattern: {chosen[0]}x{chosen[1]} inner corners\n")
            f.write(f"square_size_m: {SQUARE_SIZE}\n")
            f.write(f"num_images: {n_imgs}\n")
            f.write(f"reprojection_rms_px: {rms:.6f}\n")
            f.write(f"intrinsics: [{fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}]\n")
            f.write(f"distortion_model: radial-tangential\n")
            f.write(f"distortion_coefficients: [{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}]\n")
            f.write(f"cv2_cross_check_rms_px: {ret_cv:.6f}\n")
            f.write(f"static_board_check_rms_px: {rms_static:.2f}\n")
        print(f"\n结果已写入: {out_path}")
    except OSError as e:
        print(f"\n[警告] 无法写入结果文件: {e}")


if __name__ == '__main__':
    # 管道/重定向输出统一使用 UTF-8，避免 Windows 默认代码页导致的乱码
    try:
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="MinecraftSim EuRoC 相机标定工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'mav0',
        nargs='?',
        help='mav0 目录路径（包含 cam0/ 或 cam1/ 以及 state_groundtruth_estimate0/）',
        default=os.path.dirname(os.path.abspath(__file__))
    )
    args = parser.parse_args()

    main(args.mav0)
