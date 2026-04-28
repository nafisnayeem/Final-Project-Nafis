import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Dynamics-learning PINN for UAV maneuver generalization
# ------------------------------------------------------------
# This version is intentionally DIFFERENT from the sensor-conditioned PINN:
#   - Sensor-conditioned PINN learns observer mapping:   (y_t, u_t, ...) -> x_hat_t
#   - Dynamics-learning PINN learns transition mapping:  (x_t, u_t)      -> x_{t+1}
#
# For fair data generation and comparison, this script uses the same EKF / IEKF
# observer-compatible process and measurement definitions as your EKF/IEKF source:
#   x  = [pn, pe, pd, phi, theta, psi, u, v, w, Vwx, Vwy, Vwz]  (12 states)
#   uin = [Amx, Amy, Amz, p, q, r]                              (6 inputs)
#   z  = [pn, pe, pd, phi, theta, psi, VIx, VIy, VIz, Vt]       (10 measurements)
#
# The PINN learns a hybrid transition model:
#   x_{t+1} = x_t + dt * f_phys(x_t, uin_t) + NN_theta(x_t, uin_t)
#
# Included improvements:
#   * richer excitation maneuvers for state-space coverage / phase portrait coverage
#   * empirical local observability check for the EKF/IEKF measurement set
#   * training/validation/test split (turn is unseen test)
#   * scaling / normalization from training data only
#   * curriculum transfer learning across maneuvers
#   * mini-batch Adam + SciPy L-BFGS-B
#   * one-step + rollout + measurement-consistency + physics-prior + Jacobian regularization losses
#   * explicit auto-differentiation for Jacobian smoothness penalty
#   * beta clamp and no-wind/data consistency fixes from your improvement notes
# ============================================================


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_MANEUVERS = [
    "straight",
    "doublet",
    "chirp",
    "circle",
    "helical_circle",
    "rich",
    "figure8",
    "spiral",
]
TEST_MANEUVER = "turn"

CURRICULUM_PHASES = [
    ["straight"],
    ["straight", "doublet"],
    ["straight", "doublet", "chirp"],
    ["straight", "doublet", "chirp", "circle"],
    ["straight", "doublet", "chirp", "circle", "helical_circle"],
    ["straight", "doublet", "chirp", "circle", "helical_circle", "rich"],
    ["straight", "doublet", "chirp", "circle", "helical_circle", "rich", "figure8"],
    ["straight", "doublet", "chirp", "circle", "helical_circle", "rich", "figure8", "spiral"],
]

VAL_FRACTION = 0.15
WINDOW_HORIZON = 12
BATCH_SIZE = 256
ADAM_EPOCHS_PER_PHASE = 220
ACTIVATION_SWEEP_EPOCHS = 35
PHYSICS_RAMP_EPOCHS = 60
LBFGSB_MAXITER = 60
LBFGSB_SUBSET_WINDOWS = 2000

PINN_HIDDEN = 256
PINN_BLOCKS = 6
TIME_FEATURES = 0  # dynamics is Markovian here; keep time out of the transition map

USE_STATE_INPUT_NOISE = True
STATE_INPUT_NOISE_STD = 0.01
USE_SCHEDULED_SAMPLING = True
SCHEDULED_SAMPLING_MAX = 0.70

SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

# Loss weights
W_ONESTEP = 1.0
W_ROLLOUT = 1.0
W_MEAS = 0.6
W_PHYS_PRIOR = 0.15
W_IC = 1.0
W_JAC = 1e-4
W_REG = 1e-7

# EKF / IEKF settings
EKF_Q_SCALE = 1.0
EKF_R_SCALE = 1.0
EKF_NRK = 4
IEKF_NUM_UPDATE_ITERS = 2

# Sampling / sim
TF = 120.0
TS = 0.05


# ============================================================
# 2. AIRCRAFT CONSTANTS
# ============================================================
g = 9.81
rho = 1.237

m = 3.311
c = 0.254
b = 1.80
S = 0.457

Ixx = 0.319
Iyy = 0.267
Izz = 0.471
Ixz = 0.024
Ixy = 0.0
Iyz = 0.0

Inertia = np.array([
    [Ixx, -Ixy, -Ixz],
    [-Ixy, Iyy, -Iyz],
    [-Ixz, -Iyz, Izz],
], dtype=float)

PropDia = 0.254
nProp = 2
etaProp = 0.90

# Longitudinal coefficients
Cj2 = -0.13096
Cj = -0.04005
Cj0 = 0.115918

Cx0 = 0.009 - 0.4374
Cxq = 0.0
Cxde = 0.051
Cxalpha = 0.282
Cxalpha2 = 3.173 + 0.119

Cz0 = -0.225
Czq = -12.54
Czde = 0.0
Czalpha = -4.436 - 0.015

Cm0 = 0.008
Cmq = -14.019
Cmde = -0.415
Cmalpha = -0.444 - 0.027
Cmdalpha = 0.514 + 0.036

# Lateral-directional coefficients
Cy0 = 0.0
Cyp = 0.221
Cyr = 0.230
Cyda = 0.118
Cydr = 0.136
Cybeta = -0.410 - 0.115

Cl0 = 0.0
Clp = -0.386
Clr = 0.0
Clda = -0.137
Cldr = 0.0
Clbeta = -0.035 - 0.004

Cn0 = 0.0
Cnp = 0.0
Cnr = -0.119
Cnda = 0.013
Cndr = -0.068
Cnbeta = 0.083 + 0.020

VEq = 18.165
thetaEq = 0.0277
deEq = -0.012118

# IMPORTANT FIX FROM IMPROVEMENT NOTES:
# the physics assumes no wind, so keep truth wind zero and do not inject random wind drift.
SIGMA_V = 0.01
SIGMA_OMEGA = 0.001
SIGMA_WIND = 0.0
SIGMA_MEAS = 0.1
SIGMA_AM = 0.03
EPS = 1e-8

STATE_DIM = 12
MEAS_DIM = 10
UIN_DIM = 6
FULL_STATE_DIM = 15

INERTIA_T = torch.tensor(Inertia, dtype=torch.float32, device=device)


# ============================================================
# 3. UTILS
# ============================================================
def set_all_seeds(seed: int = 1234) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def safe_asin_np(x: np.ndarray) -> np.ndarray:
    return np.arcsin(np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6))


def safe_asin_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.asin(torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6))


def smooth_signal(x: np.ndarray, window: int = SAVGOL_WINDOW, poly: int = SAVGOL_POLY) -> np.ndarray:
    if x.shape[0] < window:
        return x.copy()
    if window % 2 == 0:
        window += 1
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        out[:, j] = savgol_filter(x[:, j], window_length=window, polyorder=poly, mode="interp")
    return out


def robust_mse(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    ax = torch.abs(x)
    quad = torch.minimum(ax, torch.tensor(delta, device=x.device))
    lin = ax - quad
    return torch.mean(0.5 * quad ** 2 + delta * lin)


# ============================================================
# 4. ROTATION / KINEMATICS / OBSERVER PROCESS MODEL
# ============================================================
def rotation_matrix_np(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth, cth * sphi, cth * cphi],
    ], dtype=float)


def measurement_from_state12_np(x12: np.ndarray) -> np.ndarray:
    x12 = np.atleast_2d(x12)
    N = x12.shape[0]
    z = np.zeros((N, 10), dtype=float)
    z[:, 0:6] = x12[:, 0:6]
    for i in range(N):
        phi, theta, psi = x12[i, 3:6]
        vr = x12[i, 6:9]
        vw = x12[i, 9:12]
        RIB = rotation_matrix_np(phi, theta, psi)
        VI = RIB @ vr + vw
        z[i, 6:9] = VI
        z[i, 9] = np.linalg.norm(vr)
    return z


def measurement_from_state12_torch(x12: torch.Tensor) -> torch.Tensor:
    pos_ang = x12[:, 0:6]
    phi = x12[:, 3]
    theta = x12[:, 4]
    psi = x12[:, 5]
    u = x12[:, 6]
    v = x12[:, 7]
    w = x12[:, 8]
    Vwx = x12[:, 9]
    Vwy = x12[:, 10]
    Vwz = x12[:, 11]

    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    VIx = cpsi * cth * u + (cpsi * sth * sphi - spsi * cphi) * v + (cpsi * sth * cphi + spsi * sphi) * w + Vwx
    VIy = spsi * cth * u + (spsi * sth * sphi + cpsi * cphi) * v + (spsi * sth * cphi - cpsi * sphi) * w + Vwy
    VIz = -sth * u + cth * sphi * v + cth * cphi * w + Vwz
    Vt = torch.sqrt(u * u + v * v + w * w + 1e-8)
    return torch.cat([pos_ang, torch.stack([VIx, VIy, VIz, Vt], dim=1)], dim=1)


def observer_rhs_from_state12_np(x12: np.ndarray, uin: np.ndarray) -> np.ndarray:
    # x = [pn, pe, pd, phi, theta, psi, u, v, w, Vwx, Vwy, Vwz]
    # uin = [Amx, Amy, Amz, p, q, r]
    pn, pe, pd, phi, theta, psi, u, v, w, Vwx, Vwy, Vwz = x12
    Amx, Amy, Amz, p, q, r = uin

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    pn_dot = Vwx - v * (cth * spsi - cpsi * sphi * sth) + w * (sphi * spsi + cphi * cpsi * sth) + u * cth * cpsi
    pe_dot = Vwy + v * (cphi * cpsi + sphi * sth * spsi) - w * (cpsi * sphi - cphi * sth * spsi) + u * cth * spsi
    pd_dot = Vwz - u * sth + w * cphi * cth + v * cth * sphi

    tan_th = sth / max(cth, 1e-6)
    phi_dot = p + q * sphi * tan_th + r * cphi * tan_th
    theta_dot = q * cphi - r * sphi
    psi_dot = q * sphi / max(cth, 1e-6) + r * cphi / max(cth, 1e-6)

    u_dot = Amx - q * w + r * v - g * sth
    v_dot = Amy + p * w - r * u + g * cth * sphi
    w_dot = Amz - p * v + q * u + g * cphi * cth

    return np.array([pn_dot, pe_dot, pd_dot, phi_dot, theta_dot, psi_dot, u_dot, v_dot, w_dot, 0.0, 0.0, 0.0], dtype=float)


def observer_rhs_from_state12_torch(x12: torch.Tensor, uin: torch.Tensor) -> torch.Tensor:
    pn, pe, pd, phi, theta, psi, u, v, w, Vwx, Vwy, Vwz = torch.unbind(x12, dim=1)
    Amx, Amy, Amz, p, q, r = torch.unbind(uin, dim=1)

    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    pn_dot = Vwx - v * (cth * spsi - cpsi * sphi * sth) + w * (sphi * spsi + cphi * cpsi * sth) + u * cth * cpsi
    pe_dot = Vwy + v * (cphi * cpsi + sphi * sth * spsi) - w * (cpsi * sphi - cphi * sth * spsi) + u * cth * spsi
    pd_dot = Vwz - u * sth + w * cphi * cth + v * cth * sphi

    tan_th = sth / torch.clamp(cth, min=1e-6)
    phi_dot = p + q * sphi * tan_th + r * cphi * tan_th
    theta_dot = q * cphi - r * sphi
    psi_dot = q * sphi / torch.clamp(cth, min=1e-6) + r * cphi / torch.clamp(cth, min=1e-6)

    u_dot = Amx - q * w + r * v - g * sth
    v_dot = Amy + p * w - r * u + g * cth * sphi
    w_dot = Amz - p * v + q * u + g * cphi * cth

    zero = torch.zeros_like(u_dot)
    return torch.stack([pn_dot, pe_dot, pd_dot, phi_dot, theta_dot, psi_dot, u_dot, v_dot, w_dot, zero, zero, zero], dim=1)


# ============================================================
# 5. FULL 15-STATE PLANT FOR DATA GENERATION
# ============================================================
def fixed_wing_eom_np(t, x, tmaneuver, davec, devec, drvec, drps_vec):
    # x = [pn, pe, pd, phi, theta, psi, u, v, w, p, q, r, Vwx, Vwy, Vwz]
    X = x[0:3]
    phi, theta, psi = x[3:6]
    u, v, w = x[6:9]
    p, q, r = x[9:12]

    RotMat = rotation_matrix_np(phi, theta, psi)
    LMat = np.array([
        [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi) / max(np.cos(theta), 1e-6), np.cos(phi) / max(np.cos(theta), 1e-6)],
    ], dtype=float)

    da = np.interp(t, tmaneuver, davec)
    de = np.interp(t, tmaneuver, devec)
    dr = np.interp(t, tmaneuver, drvec)
    drps = np.interp(t, tmaneuver, drps_vec)

    Vt = max(np.sqrt(u * u + v * v + w * w), 1e-6)
    alpha = np.arctan2(w, u)
    beta = safe_asin_np(np.array([v / Vt]))[0]

    phat = p * b / (2.0 * Vt)
    qhat = q * c / (2.0 * Vt)
    rhat = r * b / (2.0 * Vt)
    J = Vt / max(drps * PropDia, 1e-6)

    CJ = Cj0 + Cj * J + Cj2 * J * J

    Cx = Cx0 + Cxq * qhat + Cxde * de + Cxalpha * alpha + Cxalpha2 * alpha * alpha
    Cy = Cy0 + Cyp * phat + Cyr * rhat + Cyda * da + Cydr * dr + Cybeta * beta
    Cz = Cz0 + Czq * qhat + Czde * de + Czalpha * alpha

    dynpres = 0.5 * rho * Vt * Vt
    FX = dynpres * S * Cx + PropDia ** 4 * rho * etaProp * nProp * drps ** 2 * CJ
    FY = dynpres * S * Cy
    FZ = dynpres * S * Cz

    Gravity_body = RotMat.T @ np.array([0.0, 0.0, m * g])
    Force = np.array([FX, FY, FZ]) + Gravity_body

    Cl = Cl0 + Clp * phat + Clr * rhat + Clda * da + Cldr * dr + Clbeta * beta
    Cm = Cm0 + Cmq * qhat + Cmde * de + Cmalpha * alpha
    Cn = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da + Cndr * dr + Cnbeta * beta

    L = dynpres * S * b * Cl
    M = dynpres * S * c * Cm
    N = dynpres * S * b * Cn
    Moment = np.array([L, M, N])

    WX = WY = WZ = 0.0
    Wdot = np.zeros(3)
    Vw = np.array([WX, WY, WZ])

    Xdot = RotMat @ np.array([u, v, w]) + Vw
    Thetadot = LMat @ np.array([p, q, r])
    Vdot = (1.0 / m) * Force + np.cross(np.array([u, v, w]), np.array([p, q, r]))
    omegadot = np.linalg.solve(Inertia, Moment + np.cross(Inertia @ np.array([p, q, r]), np.array([p, q, r])))
    Vwdot = Wdot

    return np.concatenate([Xdot, Thetadot, Vdot, omegadot, Vwdot])


def rk4_fixed_step_np(f, t0, tf, x0, dt, args):
    n_steps = int(round((tf - t0) / dt))
    t = np.linspace(t0, tf, n_steps + 1)
    x = np.zeros((n_steps + 1, len(x0)))
    x[0] = x0
    for i in range(n_steps):
        ti = t[i]
        xi = x[i]
        k1 = f(ti, xi, *args)
        k2 = f(ti + dt / 2.0, xi + dt / 2.0 * k1, *args)
        k3 = f(ti + dt / 2.0, xi + dt / 2.0 * k2, *args)
        k4 = f(ti + dt, xi + dt * k3, *args)
        x[i + 1] = xi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return t, x


# ============================================================
# 6. MANEUVERS / EXCITATION DESIGN
# ============================================================
def _smooth_window(t_exc: np.ndarray, ramp_time: float = 2.0) -> np.ndarray:
    ramp = np.clip(t_exc / max(ramp_time, 1e-6), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * ramp))


def build_controls_np(tvec: np.ndarray, maneuver: str, de_eq_local: float):
    ts = tvec[1] - tvec[0]
    tm = tvec.copy()
    N = len(tvec)

    da = np.zeros(N)
    de = np.ones(N) * de_eq_local
    dr = np.zeros(N)
    drps = np.ones(N) * 220.0

    deg = np.pi / 180.0
    t0 = 4.0
    idx = tm >= t0
    te = tm[idx] - t0
    env = _smooth_window(te, ramp_time=2.5)
    T = max(tm[-1] - t0, 1e-6)

    if maneuver in ("none", "straight"):
        pass

    elif maneuver == "doublet":
        n2 = int(round(2.0 / ts))
        i0 = int(round(8.0 / ts))
        angle = 5.0 * deg
        if i0 + 2 * n2 < N:
            de[i0:i0 + n2] += angle
            de[i0 + n2:i0 + 2 * n2] -= angle

    elif maneuver == "chirp":
        f0a, f1a = 0.03, 0.60
        f0e, f1e = 0.02, 0.45
        f0r, f1r = 0.03, 0.50
        ka = (f1a - f0a) / T
        ke = (f1e - f0e) / T
        kr = (f1r - f0r) / T
        pha = 2 * np.pi * (f0a * te + 0.5 * ka * te ** 2)
        phe = 2 * np.pi * (f0e * te + 0.5 * ke * te ** 2)
        phr = 2 * np.pi * (f0r * te + 0.5 * kr * te ** 2)
        da[idx] = env * (2.8 * deg) * np.sin(pha)
        de[idx] = de_eq_local + env * (3.0 * deg) * np.sin(phe + 0.7)
        dr[idx] = env * (2.0 * deg) * np.sin(phr + 1.2)

    elif maneuver == "circle":
        i0 = int(round(4.0 / ts))
        da[i0:] = 0.35 * deg
        dr[i0:] = 1.0 * deg
        drps[:] = 250.0

    elif maneuver == "helical_circle":
        i0 = int(round(4.0 / ts))
        da[i0:] = 0.42 * deg
        dr[i0:] = 1.1 * deg
        de[idx] = de_eq_local + env * (1.6 * deg) * np.sin(2 * np.pi * 0.06 * te)
        drps[:] = 250.0

    elif maneuver == "rich":
        da[idx] = env * (2.0 * deg) * (
            np.sin(2 * np.pi * 0.08 * te + 0.1)
            + 0.6 * np.sin(2 * np.pi * 0.22 * te + 1.0)
            + 0.35 * np.sin(2 * np.pi * 0.47 * te + 2.1)
        )
        dr[idx] = env * (2.7 * deg) * (
            0.8 * np.sin(2 * np.pi * 0.05 * te + 0.4)
            + 0.5 * np.sin(2 * np.pi * 0.18 * te + 1.7)
            + 0.3 * np.sin(2 * np.pi * 0.38 * te + 2.5)
        )
        de[idx] = de_eq_local + env * (4.0 * deg) * (
            0.9 * np.sin(2 * np.pi * 0.04 * te + 0.3)
            + 0.6 * np.sin(2 * np.pi * 0.14 * te + 1.3)
            + 0.4 * np.sin(2 * np.pi * 0.33 * te + 2.0)
        )

    elif maneuver == "figure8":
        da[idx] = env * (1.9 * deg) * np.sin(2 * np.pi * 0.025 * te) + 0.55 * deg * env * np.sin(2 * np.pi * 0.11 * te + 0.8)
        dr[idx] = 1.0 * deg * env * np.sin(2 * np.pi * 0.025 * te + np.pi / 2)
        de[idx] = de_eq_local + env * (2.2 * deg) * np.sin(2 * np.pi * 0.05 * te + 0.5)

    elif maneuver == "spiral":
        da[idx] = env * (0.9 * deg + 1.0 * deg * np.sin(2 * np.pi * 0.04 * te))
        dr[idx] = env * (1.4 * deg * np.sin(2 * np.pi * 0.03 * te + 1.0))
        de[idx] = de_eq_local + env * (1.2 * deg + 1.8 * deg * np.sin(2 * np.pi * 0.03 * te + 0.7))
        drps[idx] = 235.0 + 15.0 * env * np.sin(2 * np.pi * 0.025 * te)

    elif maneuver == "turn":
        n2 = int(round(2.0 / ts))
        i0 = int(round(4.0 / ts))
        if i0 + n2 < N:
            da[i0:i0 + n2] += 0.25 * deg
            dr[i0:i0 + n2] += 1.0 * deg

    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")

    da = np.clip(da, -6.0 * deg, 6.0 * deg)
    de = np.clip(de, de_eq_local - 12.0 * deg, de_eq_local + 12.0 * deg)
    dr = np.clip(dr, -8.0 * deg, 8.0 * deg)
    drps = np.clip(drps, 200.0, 260.0)

    return tm, da, de, dr, drps


# ============================================================
# 7. DATA GENERATION
# ============================================================
def state15_to_state12(x15: np.ndarray) -> np.ndarray:
    return np.column_stack([x15[:, 0:9], x15[:, 12:15]])


def add_noise_and_construct_io_np(
    tvec: np.ndarray,
    x_true15: np.ndarray,
    tmaneuver: np.ndarray,
    davec: np.ndarray,
    devec: np.ndarray,
    drvec: np.ndarray,
    drps_vec: np.ndarray,
) -> Dict[str, np.ndarray]:
    N = len(tvec)
    x_noisy15 = x_true15.copy()
    x_noisy15[:, 6:9] += SIGMA_V * np.random.randn(N, 3)
    x_noisy15[:, 9:12] += SIGMA_OMEGA * np.random.randn(N, 3)
    x_noisy15[:, 12:15] += SIGMA_WIND * np.random.randn(N, 3)  # zero because truth has no wind

    Am = np.zeros((N, 3), dtype=float)
    pqr_meas = x_noisy15[:, 9:12].copy()

    for i, ti in enumerate(tvec):
        ub, vb, wb = x_noisy15[i, 6:9]
        pb, qb, rb = x_noisy15[i, 9:12]
        da_i = np.interp(ti, tmaneuver, davec)
        de_i = np.interp(ti, tmaneuver, devec)
        dr_i = np.interp(ti, tmaneuver, drvec)
        drps_i = np.interp(ti, tmaneuver, drps_vec)

        Vt = max(np.sqrt(ub * ub + vb * vb + wb * wb), 1e-6)
        alpha = np.arctan2(wb, ub)
        beta = safe_asin_np(np.array([vb / Vt]))[0]

        phat = pb * b / (2.0 * Vt)
        qhat = qb * c / (2.0 * Vt)
        rhat = rb * b / (2.0 * Vt)
        J = Vt / max(drps_i * PropDia, 1e-6)

        CJm = Cj0 + Cj * J + Cj2 * J * J
        Cxm = Cx0 + Cxq * qhat + Cxde * de_i + Cxalpha * alpha + Cxalpha2 * alpha * alpha
        Cym = Cy0 + Cyp * phat + Cyr * rhat + Cyda * da_i + Cydr * dr_i + Cybeta * beta
        Czm = Cz0 + Czq * qhat + Czde * de_i + Czalpha * alpha

        dynpres = 0.5 * rho * Vt * Vt
        Xm = dynpres * S * Cxm + PropDia ** 4 * rho * etaProp * nProp * drps_i ** 2 * CJm
        Ym = dynpres * S * Cym
        Zm = dynpres * S * Czm
        Forcem = np.array([Xm, Ym, Zm])
        Am[i] = Forcem / m + SIGMA_AM * np.random.randn(3)

    x_true12 = state15_to_state12(x_true15)
    x_noisy12 = state15_to_state12(x_noisy15)

    z_true = measurement_from_state12_np(x_true12)
    z_meas = z_true + SIGMA_MEAS * np.random.randn(*z_true.shape)
    z_meas[:, 3] = wrap_angle_np(z_meas[:, 3])
    z_meas[:, 4] = wrap_angle_np(z_meas[:, 4])
    z_meas[:, 5] = wrap_angle_np(z_meas[:, 5])

    uin = np.column_stack([Am, pqr_meas])
    controls = np.column_stack([davec, devec, drvec])

    return {
        "x_true12": x_true12,
        "x_noisy12": x_noisy12,
        "z_true": z_true,
        "z_meas": z_meas,
        "uin": uin,
        "controls": controls,
        "drps": drps_vec.copy(),
        "Am": Am,
        "pqr_meas": pqr_meas,
    }


def simulate_maneuver_np(maneuver: str, tf: float = TF, ts: float = TS) -> Dict[str, np.ndarray]:
    x0 = np.array([0.0, 0.0, -50.0])
    Theta0 = np.array([0.0, thetaEq, 0.0])
    v0 = np.array([VEq * np.cos(thetaEq), 0.0, VEq * np.sin(thetaEq)])
    omega0 = np.array([0.0, 0.0, 0.0])
    Vw0 = np.array([0.0, 0.0, 0.0])
    IC = np.concatenate([x0, Theta0, v0, omega0, Vw0])

    tvec = np.arange(0.0, tf + ts, ts)
    tm, da, de, dr, drps = build_controls_np(tvec, maneuver, deEq)

    t, x15 = rk4_fixed_step_np(
        fixed_wing_eom_np,
        0.0,
        tf,
        IC,
        ts,
        (tm, da, de, dr, drps),
    )

    io = add_noise_and_construct_io_np(t, x15, tm, da, de, dr, drps)
    tau = (t - t[0]) / max(t[-1] - t[0], 1e-8)

    return {
        "maneuver": maneuver,
        "t": t,
        "tau": tau,
        "x15_true": x15,
        "x12_true": io["x_true12"],
        "x12_noisy": io["x_noisy12"],
        "z_true": io["z_true"],
        "z_meas": io["z_meas"],
        "uin": io["uin"],
        "controls": io["controls"],
        "drps": io["drps"],
        "Am": io["Am"],
        "pqr_meas": io["pqr_meas"],
    }


def generate_all_datasets(tf: float = TF, ts: float = TS) -> Dict[str, Dict[str, np.ndarray]]:
    all_mans = TRAIN_MANEUVERS + [TEST_MANEUVER]
    datasets = {}
    for man in all_mans:
        print(f"Simulating {man} ...")
        datasets[man] = simulate_maneuver_np(man, tf=tf, ts=ts)
    return datasets


# ============================================================
# 8. OBSERVABILITY / COVERAGE DIAGNOSTICS
# ============================================================
def fd_jacobian_f(x: np.ndarray, uin: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = x.size
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fp = observer_rhs_from_state12_np(x + dx, uin)
        fm = observer_rhs_from_state12_np(x - dx, uin)
        A[:, i] = (fp - fm) / (2.0 * eps)
    return A


def fd_jacobian_h(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = x.size
    H = np.zeros((MEAS_DIM, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        hp = measurement_from_state12_np((x + dx)[None, :])[0]
        hm = measurement_from_state12_np((x - dx)[None, :])[0]
        H[:, i] = (hp - hm) / (2.0 * eps)
    return H


def local_observability_rank(x: np.ndarray, uin: np.ndarray, dt: float = TS, order: int = STATE_DIM) -> int:
    A = fd_jacobian_f(x, uin)
    H = fd_jacobian_h(x)
    Ad = np.eye(STATE_DIM) + dt * A
    blocks = [H]
    Apow = np.eye(STATE_DIM)
    for _ in range(1, order):
        Apow = Apow @ Ad
        blocks.append(H @ Apow)
    O = np.vstack(blocks)
    return np.linalg.matrix_rank(O, tol=1e-5)


def report_observability(datasets: Dict[str, Dict[str, np.ndarray]], stride: int = 150) -> None:
    print("\n=== Empirical local observability check using EKF/IEKF measurement set ===")
    for man in TRAIN_MANEUVERS + [TEST_MANEUVER]:
        data = datasets[man]
        ranks = []
        for i in range(0, len(data["t"]), stride):
            ranks.append(local_observability_rank(data["x12_true"][i], data["uin"][i]))
        print(f"{man:>14s}: min rank = {min(ranks):2d}, max rank = {max(ranks):2d}, mean rank = {np.mean(ranks):.2f}")


def plot_phase_portraits(datasets: Dict[str, Dict[str, np.ndarray]], save_path: str = "dyn_phase_portraits.png") -> None:
    pairs = [
        (6, 8, "u vs w"),
        (3, 5, "phi vs psi"),
        (0, 1, "pn vs pe"),
        (4, 6, "theta vs u"),
        (7, 10, "v vs Vwy"),
        (2, 11, "pd vs Vwz"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for man in TRAIN_MANEUVERS:
        x = datasets[man]["x12_true"]
        for ax, (i, j, ttl) in zip(axes, pairs):
            ax.plot(x[:, i], x[:, j], lw=1.0, alpha=0.8, label=man)
            ax.set_title(ttl)
            ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


# ============================================================
# 9. NORMALIZATION / WINDOWS / DATASET
# ============================================================
@dataclass
class DynNormalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    u_mean: np.ndarray
    u_std: np.ndarray
    corr_mean: np.ndarray
    corr_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray


def split_indices_contiguous(n: int, val_fraction: float = VAL_FRACTION) -> Tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(round(n * val_fraction)))
    split = n - n_val
    idx_train = np.arange(0, split)
    idx_val = np.arange(split, n)
    return idx_train, idx_val


def build_phase_arrays(
    datasets: Dict[str, Dict[str, np.ndarray]],
    maneuvers: List[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    train_pack = {}
    val_pack = {}
    for man in maneuvers:
        n = len(datasets[man]["t"])
        idx_tr, idx_va = split_indices_contiguous(n, VAL_FRACTION)
        train_pack[man] = {k: v[idx_tr] if isinstance(v, np.ndarray) and v.shape[0] == n else v for k, v in datasets[man].items()}
        val_pack[man] = {k: v[idx_va] if isinstance(v, np.ndarray) and v.shape[0] == n else v for k, v in datasets[man].items()}
    return train_pack, val_pack


def fit_normalizer(train_pack: Dict[str, Dict[str, np.ndarray]], dt: float = TS) -> DynNormalizer:
    X = np.vstack([train_pack[m]["x12_true"][:-1] for m in train_pack])
    U = np.vstack([train_pack[m]["uin"][:-1] for m in train_pack])
    Z = np.vstack([train_pack[m]["z_meas"][1:] for m in train_pack])

    phys_next_all = []
    next_all = []
    for m in train_pack:
        x = train_pack[m]["x12_true"]
        u = train_pack[m]["uin"]
        x_curr = x[:-1]
        x_next = x[1:]
        rhs = np.array([observer_rhs_from_state12_np(xi, ui) for xi, ui in zip(x_curr, u[:-1])])
        x_phys_next = x_curr + dt * rhs
        phys_next_all.append(x_phys_next)
        next_all.append(x_next)
    phys_next_all = np.vstack(phys_next_all)
    next_all = np.vstack(next_all)
    corr = next_all - phys_next_all

    def stats(arr):
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        s = np.where(s < 1e-8, 1.0, s)
        return m, s

    xm, xs = stats(X)
    um, us = stats(U)
    cm, cs = stats(corr)
    zm, zs = stats(Z)
    return DynNormalizer(xm, xs, um, us, cm, cs, zm, zs)


def standardize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / std


class DynWindowDataset(Dataset):
    def __init__(self, pack: Dict[str, Dict[str, np.ndarray]], normalizer: DynNormalizer, horizon: int = WINDOW_HORIZON):
        self.horizon = horizon
        self.samples = []
        man_map = {m: k for k, m in enumerate(pack.keys())}
        for man in pack:
            d = pack[man]
            n = len(d["t"])
            for s in range(0, n - horizon - 1):
                e = s + horizon + 1
                self.samples.append((man_map[man], man, s, e))
        self.pack = pack
        self.normalizer = normalizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, man, s, e = self.samples[idx]
        d = self.pack[man]
        x_seq = d["x12_true"][s:e].astype(np.float32)      # [H+1, 12]
        u_seq = d["uin"][s:e-1].astype(np.float32)         # [H, 6]
        z_next_seq = d["z_meas"][s+1:e].astype(np.float32) # [H, 10]
        tau_seq = d["tau"][s:e].astype(np.float32)

        x_seq_n = standardize(x_seq, self.normalizer.x_mean, self.normalizer.x_std).astype(np.float32)
        u_seq_n = standardize(u_seq, self.normalizer.u_mean, self.normalizer.u_std).astype(np.float32)
        z_seq_n = standardize(z_next_seq, self.normalizer.z_mean, self.normalizer.z_std).astype(np.float32)

        return {
            "x_seq": torch.tensor(x_seq, dtype=torch.float32),
            "u_seq": torch.tensor(u_seq, dtype=torch.float32),
            "z_next_seq": torch.tensor(z_next_seq, dtype=torch.float32),
            "x_seq_n": torch.tensor(x_seq_n, dtype=torch.float32),
            "u_seq_n": torch.tensor(u_seq_n, dtype=torch.float32),
            "z_next_seq_n": torch.tensor(z_seq_n, dtype=torch.float32),
            "tau_seq": torch.tensor(tau_seq, dtype=torch.float32),
        }


# ============================================================
# 10. NETWORK
# ============================================================
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "swish": Swish,
    "sine": Sine,
}


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation_cls):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.ln = nn.LayerNorm(width)
        self.act = activation_cls()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.ln(x + h)


class DynamicsPINN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = STATE_DIM, hidden: int = 256, blocks: int = 6, activation_cls=nn.Tanh):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, activation_cls) for _ in range(blocks)])
        self.act = activation_cls()
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)


# ============================================================
# 11. DYNAMICS MODEL / LOSS
# ============================================================
def state_weights(device_: torch.device) -> torch.Tensor:
    return torch.tensor([10.0, 10.0, 10.0, 4.0, 4.0, 8.0, 6.0, 6.0, 6.0, 2.0, 2.0, 2.0], device=device_)


def meas_weights(device_: torch.device) -> torch.Tensor:
    return torch.tensor([10.0, 10.0, 10.0, 4.0, 4.0, 8.0, 6.0, 6.0, 6.0, 8.0], device=device_)


def normalize_tensor(arr: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=torch.float32, device=arr.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=arr.device)
    return (arr - mean_t) / std_t


def denormalize_tensor(arr: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=torch.float32, device=arr.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=arr.device)
    return arr * std_t + mean_t


def angle_wrap_residual_torch(x_pred: torch.Tensor, x_true: torch.Tensor, angle_indices: List[int] = None) -> torch.Tensor:
    """
    Out-of-place angle-wrapped residual.

    Minimal autograd fix:
    The previous version used in-place slice assignment:
        res[:, idx] = wrap_angle_torch(res[:, idx])
    During rollout training, the same tensors are reused through the recurrent
    graph. That in-place update can change the version counter of a tensor
    needed by backward, producing:
        RuntimeError: one of the variables needed for gradient computation
        has been modified by an inplace operation.
    """
    res = x_pred - x_true
    if angle_indices is None:
        angle_indices = [3, 4, 5]

    cols = []
    for j in range(res.shape[1]):
        col = res[:, j:j + 1]
        if j in angle_indices:
            col = wrap_angle_torch(col)
        cols.append(col)
    return torch.cat(cols, dim=1)


def wrap_state_angles_torch(x: torch.Tensor, angle_indices: List[int] = None) -> torch.Tensor:
    """
    Out-of-place state angle wrapping for tensors that may still be connected
    to the autograd graph.
    """
    if angle_indices is None:
        angle_indices = [3, 4, 5]

    cols = []
    for j in range(x.shape[1]):
        col = x[:, j:j + 1]
        if j in angle_indices:
            col = wrap_angle_torch(col)
        cols.append(col)
    return torch.cat(cols, dim=1)


def hybrid_step(model: nn.Module, x_curr_phys: torch.Tensor, u_curr_phys: torch.Tensor, normalizer: DynNormalizer, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rhs = observer_rhs_from_state12_torch(x_curr_phys, u_curr_phys)
    x_phys_next = x_curr_phys + dt * rhs

    x_n = normalize_tensor(x_curr_phys, normalizer.x_mean, normalizer.x_std)
    u_n = normalize_tensor(u_curr_phys, normalizer.u_mean, normalizer.u_std)
    net_in = torch.cat([x_n, u_n], dim=1)
    corr_n = model(net_in)
    corr_phys = denormalize_tensor(corr_n, normalizer.corr_mean, normalizer.corr_std)
    x_pred_next = x_phys_next + corr_phys
    return x_pred_next, x_phys_next, corr_phys


def jacobian_smoothness_penalty(model: nn.Module, x_curr_phys: torch.Tensor, u_curr_phys: torch.Tensor, normalizer: DynNormalizer) -> torch.Tensor:
    # Explicit auto-diff use, as requested.
    x_in = normalize_tensor(x_curr_phys, normalizer.x_mean, normalizer.x_std).detach().clone().requires_grad_(True)
    u_in = normalize_tensor(u_curr_phys, normalizer.u_mean, normalizer.u_std).detach().clone()
    net_in = torch.cat([x_in, u_in], dim=1)
    corr_n = model(net_in)
    v = torch.randn_like(corr_n)
    jvp_like = torch.autograd.grad((corr_n * v).sum(), x_in, create_graph=True, retain_graph=True)[0]
    return torch.mean(jvp_like ** 2)


def rollout_loss_fn(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    normalizer: DynNormalizer,
    epoch: int,
    dt: float = TS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_seq = batch["x_seq"].to(device)          # [B, H+1, 12]
    u_seq = batch["u_seq"].to(device)          # [B, H, 6]
    z_next_seq = batch["z_next_seq"].to(device)

    B, Hp1, _ = x_seq.shape
    H = Hp1 - 1

    # scheduled sampling ratio increases during Adam training
    if USE_SCHEDULED_SAMPLING:
        ss_ratio = min(SCHEDULED_SAMPLING_MAX, SCHEDULED_SAMPLING_MAX * (epoch + 1) / max(ADAM_EPOCHS_PER_PHASE, 1))
    else:
        ss_ratio = 1.0

    x_prev = x_seq[:, 0, :]
    one_step_terms = []
    rollout_terms = []
    meas_terms = []
    phys_prior_terms = []
    ic_terms = []

    for k in range(H):
        u_k = u_seq[:, k, :]
        x_true_next = x_seq[:, k + 1, :]
        z_true_next = z_next_seq[:, k, :]

        x_pred_next, x_phys_next, corr_phys = hybrid_step(model, x_prev, u_k, normalizer, dt)

        res1 = angle_wrap_residual_torch(x_pred_next, x_true_next)
        one_step_terms.append(robust_mse(res1 * state_weights(device)))

        z_pred_next = measurement_from_state12_torch(x_pred_next)
        z_res = angle_wrap_residual_torch(
            z_pred_next,
            z_true_next,
            angle_indices=[3, 4, 5],
        )
        meas_terms.append(robust_mse(z_res * meas_weights(device)))

        # physics-prior term: keep learned correction modest unless data demands otherwise
        corr_n = normalize_tensor(corr_phys, normalizer.corr_mean, normalizer.corr_std)
        phys_prior_terms.append(robust_mse(corr_n))

        if k == 0:
            ic_res = angle_wrap_residual_torch(x_pred_next[:, [0, 1, 2, 5]], x_true_next[:, [0, 1, 2, 5]], angle_indices=[3])
            ic_terms.append(robust_mse(ic_res))

        # rollout term accumulates as model operates on its own state distribution
        rollout_terms.append(robust_mse(res1 * state_weights(device)))

        if k < H - 1:
            if self_training_mask(B, ss_ratio, device):
                x_prev = x_pred_next
            else:
                x_prev = x_true_next

            if USE_STATE_INPUT_NOISE:
                noise = STATE_INPUT_NOISE_STD * torch.randn_like(x_prev)
                x_prev = x_prev + noise * torch.tensor(normalizer.x_std, dtype=torch.float32, device=device)
                x_prev = wrap_state_angles_torch(x_prev, angle_indices=[3, 4, 5])

    jac_penalty = jacobian_smoothness_penalty(model, x_seq[:, 0, :], u_seq[:, 0, :], normalizer)

    reg = torch.zeros((), device=device)
    for p in model.parameters():
        reg = reg + torch.sum(p ** 2)

    one_step = torch.stack(one_step_terms).mean()
    rollout = torch.stack(rollout_terms).mean()
    meas = torch.stack(meas_terms).mean()
    phys_prior = torch.stack(phys_prior_terms).mean()
    ic = torch.stack(ic_terms).mean() if len(ic_terms) else torch.zeros((), device=device)

    total = (
        W_ONESTEP * one_step
        + W_ROLLOUT * rollout
        + W_MEAS * meas
        + W_PHYS_PRIOR * phys_prior
        + W_IC * ic
        + W_JAC * jac_penalty
        + W_REG * reg
    )

    stats = {
        "total": float(total.detach().cpu()),
        "one": float(one_step.detach().cpu()),
        "roll": float(rollout.detach().cpu()),
        "meas": float(meas.detach().cpu()),
        "phys": float(phys_prior.detach().cpu()),
        "ic": float(ic.detach().cpu()),
        "jac": float(jac_penalty.detach().cpu()),
    }
    return total, stats


def self_training_mask(batch_size: int, ss_ratio: float, device_: torch.device) -> bool:
    # Keep a single decision per batch for cleaner rollout behavior.
    return torch.rand(1, device=device_).item() < ss_ratio


# ============================================================
# 12. ACTIVATION SWEEP / TRAINING
# ============================================================
def make_loaders(train_pack, val_pack, normalizer):
    train_ds = DynWindowDataset(train_pack, normalizer, WINDOW_HORIZON)
    val_ds = DynWindowDataset(val_pack, normalizer, WINDOW_HORIZON)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(2 * BATCH_SIZE, max(len(val_ds), 1)), shuffle=False, drop_last=False)
    return train_loader, val_loader, train_ds, val_ds


def evaluate_model(model, loader, normalizer, epoch_proxy: int = ADAM_EPOCHS_PER_PHASE) -> Dict[str, float]:
    model.eval()
    meters = []
    with torch.enable_grad():
        for batch in loader:
            loss, stats = rollout_loss_fn(model, batch, normalizer, epoch_proxy, TS)
            meters.append(stats)
    out = {}
    for k in meters[0]:
        out[k] = float(np.mean([m[k] for m in meters]))
    return out


def activation_sweep(train_loader, val_loader, normalizer, in_dim: int) -> str:
    print("\n=== Activation sweep for dynamics model ===")
    best_name = None
    best_val = np.inf
    for name, act_cls in ACTIVATIONS.items():
        set_all_seeds(SEED)
        model = DynamicsPINN(in_dim=in_dim, hidden=PINN_HIDDEN, blocks=3, activation_cls=act_cls).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(ACTIVATION_SWEEP_EPOCHS):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                loss, _ = rollout_loss_fn(model, batch, normalizer, ep, TS)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        val_stats = evaluate_model(model, val_loader, normalizer, epoch_proxy=ACTIVATION_SWEEP_EPOCHS)
        print(f"{name:>8s}: val total = {val_stats['total']:.4e}")
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            best_name = name
    print(f"Selected activation: {best_name}")
    return best_name


def train_phase(model, train_loader, val_loader, normalizer, phase_name: str):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = np.inf
    history = {"train": [], "val": []}

    for ep in range(ADAM_EPOCHS_PER_PHASE):
        model.train()
        batch_stats = []
        for batch in train_loader:
            opt.zero_grad()
            loss, stats = rollout_loss_fn(model, batch, normalizer, ep, TS)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_stats.append(stats)

        train_mean = {k: float(np.mean([s[k] for s in batch_stats])) for k in batch_stats[0]}
        val_mean = evaluate_model(model, val_loader, normalizer, epoch_proxy=ep)
        history["train"].append(train_mean)
        history["val"].append(val_mean)

        if val_mean["total"] < best_val:
            best_val = val_mean["total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep == 0 or (ep + 1) % 20 == 0:
            print(
                f"[{phase_name}] epoch {ep+1:4d}/{ADAM_EPOCHS_PER_PHASE} | "
                f"train {train_mean['total']:.4e} | val {val_mean['total']:.4e} | "
                f"one {val_mean['one']:.3e} roll {val_mean['roll']:.3e} meas {val_mean['meas']:.3e} phys {val_mean['phys']:.3e}"
            )

    model.load_state_dict(best_state)
    return history


def flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_params_from_flat(model: nn.Module, flat: np.ndarray) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            val = torch.from_numpy(flat[idx:idx+n].reshape(p.shape)).to(device=device, dtype=p.dtype)
            p.copy_(val)
            idx += n


def lbfgsb_finetune(model, train_ds: Dataset, normalizer: DynNormalizer, maxiter: int = LBFGSB_MAXITER):
    print("Starting SciPy L-BFGS-B fine tuning ...")
    subset_n = min(LBFGSB_SUBSET_WINDOWS, len(train_ds))
    idx = np.random.choice(len(train_ds), size=subset_n, replace=False)
    batch_list = [train_ds[int(i)] for i in idx]

    def collate_subset(items):
        out = {}
        for key in items[0].keys():
            out[key] = torch.stack([it[key] for it in items], dim=0).to(device)
        return out

    big_batch = collate_subset(batch_list)
    x0 = flatten_params(model).astype(np.float64)

    def fun_and_grad(flat):
        set_params_from_flat(model, flat.astype(np.float32))
        model.zero_grad(set_to_none=True)
        loss, _ = rollout_loss_fn(model, big_batch, normalizer, ADAM_EPOCHS_PER_PHASE, TS)
        loss.backward()
        grads = []
        for p in model.parameters():
            grads.append(p.grad.detach().cpu().numpy().ravel().astype(np.float64))
        return float(loss.detach().cpu()), np.concatenate(grads)

    res = minimize(
        fun=lambda x: fun_and_grad(x)[0],
        jac=lambda x: fun_and_grad(x)[1],
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": True, "maxcor": 50},
    )
    set_params_from_flat(model, res.x.astype(np.float32))
    print(f"L-BFGS-B done. success={res.success}, message={res.message}")
    return res


# ============================================================
# 13. PREDICTION / ROLLOUT / EKF / IEKF COMPARISON
# ============================================================
def one_step_predict(model: nn.Module, data: Dict[str, np.ndarray], normalizer: DynNormalizer, dt: float = TS) -> np.ndarray:
    model.eval()
    x = torch.tensor(data["x12_true"][:-1], dtype=torch.float32, device=device)
    u = torch.tensor(data["uin"][:-1], dtype=torch.float32, device=device)
    with torch.no_grad():
        x_next_pred, _, _ = hybrid_step(model, x, u, normalizer, dt)
    y = np.vstack([data["x12_true"][0:1], x_next_pred.cpu().numpy()])
    return y


def free_rollout_predict(model: nn.Module, data: Dict[str, np.ndarray], normalizer: DynNormalizer, dt: float = TS, x0: np.ndarray = None) -> np.ndarray:
    model.eval()
    n = len(data["t"])
    out = np.zeros((n, STATE_DIM), dtype=float)
    out[0] = data["x12_true"][0].copy() if x0 is None else x0.copy()

    x_prev = torch.tensor(out[0:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        for k in range(n - 1):
            u_k = torch.tensor(data["uin"][k:k+1], dtype=torch.float32, device=device)
            x_pred, _, _ = hybrid_step(model, x_prev, u_k, normalizer, dt)
            xp = x_pred.cpu().numpy()[0]
            xp[3] = wrap_angle_np(np.array([xp[3]]))[0]
            xp[4] = wrap_angle_np(np.array([xp[4]]))[0]
            xp[5] = wrap_angle_np(np.array([xp[5]]))[0]
            out[k + 1] = xp
            x_prev = x_pred
    return out


def rmse_with_wrapped_angles(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    err = y_pred - y_true
    for idx in [3, 4, 5]:
        err[:, idx] = wrap_angle_np(err[:, idx])
    return np.sqrt(np.mean(err ** 2, axis=0))


def rk4_observer_step(x: np.ndarray, uin: np.ndarray, dt: float = TS, n_sub: int = EKF_NRK) -> np.ndarray:
    h = dt / n_sub
    xk = x.copy()
    for _ in range(n_sub):
        k1 = observer_rhs_from_state12_np(xk, uin)
        k2 = observer_rhs_from_state12_np(xk + 0.5 * h * k1, uin)
        k3 = observer_rhs_from_state12_np(xk + 0.5 * h * k2, uin)
        k4 = observer_rhs_from_state12_np(xk + h * k3, uin)
        xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xk[3] = wrap_angle_np(np.array([xk[3]]))[0]
        xk[4] = wrap_angle_np(np.array([xk[4]]))[0]
        xk[5] = wrap_angle_np(np.array([xk[5]]))[0]
    return xk


def W_invariant_np(x: np.ndarray) -> np.ndarray:
    phi, theta, psi = x[3:6]
    RIB = rotation_matrix_np(phi, theta, psi)
    W = np.eye(STATE_DIM)
    W[6:9, 6:9] = RIB
    return W


def numerical_jacobian_f_discrete(x: np.ndarray, uin: np.ndarray, dt: float = TS, eps: float = 1e-5) -> np.ndarray:
    n = x.size
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fp = rk4_observer_step(x + dx, uin, dt)
        fm = rk4_observer_step(x - dx, uin, dt)
        A[:, i] = (fp - fm) / (2 * eps)
    return A


def numerical_jacobian_h(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    H = np.zeros((MEAS_DIM, STATE_DIM), dtype=float)
    for i in range(STATE_DIM):
        dx = np.zeros(STATE_DIM)
        dx[i] = eps
        hp = measurement_from_state12_np((x + dx)[None, :])[0]
        hm = measurement_from_state12_np((x - dx)[None, :])[0]
        H[:, i] = (hp - hm) / (2 * eps)
    return H


def ekf_filter(data: Dict[str, np.ndarray], invariant: bool = False) -> np.ndarray:
    z = data["z_meas"]
    uin = data["uin"]
    n = len(data["t"])
    xhat = np.zeros((n, STATE_DIM), dtype=float)
    xhat[0] = data["x12_true"][0].copy()

    # based on EKF/IEKF source magnitudes, but adjusted to match the no-wind truth used here
    q_pos = 1e-5
    q_ang = 1e-5
    q_vel = 1e-2
    q_wind = 1e-3
    Q = EKF_Q_SCALE * np.diag([q_pos, q_pos, q_pos, q_ang, q_ang, q_ang, q_vel, q_vel, q_vel, q_wind, q_wind, q_wind])
    R = EKF_R_SCALE * (0.1 ** 2) * np.eye(MEAS_DIM)
    P = 5.0 * np.eye(STATE_DIM)
    if invariant:
        W0 = W_invariant_np(xhat[0])
        P = W0.T @ P @ W0

    I = np.eye(STATE_DIM)
    for k in range(n - 1):
        x_pred = rk4_observer_step(xhat[k], uin[k], TS)
        A = numerical_jacobian_f_discrete(xhat[k], uin[k], TS)
        P_pred = A @ P @ A.T + Q

        x_upd = x_pred.copy()
        P_upd = P_pred.copy()

        # IEKF source uses invariant-form update x+ = x- + W(x-) K nu.
        n_iters = IEKF_NUM_UPDATE_ITERS if invariant else 1
        for _ in range(n_iters):
            y_pred = measurement_from_state12_np(x_upd[None, :])[0]
            nu = z[k + 1] - y_pred
            nu[3] = wrap_angle_np(np.array([nu[3]]))[0]
            nu[4] = wrap_angle_np(np.array([nu[4]]))[0]
            nu[5] = wrap_angle_np(np.array([nu[5]]))[0]

            H = numerical_jacobian_h(x_upd)
            S = H @ P_upd @ H.T + R
            K = P_upd @ H.T @ np.linalg.inv(S)
            if invariant:
                x_upd = x_pred + W_invariant_np(x_pred) @ (K @ nu)
            else:
                x_upd = x_upd + K @ nu
            x_upd[3] = wrap_angle_np(np.array([x_upd[3]]))[0]
            x_upd[4] = wrap_angle_np(np.array([x_upd[4]]))[0]
            x_upd[5] = wrap_angle_np(np.array([x_upd[5]]))[0]
            P_upd = (I - K @ H) @ P_upd

        xhat[k + 1] = x_upd
        P = P_upd

    return xhat


def plot_turn_results(data, one_step_pred, rollout_pred, ekf_pred, iekf_pred, save_prefix="dynamics_pinn_turn"):
    y_true = data["x12_true"]
    t = data["t"]

    rmse_one = rmse_with_wrapped_angles(y_true, one_step_pred)
    rmse_roll = rmse_with_wrapped_angles(y_true, rollout_pred)
    rmse_ekf = rmse_with_wrapped_angles(y_true, ekf_pred)
    rmse_iekf = rmse_with_wrapped_angles(y_true, iekf_pred)

    names = ["pn", "pe", "pd", "phi", "theta", "psi", "u", "v", "w", "Vwx", "Vwy", "Vwz"]

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    axes = axes.flatten()
    for i in range(12):
        axes[i].plot(t, y_true[:, i], "k", lw=1.3, label="Truth")
        axes[i].plot(t, rollout_pred[:, i], "r--", lw=1.2, label="Dyn-PINN rollout")
        axes[i].plot(t, ekf_pred[:, i], "b-.", lw=1.0, label="EKF")
        axes[i].plot(t, iekf_pred[:, i], color="tab:orange", ls=":", lw=1.2, label="IEKF")
        axes[i].set_title(f"{names[i]} | roll RMSE={rmse_roll[i]:.4f}")
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Dynamics-learning PINN vs EKF / IEKF on unseen TURN maneuver")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_states.png", dpi=220)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ind = np.arange(12)
    width = 0.2
    ax2.bar(ind - 1.5 * width, rmse_one, width, label="PINN one-step")
    ax2.bar(ind - 0.5 * width, rmse_roll, width, label="PINN rollout")
    ax2.bar(ind + 0.5 * width, rmse_ekf, width, label="EKF")
    ax2.bar(ind + 1.5 * width, rmse_iekf, width, label="IEKF")
    ax2.set_xticks(ind)
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_ylabel("RMSE")
    ax2.set_title("TURN RMSE comparison")
    ax2.grid(True, axis="y")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_rmse.png", dpi=220)
    plt.close(fig2)

    return rmse_one, rmse_roll, rmse_ekf, rmse_iekf


# ============================================================
# 14. MAIN
# ============================================================
def main():
    set_all_seeds(SEED)
    datasets = generate_all_datasets(tf=TF, ts=TS)

    report_observability(datasets, stride=140)
    plot_phase_portraits(datasets)

    # Use a single normalizer from the full training maneuver set so transfer learning phases stay consistent.
    full_train_pack, _ = build_phase_arrays(datasets, TRAIN_MANEUVERS)
    normalizer = fit_normalizer(full_train_pack, dt=TS)

    histories = {}
    model = None
    input_dim = STATE_DIM + UIN_DIM

    for phase_idx, mans in enumerate(CURRICULUM_PHASES):
        print(f"\n================ PHASE {phase_idx+1}/{len(CURRICULUM_PHASES)}: {mans} ================")
        train_pack, val_pack = build_phase_arrays(datasets, mans)
        train_loader, val_loader, train_ds, val_ds = make_loaders(train_pack, val_pack, normalizer)

        if model is None:
            act_name = activation_sweep(train_loader, val_loader, normalizer, input_dim)
            model = DynamicsPINN(in_dim=input_dim, hidden=PINN_HIDDEN, blocks=PINN_BLOCKS, activation_cls=ACTIVATIONS[act_name]).to(device)

        history = train_phase(model, train_loader, val_loader, normalizer, phase_name="+".join(mans))
        histories["+".join(mans)] = history
        lbfgsb_finetune(model, train_ds, normalizer, maxiter=LBFGSB_MAXITER)

    # Unseen turn evaluation
    turn_data = datasets[TEST_MANEUVER]
    y_one = one_step_predict(model, turn_data, normalizer, TS)
    y_roll = free_rollout_predict(model, turn_data, normalizer, TS)
    y_ekf = ekf_filter(turn_data, invariant=False)
    y_iekf = ekf_filter(turn_data, invariant=True)

    rmse_one, rmse_roll, rmse_ekf, rmse_iekf = plot_turn_results(turn_data, y_one, y_roll, y_ekf, y_iekf)

    names = ["pn", "pe", "pd", "phi", "theta", "psi", "u", "v", "w", "Vwx", "Vwy", "Vwz"]
    print("\n=== TURN RMSE COMPARISON ===")
    print("\nPINN one-step RMSE")
    for n, r in zip(names, rmse_one):
        print(f"{n:>6s}: {r:.6f}")
    print("\nPINN free-rollout RMSE")
    for n, r in zip(names, rmse_roll):
        print(f"{n:>6s}: {r:.6f}")
    print("\nEKF RMSE")
    for n, r in zip(names, rmse_ekf):
        print(f"{n:>6s}: {r:.6f}")
    print("\nIEKF RMSE")
    for n, r in zip(names, rmse_iekf):
        print(f"{n:>6s}: {r:.6f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "normalizer": normalizer.__dict__,
            "train_maneuvers": TRAIN_MANEUVERS,
            "test_maneuver": TEST_MANEUVER,
            "config": {
                "batch_size": BATCH_SIZE,
                "val_fraction": VAL_FRACTION,
                "window_horizon": WINDOW_HORIZON,
                "scheduled_sampling_max": SCHEDULED_SAMPLING_MAX,
            },
        },
        "dynamics_learning_pinn_uav.pth",
    )
    np.savez(
        "dynamics_turn_results.npz",
        t=turn_data["t"],
        y_true=turn_data["x12_true"],
        y_one=y_one,
        y_roll=y_roll,
        y_ekf=y_ekf,
        y_iekf=y_iekf,
        rmse_one=rmse_one,
        rmse_roll=rmse_roll,
        rmse_ekf=rmse_ekf,
        rmse_iekf=rmse_iekf,
    )
    print("\nSaved outputs:")
    print("  - dynamics_learning_pinn_uav.pth")
    print("  - dynamics_turn_results.npz")
    print("  - dyn_phase_portraits.png")
    print("  - dynamics_pinn_turn_states.png")
    print("  - dynamics_pinn_turn_rmse.png")


if __name__ == "__main__":
    main()