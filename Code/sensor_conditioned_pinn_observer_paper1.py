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
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Sensor-conditioned PINN observer for UAV state estimation
# Minimal autograd/in-place fix version
# ============================================================

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADAM_EPOCHS_PER_PHASE = 350
ACTIVATION_SWEEP_EPOCHS = 40
BATCH_SIZE = 256
VAL_FRACTION = 0.15
LBFGSB_MAXITER = 80
PHYSICS_RAMP_EPOCHS = 75
HISTORY = 1
AUX_USE_CONTROLS = True
STRICT_EKF_IO_ONLY = False
TIME_FOURIER_FEATURES = 4
PINN_HIDDEN = 256
PINN_BLOCKS = 6
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

W_STATE = 1.0
W_MEAS = 1.0
W_PHYS = 2.0
W_IC = 5.0
W_ROLLOUT = 0.5
W_ACCEL = 0.75
W_REG = 1e-6

TRAIN_MANEUVERS = [
    "straight", "doublet", "chirp", "circle", "helical_circle", "rich", "figure8",
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
]

g = 9.81
rho = 1.237
m = 3.311
c = 0.254
b = 1.80
S = 0.457
Ixx, Iyy, Izz = 0.319, 0.267, 0.471
Ixz, Ixy, Iyz = 0.024, 0.0, 0.0
Inertia = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]], dtype=float)
PropDia = 0.254
nProp = 2
etaProp = 0.90

Cj2, Cj, Cj0 = -0.13096, -0.04005, 0.115918
Cx0, Cxq, Cxde, Cxalpha, Cxalpha2 = 0.009 - 0.4374, 0.0, 0.051, 0.282, 3.173 + 0.119
Cz0, Czq, Czde, Czalpha = -0.225, -12.54, 0.0, -4.436 - 0.015
Cm0, Cmq, Cmde, Cmalpha, Cmdalpha = 0.008, -14.019, -0.415, -0.444 - 0.027, 0.514 + 0.036
Cy0, Cyp, Cyr, Cyda, Cydr, Cybeta = 0.0, 0.221, 0.230, 0.118, 0.136, -0.410 - 0.115
Cl0, Clp, Clr, Clda, Cldr, Clbeta = 0.0, -0.386, 0.0, -0.137, 0.0, -0.035 - 0.004
Cn0, Cnp, Cnr, Cnda, Cndr, Cnbeta = 0.0, 0.0, -0.119, 0.013, -0.068, 0.083 + 0.020

VEq = 18.165
thetaEq = 0.0277
deEq = -0.012118
VonKarmanFlag = 3
Amp = 0.0
Omega = 0.0
Omega2 = 0.0
Phase = 0.0
Phase2 = 0.0

SIGMA_V = 0.01
SIGMA_OMEGA = 0.001
SIGMA_WIND = 0.0
SIGMA_MEAS = 0.1
SIGMA_AM = 0.03
SIGMA_ALPHA = 0.03

EPS = 1e-8
STATE_DIM = 12
MEAS_DIM = 10
UIN_DIM = 6
TF = 120.0
TS = 0.05
INERTIA_T = torch.tensor(Inertia, dtype=torch.float32, device=device)


def set_all_seeds(seed: int = 1234) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def safe_asin(x):
    return np.arcsin(np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6))


def safe_asin_torch(x):
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


# -------- Minimal autograd-safe angle helpers --------
def wrap_columns_torch(x: torch.Tensor, angle_indices: List[int]) -> torch.Tensor:
    cols = []
    for j in range(x.shape[1]):
        col = x[:, j:j + 1]
        if j in angle_indices:
            col = wrap_angle_torch(col)
        cols.append(col)
    return torch.cat(cols, dim=1)


def residual_with_wrapped_columns(pred: torch.Tensor, true: torch.Tensor, angle_indices: List[int]) -> torch.Tensor:
    return wrap_columns_torch(pred - true, angle_indices)


def rotation_matrix_np(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth, cth * sphi, cth * cphi],
    ], dtype=float)


def euler_kinematics_np(phi, theta, p, q, r):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    tan_th = sth / max(cth, 1e-6)
    phi_dot = p + q * sphi * tan_th + r * cphi * tan_th
    theta_dot = q * cphi - r * sphi
    psi_dot = q * sphi / max(cth, 1e-6) + r * cphi / max(cth, 1e-6)
    return np.array([phi_dot, theta_dot, psi_dot], dtype=float)


def state15_to_state12(x15: np.ndarray) -> np.ndarray:
    return np.column_stack([x15[:, 0:9], x15[:, 12:15]])


def measurement_from_state15_np(x15: np.ndarray) -> np.ndarray:
    N = x15.shape[0]
    z = np.zeros((N, 10), dtype=float)
    z[:, 0:6] = x15[:, 0:6]
    for i in range(N):
        phi, theta, psi = x15[i, 3:6]
        vr = x15[i, 6:9]
        vw = x15[i, 12:15]
        RIB = rotation_matrix_np(phi, theta, psi)
        VI = RIB @ vr + vw
        z[i, 6:9] = VI
        z[i, 9] = np.linalg.norm(vr)
    return z


def measurement_from_state12_np(x12: np.ndarray, pqr: np.ndarray = None) -> np.ndarray:
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


def observer_rhs_from_state12_np(x12: np.ndarray, uin: np.ndarray) -> np.ndarray:
    pn, pe, pd, phi, theta, psi, u, v, w, Vwx, Vwy, Vwz = x12
    Amx, Amy, Amz, p, q, r = uin
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    pn_dot = Vwx - v * (cth * spsi - cpsi * sphi * sth) + w * (sphi * spsi + cphi * cpsi * sth) + u * cth * cpsi
    pe_dot = Vwy + v * (cphi * cpsi + sphi * sth * spsi) - w * (cpsi * sphi - cphi * sth * spsi) + u * cth * spsi
    pd_dot = Vwz - u * sth + w * cphi * cth + v * cth * sphi
    eul_dot = euler_kinematics_np(phi, theta, p, q, r)
    u_dot = Amx - q * w + r * v - g * sth
    v_dot = Amy + p * w - r * u + g * cth * sphi
    w_dot = Amz - p * v + q * u + g * cphi * cth
    return np.array([pn_dot, pe_dot, pd_dot, eul_dot[0], eul_dot[1], eul_dot[2], u_dot, v_dot, w_dot, 0.0, 0.0, 0.0])


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


def measurement_from_state12_torch(x12: torch.Tensor) -> torch.Tensor:
    pos_ang = x12[:, 0:6]
    phi, theta, psi = x12[:, 3], x12[:, 4], x12[:, 5]
    u, v, w = x12[:, 6], x12[:, 7], x12[:, 8]
    Vwx, Vwy, Vwz = x12[:, 9], x12[:, 10], x12[:, 11]
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)
    VIx = cpsi * cth * u + (cpsi * sth * sphi - spsi * cphi) * v + (cpsi * sth * cphi + spsi * sphi) * w + Vwx
    VIy = spsi * cth * u + (spsi * sth * sphi + cpsi * cphi) * v + (spsi * sth * cphi - cpsi * sphi) * w + Vwy
    VIz = -sth * u + cth * sphi * v + cth * cphi * w + Vwz
    Vt = torch.sqrt(u * u + v * v + w * w + 1e-8)
    return torch.cat([pos_ang, torch.stack([VIx, VIy, VIz, Vt], dim=1)], dim=1)


def fixed_wing_eom_np(t, x, tmaneuver, davec, devec, drvec, drps_vec):
    phi, theta, psi = x[3:6]
    u, v, w = x[6:9]
    p, q, r = x[9:12]
    RotMat = rotation_matrix_np(phi, theta, psi)
    LMat = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                     [0.0, np.cos(phi), -np.sin(phi)],
                     [0.0, np.sin(phi) / max(np.cos(theta), 1e-6), np.cos(phi) / max(np.cos(theta), 1e-6)]], dtype=float)
    da = np.interp(t, tmaneuver, davec)
    de = np.interp(t, tmaneuver, devec)
    dr = np.interp(t, tmaneuver, drvec)
    drps = np.interp(t, tmaneuver, drps_vec)
    Vt = max(np.sqrt(u * u + v * v + w * w), 1e-6)
    alpha = np.arctan2(w, u)
    beta = safe_asin(v / Vt)
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
    Moment = np.array([dynpres * S * b * Cl, dynpres * S * c * Cm, dynpres * S * b * Cn])
    Vw = np.array([0.0, 0.0, 0.0])
    Xdot = RotMat @ np.array([u, v, w]) + Vw
    Thetadot = LMat @ np.array([p, q, r])
    Vdot = (1.0 / m) * Force + np.cross(np.array([u, v, w]), np.array([p, q, r]))
    omegadot = np.linalg.solve(Inertia, Moment + np.cross(Inertia @ np.array([p, q, r]), np.array([p, q, r])))
    Vwdot = np.zeros(3)
    return np.concatenate([Xdot, Thetadot, Vdot, omegadot, Vwdot])


def rk4_fixed_step_np(f, t0, tf, x0, dt, args):
    n_steps = int(round((tf - t0) / dt))
    t = np.linspace(t0, tf, n_steps + 1)
    x = np.zeros((n_steps + 1, len(x0)))
    x[0] = x0
    for i in range(n_steps):
        ti, xi = t[i], x[i]
        k1 = f(ti, xi, *args)
        k2 = f(ti + dt / 2.0, xi + dt / 2.0 * k1, *args)
        k3 = f(ti + dt / 2.0, xi + dt / 2.0 * k2, *args)
        k4 = f(ti + dt, xi + dt * k3, *args)
        x[i + 1] = xi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return t, x


def _smooth_window(t_exc, ramp_time=2.0):
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
        n2 = int(round(2.0 / ts)); i0 = int(round(8.0 / ts)); angle = 5.0 * deg
        if i0 + 2 * n2 < N:
            de[i0:i0 + n2] += angle; de[i0 + n2:i0 + 2 * n2] -= angle
    elif maneuver == "chirp":
        f0a, f1a = 0.03, 0.55; f0e, f1e = 0.02, 0.45; f0r, f1r = 0.03, 0.50
        ka, ke, kr = (f1a - f0a) / T, (f1e - f0e) / T, (f1r - f0r) / T
        da[idx] = env * (2.5 * deg) * np.sin(2 * np.pi * (f0a * te + 0.5 * ka * te ** 2))
        de[idx] = de_eq_local + env * (3.0 * deg) * np.sin(2 * np.pi * (f0e * te + 0.5 * ke * te ** 2) + 0.7)
        dr[idx] = env * (2.0 * deg) * np.sin(2 * np.pi * (f0r * te + 0.5 * kr * te ** 2) + 1.2)
    elif maneuver == "circle":
        i0 = int(round(4.0 / ts)); da[i0:] = 0.35 * deg; dr[i0:] = 1.0 * deg; de[:] = de_eq_local; drps[:] = 250.0
    elif maneuver == "helical_circle":
        i0 = int(round(4.0 / ts)); da[i0:] = 0.40 * deg; dr[i0:] = 1.1 * deg
        de[idx] = de_eq_local + env * (1.5 * deg) * np.sin(2 * np.pi * 0.06 * te); drps[:] = 250.0
    elif maneuver == "rich":
        da[idx] = env * (1.8 * deg) * (np.sin(2 * np.pi * 0.08 * te + 0.1) + 0.6 * np.sin(2 * np.pi * 0.22 * te + 1.0) + 0.35 * np.sin(2 * np.pi * 0.47 * te + 2.1))
        dr[idx] = env * (2.5 * deg) * (0.8 * np.sin(2 * np.pi * 0.05 * te + 0.4) + 0.5 * np.sin(2 * np.pi * 0.18 * te + 1.7) + 0.3 * np.sin(2 * np.pi * 0.38 * te + 2.5))
        de[idx] = de_eq_local + env * (3.8 * deg) * (0.9 * np.sin(2 * np.pi * 0.04 * te + 0.3) + 0.6 * np.sin(2 * np.pi * 0.14 * te + 1.3) + 0.4 * np.sin(2 * np.pi * 0.33 * te + 2.0))
    elif maneuver == "figure8":
        bank_cmd = env * (1.8 * deg) * np.sin(2 * np.pi * 0.025 * te)
        da[idx] = bank_cmd + 0.5 * deg * env * np.sin(2 * np.pi * 0.11 * te + 0.8)
        dr[idx] = 0.9 * deg * env * np.sin(2 * np.pi * 0.025 * te + np.pi / 2)
        de[idx] = de_eq_local + env * (2.2 * deg) * np.sin(2 * np.pi * 0.05 * te + 0.5)
    elif maneuver == "turn":
        n2 = int(round(2.0 / ts)); i0 = int(round(4.0 / ts))
        if i0 + n2 < N:
            da[i0:i0 + n2] += 0.25 * deg; dr[i0:i0 + n2] += 1.0 * deg
    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")
    da = np.clip(da, -6.0 * deg, 6.0 * deg)
    de = np.clip(de, de_eq_local - 12.0 * deg, de_eq_local + 12.0 * deg)
    dr = np.clip(dr, -8.0 * deg, 8.0 * deg)
    return tm, da, de, dr, drps


def add_noise_and_construct_observer_io_np(tvec, x_true15, tmaneuver, davec, devec, drvec, drps_vec):
    N = len(tvec)
    x_noisy15 = x_true15.copy()
    x_noisy15[:, 6:9] += SIGMA_V * np.random.randn(N, 3)
    x_noisy15[:, 9:12] += SIGMA_OMEGA * np.random.randn(N, 3)
    x_noisy15[:, 12:15] += SIGMA_WIND * np.random.randn(N, 3)
    Am = np.zeros((N, 3)); Alpham = np.zeros((N, 3))
    for i, ti in enumerate(tvec):
        ub, vb, wb = x_noisy15[i, 6:9]; pb, qb, rb = x_noisy15[i, 9:12]
        da_i = np.interp(ti, tmaneuver, davec); de_i = np.interp(ti, tmaneuver, devec); dr_i = np.interp(ti, tmaneuver, drvec); drps_i = np.interp(ti, tmaneuver, drps_vec)
        Vt = max(np.sqrt(ub * ub + vb * vb + wb * wb), 1e-6)
        alpha = np.arctan2(wb, ub); beta = safe_asin(vb / Vt)
        phat = pb * b / (2.0 * Vt); qhat = qb * c / (2.0 * Vt); rhat = rb * b / (2.0 * Vt); J = Vt / max(drps_i * PropDia, 1e-6)
        CJm = Cj0 + Cj * J + Cj2 * J * J
        Cxm = Cx0 + Cxq * qhat + Cxde * de_i + Cxalpha * alpha + Cxalpha2 * alpha * alpha
        Cym = Cy0 + Cyp * phat + Cyr * rhat + Cyda * da_i + Cydr * dr_i + Cybeta * beta
        Czm = Cz0 + Czq * qhat + Czde * de_i + Czalpha * alpha
        dynpres = 0.5 * rho * Vt * Vt
        Xm = dynpres * S * Cxm + PropDia ** 4 * rho * etaProp * nProp * drps_i ** 2 * CJm
        Ym = dynpres * S * Cym; Zm = dynpres * S * Czm
        Clm = Cl0 + Clp * phat + Clr * rhat + Clda * da_i + Cldr * dr_i + Clbeta * beta
        Cmm = Cm0 + Cmq * qhat + Cmde * de_i + Cmalpha * alpha
        Cnm = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da_i + Cndr * dr_i + Cnbeta * beta
        Momentm = np.array([dynpres * S * b * Clm, dynpres * S * c * Cmm, dynpres * S * b * Cnm])
        Am[i] = np.array([Xm, Ym, Zm]) / m + SIGMA_AM * np.random.randn(3)
        Alpham[i] = np.linalg.solve(Inertia, Momentm) + SIGMA_ALPHA * np.random.randn(3)
    z_true = measurement_from_state15_np(x_true15)
    z_meas = z_true + SIGMA_MEAS * np.random.randn(*z_true.shape)
    z_meas[:, 3] = wrap_angle(z_meas[:, 3]); z_meas[:, 4] = wrap_angle(z_meas[:, 4]); z_meas[:, 5] = wrap_angle(z_meas[:, 5])
    pqr_meas = x_noisy15[:, 9:12]
    uin = np.column_stack([Am, pqr_meas])
    return {"x_true15": x_true15, "x_true12": state15_to_state12(x_true15), "x_noisy12": state15_to_state12(x_noisy15), "z_true": z_true, "z_meas": z_meas, "uin": uin, "Am": Am, "Alpham": Alpham, "pqr_meas": pqr_meas, "controls": np.column_stack([davec, devec, drvec]), "drps": drps_vec.copy()}


def make_network_input(tau, uin, z_meas, controls, drps):
    feats = [tau.reshape(-1, 1), uin, z_meas]
    if not STRICT_EKF_IO_ONLY and AUX_USE_CONTROLS:
        feats.extend([controls, drps.reshape(-1, 1)])
    return np.concatenate(feats, axis=1)


def simulate_maneuver_np(maneuver: str, tf: float = TF, ts: float = TS):
    x0 = np.array([0.0, 0.0, -50.0])
    Theta0 = np.array([0.0, thetaEq, 0.0])
    v0 = np.array([VEq * np.cos(thetaEq), 0.0, VEq * np.sin(thetaEq)])
    omega0 = np.array([0.0, 0.0, 0.0])
    Vw0 = np.array([0.0, 0.0, 0.0])
    IC = np.concatenate([x0, Theta0, v0, omega0, Vw0])
    tvec = np.arange(0.0, tf + ts, ts)
    tm, da, de, dr, drps = build_controls_np(tvec, maneuver, deEq)
    t, x15 = rk4_fixed_step_np(fixed_wing_eom_np, 0.0, tf, IC, ts, (tm, da, de, dr, drps))
    io = add_noise_and_construct_observer_io_np(t, x15, tm, da, de, dr, drps)
    tau = (t - t[0]) / max(t[-1] - t[0], 1e-8)
    X_in = make_network_input(tau, io["uin"], io["z_meas"], io["controls"], io["drps"])
    return {"maneuver": maneuver, "t": t, "tau": tau, "x15_true": x15, "x12_true": io["x_true12"], "x12_noisy": io["x_noisy12"], "z_true": io["z_true"], "z_meas": io["z_meas"], "uin": io["uin"], "Am": io["Am"], "Alpham": io["Alpham"], "pqr_meas": io["pqr_meas"], "controls": io["controls"], "drps": io["drps"], "X_in": X_in}


def generate_all_datasets(tf: float = TF, ts: float = TS):
    out = {}
    for man in TRAIN_MANEUVERS + [TEST_MANEUVER]:
        print(f"Simulating {man} ...")
        out[man] = simulate_maneuver_np(man, tf=tf, ts=ts)
    return out


def fd_jacobian_f(x, uin, eps=1e-5):
    n = x.size; A = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (observer_rhs_from_state12_np(x + dx, uin) - observer_rhs_from_state12_np(x - dx, uin)) / (2.0 * eps)
    return A


def fd_jacobian_h(x, eps=1e-5):
    n = x.size; H = np.zeros((MEAS_DIM, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        hp = measurement_from_state12_np((x + dx)[None, :])[0]
        hm = measurement_from_state12_np((x - dx)[None, :])[0]
        H[:, i] = (hp - hm) / (2.0 * eps)
    return H


def local_observability_rank(x, uin, dt=TS, order=STATE_DIM):
    A = fd_jacobian_f(x, uin); H = fd_jacobian_h(x); Ad = np.eye(STATE_DIM) + dt * A
    blocks = [H]; Apow = np.eye(STATE_DIM)
    for _ in range(1, order):
        Apow = Apow @ Ad; blocks.append(H @ Apow)
    return np.linalg.matrix_rank(np.vstack(blocks), tol=1e-5)


def report_observability(datasets, stride=150):
    print("\n=== Empirical local observability check (rank of stacked [H; HA; ...]) ===")
    for man in TRAIN_MANEUVERS + [TEST_MANEUVER]:
        data = datasets[man]
        ranks = [local_observability_rank(data["x12_true"][i], data["uin"][i]) for i in range(0, len(data["t"]), stride)]
        print(f"{man:>14s}: min rank = {min(ranks):2d}, max rank = {max(ranks):2d}, mean rank = {np.mean(ranks):.2f}")


def plot_phase_portraits(datasets, save_path="phase_portraits_training.png"):
    pairs = [(3, 0, "phi vs pn"), (4, 6, "theta vs u"), (5, 11, "psi vs Vwz"), (6, 8, "u vs w"), (1, 2, "pe vs pd"), (7, 10, "v vs Vwy")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.flatten()
    for man in TRAIN_MANEUVERS:
        x = datasets[man]["x12_true"]
        for ax, (i, j, ttl) in zip(axes, pairs):
            ax.plot(x[:, i], x[:, j], lw=1.0, alpha=0.8, label=man); ax.set_title(ttl); ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, ncol=4, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(save_path, dpi=220); plt.close(fig)


@dataclass
class Normalizer:
    x_mean: np.ndarray; x_std: np.ndarray; y_mean: np.ndarray; y_std: np.ndarray; z_mean: np.ndarray; z_std: np.ndarray; u_mean: np.ndarray; u_std: np.ndarray; a_mean: np.ndarray; a_std: np.ndarray


def split_indices_contiguous(n: int, val_fraction: float = VAL_FRACTION):
    n_val = max(1, int(round(n * val_fraction))); split = n - n_val
    return np.arange(0, split), np.arange(split, n)


def build_phase_arrays(datasets, maneuvers):
    train_pack, val_pack = {}, {}
    for man in maneuvers:
        n = len(datasets[man]["t"]); idx_tr, idx_va = split_indices_contiguous(n, VAL_FRACTION)
        train_pack[man] = {k: v[idx_tr] if isinstance(v, np.ndarray) and v.shape[0] == n else v for k, v in datasets[man].items()}
        val_pack[man] = {k: v[idx_va] if isinstance(v, np.ndarray) and v.shape[0] == n else v for k, v in datasets[man].items()}
    return train_pack, val_pack


def fit_normalizer(train_pack):
    X = np.vstack([train_pack[m]["X_in"] for m in train_pack]); Y = np.vstack([train_pack[m]["x12_true"] for m in train_pack])
    Z = np.vstack([train_pack[m]["z_meas"] for m in train_pack]); U = np.vstack([train_pack[m]["uin"] for m in train_pack]); A = np.vstack([train_pack[m]["Am"] for m in train_pack])
    def stats(arr):
        mean = arr.mean(axis=0); std = arr.std(axis=0); std = np.where(std < 1e-8, 1.0, std); return mean, std
    xm, xs = stats(X); ym, ys = stats(Y); zm, zs = stats(Z); um, us = stats(U); am, astd = stats(A)
    return Normalizer(xm, xs, ym, ys, zm, zs, um, us, am, astd)


def standardize(arr, mean, std): return (arr - mean) / std


class ObserverDataset(Dataset):
    def __init__(self, pack, normalizer):
        X_list = []; Y_list = []; Z_list = []; U_list = []; A_list = []; T_list = []; man_id = []
        for k, man in enumerate(pack):
            d = pack[man]
            X_list.append(standardize(d["X_in"], normalizer.x_mean, normalizer.x_std))
            Y_list.append(standardize(d["x12_true"], normalizer.y_mean, normalizer.y_std))
            Z_list.append(standardize(d["z_meas"], normalizer.z_mean, normalizer.z_std))
            U_list.append(standardize(d["uin"], normalizer.u_mean, normalizer.u_std))
            A_list.append(standardize(d["Am"], normalizer.a_mean, normalizer.a_std))
            T_list.append(d["tau"][:, None]); man_id.append(np.full((len(d["tau"]), 1), k, dtype=np.int64))
        self.X = torch.tensor(np.vstack(X_list), dtype=torch.float32)
        self.Y = torch.tensor(np.vstack(Y_list), dtype=torch.float32)
        self.Z = torch.tensor(np.vstack(Z_list), dtype=torch.float32)
        self.U = torch.tensor(np.vstack(U_list), dtype=torch.float32)
        self.A = torch.tensor(np.vstack(A_list), dtype=torch.float32)
        self.tau = torch.tensor(np.vstack(T_list), dtype=torch.float32)
        self.man_id = torch.tensor(np.vstack(man_id), dtype=torch.long)
        self.n = self.X.shape[0]
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return {"X": self.X[idx], "Y": self.Y[idx], "Z": self.Z[idx], "U": self.U[idx], "A": self.A[idx], "tau": self.tau[idx], "man_id": self.man_id[idx]}


class Sine(nn.Module):
    def forward(self, x): return torch.sin(x)

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "swish": Swish, "sine": Sine}


class FourierTimeEncoding(nn.Module):
    def __init__(self, n_freq=4): super().__init__(); self.n_freq = n_freq
    def forward(self, tau):
        outs = [tau]
        for k in range(self.n_freq):
            freq = 2.0 ** k * math.pi; outs.append(torch.sin(freq * tau)); outs.append(torch.cos(freq * tau))
        return torch.cat(outs, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width, activation_cls):
        super().__init__(); self.fc1 = nn.Linear(width, width); self.fc2 = nn.Linear(width, width); self.ln = nn.LayerNorm(width); self.act = activation_cls()
    def forward(self, x):
        h = self.act(self.fc1(x)); h = self.fc2(h); return self.ln(x + h)


class SensorConditionedPINN(nn.Module):
    def __init__(self, base_in_dim, hidden=256, blocks=6, activation_cls=nn.Tanh):
        super().__init__(); self.time_enc = FourierTimeEncoding(TIME_FOURIER_FEATURES); self.base_in_dim = base_in_dim
        self.time_encoded_dim = 1 + 2 * TIME_FOURIER_FEATURES; self.other_dim = base_in_dim - 1
        in_dim = self.time_encoded_dim + self.other_dim
        self.fc_in = nn.Linear(in_dim, hidden); self.blocks = nn.ModuleList([ResidualBlock(hidden, activation_cls) for _ in range(blocks)]); self.act = activation_cls(); self.fc_out = nn.Linear(hidden, STATE_DIM)
    def forward(self, x):
        tau = x[:, :1]; other = x[:, 1:]; feat = torch.cat([self.time_enc(tau), other], dim=1)
        h = self.act(self.fc_in(feat))
        for blk in self.blocks: h = blk(h)
        return self.fc_out(h)


def robust_mse(res, delta=1.0):
    abs_r = torch.abs(res); quad = torch.minimum(abs_r, torch.tensor(delta, device=res.device)); lin = abs_r - quad
    return torch.mean(0.5 * quad ** 2 + delta * lin)


def denorm_y(y_std, y_mean, y_hat_norm): return y_hat_norm * y_std + y_mean

def denorm_u(u_std, u_mean, u_norm): return u_norm * u_std + u_mean

def denorm_z(z_std, z_mean, z_norm): return z_norm * z_std + z_mean

def denorm_a(a_std, a_mean, a_norm): return a_norm * a_std + a_mean


def state_residual_with_wrapping(y_pred_norm, y_true_norm, y_mean_t, y_std_t):
    y_pred_phys = denorm_y(y_std_t, y_mean_t, y_pred_norm); y_true_phys = denorm_y(y_std_t, y_mean_t, y_true_norm)
    res = y_pred_norm - y_true_norm
    cols = []
    for j in range(res.shape[1]):
        col = res[:, j:j + 1]
        if j == 5:
            col = wrap_angle_torch(y_pred_phys[:, 5:6] - y_true_phys[:, 5:6]) / y_std_t[5]
        cols.append(col)
    res = torch.cat(cols, dim=1)
    weights = torch.tensor([10.0, 10.0, 10.0, 3.0, 3.0, 10.0, 6.0, 6.0, 6.0, 2.0, 2.0, 2.0], device=res.device, dtype=torch.float32)
    return res, weights


def make_tau_requires_grad(x_batch):
    # Keep tau as the actual leaf tensor used for autograd. Do not ask grad wrt xg[:, :1].
    other = x_batch[:, 1:].detach().clone()
    tau = x_batch[:, :1].detach().clone().requires_grad_(True)
    xg = torch.cat([tau, other], dim=1)
    return xg, tau


def physics_time_derivative_only(model, x_batch_norm, tf_sec, y_std_t, y_mean_t):
    xg, tau = make_tau_requires_grad(x_batch_norm)
    y_pred_norm = model(xg)
    y_pred_phys = denorm_y(y_std_t, y_mean_t, y_pred_norm)
    grads = []
    for j in range(y_pred_phys.shape[1]):
        gj = torch.autograd.grad(y_pred_phys[:, j].sum(), tau, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if gj is None:
            gj = torch.zeros_like(tau)
        grads.append(gj)
    dY_dtau = torch.cat(grads, dim=1)
    dY_dt = dY_dtau * (1.0 / max(tf_sec, 1e-8))
    return dY_dt, y_pred_norm, y_pred_phys


def observer_loss(model, batch, normalizer, tf_sec, physics_weight_eff, rollout_neighbor=None):
    y_mean_t = torch.tensor(normalizer.y_mean, dtype=torch.float32, device=device); y_std_t = torch.tensor(normalizer.y_std, dtype=torch.float32, device=device)
    z_mean_t = torch.tensor(normalizer.z_mean, dtype=torch.float32, device=device); z_std_t = torch.tensor(normalizer.z_std, dtype=torch.float32, device=device)
    u_mean_t = torch.tensor(normalizer.u_mean, dtype=torch.float32, device=device); u_std_t = torch.tensor(normalizer.u_std, dtype=torch.float32, device=device)
    a_mean_t = torch.tensor(normalizer.a_mean, dtype=torch.float32, device=device); a_std_t = torch.tensor(normalizer.a_std, dtype=torch.float32, device=device)
    X = batch["X"].to(device); Y = batch["Y"].to(device); Z = batch["Z"].to(device); U = batch["U"].to(device); A = batch["A"].to(device)
    dYdt, Y_pred_norm, Y_pred_phys = physics_time_derivative_only(model, X, tf_sec, y_std_t, y_mean_t)
    res_state, w_state = state_residual_with_wrapping(Y_pred_norm, Y, y_mean_t, y_std_t)
    state_loss = robust_mse(res_state * w_state)
    Z_pred_phys = measurement_from_state12_torch(Y_pred_phys); Z_true_phys = denorm_z(z_std_t, z_mean_t, Z)
    meas_res = residual_with_wrapped_columns(Z_pred_phys, Z_true_phys, [3, 4, 5])
    z_weights = torch.tensor([10.0, 10.0, 10.0, 4.0, 4.0, 8.0, 6.0, 6.0, 6.0, 8.0], device=device)
    meas_loss = robust_mse((meas_res / z_std_t) * z_weights)
    U_phys = denorm_u(u_std_t, u_mean_t, U)
    rhs = observer_rhs_from_state12_torch(Y_pred_phys, U_phys)
    phys_loss = robust_mse(dYdt - rhs)
    A_true_phys = denorm_a(a_std_t, a_mean_t, A)
    phi, theta = Y_pred_phys[:, 3], Y_pred_phys[:, 4]
    u_b, v_b, w_b = Y_pred_phys[:, 6], Y_pred_phys[:, 7], Y_pred_phys[:, 8]
    p, q, r = U_phys[:, 3], U_phys[:, 4], U_phys[:, 5]
    Amx_pred = dYdt[:, 6] + q * w_b - r * v_b + g * torch.sin(theta)
    Amy_pred = dYdt[:, 7] - p * w_b + r * u_b - g * torch.cos(theta) * torch.sin(phi)
    Amz_pred = dYdt[:, 8] + p * v_b - q * u_b - g * torch.cos(phi) * torch.cos(theta)
    A_pred_phys = torch.stack([Amx_pred, Amy_pred, Amz_pred], dim=1)
    accel_loss = robust_mse((A_pred_phys - A_true_phys) / a_std_t)
    tau = X[:, 0]
    ic_mask = tau < (10.0 / max(tf_sec / 0.05, 10.0))
    if torch.any(ic_mask):
        pred_ic_phys = denorm_y(y_std_t, y_mean_t, Y_pred_norm[ic_mask]); true_ic_phys = denorm_y(y_std_t, y_mean_t, Y[ic_mask])
        ic_res_full = Y_pred_norm[ic_mask] - Y[ic_mask]
        cols = []
        for j in range(ic_res_full.shape[1]):
            col = ic_res_full[:, j:j+1]
            if j == 5:
                col = wrap_angle_torch(pred_ic_phys[:, 5:6] - true_ic_phys[:, 5:6]) / y_std_t[5]
            cols.append(col)
        ic_res_full = torch.cat(cols, dim=1)
        ic_loss = robust_mse(ic_res_full[:, [0, 1, 2, 5]])
    else:
        ic_loss = torch.zeros((), device=device)
    rollout_loss = torch.zeros((), device=device)
    if rollout_neighbor is not None:
        Yn = rollout_neighbor["Y"].to(device); dt = rollout_neighbor["dt"]
        Y_pred_roll_phys = Y_pred_phys + dt * rhs; Yn_phys = denorm_y(y_std_t, y_mean_t, Yn)
        roll_res = residual_with_wrapped_columns(Y_pred_roll_phys, Yn_phys, [5])
        rollout_loss = robust_mse(roll_res)
    reg_loss = torch.zeros((), device=device)
    for p_ in model.parameters(): reg_loss = reg_loss + torch.sum(p_ ** 2)
    total = W_STATE * state_loss + W_MEAS * meas_loss + physics_weight_eff * W_PHYS * phys_loss + W_IC * ic_loss + W_ROLLOUT * rollout_loss + W_ACCEL * accel_loss + W_REG * reg_loss
    stats = {"total": float(total.detach().cpu()), "state": float(state_loss.detach().cpu()), "meas": float(meas_loss.detach().cpu()), "phys": float(phys_loss.detach().cpu()), "ic": float(ic_loss.detach().cpu()), "roll": float(rollout_loss.detach().cpu()), "accel": float(accel_loss.detach().cpu())}
    return total, stats


def make_loaders(train_pack, val_pack, normalizer):
    train_ds = ObserverDataset(train_pack, normalizer); val_ds = ObserverDataset(val_pack, normalizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(2 * BATCH_SIZE, max(len(val_ds), 1)), shuffle=False, drop_last=False)
    return train_loader, val_loader, train_ds, val_ds


def evaluate_model(model, loader, normalizer, tf_sec):
    model.eval(); meters = []
    with torch.enable_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, stats = observer_loss(model, batch, normalizer, tf_sec, physics_weight_eff=W_PHYS)
            meters.append(stats)
    return {k: float(np.mean([m[k] for m in meters])) for k in meters[0]}


def activation_sweep(train_loader, val_loader, normalizer, input_dim, tf_sec):
    print("\n=== Activation sweep ===")
    best_name = None; best_val = np.inf
    for name, act_cls in ACTIVATIONS.items():
        set_all_seeds(SEED)
        model = SensorConditionedPINN(input_dim, hidden=PINN_HIDDEN, blocks=3, activation_cls=act_cls).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(ACTIVATION_SWEEP_EPOCHS):
            model.train(); lam = min(1.0, (ep + 1) / max(PHYSICS_RAMP_EPOCHS, 1))
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(); loss, _ = observer_loss(model, batch, normalizer, tf_sec, physics_weight_eff=lam)
                if not torch.isfinite(loss): continue
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val_stats = evaluate_model(model, val_loader, normalizer, tf_sec)
        print(f"{name:>8s}: val total = {val_stats['total']:.4e}")
        if val_stats["total"] < best_val: best_val = val_stats["total"]; best_name = name
    print(f"Selected activation: {best_name}")
    return best_name


def train_phase(model, train_loader, val_loader, normalizer, tf_sec, phase_name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_val = np.inf; history = {"train": [], "val": []}
    for ep in range(ADAM_EPOCHS_PER_PHASE):
        model.train(); batch_stats = []; lam = min(1.0, (ep + 1) / max(PHYSICS_RAMP_EPOCHS, 1))
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(); loss, stats = observer_loss(model, batch, normalizer, tf_sec, physics_weight_eff=lam)
            if not torch.isfinite(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); batch_stats.append(stats)
        if not batch_stats:
            continue
        train_mean = {k: float(np.mean([s[k] for s in batch_stats])) for k in batch_stats[0]}
        val_mean = evaluate_model(model, val_loader, normalizer, tf_sec)
        history["train"].append(train_mean); history["val"].append(val_mean)
        if val_mean["total"] < best_val:
            best_val = val_mean["total"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 25 == 0:
            print(f"[{phase_name}] epoch {ep+1:4d}/{ADAM_EPOCHS_PER_PHASE} | train {train_mean['total']:.4e} | val {val_mean['total']:.4e} | state {val_mean['state']:.3e} meas {val_mean['meas']:.3e} phys {val_mean['phys']:.3e}")
    model.load_state_dict(best_state); return history


def _flatten_params(model): return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def _set_params_from_flat(model, flat):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel(); p.copy_(torch.from_numpy(flat[idx:idx+n].reshape(p.shape)).to(device=device, dtype=p.dtype)); idx += n


def lbfgsb_finetune(model, train_loader, normalizer, tf_sec, maxiter=80):
    print("Starting SciPy L-BFGS-B fine tuning ...")
    batches = [{k: v.to(device) for k, v in batch.items()} for batch in train_loader]
    x0 = _flatten_params(model).astype(np.float64)
    def fun_and_grad(flat):
        _set_params_from_flat(model, flat.astype(np.float32)); model.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for batch in batches:
            loss, _ = observer_loss(model, batch, normalizer, tf_sec, physics_weight_eff=1.0); total = total + loss
        total = total / max(len(batches), 1); total.backward()
        grads = []
        for p in model.parameters():
            if p.grad is None: grads.append(np.zeros(p.numel(), dtype=np.float64))
            else: grads.append(p.grad.detach().cpu().numpy().ravel().astype(np.float64))
        return float(total.detach().cpu()), np.concatenate(grads)
    res = minimize(fun=lambda x: fun_and_grad(x)[0], x0=x0, jac=lambda x: fun_and_grad(x)[1], method="L-BFGS-B", options={"maxiter": maxiter, "disp": True, "maxcor": 50})
    _set_params_from_flat(model, res.x.astype(np.float32)); print(f"L-BFGS-B done. success={res.success}, message={res.message}"); return res


def predict_trajectory(model, data, normalizer):
    y_mean_t = torch.tensor(normalizer.y_mean, dtype=torch.float32, device=device); y_std_t = torch.tensor(normalizer.y_std, dtype=torch.float32, device=device)
    X = torch.tensor(standardize(data["X_in"], normalizer.x_mean, normalizer.x_std), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        Yhat_norm = model(X); return denorm_y(y_std_t, y_mean_t, Yhat_norm).cpu().numpy()


def rmse_with_wrapped_angles(y_true, y_pred):
    err = y_pred - y_true
    err[:, 3] = wrap_angle(err[:, 3]); err[:, 4] = wrap_angle(err[:, 4]); err[:, 5] = wrap_angle(err[:, 5])
    return np.sqrt(np.mean(err ** 2, axis=0))


def plot_turn_results(data, y_pred, save_prefix="sensor_conditioned_turn"):
    y_true = data["x12_true"]; t = data["t"]; rmse = rmse_with_wrapped_angles(y_true, y_pred)
    names = ["pn", "pe", "pd", "phi", "theta", "psi", "u", "v", "w", "Vwx", "Vwy", "Vwz"]
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True); axes = axes.flatten()
    for i in range(12):
        axes[i].plot(t, y_true[:, i], "k", lw=1.3, label="Truth"); axes[i].plot(t, y_pred[:, i], "r--", lw=1.2, label="PINN")
        axes[i].set_title(f"{names[i]} | RMSE={rmse[i]:.4f}"); axes[i].grid(True)
        if i == 0: axes[i].legend()
    axes[-1].set_xlabel("Time [s]"); fig.suptitle("Sensor-conditioned PINN observer on unseen TURN maneuver"); fig.tight_layout(); fig.savefig(f"{save_prefix}_states.png", dpi=220); plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(10, 4.5)); ax2.bar(np.arange(12), rmse); ax2.set_xticks(np.arange(12)); ax2.set_xticklabels(names, rotation=45); ax2.set_ylabel("RMSE"); ax2.set_title("TURN RMSE by state"); ax2.grid(True, axis="y"); fig2.tight_layout(); fig2.savefig(f"{save_prefix}_rmse.png", dpi=220); plt.close(fig2)
    return rmse


def main():
    set_all_seeds(SEED)
    tf = TF; ts = TS
    datasets = generate_all_datasets(tf=tf, ts=ts)
    report_observability(datasets, stride=140)
    plot_phase_portraits(datasets)
    histories = {}; model = None; final_normalizer = None; input_dim = datasets[TRAIN_MANEUVERS[0]]["X_in"].shape[1]
    for phase_idx, mans in enumerate(CURRICULUM_PHASES):
        print(f"\n================ PHASE {phase_idx+1}/{len(CURRICULUM_PHASES)}: {mans} ================")
        train_pack, val_pack = build_phase_arrays(datasets, mans)
        normalizer = fit_normalizer(train_pack)
        train_loader, val_loader, train_ds, val_ds = make_loaders(train_pack, val_pack, normalizer)
        if model is None:
            act_name = activation_sweep(train_loader, val_loader, normalizer, input_dim, tf)
            model = SensorConditionedPINN(input_dim, hidden=PINN_HIDDEN, blocks=PINN_BLOCKS, activation_cls=ACTIVATIONS[act_name]).to(device)
        history = train_phase(model, train_loader, val_loader, normalizer, tf, phase_name="+".join(mans))
        histories["+".join(mans)] = history
        lbfgsb_finetune(model, train_loader, normalizer, tf, maxiter=LBFGSB_MAXITER)
        final_normalizer = normalizer
    turn_data = datasets[TEST_MANEUVER]
    y_turn_pred = predict_trajectory(model, turn_data, final_normalizer)
    rmse = plot_turn_results(turn_data, y_turn_pred)
    print("\n=== TURN RMSE (sensor-conditioned PINN observer) ===")
    names = ["pn", "pe", "pd", "phi", "theta", "psi", "u", "v", "w", "Vwx", "Vwy", "Vwz"]
    for n, r in zip(names, rmse): print(f"{n:>6s}: {r:.6f}")
    torch.save({"model_state": model.state_dict(), "normalizer": final_normalizer.__dict__, "train_maneuvers": TRAIN_MANEUVERS, "test_maneuver": TEST_MANEUVER, "config": {"batch_size": BATCH_SIZE, "val_fraction": VAL_FRACTION, "history": HISTORY, "strict_ekf_io_only": STRICT_EKF_IO_ONLY, "aux_use_controls": AUX_USE_CONTROLS}}, "sensor_conditioned_pinn_observer.pth")
    np.savez("sensor_conditioned_turn_results.npz", t=turn_data["t"], y_true=turn_data["x12_true"], y_pred=y_turn_pred, rmse=rmse)
    print("\nSaved:"); print("  - sensor_conditioned_pinn_observer.pth"); print("  - sensor_conditioned_turn_results.npz"); print("  - phase_portraits_training.png"); print("  - sensor_conditioned_turn_states.png"); print("  - sensor_conditioned_turn_rmse.png")


if __name__ == "__main__":
    main()
