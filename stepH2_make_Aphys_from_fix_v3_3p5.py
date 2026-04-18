# scripts_hybrid/stepH2_make_Aphys_from_fix_v2_est_fnfix.py
# Purpose:
#   Build physics-based A_phys(t) for FIX, but estimate fn from FIX itself (fn_fix)
#   because sweep & fix are in different environments and fix may not be tuned exactly to resonance.
#
# Output:
#   processed_hybrid/01_phys_fix/{FIX_NAME}_Aphys.csv
#   processed_hybrid/01_phys_fix/plots/*.png

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"

# ===================== YOU EDIT HERE =====================
FIX_NAME   = "3.5nl-fix17.07std"   # no suffix
SWEEP_NAME = "2.5nl-swp"        # no suffix (optional reference Qm/fn)
USE_SWEEP_QM = True             # use Qm from sweep params (recommended)
USE_SWEEP_FN_AS_REFERENCE = True  # only for plotting reference line

FS_DEFAULT = 100000

# bandpass range for current carrier
BPF_LO = 15000.0
BPF_HI = 25000.0

# windowing for frequency tracking (robust)
FTRACK_WIN_SEC = 0.20           # 0.2s window
FTRACK_HOP_SEC = 0.05           # 0.05s hop
FTRACK_FMIN = 15000.0
FTRACK_FMAX = 25000.0

# Choose steady region to estimate fn_fix (seconds)
# If you "wait for it to be stable", set this to later region.
FN_EST_T0 = 1.0
FN_EST_T1 = 6.0

# RMS envelope
RMS_WIN_MS = 50

COH_NOTE = "fn_fix estimated from FIX via Welch peak tracking"
# ========================================================

# ===== (B) Downsample control =====
DS_FACTOR = 100  # 100k -> 1k

CLEAN_DIR = os.path.join(ROOT, r"processed\01_clean_fix")
PARAM_DIR = os.path.join(ROOT, r"processed_hybrid\00_params_sweep")
OUT_DIR   = os.path.join(ROOT, r"processed_hybrid\01_phys_fix")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EPS = 1e-12

def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)

def rolling_rms_valid(x, win):
    x = x.astype(np.float64)
    x2 = x * x
    k = np.ones(win, dtype=np.float64) / win
    return np.sqrt(np.convolve(x2, k, mode="valid"))

def estimate_fs_from_time(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return None
    dt_med = np.median(dt)
    if dt_med <= 0:
        return None
    return 1.0 / dt_med

def resonance_gain(f, fn, Qm):
    r = f / (fn + EPS)
    zeta = 1.0 / (2.0 * (Qm + EPS))
    return 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * zeta * r)**2 + EPS)

def peak_freq_welch(x_seg, fs, fmin, fmax):
    # Welch PSD peak (robust to noise)
    f, Pxx = welch(x_seg, fs=fs, nperseg=min(len(x_seg), 4096))
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return np.nan
    f2 = f[m]
    p2 = Pxx[m]
    if not np.all(np.isfinite(p2)) or len(p2) < 3:
        return np.nan
    return float(f2[np.argmax(p2)])

def track_frequency_welch(cur_bp, t, fs, win_sec, hop_sec, fmin, fmax):
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    win = max(win, 1024)
    hop = max(hop, 1)

    f_list = []
    tc_list = []
    idx0 = 0
    N = len(cur_bp)

    while idx0 + win <= N:
        seg = cur_bp[idx0:idx0+win]
        fpk = peak_freq_welch(seg, fs, fmin, fmax)
        tc = float(t[idx0 + win//2])
        f_list.append(fpk)
        tc_list.append(tc)
        idx0 += hop

    return np.array(tc_list, dtype=np.float64), np.array(f_list, dtype=np.float64)

def main():
    clean_path = os.path.join(CLEAN_DIR, f"{FIX_NAME}_clean.csv")
    param_path = os.path.join(PARAM_DIR, f"params_{SWEEP_NAME}.json")

    if not os.path.exists(clean_path):
        raise FileNotFoundError(clean_path)

    df = pd.read_csv(clean_path)

    col_t   = get_col(df, ["t_sec", "Time", "time", "t"])
    col_cur = get_col(df, ["Current", "current"])
    if col_cur is None:
        raise ValueError(f"Missing Current column in {clean_path}. columns={list(df.columns)}")

    cur = pd.to_numeric(df[col_cur], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(cur)
    cur = cur[m]

    # time & fs
    fs = FS_DEFAULT
    t = None
    if col_t is not None:
        t_try = pd.to_numeric(df[col_t], errors="coerce").to_numpy(dtype=np.float64)[m]
        if np.isfinite(t_try).sum() > 0:
            fs_est = estimate_fs_from_time(t_try)
            if fs_est is not None and 1000 < fs_est < 1e6:
                fs = float(fs_est)
                t = t_try
    if t is None:
        t = np.arange(len(cur), dtype=np.float64) / fs

    # optional sweep params
    fn_sweep = None
    Qm = None
    if os.path.exists(param_path):
        with open(param_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        if "fn_hz" in params:
            fn_sweep = float(params["fn_hz"])
        if "Qm" in params:
            Qm = float(params["Qm"])

    if Qm is None:
        # fallback if no sweep Qm
        Qm = 35.0  # reasonable default, but better to use sweep Qm
    if (not USE_SWEEP_QM) and (Qm is not None):
        # still keep, but here for clarity
        pass

    print("\n================ Phase 2 (v2): A_phys with fn_fix ================")
    print("FIX_NAME   :", FIX_NAME)
    print("CLEAN_PATH :", clean_path)
    print("FS         :", fs)
    print("BPF        :", BPF_LO, "-", BPF_HI, "Hz")
    print("FTRACK win/hop:", FTRACK_WIN_SEC, "/", FTRACK_HOP_SEC, "sec")
    print("FN_EST region:", FN_EST_T0, "->", FN_EST_T1, "sec")
    print("Qm_used    :", Qm)
    if fn_sweep is not None:
        print("fn_sweep(ref):", fn_sweep)
    print("DS_FACTOR  :", DS_FACTOR)
    print("=================================================================\n")

    # bandpass
    cur_bp = butter_bandpass(cur, fs, BPF_LO, BPF_HI, order=4)

    # robust frequency tracking
    t_fk, f_pk = track_frequency_welch(
        cur_bp, t, fs,
        win_sec=FTRACK_WIN_SEC,
        hop_sec=FTRACK_HOP_SEC,
        fmin=FTRACK_FMIN, fmax=FTRACK_FMAX
    )

    # pick region to estimate fn_fix
    region = (t_fk >= FN_EST_T0) & (t_fk <= FN_EST_T1) & np.isfinite(f_pk)
    if np.sum(region) < 5:
        # fallback: choose top-energy region automatically (use PSD peak values by magnitude proxy)
        # Here: choose median of all valid peaks
        fn_fix = float(np.nanmedian(f_pk))
        print("⚠️ Not enough points in FN_EST region; fallback to median(all) f_peak.")
    else:
        fn_fix = float(np.median(f_pk[region]))

    print(f"Estimated fn_fix = {fn_fix:.3f} Hz (from FIX)")

    # RMS envelope (on raw current)
    rms_win = int(fs * RMS_WIN_MS / 1000.0)
    rms_win = max(10, rms_win)
    I_rms = rolling_rms_valid(cur_bp, rms_win)
    t_rms = t[rms_win - 1:]

    # align f_peak(t_fk) onto t_rms by interpolation (1k-friendly later)
    # (f_peak sampled at hop_sec; I_rms is dense)
    f_interp = np.interp(t_rms, t_fk, np.nan_to_num(f_pk, nan=fn_fix))
    # clamp outliers (optional safety)
    f_interp = np.clip(f_interp, FTRACK_FMIN, FTRACK_FMAX)

    # compute G_res with fn_fix
    G = resonance_gain(f_interp, fn=fn_fix, Qm=Qm)

    # scale constant (still C=1)
    C = 1.0
    A_phys = C * I_rms * G

    # ===== downsample to 1kHz grid (systematic) =====
    FS_DS = fs / DS_FACTOR  # should be ~1000
    if abs(FS_DS - 1000) > 5:
        print(f"⚠️ FS_DS is not ~1000Hz: {FS_DS:.3f} (check FS_DEFAULT or time column)")

    # use uniform 1k grid starting from first available t_rms
    t0 = float(t_rms[0])
    t_ds = t0 + np.arange(int((t_rms[-1] - t0) * FS_DS) + 1, dtype=np.float64) / FS_DS

    # resample envelope-like signals (safe because RMS already smooth)
    I_rms_ds    = np.interp(t_ds, t_rms, I_rms)
    f_interp_ds = np.interp(t_ds, t_rms, f_interp)
    G_ds        = resonance_gain(f_interp_ds, fn=fn_fix, Qm=Qm)
    A_phys_ds   = (1.0 * I_rms_ds * G_ds)

    # replace streams for saving
    t_use = t_ds
    I_use = I_rms_ds
    f_use = f_interp_ds
    G_use = G_ds
    A_use = A_phys_ds

    # save csv
    out = pd.DataFrame({
        "t_sec": t_use.astype(np.float64),
        "I_rms": I_use.astype(np.float64),
        "f_inst": f_use.astype(np.float64),   # using Welch-peak tracking (robust)
        "G_res": G_use.astype(np.float64),
        "A_phys": A_use.astype(np.float64),
        "fn_used": np.full(len(t_use), fn_fix, dtype=np.float64),
        "Qm_used": np.full(len(t_use), Qm, dtype=np.float64),
        "fn_sweep_ref": np.full(len(t_use), fn_sweep if fn_sweep is not None else np.nan, dtype=np.float64),
    })
    out_path = os.path.join(OUT_DIR, f"{FIX_NAME}_Aphys.csv")
    out.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)
    print(f"Saved @ ~{FS_DS:.1f} Hz | len={len(out)} | t span={out['t_sec'].iloc[0]:.3f}->{out['t_sec'].iloc[-1]:.3f} s")
    print(f"A_phys min/max/mean = {A_use.min():.6f} / {A_use.max():.6f} / {A_use.mean():.6f}")

    # ===================== plots =====================

    # 1) bandpassed current snippet
    fig = plt.figure()
    Nshow = int(0.02 * fs)  # 20ms
    Nshow = min(Nshow, len(cur_bp))
    plt.plot(t[:Nshow], cur_bp[:Nshow])
    plt.xlabel("Time (s)")
    plt.ylabel("Current (bandpassed)")
    plt.title(f"{FIX_NAME} | Current bandpass {BPF_LO/1000:.1f}-{BPF_HI/1000:.1f} kHz")
    p1 = os.path.join(PLOT_DIR, f"{FIX_NAME}_cur_bp.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) frequency track
    fig = plt.figure(figsize=(10,4))
    plt.plot(t_fk, f_pk, lw=1.0, label="f_peak (Welch tracking)")
    plt.axhline(fn_fix, linestyle="--", label=f"fn_fix={fn_fix:.1f}")
    if (fn_sweep is not None) and USE_SWEEP_FN_AS_REFERENCE:
        plt.axhline(fn_sweep, linestyle=":", label=f"fn_sweep(ref)={fn_sweep:.1f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{FIX_NAME} | Robust frequency tracking (Welch peak)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p2 = os.path.join(PLOT_DIR, f"{FIX_NAME}_f_track_welch.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) G_res
    fig = plt.figure(figsize=(10,4))
    plt.plot(out["t_sec"], out["G_res"], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("G_res (arb.)")
    plt.title(f"{FIX_NAME} | G_res(f_peak; fn_fix, Qm)")
    plt.grid(True, alpha=0.25)
    p3 = os.path.join(PLOT_DIR, f"{FIX_NAME}_G_res.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) A_phys
    fig = plt.figure(figsize=(10,4))
    plt.plot(out["t_sec"], out["A_phys"], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("A_phys (arb.)")
    plt.title(f"{FIX_NAME} | A_phys = I_rms * G_res | {COH_NOTE}")
    plt.grid(True, alpha=0.25)
    p4 = os.path.join(PLOT_DIR, f"{FIX_NAME}_A_phys.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved plots:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print(" ", p4)
    print("=================================================================\n")

if __name__ == "__main__":
    main()
