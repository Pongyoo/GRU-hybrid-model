# check_disp_raw.py
# Purpose: Quick check displacement RAW (before RMS, before downsample)
# Output: 2 plots -> (1) time 20ms, (2) PSD/FFT (Welch)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# ===================== YOU EDIT HERE =====================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"
CLEAN_PATH = os.path.join(ROOT, r"processed\01_clean_fix", f"{BASE}_clean.csv")

FS_RAW = 100000  # expected raw sampling rate
T_SHOW_SEC = 0.02  # 20 ms
PSD_FMIN = 0
PSD_FMAX = 30000   # show up to 30 kHz
OUT_DIR = os.path.join(ROOT, r"processed_hybrid\debug_disp_raw")
# =========================================================

def read_clean(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def pick_disp_col(cols):
    # your clean file used "Displacement" style
    for c in cols:
        if "displac" in str(c).lower():
            return c
    # fallback: common alternatives
    for c in cols:
        s = str(c).lower()
        if ("disp" in s) or ("ldv" in s) or ("laser" in s):
            return c
    return None

def estimate_fs_from_time(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return None
    dt_med = np.median(dt)
    if dt_med <= 0:
        return None
    return 1.0 / dt_med

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(CLEAN_PATH)

    df = read_clean(CLEAN_PATH)

    if "Time" not in df.columns:
        raise ValueError(f"Missing 'Time' column in {CLEAN_PATH}. columns={list(df.columns)}")

    disp_col = pick_disp_col(df.columns)
    if disp_col is None:
        raise ValueError(f"Cannot find displacement column in {CLEAN_PATH}. columns={list(df.columns)}")

    # ===== FIX (only change block): do NOT mask by Time =====
    t_raw = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=np.float64)
    x_raw = pd.to_numeric(df[disp_col], errors="coerce").to_numpy(dtype=np.float64)

    # --- keep samples by displacement only (Time may be empty) ---
    m = np.isfinite(x_raw)
    x = x_raw[m]

    if len(x) == 0:
        raise ValueError("No finite displacement samples after numeric conversion.")

    # --- build / clean time ---
    fs = float(FS_RAW)

    t_tmp = t_raw[m]  # time aligned with x (may still be all-NaN)
    if np.isfinite(t_tmp).sum() > 10:
        fs_est = estimate_fs_from_time(t_tmp)
        if fs_est is not None and 1000 < fs_est < 1e6:
            fs = float(fs_est)
            t = t_tmp
        else:
            t = np.arange(len(x), dtype=np.float64) / fs
    else:
        # Time column is empty -> uniform time
        t = np.arange(len(x), dtype=np.float64) / fs
    # ===== END FIX =====

    print("==== DISP RAW CHECK ====")
    print("CLEAN_PATH:", CLEAN_PATH)
    print("disp_col  :", disp_col)
    print("len       :", len(x))
    print("fs_used   :", fs)
    print("t span    :", float(t[0]), "->", float(t[-1]))
    print("========================")

    # -------- Plot 1: time-domain 20 ms ----------
    Nshow = int(T_SHOW_SEC * fs)
    Nshow = min(Nshow, len(x))

    plt.figure(figsize=(10,4))
    plt.plot(t[:Nshow], x[:Nshow], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (raw)")
    plt.title(f"{BASE} | Displacement RAW time-domain (first {T_SHOW_SEC*1000:.0f} ms)")
    plt.grid(True, alpha=0.25)
    p1 = os.path.join(OUT_DIR, f"{BASE}_disp_raw_20ms.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=180)
    plt.close()

    # -------- Plot 2: PSD (Welch) ----------
    # use first few seconds (or full if shorter) for stable PSD
    # (Welch needs enough samples)
    sec_use = min(5.0, (t[-1]-t[0]))
    Npsd = int(sec_use * fs)
    Npsd = min(Npsd, len(x))
    x_psd = x[:Npsd]

    nperseg = min(4096, len(x_psd))
    f, Pxx = welch(x_psd, fs=fs, nperseg=nperseg)

    # limit display range
    mm = (f >= PSD_FMIN) & (f <= PSD_FMAX)

    plt.figure(figsize=(10,4))
    plt.plot(f[mm], Pxx[mm], lw=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (Welch)")
    plt.title(f"{BASE} | Displacement RAW PSD (Welch) | using {sec_use:.1f}s")
    plt.grid(True, alpha=0.25)
    p2 = os.path.join(OUT_DIR, f"{BASE}_disp_raw_psd.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=180)
    plt.close()

    print("✅ Saved:")
    print(" ", p1)
    print(" ", p2)
    print("OUT_DIR:", OUT_DIR)

if __name__ == "__main__":
    main()
