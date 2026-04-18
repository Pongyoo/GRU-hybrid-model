# ============================================================
# stepH3_make_residual_dataset_pack_v2.py
# Phase 3 (v2): build residual label + pack sequences dataset (.npz)
#
# (PATCH) แก้ 2 จุด:
#  1) compute_y_logratio(): กลับไปเป็น horizon log-ratio แบบเดิม
#       y[t] = log( (A[t+H]+eps)/(A[t]+eps) )  -> len = N-H
#     (ไม่ใช้ A_ref, ไม่ return y[:-H])
#  2) ใส่บล็อก quick stats หลังได้ A_meas, A_phys_1k และก่อน loop H
#
# (ADD) เพิ่มอีก 2 อย่างตามที่ขอ:
#  3) PLOT sanity print ก่อน plot ใหญ่ (กันสลับเส้น/สลับตัว)
#  4) manual y_meas check 1 จุด ตอน H==500 (พิสูจน์นิยาม y)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from scipy.signal import butter, filtfilt


# ===========================
# ====== YOU EDIT HERE ======
# ===========================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"

BASE = "2.5nl-fix17.37"

CLEAN_PATH = os.path.join(ROOT, r"processed\01_clean_fix", f"{BASE}_clean.csv")
# CLEAN_PATH = os.path.join(ROOT, r"processed\01_clean_fix", f"{BASE}_clean.xlsx")

APHYS_PATH = os.path.join(ROOT, r"processed_hybrid\01_phys_fix", f"{BASE}_Aphys.csv")

H_LIST = [500, 1000, 2000, 4000, 5000, 8000]

SEQ_LEN = 1000
STRIDE = 50

TRIM_TAIL_SEC = 1.0

FS_RAW = 100000
DS_FACTOR = 100
FS_DS = FS_RAW // DS_FACTOR  # 1kHz

RMS_WIN_MS = 50
RMS_WIN = int(FS_DS * RMS_WIN_MS / 1000)  # 50 samples @1kHz

REF_SEC = 1.0
REF_N = max(10, int(REF_SEC * FS_DS))  # 1000 samples at 1kHz

EPS = 1e-8

DISP_FILTER_TYPE = "bandpass"   # "bandpass" หรือ "highpass"
DISP_BPF_LO = 15000.0
DISP_BPF_HI = 25000.0
DISP_HPF_CUTOFF = 5000.0
FILTER_ORDER = 4

RMS_WIN_RAW = int(FS_RAW * RMS_WIN_MS / 1000)
RMS_WIN_RAW = max(10, RMS_WIN_RAW)
# ===========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print("🖼️ saved:", path)

def read_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def downsample_take_every(x: np.ndarray, factor: int) -> np.ndarray:
    return x[::factor]

def rolling_rms_valid(x: np.ndarray, win: int) -> Optional[np.ndarray]:
    if len(x) < win:
        return None
    x2 = x.astype(np.float64) ** 2
    csum = np.cumsum(np.insert(x2, 0, 0.0))
    wsum = csum[win:] - csum[:-win]
    return np.sqrt(wsum / win)

def pick_displacement_col(cols) -> Optional[str]:
    for c in cols:
        if "displac" in str(c).lower():
            return c
    return None

def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)

def butter_highpass(x, fs, fc, order=4):
    nyq = 0.5 * fs
    fc_n = fc / nyq
    b, a = butter(order, fc_n, btype="highpass")
    return filtfilt(b, a, x)

def build_A_meas_from_clean(clean_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = read_table_any(clean_path)

    if "Time" not in df.columns:
        raise ValueError(f"Clean file missing 'Time' column: {clean_path}")

    disp_col = pick_displacement_col(df.columns)
    if disp_col is None:
        raise ValueError(
            f"Clean file missing displacement column (contains 'displac*'): {clean_path}\n"
            f"Available columns={list(df.columns)}"
        )

    t_raw = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=np.float64)
    disp_raw = pd.to_numeric(df[disp_col], errors="coerce").to_numpy(dtype=np.float64)

    m = np.isfinite(disp_raw)
    disp_raw = disp_raw[m]
    if len(disp_raw) < 1000:
        raise ValueError("Too short displacement after numeric conversion / NaN drop.")

    t_tmp = t_raw[m]
    if np.isfinite(t_tmp).sum() > 10:
        t_num = t_tmp.astype(np.float64)
    else:
        t_num = np.arange(len(disp_raw), dtype=np.float64) / float(FS_RAW)

    disp0 = disp_raw - float(np.mean(disp_raw))

    if DISP_FILTER_TYPE == "bandpass":
        disp_clean = butter_bandpass(disp0, FS_RAW, DISP_BPF_LO, DISP_BPF_HI, order=FILTER_ORDER)
    elif DISP_FILTER_TYPE == "highpass":
        disp_clean = butter_highpass(disp0, FS_RAW, DISP_HPF_CUTOFF, order=FILTER_ORDER)
    else:
        raise ValueError("DISP_FILTER_TYPE must be 'bandpass' or 'highpass'")

    rms_raw = rolling_rms_valid(disp_clean, RMS_WIN_RAW)
    if rms_raw is None:
        raise ValueError("Signal too short for RMS window (RAW)")

    t_rms_raw = t_num[RMS_WIN_RAW - 1:]

    t_1k = downsample_take_every(t_rms_raw, DS_FACTOR)
    A_meas_1k = downsample_take_every(rms_raw, DS_FACTOR)

    return t_1k.astype(np.float64), A_meas_1k.astype(np.float64)

def load_A_phys(Aphys_path: str) -> pd.DataFrame:
    dfp = pd.read_csv(Aphys_path)
    req = ["t_sec", "I_rms", "f_inst", "G_res", "A_phys"]
    for r in req:
        if r not in dfp.columns:
            raise ValueError(f"Aphys file missing '{r}': {Aphys_path}")
    return dfp

# ===================== PATCH 1: FIX y =====================
def compute_y_logratio(A: np.ndarray, H: int) -> np.ndarray:
    """
    CORRECT (horizon log-ratio):
    y[t] = log( (A[t+H] + eps) / (A[t] + eps) )
    output length = len(A) - H
    """
    A = A.astype(np.float64)
    if H <= 0:
        raise ValueError("H must be > 0")
    if len(A) <= H:
        raise ValueError(f"len(A)={len(A)} must be > H={H}")

    A0 = A[:-H]
    A1 = A[H:]
    y = np.log((A1 + EPS) / (A0 + EPS))
    return y.astype(np.float64)
# ===================== END PATCH 1 =====================

def compute_innovation(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    y[1:] = x[1:] - x[:-1]
    return y

def make_sequences(Xt: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    N = Xt.shape[0]
    last = N - seq_len
    if last <= 0:
        return None, None
    idxs = np.arange(0, last, stride, dtype=np.int64)
    X_seq = np.empty((len(idxs), seq_len, Xt.shape[1]), dtype=np.float32)
    y_out = np.empty((len(idxs),), dtype=np.float32)

    for i, s in enumerate(idxs):
        X_seq[i] = Xt[s:s + seq_len].astype(np.float32)
        y_out[i] = float(y[s + seq_len - 1])
    return X_seq, y_out


def main():
    processed = os.path.join(ROOT, "processed_hybrid")
    out_dir = os.path.join(processed, "02_residual_dataset")
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plot_dir)

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"Missing CLEAN_PATH:\n  {CLEAN_PATH}")
    if not os.path.exists(APHYS_PATH):
        raise FileNotFoundError(f"Missing APHYS_PATH:\n  {APHYS_PATH}")

    print("\n================ Phase 3 (v2): Residual Dataset Pack ================")
    print(f"BASE      : {BASE}")
    print(f"CLEAN_PATH: {CLEAN_PATH}")
    print(f"APHYS_PATH: {APHYS_PATH}")
    print(f"FS_RAW={FS_RAW} | DS_FACTOR={DS_FACTOR} | FS_DS={FS_DS}")
    print(f"RMS_WIN_MS={RMS_WIN_MS} -> RMS_WIN={RMS_WIN}")
    print(f"RMS_WIN_RAW={RMS_WIN_RAW} (raw samples)")
    print(f"DISP_FILTER_TYPE={DISP_FILTER_TYPE} | BPF={DISP_BPF_LO}-{DISP_BPF_HI} Hz | ORDER={FILTER_ORDER}")
    print(f"SEQ_LEN={SEQ_LEN} | STRIDE={STRIDE} | TRIM_TAIL_SEC={TRIM_TAIL_SEC}")
    print("=====================================================================\n")

    t_meas, A_meas = build_A_meas_from_clean(CLEAN_PATH)

    print("\n--- A_meas time sanity ---")
    print(f"t_meas range: {t_meas[0]:.3f} -> {t_meas[-1]:.3f} (span {t_meas[-1]-t_meas[0]:.3f} s)")
    print("A_meas len:", len(A_meas))
    print("A_meas min/max/mean:", float(np.min(A_meas)), float(np.max(A_meas)), float(np.mean(A_meas)))

    dfp = load_A_phys(APHYS_PATH)

    print("\n--- A_phys sanity check ---")
    print("dfp columns:", list(dfp.columns))
    print("dfp head:\n", dfp.head(3))
    print("dfp tail:\n", dfp.tail(3))

    dfp = dfp.sort_values("t_sec").drop_duplicates("t_sec", keep="first").reset_index(drop=True)

    cols_need = ["t_sec", "I_rms", "f_inst", "G_res", "A_phys"]
    for c in cols_need:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    dfp_valid = dfp.dropna(subset=["t_sec", "A_phys"]).copy()

    t_phys = dfp_valid["t_sec"].to_numpy(dtype=np.float64)
    if len(t_phys) < 5:
        raise ValueError("Aphys.csv has too few valid samples (t_sec/A_phys) to resample.")

    t0, t1 = float(t_phys[0]), float(t_phys[-1])
    print(f"t_sec range: {t0:.3f} -> {t1:.3f} (span {t1-t0:.3f} s)")
    dt = np.diff(t_phys)
    print("t_sec monotonic:", bool(np.all(dt > 0)))
    print("t_sec median dt:", float(np.median(dt)) if len(dt) > 0 else float("nan"))

    Aphys_raw = dfp_valid["A_phys"].to_numpy(dtype=np.float64)
    print("A_phys len:", len(Aphys_raw))
    print("A_phys min/max/mean:", float(np.nanmin(Aphys_raw)), float(np.nanmax(Aphys_raw)), float(np.nanmean(Aphys_raw)))
    print("A_phys NaN count:", int(np.isnan(Aphys_raw).sum()))
    print("A_phys zero-ish count:", int((np.abs(Aphys_raw) < 1e-12).sum()))

    t_min, t_max = float(t_phys[0]), float(t_phys[-1])
    mask_meas = (t_meas >= t_min) & (t_meas <= t_max)
    if mask_meas.mean() < 0.5:
        print("\n⚠️ Warning: less than 50% of t_meas is within A_phys time range.")
        print(f"t_meas span: {t_meas[0]:.3f}->{t_meas[-1]:.3f} | t_phys span: {t_min:.3f}->{t_max:.3f}")

    t_meas = t_meas[mask_meas]
    A_meas = A_meas[mask_meas]

    def interp_col(col: str) -> np.ndarray:
        v = dfp_valid[col].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum() < 5:
            raise ValueError(f"Column '{col}' has too few finite values for interp.")
        return np.interp(t_meas.astype(np.float64), t_phys[ok], v[ok]).astype(np.float64)

    I_rms_1k = interp_col("I_rms")
    f_inst_1k = interp_col("f_inst")
    G_res_1k = interp_col("G_res")
    A_phys_1k = interp_col("A_phys")

    N = len(A_meas)
    trim_n = int(max(0.0, TRIM_TAIL_SEC) * FS_DS)
    if trim_n > 0 and N > trim_n + 10:
        A_meas = A_meas[:-trim_n]
        t_meas = t_meas[:-trim_n]
        I_rms_1k = I_rms_1k[:-trim_n]
        f_inst_1k = f_inst_1k[:-trim_n]
        G_res_1k = G_res_1k[:-trim_n]
        A_phys_1k = A_phys_1k[:-trim_n]

    print(f"\nAligned length @1kHz (after resample+trim): N={len(A_meas)} (trim tail={trim_n} samples)\n")

    # ===================== PATCH 2: QUICK STATS =====================
    def quick_stats(name, x):
        x = np.asarray(x, dtype=np.float64)
        print(f"[{name}] len={len(x)} min={np.min(x):.4g} max={np.max(x):.4g} "
              f"mean={np.mean(x):.4g} std={np.std(x):.4g}")

    print("\n--- Quick stats sanity (before H loop) ---")
    quick_stats("A_meas", A_meas)
    quick_stats("A_phys_1k", A_phys_1k)

    for H_test in [500, 1000]:
        if len(A_meas) > H_test + 5 and len(A_phys_1k) > H_test + 5:
            y_m = compute_y_logratio(A_meas, H_test)
            y_p = compute_y_logratio(A_phys_1k, H_test)
            print(f"\nH={H_test}")
            quick_stats("y_meas", y_m)
            quick_stats("y_phys", y_p)
            c = np.corrcoef(y_m, y_p)[0, 1]
            print("corr(y_meas, y_phys) =", float(c))
        else:
            print(f"\nH={H_test} (skip stats: too short)")
    # ===================== END PATCH 2 =====================

    # ===================== ADD 3: PLOT SANITY =====================
    print("PLOT sanity:",
          "A_meas mean", float(np.mean(A_meas)),
          "A_phys_1k mean", float(np.mean(A_phys_1k)))
    # ===================== END ADD 3 =====================

    plt.figure(figsize=(10, 6))
    plt.plot(t_meas, A_meas, label="A_meas (RMS disp @1kHz)")
    plt.plot(t_meas, A_phys_1k, label="A_phys (resampled -> @1kHz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arb.)")
    plt.title(f"{BASE} | A_meas vs A_phys (aligned on t_meas @1kHz)")
    plt.legend()
    savefig(os.path.join(plot_dir, f"{BASE}_Ameas_vs_Aphys.png"))

    def plot_zoom(s0: int, s1: int):
        s0 = int(max(0, s0))
        s1 = int(min(len(A_meas), s1))
        if s1 <= s0 + 5:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(t_meas[s0:s1], A_meas[s0:s1], label="A_meas", lw=1.2)
        plt.plot(t_meas[s0:s1], A_phys_1k[s0:s1], label="A_phys_1k", lw=1.2, alpha=0.85)
        plt.title(f"{BASE} | A_meas vs A_phys zoom [{s0}:{s1}]")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.25)
        savefig(os.path.join(plot_dir, f"{BASE}_Ameas_Aphys_zoom_{s0}_{s1}.png"))

    plot_zoom(0, 3000)
    plot_zoom(len(A_meas)-3000, len(A_meas))


    X_phys_full = np.stack([I_rms_1k, f_inst_1k, G_res_1k, A_phys_1k], axis=1).astype(np.float64)

    meta_common = dict(
        base=BASE,
        FS_RAW=FS_RAW,
        DS_FACTOR=DS_FACTOR,
        FS_DS=FS_DS,
        RMS_WIN_MS=RMS_WIN_MS,
        RMS_WIN=RMS_WIN,
        REF_SEC=REF_SEC,
        REF_N=REF_N,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        trim_tail_sec=TRIM_TAIL_SEC,
        clean_path=os.path.abspath(CLEAN_PATH),
        Aphys_path=os.path.abspath(APHYS_PATH),
        phys_resample="interp_to_t_meas_1kHz",
        y_def="log(A[t+H]/A[t]) then residual diff",
    )

    for H in H_LIST:
        if H <= 0:
            continue
        if len(A_meas) <= H + SEQ_LEN + 5:
            print(f"Skip H={H}: too short after trim")
            continue

        y_meas0 = compute_y_logratio(A_meas, H)        # len = N - H
        y_phys0 = compute_y_logratio(A_phys_1k, H)     # len = N - H

        # ===================== ADD 4: MANUAL CHECK =====================
        if H == 500:
            i = 10000
            if (i + H) < len(A_meas) and i < len(y_meas0):
                y_manual = np.log((A_meas[i + H] + EPS) / (A_meas[i] + EPS))
                print("manual y_meas check:",
                      float(y_manual),
                      "| y_meas0[i]=", float(y_meas0[i]),
                      "| abs diff=", float(abs(y_manual - y_meas0[i])))
            else:
                print("manual y_meas check: skip (index out of range)",
                      "len(A_meas)=", len(A_meas), "len(y_meas0)=", len(y_meas0))
        # ===================== END ADD 4 =====================

        residual_raw = (y_meas0 - y_phys0).astype(np.float64)
        residual_innov = compute_innovation(residual_raw)

        Xk = X_phys_full[:-H, :]                       # len = N - H

        X_seq, y_seq = make_sequences(Xk, residual_innov, SEQ_LEN, STRIDE)
        if X_seq is None:
            print(f"Skip H={H}: cannot make sequences")
            continue

        t_y = t_meas[:-H]
        t_y = t_y[:len(residual_raw)]

        plt.figure(figsize=(10, 5))
        plt.plot(t_y, residual_raw, label="residual_raw = y_meas - y_phys", linewidth=1.0)

        med = np.median(residual_raw)
        mad = np.median(np.abs(residual_raw - med)) + 1e-12
        z = (residual_raw - med) / (1.4826 * mad)
        clipped = np.clip(z, -3.0, 3.0) * (1.4826 * mad) + med
        plt.plot(t_y, clipped, label="residual_raw (clipped k=3)", linewidth=1.0, alpha=0.9)

        plt.axhline(0, linestyle="--", linewidth=1.0)
        plt.xlabel("Time (s)")
        plt.ylabel("Residual (log-ratio space)")
        plt.title(f"{BASE} | residual(t) | H={H}")
        plt.legend()
        savefig(os.path.join(plot_dir, f"{BASE}_residual_t_H{H}.png"))

        plt.figure(figsize=(8, 5))
        plt.hist(residual_raw, bins=80)
        plt.xlabel("Residual (log-ratio space)")
        plt.ylabel("Count")
        plt.title(f"{BASE} | residual histogram | H={H}")
        savefig(os.path.join(plot_dir, f"{BASE}_residual_hist_H{H}.png"))

        out_npz = os.path.join(out_dir, f"{BASE}_H{H}.npz")
        meta = dict(meta_common)
        meta.update(dict(H=int(H), X_dim=int(X_seq.shape[-1]), n_seq=int(X_seq.shape[0])))

        np.savez_compressed(
            out_npz,
            X_seq=X_seq.astype(np.float32),
            y_seq=y_seq.astype(np.float32),
            t_sec=t_meas.astype(np.float32),
            A_meas=A_meas.astype(np.float32),
            A_phys=A_phys_1k.astype(np.float32),
            residual_raw=residual_raw.astype(np.float32),
            residual_innov=residual_innov.astype(np.float32),
            meta=json.dumps(meta),
        )

        print(f"✅ Saved dataset: {out_npz} | X_seq={X_seq.shape} y_seq={y_seq.shape}")

    print("\nDone. Plots saved to:")
    print(" ", plot_dir)
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
