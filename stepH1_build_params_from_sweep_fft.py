# stepH1_build_params_from_sweep_fft.py
# Purpose:
#   Estimate fn, Qm from raw sweep using FFT/Welch FRF (H1) + coherence
#   Plot in paper-style (FRF magnitude + coherence with fn, f1/f2, -3 dB)
#
# Input :
#   raw_sweep_old/2.5nl-swp.csv
#
# Output:
#   processed_hybrid/00_params_sweep/params_2.5nl-swp.json
#   processed_hybrid/00_params_sweep/plot_frfcoh_2.5nl-swp.png

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, csd

# =======================
# Paths
# =======================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
RAW_DIR = os.path.join(ROOT, "raw_sweep_old")
OUT_DIR = os.path.join(ROOT, "processed_hybrid", "00_params_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

# =======================
# Parameters
# =======================
FS = 100000                # sampling rate
F_MIN, F_MAX = 15000, 25000
NFFT = 16384
NSEG = 8192
COH_TH = 0.80
EPS = 1e-12

# =======================
# Utilities
# =======================
def get_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Missing columns {names}. Found: {list(df.columns)}")

def find_3db_bandwidth(f, mag, i_pk):
    """
    Find -3 dB bandwidth around a peak.
    """
    peak = float(mag[i_pk])
    thr = peak / np.sqrt(2.0)

    iL = i_pk
    while iL > 0 and mag[iL] >= thr:
        iL -= 1

    iR = i_pk
    while iR < len(mag) - 1 and mag[iR] >= thr:
        iR += 1

    fL = float(f[iL])
    fR = float(f[iR])
    BW = float(max(EPS, fR - fL))
    return fL, fR, BW, thr, peak

# =======================
# Main
# =======================
def main(level="2.5nl-swp"):
    path = os.path.join(RAW_DIR, f"{level}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    cur = pd.to_numeric(
        df[get_col(df, ["Current", "current"])],
        errors="coerce"
    ).to_numpy(np.float64)

    disp = pd.to_numeric(
        df[get_col(df, ["Displacement", "displacement"])],
        errors="coerce"
    ).to_numpy(np.float64)

    m = np.isfinite(cur) & np.isfinite(disp)
    cur, disp = cur[m], disp[m]

    # =======================
    # Welch FRF (H1)
    # =======================
    f, Sxx = welch(cur, fs=FS, nperseg=NSEG, nfft=NFFT)
    _, Syy = welch(disp, fs=FS, nperseg=NSEG, nfft=NFFT)
    _, Sxy = csd(cur, disp, fs=FS, nperseg=NSEG, nfft=NFFT)

    H1 = Sxy / (Sxx + EPS)
    coh = np.abs(Sxy) ** 2 / ((Sxx + EPS) * (Syy + EPS))
    mag = np.abs(H1)

    # =======================
    # Frequency band of interest
    # =======================
    band = (f >= F_MIN) & (f <= F_MAX)
    f_b = f[band]
    mag_b = mag[band]
    coh_b = coh[band]

    # Only use coherence-passed region for parameter estimation
    ok = coh_b >= COH_TH
    if ok.sum() < 10:
        raise RuntimeError("Not enough points pass coherence threshold.")

    f_ok = f_b[ok]
    mag_ok = mag_b[ok]
    coh_ok = coh_b[ok]

    # Sort by frequency (important for bandwidth search)
    idx = np.argsort(f_ok)
    f_s = f_ok[idx]
    m_s = mag_ok[idx]

    # Peak
    i_pk = int(np.argmax(m_s))
    fn = float(f_s[i_pk])

    # -3 dB bandwidth
    fL, fR, BW, thr, peak = find_3db_bandwidth(f_s, m_s, i_pk)
    Qm = float(fn / BW)

    # =======================
    # Save parameters
    # =======================
    params = {
        "level_name": level,
        "fs_hz": FS,
        "method": "Welch_H1",
        "f_range_hz": [F_MIN, F_MAX],
        "nfft": NFFT,
        "nperseg": NSEG,
        "coh_threshold": COH_TH,
        "fn_hz": fn,
        "bw_hz": BW,
        "Qm": Qm,
        "fL_hz": fL,
        "fR_hz": fR,
        "peak_mag": peak,
        "minus3db_level": thr
    }

    json_path = os.path.join(OUT_DIR, f"params_{level}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)

    print(f"fn = {fn} Hz | BW = {BW} Hz | Qm = {Qm}")

    # =======================
    # Plot (paper style)
    # =======================
    fig = plt.figure(figsize=(10, 6), dpi=140)

    # --- Top: FRF magnitude ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(f_s, m_s, linewidth=1.5, color="blue")
    ax1.axhline(thr, color="gray", linestyle="--", linewidth=1.2, label="-3 dB")
    ax1.axvline(fn, color="red", linestyle="--", linewidth=1.6, label="fn")
    ax1.axvline(fL, color="green", linestyle="--", linewidth=1.4, label="f1")
    ax1.axvline(fR, color="green", linestyle="--", linewidth=1.4, label="f2")

    ax1.set_title("FRF Magnitude (only coherence-passed region)")
    ax1.set_ylabel("|H1|")
    ax1.grid(True, alpha=0.3)

    txt = f"fn = {fn:.2f} Hz\nBW = {BW:.2f} Hz\nQm = {Qm:.2f}"
    ax1.text(
        0.02, 0.95, txt,
        transform=ax1.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7)
    )

    xpad = 1500
    ax1.set_xlim(max(F_MIN, fn - xpad), min(F_MAX, fn + xpad))

    # --- Bottom: Coherence ---
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(f_b, coh_b, linewidth=1.2, color="tab:blue")
    ax2.axhline(COH_TH, color="red", linestyle="--", linewidth=1.4, label="coh ≥ 0.8")

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Coherence")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"plot_frfcoh_{level}.png")
    fig.savefig(plot_path)
    plt.close(fig)

    print("✅ Saved:", json_path)
    print("✅ Saved plot:", plot_path)

if __name__ == "__main__":
    main("2.5nl-swp")
