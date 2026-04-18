# ============================================================
# stepH5_eval_hybrid_ymeas.py
# Phase 5: Evaluate Hybrid on y_meas (log-ratio) space
#
# Hybrid:
#   y_hybrid_hat(t) = y_phys(t) + r_hat(t)
# where r_hat is reconstructed by integrating predicted delta residual.
#
# Inputs:
#   - NPZ: processed_hybrid/02_residual_dataset/{base}_H{H}.npz
#   - Model: processed_hybrid/03_gru_residual/gru_residual_best.pt (from Phase4 v2)
#   - Scaler: processed_hybrid/03_gru_residual/scaler.json
#
# Outputs:
#   processed_hybrid/04_hybrid_eval/{base}_H{H}/
#     - y_compare_test.png
#     - residual_compare_test.png
#     - scatter_yhybrid_vs_ymeas.png
#     - summary_hybrid_test.json
# ============================================================

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8

# must match Phase4 v2 split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
GAP = 200

# --------------- model (same as Phase4 v2) ----------------
class GRUResidual(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        return self.fc(h_last).squeeze(-1)

# ---------------- utilities ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ✅ PATCH: make sure TEST is not empty (keep original structure otherwise)
def time_block_split(N, train_ratio=0.7, val_ratio=0.15, gap=200):
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_end = n_train
    val_start = min(train_end + gap, N)
    val_end = min(val_start + n_val, N)
    test_start = min(val_end + gap, N)

    # If test is empty, shrink gap automatically
    if test_start >= N:
        test_start = val_end  # no extra gap
        if test_start >= N:
            test_start = max(val_end - 1, 0)

    idx_train = np.arange(0, train_end)
    idx_val   = np.arange(val_start, val_end)
    idx_test  = np.arange(test_start, N)

    return idx_train, idx_val, idx_test

def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))

def compute_y_logratio(A, H, ref_n=1000):
    """
    same concept as Phase3:
      y0[t] = log((A[t+H]+eps)/(A[t]+eps))
      y0 = y0 / (y_ref + eps), y_ref=median(A0[:ref_n])
    output length = len(A) - H
    """
    A = A.astype(np.float64)
    A0 = A[:-H]
    AH = A[H:]
    y0 = np.log((AH + EPS) / (A0 + EPS))

    if len(A0) >= ref_n:
        y_ref = float(np.median(A0[:ref_n]))
    else:
        y_ref = float(np.median(A0))
    y0 = y0 / (y_ref + EPS)
    return y0.astype(np.float32)

def baseline_on_series(y_true):
    mean_pred = np.full_like(y_true, np.mean(y_true), dtype=np.float32)
    pers_pred = np.concatenate([[y_true[0]], y_true[:-1]]).astype(np.float32)
    return {
        "mean": {"RMSE": rmse(y_true, mean_pred), "R2": float(r2_score(y_true, mean_pred))},
        "persistence": {"RMSE": rmse(y_true, pers_pred), "R2": float(r2_score(y_true, pers_pred))}
    }

def load_scaler(path):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    Xm = np.array(s["Xm"], dtype=np.float32)
    Xs = np.array(s["Xs"], dtype=np.float32)
    ym = float(s["ym"])
    ys = float(s["ys"])
    return Xm, Xs, ym, ys

def apply_X_scaler(X, Xm, Xs):
    return ((X - Xm[None, None, :]) / (Xs[None, None, :] + 1e-12)).astype(np.float32)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--model_dir", type=str, default=r".\GRU_2\processed_hybrid\03_gru_residual")
    ap.add_argument("--out_dir", type=str, default=r".\GRU_2\processed_hybrid\04_hybrid_eval")
    ap.add_argument("--hidden", type=int, default=96)       # must match Phase4 v2
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()

    npz_path = os.path.abspath(args.npz)
    model_dir = os.path.abspath(args.model_dir)

    data = np.load(npz_path, allow_pickle=True)
    X0 = data["X_seq"].astype(np.float32)     # (N_seq, T, 4)
    meta = json.loads(data["meta"].item())

    A_meas = data["A_meas"].astype(np.float32)  # time series @1kHz aligned
    A_phys = data["A_phys"].astype(np.float32)  # time series @1kHz aligned
    res_raw = data["residual_raw"].astype(np.float32)  # y_meas - y_phys (length len(A)-H)

    seq_len = int(meta.get("seq_len", X0.shape[1]))
    stride = int(meta.get("stride", 50))
    H = int(meta.get("H"))
    ref_n = int(meta.get("REF_N", 1000))
    base = str(meta.get("base", "unknown"))

    N_seq, T, D0 = X0.shape
    print("\nLoaded NPZ:")
    print(f"  base={base} | H={H} | seq_len={seq_len} | stride={stride}")
    print(f"  X_seq={X0.shape} | A_meas={A_meas.shape} | A_phys={A_phys.shape} | residual_raw={res_raw.shape}")

    # -------- rebuild y_meas and y_phys from A_meas/A_phys to be explicit --------
    y_meas0 = compute_y_logratio(A_meas, H=H, ref_n=ref_n)  # length N_time-H
    y_phys0 = compute_y_logratio(A_phys, H=H, ref_n=ref_n)
    residual0 = (y_meas0 - y_phys0).astype(np.float32)

    # sanity check
    Lmin = min(len(residual0), len(res_raw))
    diff = float(np.mean(np.abs(residual0[:Lmin] - res_raw[:Lmin])))
    print(f"  sanity | mean|residual(recomputed)-residual(npz)| = {diff:.6e}")

    # -------- build lag sequence channel exactly like Phase4 v2 (past residual available) --------
    idxs = np.arange(0, N_seq * stride, stride, dtype=np.int64)

    # safety truncate if needed
    max_need = int(idxs[-1] + seq_len)
    if max_need > len(residual0):
        safe_N = int((len(residual0) - seq_len) // stride)
        safe_N = max(0, min(safe_N, N_seq))
        print(f"⚠️ residual shorter than needed. Truncate N_seq: {N_seq} -> {safe_N}")
        N_seq = safe_N
        X0 = X0[:N_seq]
        idxs = idxs[:N_seq]

    lag_seq = np.zeros((N_seq, seq_len), dtype=np.float32)
    r_end_true = np.zeros((N_seq,), dtype=np.float32)
    r_prev_true = np.zeros((N_seq,), dtype=np.float32)

    for i, s in enumerate(idxs):
        # per-timestep lag-1 residual
        lag_seq[i, 0] = residual0[s - 1] if (s - 1) >= 0 else 0.0
        lag_seq[i, 1:] = residual0[s : s + seq_len - 1]

        r_prev_true[i] = residual0[s + seq_len - 2]
        r_end_true[i]  = residual0[s + seq_len - 1]

    # target delta residual (true) at seq end
    dr_true = (r_end_true - r_prev_true).astype(np.float32)

    # append lag channel
    X = np.concatenate([X0, lag_seq[:, :, None]], axis=-1)  # dim 5
    D = X.shape[-1]

    # -------- split (same as Phase4) --------
    idx_tr, idx_va, idx_te = time_block_split(N_seq, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, gap=GAP)
    if len(idx_te) == 0:
        print("⚠️ Test split empty → using VAL as TEST")
        idx_te = idx_va

    print("\nSplit sizes:")
    print(f"  train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} (gap={GAP})")

    # -------- load scaler + normalize X --------
    scaler_path = os.path.join(model_dir, "scaler.json")
    best_path = os.path.join(model_dir, "gru_residual_best.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(model_dir, "gru_residual_final.pt")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler.json at: {scaler_path}")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing model weights at: {best_path}")

    Xm, Xs, ym, ys = load_scaler(scaler_path)
    Xn = apply_X_scaler(X, Xm, Xs)

    # -------- load model --------
    model = GRUResidual(input_dim=D, hidden_dim=args.hidden, num_layers=args.layers, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    # -------- predict delta residual on TEST indices --------
    with torch.no_grad():
        xb = torch.tensor(Xn[idx_te], dtype=torch.float32).to(DEVICE)
        dr_pred_n = model(xb).detach().cpu().numpy().astype(np.float32)

    # de-normalize delta residual
    dr_pred = (dr_pred_n * ys + ym).astype(np.float32)

    # ============================================================
    # ✅ PATCH: reconstruct residual_hat on TEST endpoints (1-step, teacher-forced)
    # (replace the old free-run integration block)
    # ============================================================
    # Map each test sequence to its residual index (t_end)
    t_end = (idxs[idx_te] + (seq_len - 1)).astype(np.int64)

    # True residual at (t_end-1) is available from residual0 (matches dr_true definition)
    t_prev = (t_end - 1).astype(np.int64)
    t_prev = np.clip(t_prev, 0, len(residual0) - 1)

    r_prev_true_test = residual0[t_prev].astype(np.float32)   # r(t_end-1)
    r_true_test      = residual0[t_end].astype(np.float32)    # r(t_end)

    # Optional: de-bias delta to prevent any small mean drift
    dr_pred_use = dr_pred.copy().astype(np.float32)
    dr_pred_use = dr_pred_use - float(np.mean(dr_pred_use))

    # 1-step residual estimate at endpoint:
    r_hat_test = (r_prev_true_test + dr_pred_use).astype(np.float32)
    # ============================================================

    # -------- build y predictions on TEST endpoints --------
    y_meas_test = y_meas0[t_end].astype(np.float32)
    y_phys_test = y_phys0[t_end].astype(np.float32)
    y_hybrid_test = (y_phys_test + r_hat_test).astype(np.float32)

    # -------- metrics --------
    metrics = {
        "base": base,
        "H": H,
        "task": "y_meas (log-ratio) prediction via hybrid residual integration",
        "Hybrid": {
            "RMSE": rmse(y_meas_test, y_hybrid_test),
            "R2": float(r2_score(y_meas_test, y_hybrid_test))
        },
        "PhysicsOnly": {
            "RMSE": rmse(y_meas_test, y_phys_test),
            "R2": float(r2_score(y_meas_test, y_phys_test))
        },
        "Baselines_on_y_meas": baseline_on_series(y_meas_test),
        "DeltaResidual_Test": {
            "RMSE": rmse(dr_true[idx_te], dr_pred),
            "R2": float(r2_score(dr_true[idx_te], dr_pred))
        },
        "split": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "gap": GAP},
        "paths": {
            "npz": npz_path,
            "model": best_path,
            "scaler": scaler_path
        },
        "meta": meta
    }

    # -------- output folder --------
    out_base = os.path.join(os.path.abspath(args.out_dir), f"{base}_H{H}")
    ensure_dir(out_base)

    # -------- plots (story-ready) --------
    # 1) y_meas vs y_phys vs y_hybrid (test)
    plt.figure(figsize=(11, 5))
    plt.plot(y_meas_test, label="y_meas (true)", linewidth=1.2)
    plt.plot(y_phys_test, label="y_phys (physics-only)", linewidth=1.2, alpha=0.85)
    plt.plot(y_hybrid_test, label="y_hybrid (physics + GRU)", linewidth=1.2, alpha=0.9)
    plt.title("Test: y_meas vs y_phys vs y_hybrid")
    plt.xlabel("Test sample index (sequence endpoints)")
    plt.ylabel("y (log-ratio)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "y_compare_test.png"), dpi=160)
    plt.close()

    # 2) residual true vs residual hat (test)
    plt.figure(figsize=(11, 5))
    plt.plot(r_true_test, label="residual_true", linewidth=1.2)
    plt.plot(r_hat_test, label="residual_hat (integrated)", linewidth=1.2, alpha=0.9)
    plt.title("Test: residual_true vs residual_hat")
    plt.xlabel("Test sample index (sequence endpoints)")
    plt.ylabel("residual")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "residual_compare_test.png"), dpi=160)
    plt.close()

    # 3) scatter y_hybrid vs y_meas
    plt.figure(figsize=(6, 6))
    plt.scatter(y_meas_test, y_hybrid_test, s=10, alpha=0.7)
    lo = float(min(y_meas_test.min(), y_hybrid_test.min()))
    hi = float(max(y_meas_test.max(), y_hybrid_test.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title("Scatter: y_hybrid vs y_meas (test)")
    plt.xlabel("y_meas (true)")
    plt.ylabel("y_hybrid (pred)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "scatter_yhybrid_vs_ymeas.png"), dpi=160)
    plt.close()

    # save summary
    with open(os.path.join(out_base, "summary_hybrid_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # -------- print summary --------
    print("\n================ Phase 5: Hybrid Evaluation (TEST) ================")
    print(f"[Hybrid]     RMSE={metrics['Hybrid']['RMSE']:.6f} | R2={metrics['Hybrid']['R2']:.4f}")
    print(f"[PhysicsOnly] RMSE={metrics['PhysicsOnly']['RMSE']:.6f} | R2={metrics['PhysicsOnly']['R2']:.4f}")

    print("\nBaselines on y_meas (TEST endpoints):")
    for k, v in metrics["Baselines_on_y_meas"].items():
        print(f"  {k:12s} | RMSE={v['RMSE']:.6f} | R2={v['R2']:.4f}")

    print("\nSaved outputs to:")
    print(" ", out_base)
    print("  - y_compare_test.png")
    print("  - residual_compare_test.png")
    print("  - scatter_yhybrid_vs_ymeas.png")
    print("  - summary_hybrid_test.json")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
