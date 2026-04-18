# ============================================================
# stepH5_eval_hybrid_ymeas_noAR_v3.py  [PATCHED stride-consistent]
# Phase 5: Evaluate Hybrid on y_meas (log-ratio) space
#
# PATCH:
#  - Make delta residual consistent with stride:
#      dr_true = r[t_end] - r[t_end - stride]
#  - Use mask to drop sequences where (t_end - stride) < 0
#  - "last-init" uses residual at (t_test_start_end - stride), not (t_end - 1)
#  - Keep everything else unchanged
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

def rollout(r0, dr_series):
    """r_hat[k] = r_hat[k-1] + dr[k], with r_hat[0] = r0 + dr[0]"""
    dr_series = dr_series.astype(np.float32)
    r_hat = np.zeros_like(dr_series, dtype=np.float32)
    if len(dr_series) == 0:
        return r_hat
    r_hat[0] = np.float32(r0) + dr_series[0]
    for k in range(1, len(dr_series)):
        r_hat[k] = r_hat[k-1] + dr_series[k]
    return r_hat

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--model_dir", type=str, default=r".\GRU_2\processed_hybrid\03_gru_residual")
    ap.add_argument("--out_dir", type=str, default=r".\GRU_2\processed_hybrid\04_hybrid_eval")
    ap.add_argument("--hidden", type=int, default=96)       # must match Phase4 v2
    ap.add_argument("--layers", type=int, default=1)

    # (optional) ใส่ tag ได้ แต่ถ้าไม่ใส่ก็ยังแยกโฟลเดอร์ด้วยชื่อ npz_base อยู่แล้ว
    ap.add_argument("--tag", type=str, default="",
                    help="optional subfolder tag (e.g., noAR_rollout). leave empty to skip")
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
    residual0 = res_raw.astype(np.float32)

    # -------- build residual endpoints exactly like Phase4 (NO-AR input) --------
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

    # ----- PATCH START: stride-consistent indices + drop invalid -----
    end_idx  = (idxs + (seq_len - 1)).astype(np.int64)
    prev_idx = (end_idx - stride).astype(np.int64)

    valid = prev_idx >= 0
    if not np.all(valid):
        drop_n = int(np.sum(~valid))
        print(f"⚠️ Drop {drop_n} sequences where prev_idx < 0 (need stride-back)")

    X0 = X0[valid]
    idxs = idxs[valid]
    end_idx = end_idx[valid]
    prev_idx = prev_idx[valid]

    N_seq = X0.shape[0]
    # ----- PATCH END -----

    # true endpoints (for delta target)
    r_end_true = residual0[end_idx].astype(np.float32)
    r_prev_true = residual0[prev_idx].astype(np.float32)

    # ✅ PATCH: stride-step delta residual (true)
    dr_true = (r_end_true - r_prev_true).astype(np.float32)

    # -------- NO-AR input --------
    X = X0
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

    # -------- debug stats: delta residual --------
    print("dr_pred mean/std/min/max:",
        float(dr_pred.mean()),
        float(dr_pred.std()),
        float(dr_pred.min()),
        float(dr_pred.max()))

    print("dr_true mean/std/min/max:",
        float(dr_true[idx_te].mean()),
        float(dr_true[idx_te].std()),
        float(dr_true[idx_te].min()),
        float(dr_true[idx_te].max()))

    # ============================================================
    # ✅ FIX: free-run rollout (no teacher-forcing)
    #   - sort endpoints by time
    #   - init=last uses residual at (t_first_end - stride)
    #   - init=mean uses mean residual of TRAIN endpoints
    # ============================================================

    # ---- sort endpoints by time ----
    t_end = end_idx[idx_te].astype(np.int64)
    order = np.argsort(t_end)
    t_end_s = t_end[order]
    dr_pred_s = dr_pred[order].astype(np.float32)

    # ---- init candidates ----
    # ✅ PATCH: last-init should use residual at (first test end - stride)
    test_start_seq = int(np.min(idx_te))
    t_end_first = int(end_idx[test_start_seq])
    t0 = int(t_end_first - stride)
    t0 = int(np.clip(t0, 0, len(residual0) - 1))
    r0_last = float(residual0[t0])

    # mean-init: ใช้ค่าเฉลี่ย residual ใน train endpoints
    t_end_train = end_idx[idx_tr].astype(np.int64)
    t_end_train = np.clip(t_end_train, 0, len(residual0) - 1)
    r0_mean = float(np.mean(residual0[t_end_train].astype(np.float32)))

    # ---- rollout in sorted time ----
    r_hat_last_s = rollout(r0_last, dr_pred_s)
    r_hat_mean_s = rollout(r0_mean, dr_pred_s)

    # ---- test truth aligned with sorted time ----
    y_meas_test = y_meas0[t_end_s].astype(np.float32)
    y_phys_test = y_phys0[t_end_s].astype(np.float32)
    r_true_test = residual0[t_end_s].astype(np.float32)

    y_hybrid_last = (y_phys_test + r_hat_last_s).astype(np.float32)
    y_hybrid_mean = (y_phys_test + r_hat_mean_s).astype(np.float32)

    # ============================================================

    # -------- metrics --------
    metrics = {
        "base": base,
        "H": H,
        "task": "y_meas (log-ratio) prediction via hybrid residual rollout (no teacher-forcing)",
        "Hybrid_last": {
            "RMSE": rmse(y_meas_test, y_hybrid_last),
            "R2": float(r2_score(y_meas_test, y_hybrid_last))
        },
        "Hybrid_mean": {
            "RMSE": rmse(y_meas_test, y_hybrid_mean),
            "R2": float(r2_score(y_meas_test, y_hybrid_mean))
        },
        "PhysicsOnly": {
            "RMSE": rmse(y_meas_test, y_phys_test),
            "R2": float(r2_score(y_meas_test, y_phys_test))
        },
        "Baselines_on_y_meas": baseline_on_series(y_meas_test),
        "DeltaResidual_Test": {
            "RMSE": rmse(dr_true[idx_te][order], dr_pred_s),
            "R2": float(r2_score(dr_true[idx_te][order], dr_pred_s))
        },
        "split": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "gap": GAP},
        "paths": {
            "npz": npz_path,
            "model": best_path,
            "scaler": scaler_path
        },
        "meta": meta,
        "rollout_init": {
            "r0_last": r0_last,
            "r0_mean": r0_mean,
            "t0_last_source": t0
        }
    }

    # -------- output folder (หัวข้อ 4A) --------
    npz_base = os.path.splitext(os.path.basename(args.npz))[0]  # เช่น 2.5nl-fix17.37_H5000
    if args.tag.strip() != "":
        out_base = os.path.join(os.path.abspath(args.out_dir), args.tag, npz_base)
    else:
        out_base = os.path.join(os.path.abspath(args.out_dir), npz_base)
    ensure_dir(out_base)

    # -------- plots --------
    # 1) last-init
    plt.figure(figsize=(11, 5))
    plt.plot(y_meas_test, label="y_meas (true)", linewidth=1.2)
    plt.plot(y_phys_test, label="y_phys (physics-only)", linewidth=1.2, alpha=0.85)
    plt.plot(y_hybrid_last, label="y_hybrid_last (rollout, init=last)", linewidth=1.2, alpha=0.9)
    plt.title("Test (sorted): y_meas vs y_phys vs y_hybrid_last")
    plt.xlabel("Test sample index (sorted endpoints)")
    plt.ylabel("y (log-ratio)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "y_compare_test_last.png"), dpi=160)
    plt.close()

    # 2) mean-init
    plt.figure(figsize=(11, 5))
    plt.plot(y_meas_test, label="y_meas (true)", linewidth=1.2)
    plt.plot(y_phys_test, label="y_phys (physics-only)", linewidth=1.2, alpha=0.85)
    plt.plot(y_hybrid_mean, label="y_hybrid_mean (rollout, init=mean)", linewidth=1.2, alpha=0.9)
    plt.title("Test (sorted): y_meas vs y_phys vs y_hybrid_mean")
    plt.xlabel("Test sample index (sorted endpoints)")
    plt.ylabel("y (log-ratio)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "y_compare_test_mean.png"), dpi=160)
    plt.close()

    # 3) combined
    plt.figure(figsize=(11, 5))
    plt.plot(y_meas_test, label="y_meas (true)", linewidth=1.2)
    plt.plot(y_phys_test, label="y_phys (physics-only)", linewidth=1.2, alpha=0.7)
    plt.plot(y_hybrid_last, label="y_hybrid_last", linewidth=1.2, alpha=0.9)
    plt.plot(y_hybrid_mean, label="y_hybrid_mean", linewidth=1.2, alpha=0.9)
    plt.title("Test (sorted): y_meas vs y_phys vs (hybrid_last + hybrid_mean)")
    plt.xlabel("Test sample index (sorted endpoints)")
    plt.ylabel("y (log-ratio)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "y_compare_test_both.png"), dpi=160)
    plt.close()

    # 4) residual compare (คงเดิม แต่ใช้ last-init เป็น default)
    plt.figure(figsize=(11, 5))
    plt.plot(r_true_test, label="residual_true", linewidth=1.2)
    plt.plot(r_hat_last_s, label="residual_hat_last (rollout)", linewidth=1.2, alpha=0.9)
    plt.title("Test (sorted): residual_true vs residual_hat_last")
    plt.xlabel("Test sample index (sorted endpoints)")
    plt.ylabel("residual")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "residual_compare_test.png"), dpi=160)
    plt.close()

    # 5) scatter y_hybrid_last vs y_meas (คงเดิมชื่อไฟล์)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_meas_test, y_hybrid_last, s=10, alpha=0.7)
    lo = float(min(y_meas_test.min(), y_hybrid_last.min()))
    hi = float(max(y_meas_test.max(), y_hybrid_last.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title("Scatter: y_hybrid_last vs y_meas (test, sorted)")
    plt.xlabel("y_meas (true)")
    plt.ylabel("y_hybrid_last (pred)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "scatter_yhybrid_vs_ymeas.png"), dpi=160)
    plt.close()

    # save summary
    with open(os.path.join(out_base, "summary_hybrid_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # -------- print summary --------
    print("\n================ Phase 5: Hybrid Evaluation (TEST) ================")
    print(f"[Hybrid_last] RMSE={metrics['Hybrid_last']['RMSE']:.6f} | R2={metrics['Hybrid_last']['R2']:.4f}")
    print(f"[Hybrid_mean] RMSE={metrics['Hybrid_mean']['RMSE']:.6f} | R2={metrics['Hybrid_mean']['R2']:.4f}")
    print(f"[PhysicsOnly] RMSE={metrics['PhysicsOnly']['RMSE']:.6f} | R2={metrics['PhysicsOnly']['R2']:.4f}")

    print("\nBaselines on y_meas (TEST endpoints):")
    for k, v in metrics["Baselines_on_y_meas"].items():
        print(f"  {k:12s} | RMSE={v['RMSE']:.6f} | R2={v['R2']:.4f}")

    print("\nSaved outputs to:")
    print(" ", out_base)
    print("  - y_compare_test_last.png")
    print("  - y_compare_test_mean.png")
    print("  - y_compare_test_both.png")
    print("  - residual_compare_test.png")
    print("  - scatter_yhybrid_vs_ymeas.png")
    print("  - summary_hybrid_test.json")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
