# ============================================================
# stepH4_train_gru_residual_withinfile.py  (v2)
# Phase 4: Train GRU as Residual AR-Compensator (leakage-safe)
#
# Key changes:
#  - Use residual_raw history as an extra sequence channel (per-timestep lag)
#  - Predict delta_residual_raw (r[t] - r[t-1]) to avoid persistence dominating
#  - Normalize X and y using TRAIN-only stats
#  - within-file time-block split + GAP
#
# Run:
#   python .\GRU_2\scripts_hybrid\stepH4_train_gru_residual_withinfile.py --npz .\GRU_2\processed_hybrid\02_residual_dataset\2.5nl-fix17.37_H5000.npz
# ============================================================

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- config ----------------
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
HIDDEN = 96
NUM_LAYERS = 1
DROPOUT = 0.0
GAP = 200         # safety gap (in sequence index)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- model ----------------
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


# ---------------- utils ----------------
def time_block_split(N, train_ratio=0.7, val_ratio=0.15, gap=200):
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_end = n_train
    val_start = train_end + gap
    val_end = min(val_start + n_val, N)
    test_start = min(val_end + gap, N)

    idx_train = np.arange(0, train_end)
    idx_val   = np.arange(val_start, val_end)
    idx_test  = np.arange(test_start, N)

    return idx_train, idx_val, idx_test


def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))


def baseline_metrics(y_true):
    mean_pred = np.full_like(y_true, np.mean(y_true), dtype=np.float32)
    pers_pred = np.concatenate([[0.0], y_true[:-1]]).astype(np.float32)

    return {
        "mean": {"RMSE": rmse(y_true, mean_pred), "R2": float(r2_score(y_true, mean_pred))},
        "persistence": {"RMSE": rmse(y_true, pers_pred), "R2": float(r2_score(y_true, pers_pred))}
    }


def fit_standardize_train(X_tr, y_tr, eps=1e-8):
    # X: (N, T, D) -> normalize per feature over all train samples+timesteps
    Xm = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    Xs = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + eps

    ym = y_tr.mean()
    ys = y_tr.std() + eps
    return Xm, Xs, float(ym), float(ys)


def apply_standardize(X, y, Xm, Xs, ym, ys):
    Xn = (X - Xm[None, None, :]) / Xs[None, None, :]
    yn = (y - ym) / ys
    return Xn.astype(np.float32), yn.astype(np.float32)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, default=r".\GRU_2\processed_hybrid\03_gru_residual")
    args = ap.parse_args()

    #out_dir = os.path.abspath(args.out)
    #ensure_dir(out_dir)

    # --- NEW: make unique subfolder per npz (avoid overwrite) ---
    npz_base = os.path.splitext(os.path.basename(args.npz))[0]  # e.g., 3.5nl-fix17.07std_H5000
    out_dir = os.path.abspath(os.path.join(args.out, npz_base))
    ensure_dir(out_dir)


    data = np.load(args.npz, allow_pickle=True)
    X = data["X_seq"].astype(np.float32)            # (N_seq, T, 4)
    meta = json.loads(data["meta"].item())
    res_raw = data["residual_raw"].astype(np.float32)  # (N_time,)

    seq_len = int(meta.get("seq_len", X.shape[1]))
    stride  = int(meta.get("stride", 50))

    N_seq, T, D0 = X.shape
    assert T == seq_len, f"SEQ_LEN mismatch: X has T={T}, meta seq_len={seq_len}"

    print("\nLoaded dataset:")
    print(f"  X shape = {X.shape} (features={D0})")
    print(f"  residual_raw length = {len(res_raw)}")
    print(f"  meta: seq_len={seq_len}, stride={stride}, H={meta.get('H')}")

    # --------------------------------------------------------
    # Build AR-style sequence channel from residual_raw:
    # For each sequence i starting at s=i*stride:
    #   lag_seq[j] = residual_raw[s + j - 1]  (lag-1 per timestep)
    #   target at end: r_end = residual_raw[s + seq_len - 1]
    #   delta target: dr = r_end - residual_raw[s + seq_len - 2]
    # --------------------------------------------------------
    idxs = np.arange(0, N_seq * stride, stride, dtype=np.int64)

    # safety check (avoid out-of-bound)
    max_need = int(idxs[-1] + seq_len)
    if max_need > len(res_raw):
        # truncate N_seq to safe
        safe_N = int((len(res_raw) - seq_len) // stride)
        safe_N = max(0, min(safe_N, N_seq))
        print(f"⚠️ residual_raw shorter than needed. Truncate N_seq: {N_seq} -> {safe_N}")
        N_seq = safe_N
        X = X[:N_seq]
        idxs = idxs[:N_seq]

    # build lag sequence channel
    lag_seq = np.zeros((N_seq, seq_len), dtype=np.float32)
    y_end   = np.zeros((N_seq,), dtype=np.float32)
    y_prev  = np.zeros((N_seq,), dtype=np.float32)

    for i, s in enumerate(idxs):
        # lag-1 per timestep
        # j=0 uses s-1 (undefined) -> set 0.0
        if s - 1 >= 0:
            lag_seq[i, 0] = res_raw[s - 1]
        else:
            lag_seq[i, 0] = 0.0
        lag_seq[i, 1:] = res_raw[s : s + seq_len - 1]

        y_prev[i] = res_raw[s + seq_len - 2]
        y_end[i]  = res_raw[s + seq_len - 1]

    # delta target
    y = (y_end - y_prev).astype(np.float32)

    # append as new channel (per timestep)
    lag_seq_3d = lag_seq[:, :, None]  # (N_seq, T, 1)
    X = np.concatenate([X, lag_seq_3d], axis=-1)  # (N_seq, T, D0+1)
    D = X.shape[-1]

    print("\n=== Task definition ===")
    print("Input : [I_rms, f_inst, G_res, A_phys] + residual_raw lag sequence")
    print("Target: delta_residual_raw at sequence end (r[t]-r[t-1])")
    print(f"Final X dim = {D}")

    # -------- split --------
    idx_tr, idx_va, idx_te = time_block_split(N_seq, gap=GAP)
    if len(idx_te) == 0:
        print("⚠️ Test split empty → using VAL as TEST")
        idx_te = idx_va

    print("\nSplit sizes:")
    print(f"  train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} (gap={GAP})")

    # -------- normalize (train-only) --------
    Xm, Xs, ym, ys = fit_standardize_train(X[idx_tr], y[idx_tr])
    Xn, yn = apply_standardize(X, y, Xm, Xs, ym, ys)

    # save scaler
    with open(os.path.join(out_dir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"Xm": Xm.tolist(), "Xs": Xs.tolist(), "ym": ym, "ys": ys}, f, indent=2)

    def make_loader(idxs, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xn[idxs], dtype=torch.float32),
            torch.tensor(yn[idxs], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(idx_tr, shuffle=True)
    val_loader   = make_loader(idx_va, shuffle=False)
    test_loader  = make_loader(idx_te, shuffle=False)

    # -------- model --------
    model = GRUResidual(D, HIDDEN, NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    hist = {"train": [], "val": []}

    # -------- training --------
    print("\n=== Training GRU (Residual AR) ===")
    best_val = 1e9
    best_path = os.path.join(out_dir, "gru_residual_best.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_loss += loss_fn(model(xb), yb).item()
        va_loss /= max(1, len(val_loader))

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train={tr_loss:.3e} | val={va_loss:.3e}")

    # load best
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    # -------- test --------
    y_true_n, y_pred_n = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yp = model(xb).cpu().numpy()
            y_pred_n.append(yp)
            y_true_n.append(yb.numpy())

    y_true_n = np.concatenate(y_true_n).astype(np.float32)
    y_pred_n = np.concatenate(y_pred_n).astype(np.float32)

    # de-normalize to original delta scale
    y_true = (y_true_n * ys + ym).astype(np.float32)
    y_pred = (y_pred_n * ys + ym).astype(np.float32)

    metrics = {
        "GRU": {"RMSE": rmse(y_true, y_pred), "R2": float(r2_score(y_true, y_pred))},
        "Baseline": baseline_metrics(y_true),
        "meta": meta
    }

    print("\n=== GRU Residual-AR Test (delta residual_raw) ===")
    print(f"RMSE = {metrics['GRU']['RMSE']:.6f}")
    print(f"R2   = {metrics['GRU']['R2']:.4f}")

    print("\n=== Baselines ===")
    for k, v in metrics["Baseline"].items():
        print(f"{k:12s} | RMSE={v['RMSE']:.6f} | R2={v['R2']:.4f}")

    # -------- plots --------
    plt.figure()
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized)")
    plt.title("Training history (GRU delta-residual AR)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(y_true, label="true", linewidth=1.2)
    plt.plot(y_pred, label="pred", linewidth=1.2, alpha=0.85)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.legend()
    plt.title("delta residual_raw prediction (test)")
    plt.savefig(os.path.join(out_dir, "delta_residual_pred_test.png"), dpi=150)
    plt.close()

    # save model + metrics
    torch.save(model.state_dict(), os.path.join(out_dir, "gru_residual_final.pt"))
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Saved outputs to:", out_dir)
    print(" - gru_residual_best.pt / gru_residual_final.pt")
    print(" - loss_curve.png / delta_residual_pred_test.png")
    print(" - scaler.json / run_summary.json")


if __name__ == "__main__":
    main()
