import os
import time
import random
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def train(config, data, model, device, criterion):
    # ==================================================
    # Configuration
    # ==================================================
    num_runs = int(config["num_runs"])
    nepochs = int(config["epochs"])
    lr = float(config["lr"])
    start_run = int(config.get("start_run", 0))

    batch_size = int(config.get("batch_size", 2))
    validate_every = int(config.get("validate_every", 50))

    use_early_stopping = bool(config.get("early_stopping", False))
    patience = int(config.get("patience", 20))
    min_delta = float(config.get("min_delta", 1e-4))

    save_plots = bool(config.get("save_plots", True))
    save_test_results = bool(config.get("save_test_results", True))
    compute_full_metrics = bool(config.get("compute_full_metrics", True))

    num_workers = int(config.get("num_workers", 2))
    shuffle_train = bool(config.get("shuffle_train", False))

    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    prcnt_plt = (
        torch.arange(0, 100.0001, 1, dtype=torch.float32) / 100.0
    ).cpu().numpy()

    val_slopes = []

    # ==================================================
    # Data (KEEP ON CPU for DataLoader + pin_memory)
    # ==================================================
    XTrain = data["XTrain"]
    zTrain = data["zTrain"]

    XVal = data["XVal"]
    ZVal = data["ZVal"]

    XTest = data["XTest"]
    ZTest = data["ZTest"]

    # ==================================================
    # Batch size strategy
    # ==================================================
    if config.get("batch_mode", "mini") == "fraction":
        n_splits = int(config.get("n_splits", 2))
        batch_size = max(1, len(XTrain) // n_splits)

    print(f"Batch mode: {config.get('batch_mode', 'mini')}, batch_size: {batch_size}")

    # ==================================================
    # DataLoader
    # ==================================================
    train_loader = DataLoader(
        TensorDataset(XTrain, zTrain),
        batch_size=batch_size,
        shuffle=shuffle_train,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # ==================================================
    # AMP setup
    # ==================================================
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ==================================================
    # Loop over runs
    # ==================================================
    for run in range(start_run, num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)

        model.apply(reset_weights)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()
        losses = []

        best_val_loss = np.inf
        best_state_dict = None
        epochs_no_improve = 0
        stop_epoch = nepochs

        # ==================================================
        # Training loop (AMP + batching)
        # ==================================================
        for epoch in range(nepochs):
            model.train()
            epoch_loss = 0.0

            for X_batch, z_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                z_batch = z_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    Y_pred = model(X_batch)
                    loss = criterion(Y_pred, z_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(train_loader.dataset)
            losses.append(epoch_loss)

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------
            if epoch % validate_every == 0:
                model.eval()
                with torch.no_grad():
                    Y_val = predict_in_batches(model, XVal, batch_size, device)
                    val_loss = criterion(Y_val, ZVal).item()

                print(
                    f"Epoch {epoch + 1}/{nepochs} | "
                    f"Train loss: {epoch_loss:.6f} | "
                    f"Val loss: {val_loss:.6f}"
                )

                if use_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_state_dict = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        stop_epoch = epoch + 1
                        print(f"Early stopping triggered at epoch {stop_epoch}")
                        break

        # --------------------------------------------------
        # Restore best model
        # --------------------------------------------------
        if use_early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # --------------------------------------------------
        # Save model
        # --------------------------------------------------
        flnm = os.path.join(output_dir, f"{type(model).__name__}_Run{run + 1}")

        np.savetxt(f"{flnm}_losses.txt", losses)
        torch.save(model.state_dict(), f"{flnm}_checkpoint.pth")

        joblib.dump(
            {
                "scaler": data["scaler"],
                "feature_names": data["feature_names"],
                "lookback": config.get("lookback", None),
                "batch_mode": config.get("batch_mode", "mini"),
                "batch_size": batch_size,
                "shuffle_train": shuffle_train,
                "amp_enabled": use_amp,
            },
            f"{flnm}_preprocessing.pkl",
        )

        # ==================================================
        # Evaluation
        # ==================================================
        eval_sets = [
            ("training", XTrain, zTrain, "Training"),
            ("validation", XVal, ZVal, "Validation"),
            ("testing", XTest, ZTest, "Testing"),
        ]

        for subset, X, Z, title in eval_sets:
            model.eval()
            with torch.no_grad():
                Y_pred = predict_in_batches(model, X, batch_size, device)
                Y_pred = Y_pred.numpy().reshape(-1)
                real = Z.numpy().reshape(-1)

            if subset == "validation":
                metrics, txt = evaluate_metrics(real, Y_pred, prcnt_plt)
                val_slopes.append((run, metrics[5]))
                print(f"  Slope on validation set: {metrics[5]:.4f}")
                print(f"  Training stopped at epoch: {stop_epoch}")
            elif compute_full_metrics:
                metrics, txt = evaluate_metrics(real, Y_pred, prcnt_plt)
            else:
                continue

            if save_plots:
                plot_results(real, Y_pred, prcnt_plt, title, f"{flnm}_{subset}.png", txt)

            if subset == "testing" and save_test_results:
                try:
                    import pandas as pd

                    mslpTest = data.get("mslpTest")
                    hourTest = data.get("hourTest")

                    if mslpTest is not None and hourTest is not None:
                        mslpTest_reset = mslpTest.reset_index(drop=True)

                        min_len = min(
                            len(mslpTest_reset),
                            len(hourTest),
                            len(Y_pred),
                            len(real),
                        )

                        df = pd.DataFrame({
                            "year": mslpTest_reset.iloc[:min_len, 0].to_numpy(),
                            "month": mslpTest_reset.iloc[:min_len, 1].to_numpy(),
                            "day": mslpTest_reset.iloc[:min_len, 2].to_numpy(),
                            "hour": np.asarray(hourTest[:min_len]),
                            "minutes": mslpTest_reset.iloc[:min_len, 4].to_numpy(),
                            "seconds": mslpTest_reset.iloc[:min_len, 5].to_numpy(),
                            "num_date": mslpTest_reset.iloc[:min_len, 6].to_numpy(),
                            "pred": Y_pred[:min_len],
                            "obs": real[:min_len],
                        })

                        np.savetxt(
                            f"{flnm}_testresults.txt",
                            df.values,
                            fmt="%.6f",
                            delimiter=" "
                        )
                except Exception as e:
                    print(f"Warning: could not save test results due to: {e}")

        print(f"Run {run + 1} completed in {time.time() - start_time:.2f} seconds")

    # ==================================================
    # Save validation slopes
    # ==================================================
    if len(val_slopes) > 0:
        slope_file = os.path.join(output_dir, "val_slopes.txt")
        np.savetxt(slope_file, val_slopes, fmt="Run %d: slope = %.6f")

        best_run = min(val_slopes, key=lambda x: abs(x[1] - 1.0))
        print(
            f"\nThe best run is Run {best_run[0] + 1}, "
            f"with a slope in the validation set = {best_run[1]:.4f}"
        )

    return val_slopes


# ======================================================
# Utilities
# ======================================================
def predict_in_batches(model, X, batch_size, device):
    model.eval()
    preds = []

    # Safer inference batch size to avoid OOM during evaluation
    eval_batch_size = min(batch_size, 2048)

    with torch.no_grad():
        for i in range(0, len(X), eval_batch_size):
            X_batch = X[i:i + eval_batch_size].to(device, non_blocking=True)
            Y_batch = model(X_batch)
            preds.append(Y_batch.cpu())

    return torch.cat(preds, dim=0)


def reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def evaluate_metrics(real, pred, prcnt):
    real = real.astype(np.float64)
    pred = pred.astype(np.float64)

    real_c = real - np.mean(real)
    pred_c = pred - np.mean(pred)

    prc = np.arange(0, 101, 1)

    target_prc = np.percentile(real_c, prc, method="linear")
    pred_prc = np.percentile(pred_c, prc, method="linear")

    mad = np.mean(np.abs(real_c - pred_c))
    madp = np.mean(np.abs(target_prc - pred_prc))
    madc = mad + madp

    corr = np.corrcoef(real_c, pred_c)[0, 1]
    rmse = np.sqrt(np.mean((real_c - pred_c) ** 2))
    slope, _ = np.polyfit(real_c, pred_c, 1)

    txt = (
        f"Slope: {slope:.3f}\n"
        f"Pearson: {corr:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"MAD: {mad:.3f}\n"
        f"MADp: {madp:.3f}\n"
        f"MADc: {madc:.3f}"
    )

    return (mad, madp, madc, corr, rmse, slope), txt


def plot_results(real, pred, prcnt, title, filename, txt):
    plt.figure(figsize=(6, 6))
    plt.scatter(real, pred, label="Data", s=6)
    plt.plot([-0.7, 1.3], [-0.7, 1.3], "r-", label="Ideal")
    plt.plot(
        np.quantile(real, prcnt),
        np.quantile(pred, prcnt),
        "k-",
        label="Percentiles",
    )
    plt.text(
        0.95,
        0.05,
        txt,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=10,
    )
    plt.xlabel("Target [m]")
    plt.ylabel("Prediction [m]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()