import os
import time
import random
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ==================================================
# Training function (MLR + MLP ONLY)
# ==================================================
def train(config, data, model, device, criterion):

    num_runs = int(config.get("num_runs", 1))
    start_run = int(config.get("start_run", 0))
    nepochs = int(config.get("epochs", 100))
    lr = float(config.get("lr", 1e-3))
    batch_size = int(config.get("batch_size", 32))

    save_plots = bool(config.get("save_plots", True))
    save_test_results = bool(config.get("save_test_results", True))
    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    
    prcnt_plt = (
        torch.arange(0, 100.0001, 1, dtype=torch.float32) / 100.0
    ).cpu().numpy()

    val_slopes = []

    # Move full datasets once
    XTrain = data["XTrain"].to(device)
    zTrain = data["zTrain"].to(device)
    XVal = data["XVal"].to(device)
    ZVal = data["ZVal"].to(device)
    XTest = data["XTest"].to(device)
    ZTest = data["ZTest"].to(device)

    for run in range(start_run, num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run)

        # Reset model weights
        model.apply(reset_weights)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # DataLoader
        g = torch.Generator()
        g.manual_seed(run)

        losses = []
        start_time = time.time()

        # ==============================================
        # Training loop
        # ==============================================
        for epoch in range(nepochs):
            model.train()
        
            optimizer.zero_grad()
        
            Y_pred = model(XTrain).squeeze(-1)
            loss = criterion(Y_pred, zTrain)
        
            loss.backward()
            optimizer.step()
        
            epoch_loss = loss.item()
            losses.append(epoch_loss)
        
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{nepochs} - Loss: {epoch_loss:.6f}")

        # ==============================================
        # Save model + losses
        # ==============================================
        flnm = os.path.join(output_dir, f"{type(model).__name__}_Run{run+1}")

        np.savetxt(f"{flnm}_losses.txt", np.array(losses))
        torch.save(model.state_dict(), f"{flnm}_checkpoint.pth")

        joblib.dump(
            {
                "scaler": data.get("scaler"),
                "feature_names": data.get("feature_names"),
            },
            f"{flnm}_preprocessing.pkl",
        )

        # ==============================================
        # Evaluation
        # ==============================================
        for subset, X, Z, title in zip(
            ["training", "validation", "testing"],
            [XTrain, XVal, XTest],
            [zTrain, ZVal, ZTest],
            ["Training", "Validation", "Testing"],
        ):
            model.eval()
            with torch.no_grad():
                Y_pred = model(X).squeeze(-1).detach().cpu().numpy().flatten()
                real = Z.detach().cpu().numpy().flatten()

            metrics, txt = evaluate_metrics(real, Y_pred, prcnt_plt)

            if subset == "validation":
                val_slopes.append((run, metrics[5]))
                print(f"  Validation slope: {metrics[5]:.4f}")

            if save_plots:
                plot_results(real, Y_pred, prcnt_plt, title, f"{flnm}_{subset}.png", txt)

            if subset == "testing" and save_test_results:
                save_testing_results(data, Y_pred, real, flnm)

        print(f"Run {run+1} completed in {time.time() - start_time:.2f} seconds")

    # ==============================================
    # Best run selection
    # ==============================================
    if len(val_slopes) > 0:
        slope_file = os.path.join(output_dir, "val_slopes.txt")
        np.savetxt(slope_file, val_slopes, fmt="Run %d: slope = %.6f")

        best_run = min(val_slopes, key=lambda x: abs(x[1] - 1.0))
        print(
            f"\nBest run: Run {best_run[0] + 1} | Validation slope = {best_run[1]:.4f}"
        )

    return val_slopes


# ==================================================
# Utilities
# ==================================================
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


def save_testing_results(data, Y_pred, real, flnm):
    try:
        import pandas as pd

        mslpTest = data.get("mslpTest")
        hourTest = data.get("hourTest")

        if mslpTest is not None and hourTest is not None:
            mslpTest_reset_index = mslpTest.reset_index(drop=True)

            df = pd.DataFrame({
                "year": mslpTest_reset_index.iloc[:, 0],
                "month": mslpTest_reset_index.iloc[:, 1],
                "day": mslpTest_reset_index.iloc[:, 2],
                "hour": hourTest,
                "minutes": mslpTest_reset_index.iloc[:, 4],
                "seconds": mslpTest_reset_index.iloc[:, 5],
                "num_date": mslpTest_reset_index.iloc[:, 6],
                "pred": Y_pred,
                "obs": real,
            })

            np.savetxt(
                f"{flnm}_testresults.txt",
                df.values,
                fmt="%.6f",
                delimiter=" "
            )

    except Exception as e:
        print(f"Warning: could not save test results due to: {e}")