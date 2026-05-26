import os
import time
import random
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt


def train(config, data, model, device, criterion):
    # ==================================================
    # Configuration
    # ==================================================
    num_runs = int(config["num_runs"])
    nepochs = int(config["epochs"])
    lr = float(config["lr"])
    start_run = int(config.get("start_run", 0))

    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    save_plots = bool(config.get("save_plots", True))
    save_test_results = bool(config.get("save_test_results", True))
    compute_full_metrics = bool(config.get("compute_full_metrics", True))

    use_early_stopping = bool(config.get("early_stopping", False))
    patience = int(config.get("patience", 20))
    min_delta = float(config.get("min_delta", 1e-4))
    validate_every = int(config.get("validate_every", 100))

    chunk_size = int(config.get("chunk_size", 1024))

    prcnt_plt = (
        torch.arange(0, 100.0001, 1, dtype=torch.float32) / 100.0
    ).cpu().numpy()

    val_slopes = []

    # ==================================================
    # Full-sequence tensors
    # ==================================================
    XTrain = data["XTrain"].to(device).contiguous()
    zTrain = data["zTrain"].to(device).contiguous()

    XVal = data["XVal"].to(device).contiguous()
    ZVal = data["ZVal"].to(device).contiguous()

    XTest = data["XTest"].to(device).contiguous()
    ZTest = data["ZTest"].to(device).contiguous()

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
        # Training loop (TBPTT + global MADp buffer)
        # ==================================================
        for epoch in range(nepochs):
            model.train()
            
            hidden = None
            epoch_loss = 0.0
            num_chunks = 0
            T = XTrain.size(1)
        
            # Global MADp buffer for this epoch
            buffer_true = []
            buffer_pred = []
        
            # Configurable parameters
            use_global_madp = bool(config.get("use_global_madp", False))
            alpha_madp = float(config.get("alpha_madp", 1.0))
            buffer_max_size = int(config.get("buffer_max_size", 50000))
            
            # -----------------------------------------
            # Sanity check: print loss configuration
            # -----------------------------------------
            if epoch % 100 == 0:
                if use_global_madp:
                    print(f"[Epoch {epoch+1}] Using GLOBAL MADp loss (alpha={alpha_madp})")
                else:
                    print(f"[Epoch {epoch+1}] Using standard loss: {config['loss_function']}")
        
            if config.get("madp_percentiles_full", False):
                percentiles_loss = torch.arange(
                    0, 100.0001, 1, dtype=torch.float32, device=device
                ) / 100.0
            else:
                percentiles_loss = torch.tensor(
                    config.get(
                        "madp_percentiles",
                        [0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
                         0.06, 0.07, 0.08, 0.09, 0.10,
                         0.20, 0.30, 0.40, 0.50,
                         0.60, 0.70, 0.80, 0.90,
                         0.95, 0.98, 0.99, 0.995, 0.999,
                         1.00]
                    ),
                    dtype=torch.float32,
                    device=device
                )
        
            for t in range(0, T, chunk_size):
                X_chunk = XTrain[:, t:t + chunk_size, :]
                z_chunk = zTrain[:, t:t + chunk_size, :]
        
                optimizer.zero_grad()
        
                Y_pred, hidden = model(X_chunk, hidden)
        
                # --------------------------------------------------
                # Standard local loss
                # --------------------------------------------------
                if use_global_madp:
                    # Local MAD term
                    mad = torch.nanmean(torch.abs(z_chunk - Y_pred))
        
                    # --------------------------------------------------
                    # Build global-like distribution:
                    # detached history + current chunk with gradient
                    # --------------------------------------------------
                    if len(buffer_true) > 0:
                        z_true_hist = torch.cat(buffer_true, dim=0)
                        z_pred_hist = torch.cat(buffer_pred, dim=0)
        
                        z_true_global = torch.cat(
                            [z_true_hist, z_chunk.reshape(-1)],
                            dim=0
                        )
        
                        z_pred_global = torch.cat(
                            [z_pred_hist, Y_pred.reshape(-1)],
                            dim=0
                        )
                    else:
                        z_true_global = z_chunk.reshape(-1)
                        z_pred_global = Y_pred.reshape(-1)
        
                    target_prc = torch.quantile(
                        z_true_global,
                        percentiles_loss
                    )
        
                    pred_prc = torch.quantile(
                        z_pred_global,
                        percentiles_loss
                    )
        
                    mad_prc = torch.mean(torch.abs(target_prc - pred_prc))
        
                    # Global-MADp-enhanced loss
                    loss = (mad + alpha_madp * mad_prc) ** 2
        
                else:
                    # Original criterion, e.g. MSE or ordinary MADc2
                    loss = criterion(Y_pred, z_chunk)
        
                loss.backward()
        
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
                optimizer.step()
        
                epoch_loss += loss.item()
                num_chunks += 1
        
                # --------------------------------------------------
                # Update buffers AFTER optimizer step
                # Store only detached tensors to avoid breaking TBPTT
                # --------------------------------------------------
                if use_global_madp:
                    buffer_true.append(z_chunk.detach().reshape(-1))
                    buffer_pred.append(Y_pred.detach().reshape(-1))
        
                    z_true_buf = torch.cat(buffer_true, dim=0)
                    z_pred_buf = torch.cat(buffer_pred, dim=0)
        
                    if z_true_buf.numel() > buffer_max_size:
                        z_true_buf = z_true_buf[-buffer_max_size:]
                        z_pred_buf = z_pred_buf[-buffer_max_size:]
        
                        buffer_true = [z_true_buf.detach()]
                        buffer_pred = [z_pred_buf.detach()]
        
                # --------------------------------------------------
                # Critical for TBPTT: truncate graph across chunks
                # --------------------------------------------------
                if hidden is not None:
                    if isinstance(hidden, tuple):  # LSTM
                        hidden = tuple(h.detach() for h in hidden)
                    else:
                        hidden = hidden.detach()
        
            epoch_loss /= num_chunks
            losses.append(epoch_loss)

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------
            if epoch % validate_every == 0:
                model.eval()
                with torch.no_grad():
                    Y_val = predict_tbptt(model, XVal, chunk_size)
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
        # Restore best model if early stopping was used
        # --------------------------------------------------
        if use_early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # --------------------------------------------------
        # Save model and losses
        # --------------------------------------------------
        flnm = os.path.join(output_dir, f"{type(model).__name__}_Run{run + 1}")

        np.savetxt(f"{flnm}_losses.txt", losses)
        torch.save(model.state_dict(), f"{flnm}_checkpoint.pth")

        joblib.dump(
            {
                "scaler": data["scaler"],
                "feature_names": data["feature_names"],
                "use_sequences": True,
                "seq_len": None,
                "chunk_size": chunk_size,
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
                Y_pred = predict_tbptt(model, X, chunk_size)
                Y_pred = Y_pred.detach().cpu().numpy().reshape(-1)
                real = Z.detach().cpu().numpy().reshape(-1)

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

                        df = pd.DataFrame({
                            "year": mslpTest_reset.iloc[:, 0],
                            "month": mslpTest_reset.iloc[:, 1],
                            "day": mslpTest_reset.iloc[:, 2],
                            "hour": hourTest,
                            "minutes": mslpTest_reset.iloc[:, 4],
                            "seconds": mslpTest_reset.iloc[:, 5],
                            "num_date": mslpTest_reset.iloc[:, 6],
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
# TBPTT prediction helper
# ======================================================
def predict_tbptt(model, X, chunk_size):
    model.eval()

    T = X.size(1)
    hidden = None
    outputs = []

    with torch.no_grad():
        for t in range(0, T, chunk_size):
            X_chunk = X[:, t:t + chunk_size, :]

            Y_chunk, hidden = model(X_chunk, hidden)
            outputs.append(Y_chunk)

            if hidden is not None:
                if isinstance(hidden, tuple):  # LSTM
                    hidden = tuple(h.detach() for h in hidden)
                else:  # RNN
                    hidden = hidden.detach()

    return torch.cat(outputs, dim=1)

# ======================================================
# Utilities
# ======================================================
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

