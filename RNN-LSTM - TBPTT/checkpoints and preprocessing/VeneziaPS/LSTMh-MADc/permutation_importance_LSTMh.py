import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.signal import detrend
import joblib

# ==================================================
# CONFIG
# ==================================================

base_path = "C:/Users/Rodrigo/Desktop/Rodrigo/02 - Research fellow/01 - Storm surge ML downscaling/surge_ml"

target_path = f"{base_path}/data/predictand/VeneziaPS"
data_path   = f"{base_path}/data/predictors/VeneziaPS"

model_path  = f"{base_path}/tests/VeneziaPS/LSTMh-MADc_globalMADp_buffer100mil"

num_runs = 40

hidden_layer_size = 60
output_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# LOAD DATA
# ==================================================

def load_file(path):
    return pd.read_csv(path, header=None, sep=r"\s+")

cmems = load_file(f"{data_path}/ssh.txt")
tide  = load_file(f"{data_path}/tide.txt")
mslp  = load_file(f"{data_path}/mslp.txt")
u10   = load_file(f"{data_path}/stressX.txt")
v10   = load_file(f"{data_path}/stressY.txt")

# ==================================================
# FILTER YEARS
# ==================================================

anio = np.array([1987, 2020])

offset = 1

shy = load_file(f"{target_path}/VeneziaPS.txt")

shy_filt = shy[
    (shy.iloc[:, 0] >= anio[0]) &
    (shy.iloc[:, 0] <= anio[-1])
]

idini = lambda df: df.index[df.iloc[:, 0] == anio[0]].tolist()
idfin = lambda df: df.index[df.iloc[:, 0] == anio[-1]].tolist()

cmems_filt = cmems.iloc[
    idini(cmems)[offset]:
    idfin(cmems)[-1] + 1
]

mslp_filt = mslp.iloc[
    idini(mslp)[offset]:
    idfin(mslp)[-1] + 1
]

tide_filt = tide.iloc[
    idini(tide)[offset]:
    idfin(tide)[-1] + 1
]

u10_filt = u10.iloc[
    idini(mslp)[offset]:
    idfin(mslp)[-1] + 1
]

v10_filt = v10.iloc[
    idini(mslp)[offset]:
    idfin(mslp)[-1] + 1
]

# ==================================================
# TEST YEARS
# ==================================================

test_years = [1995, 1996, 2020]

idx_test = [
    i
    for y in test_years
    for i in shy_filt[shy_filt.iloc[:, 0] == y].index
]

shyTest = shy_filt.loc[idx_test][:-1]

cmemsTest = cmems_filt.loc[idx_test][1:]
mslpTest  = mslp_filt.loc[idx_test][1:]
tideTest  = tide_filt.loc[idx_test][1:]
u10Test   = u10_filt.loc[idx_test][1:]
v10Test   = v10_filt.loc[idx_test][1:]

# ==================================================
# BUILD FEATURES
# ==================================================

x = cmemsTest.iloc[:, 7:14].to_numpy()

y = tideTest.iloc[:, 7].to_numpy()

w = mslpTest.iloc[:, 7:14].to_numpy()

u = u10Test.iloc[:, 7:14].to_numpy()

v = v10Test.iloc[:, 7:14].to_numpy()

# ------------------------------------------
# DETREND CMEMS
# ------------------------------------------

x_dtn = detrend(x, axis=0)

# ------------------------------------------
# FEATURE TABLE
# ------------------------------------------

cols = {}

for i in range(7):
    cols[f"x{i+1}_dtn"] = x_dtn[:, i]

cols["y"] = y

for i in range(7):

    cols[f"w{i+1}"] = w[:, i]
    cols[f"u{i+1}"] = u[:, i]
    cols[f"v{i+1}"] = v[:, i]

XTest = pd.DataFrame(cols)

# ==================================================
# TARGET
# ==================================================

ZTest = shyTest.iloc[:, 7].values.astype(np.float32)

# ==================================================
# LOAD PREPROCESSING
# ==================================================

preproc = joblib.load(
    f"{model_path}/lstmHybridModel_Run1_preprocessing.pkl"
)

scaler = preproc["scaler"]

feature_names = preproc["feature_names"]

XTest = XTest[feature_names]

# ==================================================
# SCALE INPUTS
# ==================================================

X_scaled = scaler.transform(XTest)

X_scaled = np.clip(X_scaled, -8, 8)

# ==================================================
# FULL SEQUENCE TENSORS
# ==================================================
#
# Shape:
# X -> (1, T, F)
# y -> (1, T)
#
# Entire time series provided to the LSTM
# as a single continuous sequence
#
# ==================================================

X_scaled = X_scaled[np.newaxis, :, :]

ZTest = ZTest[np.newaxis, :]

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
).to(device)

y_tensor = torch.tensor(
    ZTest,
    dtype=torch.float32
).to(device)

# ==================================================
# MODEL
# ==================================================

class lstmHybridModel(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layer_size
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=1,
            batch_first=True
        )

        self.lin = nn.Linear(
            input_size,
            hidden_layer_size
        )

        self.relu = nn.ReLU()

        self.output = nn.Linear(
            2 * hidden_layer_size,
            output_size
        )

    def forward(self, x, hidden=None):

        # ------------------------------------------
        # LSTM BRANCH
        # ------------------------------------------

        out_rnn, hidden = self.lstm(x, hidden)

        # ------------------------------------------
        # LINEAR BRANCH
        # ------------------------------------------

        out_lin = self.relu(self.lin(x))

        # ------------------------------------------
        # CONCATENATION
        # ------------------------------------------

        out = torch.cat(
            (out_lin, out_rnn),
            dim=2
        )

        # ------------------------------------------
        # OUTPUT
        # ------------------------------------------

        out = self.output(out)

        return out, hidden

# ==================================================
# PERMUTATION IMPORTANCE
# ==================================================

def permutation_importance(
    model,
    X,
    y,
    n_repeats=10,
    mode="global",
    percentile=99
):

    model.eval()

    X_np = X.detach().cpu().numpy()

    y_np = y.detach().cpu().numpy()

    # ----------------------------------------------
    # BASELINE PREDICTION
    # ----------------------------------------------

    with torch.no_grad():

        baseline_preds, _ = model(X)

    baseline_preds = (
        baseline_preds
        .detach()
        .cpu()
        .numpy()
        .squeeze(-1)
    )

    # baseline_preds -> (1, T)

    # ----------------------------------------------
    # PEAK MASK
    # ----------------------------------------------

    if mode == "peaks":

        threshold = np.percentile(
            y_np.flatten(),
            percentile
        )

        mask = y_np >= threshold

        y_eval = y_np[mask]

        baseline_eval = baseline_preds[mask]

    else:

        y_eval = y_np.flatten()

        baseline_eval = baseline_preds.flatten()

    # ----------------------------------------------
    # BASE ERROR
    # ----------------------------------------------

    base_err = np.mean(
        np.abs(y_eval - baseline_eval)
    )

    # ----------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------

    n_features = X_np.shape[2]

    importances = []

    for feature_idx in range(n_features):

        errs = []

        for repeat in range(n_repeats):

            X_perm = X_np.copy()

            # --------------------------------------
            # TEMPORAL SHUFFLING
            # --------------------------------------
            #
            # Shuffle only ONE predictor
            # along the temporal dimension
            #
            # Shape:
            # X_perm[0, :, feature_idx]
            #
            # --------------------------------------

            shuffled = X_perm[0, :, feature_idx].copy()

            np.random.shuffle(shuffled)

            X_perm[0, :, feature_idx] = shuffled

            # --------------------------------------
            # TENSOR
            # --------------------------------------

            X_perm_tensor = torch.tensor(
                X_perm,
                dtype=torch.float32
            ).to(device)

            # --------------------------------------
            # FORWARD PASS
            # --------------------------------------

            with torch.no_grad():

                preds_perm, _ = model(X_perm_tensor)

            preds_perm = (
                preds_perm
                .detach()
                .cpu()
                .numpy()
                .squeeze(-1)
            )

            # --------------------------------------
            # EVALUATION
            # --------------------------------------

            if mode == "peaks":

                preds_eval = preds_perm[mask]

            else:

                preds_eval = preds_perm.flatten()

            err = np.mean(
                np.abs(y_eval - preds_eval)
            )

            errs.append(err)

        importance = np.mean(errs) - base_err

        importances.append(importance)

    return np.array(importances)

# ==================================================
# RUN LOOP
# ==================================================

input_size = X_tensor.shape[2]

for run in range(1, num_runs + 1):

    print(f"\nRun {run}")

    # ----------------------------------------------
    # MODEL
    # ----------------------------------------------

    model = lstmHybridModel(
        input_size=input_size,
        output_size=output_size,
        hidden_layer_size=hidden_layer_size
    ).to(device)

    # ----------------------------------------------
    # LOAD CHECKPOINT
    # ----------------------------------------------

    checkpoint = torch.load(
        f"{model_path}/lstmHybridModel_Run{run}_checkpoint.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.eval()

    # ----------------------------------------------
    # GLOBAL IMPORTANCE
    # ----------------------------------------------

    print("Computing global importance...")

    imp_global = permutation_importance(
        model=model,
        X=X_tensor,
        y=y_tensor,
        n_repeats=10,
        mode="global"
    )

    # ----------------------------------------------
    # PEAK IMPORTANCE
    # ----------------------------------------------

    print("Computing peak importance...")

    imp_peaks = permutation_importance(
        model=model,
        X=X_tensor,
        y=y_tensor,
        n_repeats=10,
        mode="peaks",
        percentile=99
    )

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    np.savetxt(
        f"Run{run}_permutation_global.txt",
        imp_global
    )

    np.savetxt(
        f"Run{run}_permutation_peaks.txt",
        imp_peaks
    )

print("\nDone.")