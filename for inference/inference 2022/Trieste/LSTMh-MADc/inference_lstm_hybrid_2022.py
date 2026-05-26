# =========================================================
# inference_lstmHybrid_2022.py
# =========================================================

import os
import joblib
import torch
import numpy as np
import pandas as pd

from scipy.signal import detrend
from scipy.signal import firwin
from scipy.signal import filtfilt

import torch.nn as nn


# =========================================================
# LSTM HYBRID MODEL
# =========================================================
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

        out_rnn, hidden = self.lstm(
            x,
            hidden
        )

        out_lin = self.relu(
            self.lin(x)
        )

        out = torch.cat(
            (out_lin, out_rnn),
            dim=2
        )

        out = self.output(out)

        return out, hidden


# =========================================================
# TBPTT PREDICTION HELPER
# =========================================================
def predict_tbptt(model, X, chunk_size):

    model.eval()

    T = X.size(1)

    hidden = None

    outputs = []

    with torch.no_grad():

        for t in range(0, T, chunk_size):

            X_chunk = X[:, t:t + chunk_size, :]

            Y_chunk, hidden = model(
                X_chunk,
                hidden
            )

            outputs.append(Y_chunk)

            if hidden is not None:

                if isinstance(hidden, tuple):

                    hidden = tuple(
                        h.detach()
                        for h in hidden
                    )

                else:

                    hidden = hidden.detach()

    return torch.cat(outputs, dim=1)


# =========================================================
# CONFIGURATION
# =========================================================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_DIR = (
    r"C:\Users\Rodrigo\Desktop\Rodrigo"
    r"\02 - Research fellow"
    r"\01 - Storm surge ML downscaling"
    r"\Results april 2026\Trieste\LSTMh-MADc"
)

PREDICTOR_DIR = (
    r"C:\Users\Rodrigo\Desktop\Rodrigo"
    r"\02 - Research fellow"
    r"\01 - Storm surge ML downscaling"
    r"\surge_ml\tests\inference 2022\Trieste"
    r"\aligned_predictors_Trieste_2022"
)

OUTPUT_DIR = (
    r"C:\Users\Rodrigo\Desktop\Rodrigo"
    r"\02 - Research fellow"
    r"\01 - Storm surge ML downscaling"
    r"\surge_ml\tests\inference 2022\Trieste"
    r"\LSTMh-MADc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

raw_output_file = os.path.join(

    OUTPUT_DIR,

    "lstmHybridModel_Run9_2022_inference_RAW.txt"
)

lp_output_file = os.path.join(

    OUTPUT_DIR,

    "lstmHybridModel_Run9_2022_inference_LP13h.txt"
)

HIDDEN_LAYER_SIZE = 60


# =========================================================
# CHECKPOINT / PREPROCESSING FILES
# =========================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "lstmHybridModel_Run9_checkpoint.pth"
)

preprocessing_path = os.path.join(
    MODEL_DIR,
    "lstmHybridModel_Run9_preprocessing.pkl"
)


# =========================================================
# LOAD PREDICTOR DATA
# =========================================================

cmemsTest = pd.read_csv(
    os.path.join(
        PREDICTOR_DIR,
        "cmems_2022.csv"
    ),
    header=None
)

tideTest = pd.read_csv(
    os.path.join(
        PREDICTOR_DIR,
        "tide_2022.csv"
    ),
    header=None
)

mslpTest = pd.read_csv(
    os.path.join(
        PREDICTOR_DIR,
        "mslp_2022.csv"
    ),
    header=None
)

u10Test = pd.read_csv(
    os.path.join(
        PREDICTOR_DIR,
        "u10_2022.csv"
    ),
    header=None
)

v10Test = pd.read_csv(
    os.path.join(
        PREDICTOR_DIR,
        "v10_2022.csv"
    ),
    header=None
)


# =========================================================
# BUILD FEATURES
# =========================================================

def build_X(
    cmems,
    tide,
    mslp,
    u10,
    v10
):

    x = [
        detrend(cmems.iloc[:, i])
        for i in range(7, 14)
    ]

    y = tide.iloc[:, 7]

    w = [
        mslp.iloc[:, i]
        for i in range(7, 14)
    ]

    u = [
        u10.iloc[:, i]
        for i in range(7, 14)
    ]

    v = [
        v10.iloc[:, i]
        for i in range(7, 14)
    ]

    return pd.DataFrame(

        {f"x{i+1}_dtn": x[i]
         for i in range(7)}

        | {"y": y}

        | {f"w{i+1}": w[i]
           for i in range(7)}

        | {f"u{i+1}": u[i]
           for i in range(7)}

        | {f"v{i+1}": v[i]
           for i in range(7)}
    )


XTest = build_X(
    cmemsTest,
    tideTest,
    mslpTest,
    u10Test,
    v10Test
)


# =========================================================
# LOAD PREPROCESSING
# =========================================================

preprocessing = joblib.load(
    preprocessing_path
)

scaler = preprocessing["scaler"]

feature_names = preprocessing[
    "feature_names"
]

chunk_size = preprocessing.get(
    "chunk_size",
    1024
)


# =========================================================
# ENFORCE TRAINING FEATURE ORDER
# =========================================================

XTest = XTest[
    feature_names
]


# =========================================================
# SCALE FEATURES
# =========================================================

XTestScaled = scaler.transform(XTest)

print("\nScaled feature statistics BEFORE clipping:")

print(
    f"Mean: {np.mean(XTestScaled):.3f}"
)

print(
    f"Min : {np.min(XTestScaled):.3f}"
)

print(
    f"Max : {np.max(XTestScaled):.3f}"
)

print(
    f"Max abs value: "
    f"{np.max(np.abs(XTestScaled)):.3f}"
)

# ======================================================
# OPTIONAL ROBUSTNESS CLIPPING
# ======================================================

XTestScaled = np.clip(
    XTestScaled,
    -8,
    8
)

print("\nScaled feature statistics AFTER clipping:")

print(
    f"Mean: {np.mean(XTestScaled):.3f}"
)

print(
    f"Min : {np.min(XTestScaled):.3f}"
)

print(
    f"Max : {np.max(XTestScaled):.3f}"
)

# =========================================================
# TENSOR CONVERSION
# =========================================================

XTensor = torch.tensor(

    XTestScaled,

    dtype=torch.float32

).unsqueeze(0).to(DEVICE)


# =========================================================
# LOAD MODEL
# =========================================================

model = lstmHybridModel(
    input_size=XTensor.shape[2],
    output_size=1,
    hidden_layer_size=HIDDEN_LAYER_SIZE
).to(DEVICE)

model.load_state_dict(

    torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# =========================================================
# TBPTT INFERENCE
# =========================================================

with torch.no_grad():

    pred = predict_tbptt(
        model,
        XTensor,
        chunk_size
    )

    pred = (
        pred.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


# =========================================================
# DIAGNOSTICS
# =========================================================

print("\nPrediction statistics:")

print(
    f"Mean: {np.mean(pred):.6f}"
)

print(
    f"Max : {np.max(pred):.6f}"
)

print(
    f"Min : {np.min(pred):.6f}"
)


# ======================================================
# 13-HOUR LOW-PASS FILTER
# ======================================================

dt = 3600

fs = 1 / dt

cop = 13 * 3600

cof = 1 / cop

fir = firwin(
    numtaps=9,
    cutoff=cof,
    fs=fs,
    pass_zero=True
)

pred_lp = np.full_like(
    pred,
    np.nan
)

valid_mask = ~np.isnan(pred)

pred_lp[valid_mask] = filtfilt(
    fir,
    1,
    pred[valid_mask]
)

print("\nFiltered prediction statistics:")

print(
    f"Mean: {np.nanmean(pred_lp):.6f}"
)

print(
    f"Max : {np.nanmax(pred_lp):.6f}"
)

print(
    f"Min : {np.nanmin(pred_lp):.6f}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

try:

    mslpTest_reset_index = (
        mslpTest.reset_index(drop=True)
    )

    # ---------------------------------------------
    # RAW OUTPUT
    # ---------------------------------------------
    df_raw = pd.DataFrame({

        "year":
            mslpTest_reset_index.iloc[:, 0],

        "month":
            mslpTest_reset_index.iloc[:, 1],

        "day":
            mslpTest_reset_index.iloc[:, 2],

        "hour":
            mslpTest_reset_index.iloc[:, 3],

        "minutes":
            mslpTest_reset_index.iloc[:, 4],

        "seconds":
            mslpTest_reset_index.iloc[:, 5],

        "num_date":
            mslpTest_reset_index.iloc[:, 6],

        "pred":
            pred,
    })

    np.savetxt(

        raw_output_file,

        df_raw.values,

        fmt="%.6f",

        delimiter=" "
    )

    print(
        f"\nRaw inference saved to:\n"
        f"{raw_output_file}"
    )

    # ---------------------------------------------
    # FILTERED OUTPUT
    # ---------------------------------------------
    df_lp = pd.DataFrame({

        "year":
            mslpTest_reset_index.iloc[:, 0],

        "month":
            mslpTest_reset_index.iloc[:, 1],

        "day":
            mslpTest_reset_index.iloc[:, 2],

        "hour":
            mslpTest_reset_index.iloc[:, 3],

        "minutes":
            mslpTest_reset_index.iloc[:, 4],

        "seconds":
            mslpTest_reset_index.iloc[:, 5],

        "num_date":
            mslpTest_reset_index.iloc[:, 6],

        "pred":
            pred_lp,
    })

    np.savetxt(

        lp_output_file,

        df_lp.values,

        fmt="%.6f",

        delimiter=" "
    )

    print(
        f"\nFiltered inference saved to:\n"
        f"{lp_output_file}"
    )

except Exception as e:

    print(
        f"Warning: could not save "
        f"results due to: {e}"
    )