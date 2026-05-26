# =========================================================
# inference_rnn_simple.py
# =========================================================

import os
import glob
import joblib
import torch
import numpy as np
import pandas as pd

from scipy.signal import detrend

import torch.nn as nn


# =========================================================
# RNN SIMPLE MODEL
# =========================================================
class rnnSimpleModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_layer_sizes,
        output_size,
        num_layers=2
    ):

        super().__init__()

        hidden_size = hidden_layer_sizes[0]

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x, hidden=None):

        out, hidden = self.rnn(
            x,
            hidden
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

                hidden = hidden.detach()

    return torch.cat(outputs, dim=1)


# =========================================================
# CONFIGURATION
# =========================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Results april 2026\Trieste\RNN-MSE"

TEST_DATA_DIR = r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\aligned_test_sets_Trieste"

OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Results april 2026\inference results\Trieste\RNN-MSE"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_LAYER_SIZES = [120]
NUM_LAYERS = 2


# =========================================================
# LOAD TEST DATA
# =========================================================
cmemsTest = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "cmems_test.csv"),
    header=None
)

tideTest = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "tide_test.csv"),
    header=None
)

mslpTest = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "mslp_test.csv"),
    header=None
)

u10Test = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "u10_test.csv"),
    header=None
)

v10Test = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "v10_test.csv"),
    header=None
)

targetTest = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "target_test.csv"),
    header=None
)


# =========================================================
# BUILD FEATURES
# =========================================================
def build_X(cmems, tide, mslp, u10, v10):

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

        {f"x{i+1}_dtn": x[i] for i in range(7)}

        | {"y": y}

        | {f"w{i+1}": w[i] for i in range(7)}

        | {f"u{i+1}": u[i] for i in range(7)}

        | {f"v{i+1}": v[i] for i in range(7)}
    )


XTest = build_X(
    cmemsTest,
    tideTest,
    mslpTest,
    u10Test,
    v10Test
)

zTest = targetTest.iloc[:, 7].values


# =========================================================
# FIND CHECKPOINTS
# =========================================================
checkpoint_files = sorted(
    glob.glob(
        os.path.join(
            MODEL_DIR,
            "*_checkpoint.pth"
        )
    )
)

print(f"\nFound {len(checkpoint_files)} checkpoints.")


# =========================================================
# INFERENCE LOOP
# =========================================================
for ckpt in checkpoint_files:

    print(
        f"\nProcessing: "
        f"{os.path.basename(ckpt)}"
    )

    base_name = ckpt.replace(
        "_checkpoint.pth",
        ""
    )

    preprocessing_file = (
        f"{base_name}_preprocessing.pkl"
    )

    preprocessing = joblib.load(
        preprocessing_file
    )

    scaler = preprocessing["scaler"]

    feature_names = preprocessing[
        "feature_names"
    ]

    chunk_size = preprocessing.get(
        "chunk_size",
        1024
    )

    # =====================================================
    # SCALE FEATURES
    # =====================================================
    XTestScaled = pd.DataFrame(
        scaler.transform(XTest),
        columns=feature_names
    )

    # =====================================================
    # IMPORTANT:
    # SHAPE MUST BE [1, T, FEATURES]
    # =====================================================
    XTensor = torch.tensor(
        XTestScaled.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # =====================================================
    # LOAD MODEL
    # =====================================================
    model = rnnSimpleModel(
        input_size=XTensor.shape[2],
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        output_size=1,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=DEVICE, weights_only = True
        )
    )

    model.eval()

    # =====================================================
    # TBPTT INFERENCE
    # =====================================================
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

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    try:

        mslpTest_reset_index = (
            mslpTest.reset_index(drop=True)
        )

        hourTest = (
            mslpTest_reset_index.iloc[:, 3]
            .values
        )

        df = pd.DataFrame({

            "year":
                mslpTest_reset_index.iloc[:, 0],

            "month":
                mslpTest_reset_index.iloc[:, 1],

            "day":
                mslpTest_reset_index.iloc[:, 2],

            "hour":
                hourTest,

            "minutes":
                mslpTest_reset_index.iloc[:, 4],

            "seconds":
                mslpTest_reset_index.iloc[:, 5],

            "num_date":
                mslpTest_reset_index.iloc[:, 6],

            "pred":
                pred,

            "obs":
                zTest,
        })

        run_name = os.path.basename(
            base_name
        )

        np.savetxt(

            os.path.join(
                OUTPUT_DIR,
                f"{run_name}_testresults.txt"
            ),

            df.values,

            fmt="%.6f",

            delimiter=" "
        )

        print(
            f"Saved inference for {run_name}"
        )

    except Exception as e:

        print(
            f"Warning: could not save "
            f"results due to: {e}"
        )

print("\nInference completed.")