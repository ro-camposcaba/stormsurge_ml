# =========================================================
# inference_mlp.py
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
# MLP MODEL
# =========================================================
class mlpModel(torch.nn.Module):

    def __init__(
        self,
        device,
        inputSize,
        outputSize,
        hiddenLayerSizes=[120, 120]
    ):

        super(mlpModel, self).__init__()

        self.inputSize = inputSize
        self.outputSize = outputSize

        self.inputLr = torch.nn.Linear(
            self.inputSize,
            hiddenLayerSizes[0],
            device=device
        )

        self.hiddenLayers = torch.nn.ModuleList()

        for ihl in range(len(hiddenLayerSizes) - 1):

            hl = torch.nn.Linear(
                hiddenLayerSizes[ihl],
                hiddenLayerSizes[ihl + 1],
                device=device
            )

            self.hiddenLayers.append(hl)

        self.outputLr = torch.nn.Linear(
            hiddenLayerSizes[-1],
            self.outputSize,
            device=device
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        out = self.inputLr(x)

        out = self.relu(out)

        for hlr in self.hiddenLayers:

            out = hlr(out)

            out = self.relu(out)

        out = self.outputLr(out)

        return out


# =========================================================
# CONFIGURATION
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Results april 2026\Trieste\MLP-MADc"

TEST_DATA_DIR = r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\aligned_test_sets_Trieste"

OUTPUT_DIR = os.path.join(MODEL_DIR, r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Results april 2026\inference results\Trieste\MLP-MADc")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    x = [detrend(cmems.iloc[:, i]) for i in range(7, 14)]

    y = tide.iloc[:, 7]

    w = [mslp.iloc[:, i] for i in range(7, 14)]

    u = [u10.iloc[:, i] for i in range(7, 14)]

    v = [v10.iloc[:, i] for i in range(7, 14)]

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
    glob.glob(os.path.join(MODEL_DIR, "*_checkpoint.pth"))
)

print(f"\nFound {len(checkpoint_files)} checkpoints.")


# =========================================================
# INFERENCE LOOP
# =========================================================
for ckpt in checkpoint_files:

    print(f"\nProcessing: {os.path.basename(ckpt)}")

    base_name = ckpt.replace("_checkpoint.pth", "")

    preprocessing_file = f"{base_name}_preprocessing.pkl"

    preprocessing = joblib.load(preprocessing_file)

    scaler = preprocessing["scaler"]

    feature_names = preprocessing["feature_names"]

    XTestScaled = pd.DataFrame(
        scaler.transform(XTest),
        columns=feature_names
    )

    XTensor = torch.tensor(
        XTestScaled.to_numpy(),
        dtype=torch.float32
    ).to(DEVICE)

    model = mlpModel(
        device=DEVICE,
        inputSize=XTensor.shape[1],
        outputSize=1,
        hiddenLayerSizes=[120, 120]
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE, weights_only = True)
    )

    model.eval()

    with torch.no_grad():

        pred = model(XTensor)

        pred = pred.squeeze(-1).cpu().numpy()

    # =====================================================
    # SAVE RESULTS (ORIGINAL FORMAT)
    # =====================================================
    try:
    
        mslpTest_reset_index = mslpTest.reset_index(drop=True)
    
        # -------------------------------------------------
        # Use MSLP hours consistently
        # -------------------------------------------------
        hourTest = mslpTest_reset_index.iloc[:, 3].values
    
        df = pd.DataFrame({
    
            "year": mslpTest_reset_index.iloc[:, 0],
    
            "month": mslpTest_reset_index.iloc[:, 1],
    
            "day": mslpTest_reset_index.iloc[:, 2],
    
            "hour": hourTest,
    
            "minutes": mslpTest_reset_index.iloc[:, 4],
    
            "seconds": mslpTest_reset_index.iloc[:, 5],
    
            "num_date": mslpTest_reset_index.iloc[:, 6],
    
            "pred": pred,
    
            "obs": zTest,
        })
    
        run_name = os.path.basename(base_name)
    
        np.savetxt(
    
            os.path.join(
                OUTPUT_DIR,
                f"{run_name}_testresults.txt"
            ),
    
            df.values,
    
            fmt="%.6f",
    
            delimiter=" "
        )
    
        print(f"Saved inference for {run_name}")
    
    except Exception as e:
    
        print(f"Warning: could not save results due to: {e}")