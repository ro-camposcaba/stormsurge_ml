import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.signal import detrend
import joblib
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

base_path = "C:/Users/Rodrigo/Desktop/Rodrigo/02 - Research fellow/01 - Storm surge ML downscaling/Experiments_v6_v2/surge-ml_v2 - sliding-window"

target_path = f"{base_path}/data/predictand/VeneziaPS"
data_path = f"{base_path}/data/predictors/VeneziaPS"
model_path = f"{base_path}/experiments/VeneziaPS/MLP-MSE"

num_runs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_file(path):
    return pd.read_csv(path, header=None, sep=r"\s+")

cmems = load_file(f"{data_path}/ssh.txt")
tide  = load_file(f"{data_path}/tide.txt")
mslp  = load_file(f"{data_path}/mslp.txt")
u10   = load_file(f"{data_path}/stressX.txt")
v10   = load_file(f"{data_path}/stressY.txt")

# --------------------------------------------------
# FILTER YEARS
# --------------------------------------------------

anio = np.array([1987, 2020])
offset = 1

# Predictand
shy = load_file(f"{target_path}/VeneziaPS.txt")  
shy_filt = shy[(shy.iloc[:, 0] >= anio[0]) & (shy.iloc[:, 0] <= anio[-1])]

# Predictors
idini = lambda df: df.index[df.iloc[:, 0] == anio[0]].tolist()
idfin = lambda df: df.index[df.iloc[:, 0] == anio[-1]].tolist()

cmems_filt = cmems.iloc[idini(cmems)[offset]:idfin(cmems)[-1] + 1]
mslp_filt  = mslp.iloc[idini(mslp)[offset]:idfin(mslp)[-1] + 1]
tide_filt  = tide.iloc[idini(tide)[offset]:idfin(tide)[-1] + 1]
u10_filt   = u10.iloc[idini(mslp)[offset]:idfin(mslp)[-1] + 1]
v10_filt   = v10.iloc[idini(mslp)[offset]:idfin(mslp)[-1] + 1]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

chunksTrain = [[1987, 1992], [1997, 2018]]

def extract_chunks(df, chunks):
    return pd.concat([
        df.iloc[
            df.index[df.iloc[:,0] == ch[0]].tolist()[0] :
            df.index[df.iloc[:,0] == ch[1]].tolist()[-1] + 1
        ]
        for ch in chunks
    ])

shyTrain = extract_chunks(shy_filt, chunksTrain)
cmemsTrain = extract_chunks(cmems_filt, chunksTrain)
mslpTrain = extract_chunks(mslp_filt, chunksTrain)
tideTrain = extract_chunks(tide_filt, chunksTrain)
u10Train = extract_chunks(u10_filt, chunksTrain)
v10Train = extract_chunks(v10_filt, chunksTrain)

# TEST YEARS
test_years = [1995, 1996, 2020]

idx_test = [i for y in test_years for i in shy_filt[shy_filt.iloc[:,0]==y].index]

shyTest = shy_filt.loc[idx_test][:-1]

cmemsTest = cmems_filt.loc[idx_test][1:]
mslpTest  = mslp_filt.loc[idx_test][1:]
tideTest  = tide_filt.loc[idx_test][1:]
u10Test   = u10_filt.loc[idx_test][1:]
v10Test   = v10_filt.loc[idx_test][1:]

# --------------------------------------------------
# BUILD FEATURES (TEST)
# --------------------------------------------------

x = cmemsTest.iloc[:, 7:14].to_numpy()
y = tideTest.iloc[:, 7].to_numpy()
w = mslpTest.iloc[:, 7:14].to_numpy()
u = u10Test.iloc[:, 7:14].to_numpy()
v = v10Test.iloc[:, 7:14].to_numpy()

x_dtn = detrend(x, axis=0)

cols = {}

for i in range(7):
    cols[f"x{i+1}_dtn"] = x_dtn[:, i]

cols["y"] = y

for i in range(7):
    cols[f"w{i+1}"] = w[:, i]
    cols[f"u{i+1}"] = u[:, i]
    cols[f"v{i+1}"] = v[:, i]

XTest = pd.DataFrame(cols)

ZTest_tensor = torch.tensor(shyTest.iloc[:,7].values, dtype=torch.float32)

# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------

preproc = joblib.load(f"{model_path}/mlpModel_Run1_preprocessing.pkl")

scaler = preproc["scaler"]
feature_names = preproc["feature_names"]

XTest = XTest[feature_names]

X_scaled = scaler.transform(XTest)
X_scaled = np.clip(X_scaled, -8, 8)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

class mlpModel(torch.nn.Module):
    def __init__(self, device, inputSize, outputSize, hiddenLayerSizes=[120, 120]):
        super(mlpModel, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.inputLr = torch.nn.Linear(self.inputSize, hiddenLayerSizes[0], device=device)

        self.hiddenLayers = torch.nn.ModuleList()
        for ihl in range(len(hiddenLayerSizes) - 1):
            hl = torch.nn.Linear(hiddenLayerSizes[ihl], hiddenLayerSizes[ihl + 1], device=device)
            self.hiddenLayers.append(hl)

        self.outputLr = torch.nn.Linear(hiddenLayerSizes[-1], self.outputSize, device=device)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.inputLr(x)
        out = self.relu(out)
        for hlr in self.hiddenLayers:
            out = hlr(out)
            out = self.relu(out)
        out = self.outputLr(out)
        return out

# --------------------------------------------------
# PERMUTATION FUNCTION
# --------------------------------------------------

def permutation_importance(model, X, y, n_repeats=5, mode="global", percentile=99):
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    baseline = model(X).detach().cpu().numpy().flatten()

    if mode == "peaks":
        mask = y_np >= np.percentile(y_np, percentile)
        baseline = baseline[mask]
        y_np = y_np[mask]

    base_err = np.mean(np.abs(y_np - baseline))
    importances = []

    for i in range(X_np.shape[1]):
        errs = []

        for _ in range(n_repeats):
            X_perm = X_np.copy()
            np.random.shuffle(X_perm[:, i])

            preds = model(torch.tensor(X_perm, dtype=torch.float32).to(device))
            preds = preds.detach().cpu().numpy().flatten()

            if mode == "peaks":
                preds = preds[mask]

            errs.append(np.mean(np.abs(y_np - preds)))

        importances.append(np.mean(errs) - base_err)

    return np.array(importances)

# --------------------------------------------------
# RUN LOOP
# --------------------------------------------------

for run in range(1, num_runs + 1):

    print(f"Run {run}")

    model = mlpModel(
        device=device,
        inputSize=X_tensor.shape[1],
        outputSize=1,
        hiddenLayerSizes=[120, 120]
        )

    checkpoint = torch.load(
        f"{model_path}/mlpModel_Run{run}_checkpoint.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    imp_global = permutation_importance(
        model, X_tensor, ZTest_tensor.to(device),
        n_repeats=10, mode="global"
    )

    imp_peaks = permutation_importance(
        model, X_tensor, ZTest_tensor.to(device),
        n_repeats=10, mode="peaks"
    )

    np.savetxt(f"Run{run}_permutation_global.txt", imp_global)
    np.savetxt(f"Run{run}_permutation_peaks.txt", imp_peaks)

print("Done.")