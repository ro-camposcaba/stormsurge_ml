import torch
import pandas as pd
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler


def load_data(config, device):

    # ==================================================
    # Load data
    # ==================================================
    paths = config["data_paths"]

    shy = pd.read_csv(
        paths["predictand"],
        header=None,
        sep=r"\s+"
    )

    cmems = pd.read_csv(
        paths["cmems"],
        header=None,
        sep=r"\s+"
    )

    tide = pd.read_csv(
        paths["tide"],
        header=None,
        sep=r"\s+"
    )

    mslp = pd.read_csv(
        paths["mslp"],
        header=None,
        sep=r"\s+"
    )

    u10 = pd.read_csv(
        paths["u10"],
        header=None,
        sep=r"\s+"
    )

    v10 = pd.read_csv(
        paths["v10"],
        header=None,
        sep=r"\s+"
    )

    # ==================================================
    # Time filtering
    # ==================================================
    anio = [1987, 2020]
    
    offset = int(config.get("time_offset", 0))

    shy_filt = shy[
        (shy.iloc[:, 0] >= anio[0]) &
        (shy.iloc[:, 0] <= anio[-1])
    ]

    idini = lambda df, yr: (
        df.index[
            df.iloc[:, 0] == yr[0]
        ].tolist()[offset]
    )

    idfin = lambda df, yr: (
        df.index[
            df.iloc[:, 0] == yr[1]
        ].tolist()[-1] + 1
    )

    cmems_filt = cmems.iloc[
        idini(cmems, anio):
        idfin(cmems, anio)
    ]

    mslp_filt = mslp.iloc[
        idini(mslp, anio):
        idfin(mslp, anio)
    ]

    tide_filt = tide.iloc[
        idini(tide, anio):
        idfin(tide, anio)
    ]

    u10_filt = u10.iloc[
        idini(mslp, anio):
        idfin(mslp, anio)
    ]

    v10_filt = v10.iloc[
        idini(mslp, anio):
        idfin(mslp, anio)
    ]

    # ==================================================
    # CRITICAL FIX:
    # reset indices AFTER filtering
    # so all datasets share the same positional indexing
    # ==================================================
    shy_filt = shy_filt.reset_index(drop=True)

    cmems_filt = cmems_filt.reset_index(drop=True)

    mslp_filt = mslp_filt.reset_index(drop=True)

    tide_filt = tide_filt.reset_index(drop=True)

    u10_filt = u10_filt.reset_index(drop=True)

    v10_filt = v10_filt.reset_index(drop=True)

    # ==================================================
    # Helper: extract training chunks
    # ==================================================
    def extract_chunks(data, chunks):

        dfs = []

        for ch in chunks:

            start = (
                shy_filt.index[
                    shy_filt.iloc[:, 0] == ch[0]
                ].tolist()[0]
            )

            end = (
                shy_filt.index[
                    shy_filt.iloc[:, 0] == ch[1]
                ].tolist()[-1] + 1
            )

            dfs.append(
                data.iloc[start:end]
            )

        return pd.concat(dfs)

    # ==================================================
    # Training data
    # ==================================================
    shyTrain = extract_chunks(
        shy_filt,
        config["years"]["train"]
    )

    cmemsTrain = extract_chunks(
        cmems_filt,
        config["years"]["train"]
    )

    mslpTrain = extract_chunks(
        mslp_filt,
        config["years"]["train"]
    )

    tideTrain = extract_chunks(
        tide_filt,
        config["years"]["train"]
    )

    u10Train = extract_chunks(
        u10_filt,
        config["years"]["train"]
    )

    v10Train = extract_chunks(
        v10_filt,
        config["years"]["train"]
    )

    # ==================================================
    # Validation and test indices
    # ==================================================
    get_indices = lambda df, yrs: [

        i

        for y in yrs

        for i in df[
            df.iloc[:, 0] == y
        ].index.tolist()
    ]

    val_idx = get_indices(
        shy_filt,
        config["years"]["val"]
    )

    test_idx = get_indices(
        shy_filt,
        config["years"]["test"]
    )

    # ==================================================
    # CRITICAL FIX:
    # use iloc instead of loc
    # after resetting indices
    # ==================================================
    shyVal, shyTest = (
        shy_filt.iloc[val_idx],
        shy_filt.iloc[test_idx]
    )

    cmemsVal, cmemsTest = (
        cmems_filt.iloc[val_idx],
        cmems_filt.iloc[test_idx]
    )

    mslpVal, mslpTest = (
        mslp_filt.iloc[val_idx],
        mslp_filt.iloc[test_idx]
    )

    tideVal, tideTest = (
        tide_filt.iloc[val_idx],
        tide_filt.iloc[test_idx]
    )

    u10Val, u10Test = (
        u10_filt.iloc[val_idx],
        u10_filt.iloc[test_idx]
    )

    v10Val, v10Test = (
        v10_filt.iloc[val_idx],
        v10_filt.iloc[test_idx]
    )

    # ==================================================
    # Diagnostics
    # ==================================================
    def get_date(df, idx):

        return (

            f"{int(df.iloc[idx,0]):04d}-"

            f"{int(df.iloc[idx,1]):02d}-"

            f"{int(df.iloc[idx,2]):02d} "

            f"{int(df.iloc[idx,3]):02d}:"

            f"{int(df.iloc[idx,4]):02d}:"

            f"{int(df.iloc[idx,5]):02d}"
        )

    def print_info(name, df):

        print(

            f"{name:<15} | "

            f"start: {get_date(df, 0)} | "

            f"end: {get_date(df, -1)} | "

            f"length = {len(df)}"
        )

    print(
        "\n================ DATA CHECK ================\n"
    )

    print_info(
        "targetVal",
        shyVal
    )

    print_info(
        "predictorVal",
        tideVal
    )

    print()

    print_info(
        "targetTest",
        shyTest
    )

    print_info(
        "predictorTest",
        tideTest
    )

    print(
        "\n============================================\n"
    )

    # ==================================================
    # Save aligned testing datasets (this is optional)
    # ==================================================
    # import os

    # output_dir = "aligned_test_sets"

    # os.makedirs(
    #     output_dir,
    #     exist_ok=True
    # )

    # def save_with_tide_dates(
    #     df,
    #     filename
    # ):

    #     df_out = df.copy()

    #     df_out.iloc[:, 0:6] = (
    #         tideTest.iloc[:, 0:6].values
    #     )

    #     df_out.to_csv(

    #         os.path.join(
    #             output_dir,
    #             filename
    #         ),

    #         index=False,

    #         header=False
    #     )

    # save_with_tide_dates(
    #     shyTest,
    #     "target_test.csv"
    # )

    # save_with_tide_dates(
    #     cmemsTest,
    #     "cmems_test.csv"
    # )

    # save_with_tide_dates(
    #     tideTest,
    #     "tide_test.csv"
    # )

    # save_with_tide_dates(
    #     mslpTest,
    #     "mslp_test.csv"
    # )

    # save_with_tide_dates(
    #     u10Test,
    #     "u10_test.csv"
    # )

    # save_with_tide_dates(
    #     v10Test,
    #     "v10_test.csv"
    # )

    # print(
    #     "\nAligned testing datasets "
    #     "saved successfully."
    # )

    # ==================================================
    # Feature construction
    # ==================================================
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

    XTrain = build_X(
        cmemsTrain,
        tideTrain,
        mslpTrain,
        u10Train,
        v10Train
    )

    XVal = build_X(
        cmemsVal,
        tideVal,
        mslpVal,
        u10Val,
        v10Val
    )

    XTest = build_X(
        cmemsTest,
        tideTest,
        mslpTest,
        u10Test,
        v10Test
    )

    # ==================================================
    # Targets
    # ==================================================
    zTrain = shyTrain.iloc[:, 7]

    zVal = shyVal.iloc[:, 7]

    zTest = shyTest.iloc[:, 7]

    # ==================================================
    # Standardization
    # ==================================================
    scaler = StandardScaler()

    XTrain = pd.DataFrame(
        scaler.fit_transform(XTrain),
        columns=XTrain.columns,
        index=XTrain.index,
    )

    XVal = pd.DataFrame(
        scaler.transform(XVal),
        columns=XVal.columns,
        index=XVal.index,
    )

    XTest = pd.DataFrame(
        scaler.transform(XTest),
        columns=XTest.columns,
        index=XTest.index,
    )

    # ==================================================
    # Tensor conversion
    # ==================================================
    def to_tensor(
        x,
        seq_model=False
    ):

        t = torch.tensor(
            x.to_numpy(),
            dtype=torch.float32
        )

        if seq_model:

            t = t.unsqueeze(1)

        return t.to(device)

    # ==================================================
    # Sequence control
    # ==================================================
    use_sequences = config.get(
        "use_sequences",
        True
    )

    is_seq_model = (

        use_sequences

        and (

            "rnn"
            in config["model_type"].lower()

            or

            "lstm"
            in config["model_type"].lower()
        )
    )

    hourTest = (
        tideTest.iloc[:, 3]
        .to_numpy()
    )

    # ==================================================
    # Return dictionary
    # ==================================================
    return {

        "XTrain": to_tensor(
            XTrain,
            seq_model=is_seq_model
        ),

        "zTrain": torch.tensor(
            zTrain.to_numpy(),
            dtype=torch.float32
        ).to(device),

        "XVal": to_tensor(
            XVal,
            seq_model=is_seq_model
        ),

        "ZVal": torch.tensor(
            zVal.to_numpy(),
            dtype=torch.float32
        ).to(device),

        "XTest": to_tensor(
            XTest,
            seq_model=is_seq_model
        ),

        "ZTest": torch.tensor(
            zTest.to_numpy(),
            dtype=torch.float32
        ).to(device),

        "tideTest": tideTest,

        "hourTest": hourTest,

        "scaler": scaler,

        "feature_names": list(
            XTrain.columns
        ),
    }