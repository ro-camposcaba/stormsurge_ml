# ======================================================
# Save 2022 predictors in aligned CSV format
# ======================================================

import os
import pandas as pd


# ======================================================
# FILE PATHS
# ======================================================

base_path = (
    "C:/Users/Rodrigo/Desktop/Rodrigo/"
    "02 - Research fellow/"
    "01 - Storm surge ML downscaling/"
    "surge_ml/tests"
)

cmems_path = (
    f"{base_path}/inference 2022/predictors/VeneziaPS/"
    "cmems_7PCs_fulldomain_VeneziaPS_2022.txt"
)

tide_path = (
    f"{base_path}/inference 2022/predictors/VeneziaPS/"
    "tide_VeneziaPS_2022.txt"
)

mslp_path = (
    f"{base_path}/inference 2022/predictors/VeneziaPS/"
    "ERA5mslp_7PCs_fulldomain_VeneziaPS_2022.txt"
)

u10_path = (
    f"{base_path}/inference 2022/predictors/VeneziaPS/"
    "ERA5stressX_7PCs_fulldomain_VeneziaPS_2022.txt"
)

v10_path = (
    f"{base_path}/inference 2022/predictors/VeneziaPS/"
    "ERA5stressY_7PCs_fulldomain_VeneziaPS_2022.txt"
)


# ======================================================
# OUTPUT DIRECTORY
# ======================================================

output_dir = (
    f"{base_path}/inference 2022/"
    "aligned_predictors_VeneziaPS_2022"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ======================================================
# LOAD DATA
# ======================================================

cmems = pd.read_csv(
    cmems_path,
    header=None,
    sep=r"\s+"
)

tide = pd.read_csv(
    tide_path,
    header=None,
    sep=r"\s+"
)

mslp = pd.read_csv(
    mslp_path,
    header=None,
    sep=r"\s+"
)

u10 = pd.read_csv(
    u10_path,
    header=None,
    sep=r"\s+"
)

v10 = pd.read_csv(
    v10_path,
    header=None,
    sep=r"\s+"
)


# ======================================================
# CONFIGURATION
# ======================================================

year = 2022

# IMPORTANT:
# same offset used in the corrected data_loader
offset = 0


# ======================================================
# YEAR FILTERING
# ======================================================

def slice_year(df):

    idx = df.index[
        df.iloc[:, 0] == year
    ].tolist()

    start = (
        idx[offset]
        if len(idx) > offset
        else idx[0]
    )

    end = idx[-1] + 1

    return (
        df.iloc[start:end]
        .reset_index(drop=True)
    )


cmemsTest = slice_year(cmems)

mslpTest = slice_year(mslp)

tideTest = slice_year(tide)

u10Test = slice_year(u10)

v10Test = slice_year(v10)


# ======================================================
# USE MSLP AS COMMON TIME REFERENCE
# ======================================================

reference_dates = (
    mslpTest.iloc[:, 0:6]
    .copy()
)


# ======================================================
# HELPER FUNCTION
# ======================================================

def save_with_reference_dates(
    df,
    filename
):

    df_out = df.copy()

    # Replace first 6 columns
    # with aligned MSLP timestamps
    df_out.iloc[:, 0:6] = (
        reference_dates.values
    )

    df_out.to_csv(

        os.path.join(
            output_dir,
            filename
        ),

        index=False,

        header=False
    )


# ======================================================
# SAVE DATASETS
# ======================================================

save_with_reference_dates(
    cmemsTest,
    "cmems_2022.csv"
)

save_with_reference_dates(
    tideTest,
    "tide_2022.csv"
)

save_with_reference_dates(
    mslpTest,
    "mslp_2022.csv"
)

save_with_reference_dates(
    u10Test,
    "u10_2022.csv"
)

save_with_reference_dates(
    v10Test,
    "v10_2022.csv"
)


# ======================================================
# DIAGNOSTICS
# ======================================================

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

        f"{name:<12} | "

        f"start: {get_date(df,0)} | "

        f"end: {get_date(df,-1)} | "

        f"length = {len(df)}"
    )


print("\n=========== 2022 DATA CHECK ===========\n")

print_info("CMEMS", cmemsTest)

print_info("TIDE", tideTest)

print_info("MSLP", mslpTest)

print_info("U10", u10Test)

print_info("V10", v10Test)

print("\n=======================================\n")

print(
    "Aligned 2022 predictor datasets "
    "saved successfully."
)