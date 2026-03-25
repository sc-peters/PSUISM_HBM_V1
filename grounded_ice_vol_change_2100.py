#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Antarctic Sea Level Equivalent (SLE) from grounded ice volume change
using historical and forward PSU-ISM ensemble runs.

Produces:
    - SLE for each ensemble member
    - weighted projection
    - comparison of weighted vs unweighted distributions
"""

import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# USER INPUTS
# ======================================================

HIST_DIR = Path(
"/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/ensemble_2"
)

FWD_DIR = Path(
"/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/forward_ensemble"
)

WEIGHT_FILE = Path(
"/Users/sp53972/Documents/GitHub/PSUISM_HBM_V1/model_weights_table.csv"
)

TIME_INDEX = 80   # timestep corresponding to 2100

# ======================================================
# CONSTANTS
# ======================================================

rho_ice = 917.0
rho_water = 1028.0
A_ocean = 3.62e14

# ======================================================
# LOAD MODEL WEIGHTS
# ======================================================

weights_df = pd.read_csv(WEIGHT_FILE)

print("\nWeight file columns:", weights_df.columns)

# build dictionary: run1 -> weight
weights = dict(zip(weights_df["model_id"], weights_df["weight"]))

print("Loaded weights for", len(weights), "models")

# ======================================================
# FIND MODEL FILES
# ======================================================

hist_files = sorted(HIST_DIR.rglob("*.nc"))
fwd_files = sorted(FWD_DIR.rglob("*.nc"))

print("\nHistorical models:", len(hist_files))
print("Forward models:", len(fwd_files))

# ======================================================
# STORAGE
# ======================================================

models = []
sle_values = []
run_ids = []

# ======================================================
# PROCESS EACH MODEL
# ======================================================

for fwd_file in fwd_files:

    model_name = fwd_file.stem
    hist_file = HIST_DIR / fwd_file.name

    if not hist_file.exists():
        print("Missing historical file for", model_name)
        continue

    print("Processing:", model_name)

    # extract run number from filename
    m = re.search(r"run(\d+)", model_name)
    run_number = int(m.group(1))

    # convert to run1, run2, run3 format
    run_id = f"run{run_number}"

    # open files
    ds_hist = xr.open_dataset(hist_file)
    ds_fwd = xr.open_dataset(fwd_file)

    # grounded ice volume
    V_hist = ds_hist["totig"].values
    V_fwd = ds_fwd["totig"].values

    # final historical grounded volume
    V0 = V_hist[-1]

    # volume in 2100
    V2100 = V_fwd[TIME_INDEX]

    # grounded ice loss (positive = sea level rise)
    dV = V0 - V2100

    # convert to sea level equivalent
    sle_m = (rho_ice / rho_water) * dV / A_ocean
    sle_mm = sle_m * 1000

    models.append(model_name)
    sle_values.append(sle_mm)
    run_ids.append(run_id)

# ======================================================
# BUILD RESULTS TABLE
# ======================================================

results = pd.DataFrame({
    "model": models,
    "run_id": run_ids,
    "sle_mm_2100": sle_values
})

# map weights
results["weight"] = results["run_id"].map(weights)

# drop models without weights
results = results.dropna(subset=["weight"])

# normalize weights
results["weight"] = results["weight"] / results["weight"].sum()

print("\nResults")
print(results)

sle = results["sle_mm_2100"].values
weights = results["weight"].values

# ======================================================
# MEANS
# ======================================================

unweighted_mean = np.mean(sle)
weighted_mean = np.sum(sle * weights)

print("\nUnweighted mean:", unweighted_mean, "mm")
print("Weighted mean:", weighted_mean, "mm")

# ======================================================
# CREATE WEIGHTED SAMPLE
# ======================================================

weighted_samples = np.repeat(sle, (weights * 1000).astype(int))

# ======================================================
# PLOT DISTRIBUTIONS
# ======================================================

plt.figure(figsize=(8,6))

plt.hist(
    sle,
    bins=10,
    alpha=0.5,
    density=True,
    label="Unweighted ensemble"
)

plt.hist(
    weighted_samples,
    bins=10,
    alpha=0.5,
    density=True,
    label="Bayesian weighted ensemble"
)

plt.axvline(
    unweighted_mean,
    color="black",
    linestyle="--",
    label="Unweighted mean"
)

plt.axvline(
    weighted_mean,
    color="red",
    linestyle="--",
    label="Weighted mean"
)

plt.xlabel("Sea Level Equivalent in 2100 (mm)")
plt.ylabel("Probability Density")
plt.title("Weighted vs Unweighted Antarctic Sea Level Projections")

plt.legend()

plt.tight_layout()

plt.savefig("weighted_vs_unweighted_sle.png", dpi=300)

plt.show()

# ======================================================
# SAVE RESULTS
# ======================================================

results.to_csv("sle_ensemble_results.csv", index=False)

print("\nSaved outputs:")
print("sle_ensemble_results.csv")
print("weighted_vs_unweighted_sle.png")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Antarctic Sea Level Equivalent (SLE) from grounded ice volume change
and create ISMIP-style weighted vs unweighted projection figure.
"""

import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# USER INPUTS
# ======================================================

HIST_DIR = Path(
"/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/ensemble_2"
)

FWD_DIR = Path(
"/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/forward_ensemble"
)

WEIGHT_FILE = Path(
"/Users/sp53972/Documents/GitHub/PSUISM_HBM_V1/model_weights_table.csv"
)

TIME_INDEX = 80

# ======================================================
# CONSTANTS
# ======================================================

rho_ice = 917.0
rho_water = 1028.0
A_ocean = 3.62e14

# ======================================================
# LOAD WEIGHTS
# ======================================================

weights_df = pd.read_csv(WEIGHT_FILE)

print("Weight file columns:", weights_df.columns)

weights = dict(zip(weights_df["model_id"], weights_df["weight"]))

print("Loaded weights for", len(weights), "models")

# ======================================================
# FIND FILES
# ======================================================

hist_files = sorted(HIST_DIR.rglob("*.nc"))
fwd_files = sorted(FWD_DIR.rglob("*.nc"))

print("\nHistorical models:", len(hist_files))
print("Forward models:", len(fwd_files))

# ======================================================
# COMPUTE SLE
# ======================================================

models = []
run_ids = []
sle_values = []

for fwd_file in fwd_files:

    model_name = fwd_file.stem
    hist_file = HIST_DIR / fwd_file.name

    if not hist_file.exists():
        print("Missing historical file for", model_name)
        continue

    print("Processing:", model_name)

    # extract run number
    m = re.search(r"run(\d+)", model_name)
    run_id = f"run{int(m.group(1))}"

    ds_hist = xr.open_dataset(hist_file)
    ds_fwd = xr.open_dataset(fwd_file)

    if "totig" not in ds_hist or "totig" not in ds_fwd:
        print("Skipping", model_name, "(missing totig)")
        continue

    V_hist = ds_hist["totig"].values
    V_fwd = ds_fwd["totig"].values

    V0 = V_hist[-1]
    V2100 = V_fwd[TIME_INDEX]

    dV = V0 - V2100

    sle_m = (rho_ice / rho_water) * dV / A_ocean
    sle_mm = sle_m * 1000

    models.append(model_name)
    run_ids.append(run_id)
    sle_values.append(sle_mm)

# ======================================================
# BUILD TABLE
# ======================================================

results = pd.DataFrame({
    "model": models,
    "run_id": run_ids,
    "sle_mm_2100": sle_values
})

results["weight"] = results["run_id"].map(weights)

results = results.dropna()

results["weight"] /= results["weight"].sum()

print("\nResults")
print(results)

sle = results["sle_mm_2100"].values
weights_arr = results["weight"].values

unweighted_mean = sle.mean()
weighted_mean = np.sum(sle * weights_arr)

print("\nUnweighted mean:", unweighted_mean)
print("Weighted mean:", weighted_mean)

weighted_samples = np.repeat(sle, (weights_arr * 1000).astype(int))

# ======================================================
# FIGURE
# ======================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(8,10)
)

# ======================================================
# PANEL A — ENSEMBLE TRAJECTORIES
# ======================================================

ax = axes[0]

for _, row in results.iterrows():

    model = row["model"]
    weight = row["weight"]

    fwd_file = FWD_DIR / f"{model}.nc"

    ds = xr.open_dataset(fwd_file)

    V = ds["totig"].values

    V0 = V[0]

    sle_series = (rho_ice / rho_water) * (V0 - V) / A_ocean * 1000

    time = ds["time"].values

    ax.plot(
        time,
        sle_series,
        linewidth=1.5,
        alpha=0.2 + 3 * weight
    )

ax.set_ylabel("Sea level contribution (mm SLE)")
ax.set_title("PSU-ISM Antarctic Sea Level Projections")

# ======================================================
# PANEL B — DISTRIBUTIONS
# ======================================================

ax = axes[1]

bins = np.linspace(sle.min(), sle.max(), 10)

ax.hist(
    sle,
    bins=bins,
    density=True,
    alpha=0.5,
    label="Unweighted ensemble"
)

ax.hist(
    weighted_samples,
    bins=bins,
    density=True,
    alpha=0.5,
    label="Bayesian weighted ensemble"
)

ax.axvline(unweighted_mean, linestyle="--", color="black", label="Unweighted mean")
ax.axvline(weighted_mean, linestyle="--", color="red", label="Weighted mean")

ax.set_xlabel("Sea Level Equivalent in 2100 (mm)")
ax.set_ylabel("Probability Density")

ax.legend()

plt.tight_layout()

plt.savefig(
    "ismip_style_weighted_projection.png",
    dpi=300
)

plt.show()

# ======================================================
# SAVE TABLE
# ======================================================

results.to_csv("sle_ensemble_results.csv", index=False)

print("\nSaved:")
print("sle_ensemble_results.csv")
print("ismip_style_weighted_projection.png")




#### sle spread (ismip6)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Antarctic sea-level equivalent (SLE) time series from PSU-ISM ensemble
and make an ISMIP6-style weighted vs unweighted spread figure.
"""

import re
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# USER INPUTS
# ======================================================

FWD_DIR = Path(
    "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/forward_ensemble"
)

WEIGHT_FILE = Path(
    "/Users/sp53972/Documents/GitHub/PSUISM_HBM_V1/model_weights_table.csv"
)

START_YEAR = 2000   # change to 2015 if that is the actual start of your forward experiment

# ======================================================
# CONSTANTS
# ======================================================

rho_ice = 917.0
rho_water = 1028.0
A_ocean = 3.62e14
FILL_THRESHOLD = 1e30   # catches 9.96921e+36 fill values

# ======================================================
# HELPERS
# ======================================================

def weighted_percentile(data, weights, percentile):
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]

    cdf = np.cumsum(weights_sorted)
    cdf = cdf / cdf[-1]

    return np.interp(percentile / 100.0, cdf, data_sorted)


# ======================================================
# LOAD WEIGHTS
# ======================================================

weights_df = pd.read_csv(WEIGHT_FILE)
print("Weight file columns:", list(weights_df.columns))

weights_dict = dict(zip(weights_df["model_id"], weights_df["weight"]))
print("Loaded weights for", len(weights_dict), "models")

# ======================================================
# FIND ONLY THE TRUE FORWARD FILES
# ======================================================

fwd_files = sorted(FWD_DIR.glob("run*_fort.92.nc"))
print("Forward models found:", len(fwd_files))

# ======================================================
# STORAGE
# ======================================================

sle_series_all = []
weights_list = []
model_names = []

run_ids = []

time = None

# ======================================================
# LOOP THROUGH ENSEMBLE MEMBERS
# ======================================================

for fwd_file in fwd_files:
    model_name = fwd_file.stem

    m = re.search(r"run(\d+)", model_name)
    if m is None:
        print("Skipping file with no run id:", model_name)
        continue

    run_id = f"run{int(m.group(1))}"

    if run_id not in weights_dict:
        print("Skipping file with no weight:", model_name)
        continue

    print("Processing:", model_name)

    with xr.open_dataset(fwd_file) as ds:
        if "totig" not in ds:
            print("Skipping (no totig):", model_name)
            continue

        # -----------------------------
        # FIXED SLE CALCULATION
        # -----------------------------
        V = ds["totig"].values.astype(float)

        # remove fill values like 9.96921e+36
        V[V > FILL_THRESHOLD] = np.nan

        # interpolate over missing values
        V = pd.Series(V).interpolate(limit_direction="both").values

        # optional sanity check
        if np.any(np.isnan(V)):
            print("Skipping (NaNs remain after interpolation):", model_name)
            continue

        # use first valid forward timestep as reference
        V0 = V[0]

        # grounded ice loss -> positive SLE contribution
        dV = V0 - V

        # convert to mm SLE
        sle_series = (rho_ice / rho_water) * dV / A_ocean * 1000.0

        print("  SLE range (mm):", np.nanmin(sle_series), "to", np.nanmax(sle_series))

        
        sle_series_all.append(sle_series)
        weights_list.append(weights_dict[run_id])
        model_names.append(model_name)
        run_ids.append(run_id)

        if time is None:
            time = START_YEAR + np.arange(len(V))

# ======================================================
# CONVERT TO ARRAYS
# ======================================================

sle_series_all = np.array(sle_series_all)
weights_list = np.array(weights_list, dtype=float)

if sle_series_all.size == 0:
    raise RuntimeError("No valid ensemble members were processed.")

weights_list = weights_list / weights_list.sum()

print("Models used:", sle_series_all.shape[0])

# ======================================================
# UNWEIGHTED STATS
# ======================================================

unw_median = np.median(sle_series_all, axis=0)
unw_p16 = np.percentile(sle_series_all, 16, axis=0)
unw_p84 = np.percentile(sle_series_all, 84, axis=0)

# ======================================================
# WEIGHTED STATS
# ======================================================

w_median = []
w_p16 = []
w_p84 = []

for t in range(sle_series_all.shape[1]):
    vals = sle_series_all[:, t]

    w_median.append(weighted_percentile(vals, weights_list, 50))
    w_p16.append(weighted_percentile(vals, weights_list, 16))
    w_p84.append(weighted_percentile(vals, weights_list, 84))

w_median = np.array(w_median)
w_p16 = np.array(w_p16)
w_p84 = np.array(w_p84)

# ======================================================
# SAVE 2100 TABLE
# ======================================================

# results = pd.DataFrame({
#     "model": model_names,
#     "run_id": [f"run{int(re.search(r'run(\\d+)', m).group(1))}" for m in model_names],
#     "sle_mm_final": sle_series_all[:, -1],
#     "weight": weights_list
# })


results = pd.DataFrame({
    "model": model_names,
    "run_id": run_ids,
    "sle_mm_final": sle_series_all[:, -1],
    "weight": weights_list
})

results.to_csv("sle_ensemble_results.csv", index=False)
print("Saved sle_ensemble_results.csv")

fig, ax = plt.subplots(figsize=(9, 6))

# all ensemble members
for i in range(sle_series_all.shape[0]):
    ax.plot(time, sle_series_all[i], color="0.7", linewidth=1, alpha=0.5)

# unweighted envelope + median
ax.fill_between(
    time, unw_p16, unw_p84,
    alpha=0.30,
    label="Unweighted 16–84%"
)
ax.plot(
    time, unw_median,
    linewidth=2,
    label="Unweighted median"
)

# weighted envelope + median
ax.fill_between(
    time, w_p16, w_p84,
    alpha=0.30,
    label="Weighted 16–84%"
)
ax.plot(
    time, w_median,
    linewidth=2,
    label="Weighted median"
)

ax.set_xlabel("Year")
ax.set_ylabel("Sea level contribution (mm SLE)")
ax.set_title("Antarctic sea-level contribution — weighted vs unweighted ensemble")
ax.legend()
plt.tight_layout()
plt.savefig("ismip_style_spread.png", dpi=300)
plt.show()

print("Saved ismip_style_spread.png")

## model weights scatter

plt.figure(figsize=(6,4))

plt.scatter(
    sle_series_all[:,-1],
    weights_list
)

plt.xlabel("2100 SLE (mm)")
plt.ylabel("Model weight")
plt.title("Weight vs projection")

plt.show()