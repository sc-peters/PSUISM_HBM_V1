#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:13:07 2026

@author: sp53972
"""

# both dhdt and velocity 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIERARCHICAL BAYESIAN MODEL FOR ICE SHEET VALIDATION
Combines thickness change (dh/dt) and velocity (vx, vy) observations
"""

import numpy as np
import xarray as xr
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import re
import calendar
from datetime import datetime, timedelta

# ==============================================================================
# DATA UTILITIES
# ==============================================================================

SEC_PER_YEAR = 365.2425 * 86400.0


def _parse_origin_allow_day00(unit_str: str) -> datetime:
    """Parse 'seconds since YYYY-MM-DD hh:mm:ss' but allow DD=00."""
    m = re.search(r"since\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", unit_str)
    if not m:
        raise ValueError(f"Could not parse time unit string: {unit_str}")

    Y, Mo, D, hh, mm, ss = map(int, m.groups())

    if D == 0:
        if Mo == 1:
            Y2, Mo2 = Y - 1, 12
        else:
            Y2, Mo2 = Y, Mo - 1
        last_day = calendar.monthrange(Y2, Mo2)[1]
        return datetime(Y2, Mo2, last_day, hh, mm, ss)

    return datetime(Y, Mo, D, hh, mm, ss)

def model_decimal_years_from_ds(ds, time_name="time"):
    """
    Convert model time values to decimal years.
    Works for:
        1) seconds since YYYY-MM-DD
        2) already-year values (e.g. 2000, 2001, ...)
    Also cleans NetCDF fill values in time.
    """

    time_var = ds[time_name]
    t = np.asarray(time_var.values, dtype=float)

    # ----------------------------------------------
    # Clean obvious NetCDF fill / missing values
    # ----------------------------------------------
    fill_value = time_var.attrs.get("_FillValue", None)
    missing_value = time_var.attrs.get("missing_value", None)

    if fill_value is not None:
        t[t == fill_value] = np.nan
    if missing_value is not None:
        t[t == missing_value] = np.nan

    # also catch absurd placeholder values
    t[np.abs(t) > 1e20] = np.nan

    # ----------------------------------------------
    # Drop bad time entries
    # ----------------------------------------------
    good = np.isfinite(t)

    if not np.any(good):
        raise ValueError("No valid time values found in model time variable")

    if np.sum(~good) > 0:
        print(f"Warning: removed {np.sum(~good)} invalid time value(s) from {time_name}")

    t = t[good]

    # Try to read units
    unit_str = time_var.attrs.get("units", None)
    if unit_str is None:
        unit_str = time_var.attrs.get("unit", None)

    # --------------------------------------------------
    # CASE 1 — already years
    # --------------------------------------------------
    if unit_str is None:
        print("Time variable has no units — assuming values are already years.")
        return t

    # --------------------------------------------------
    # CASE 2 — seconds since origin
    # --------------------------------------------------
    if "since" in unit_str and "second" in unit_str:
        origin = _parse_origin_allow_day00(unit_str)

        years = np.empty_like(t, dtype=float)

        for i, sec in enumerate(t):
            dt = origin + timedelta(seconds=float(sec))
            year_start = datetime(dt.year, 1, 1)
            frac = (dt - year_start).total_seconds() / SEC_PER_YEAR
            years[i] = dt.year + frac

        return years

    # --------------------------------------------------
    # CASE 3 — already numeric years with units
    # --------------------------------------------------
    print("Time units not recognized — assuming values are years.")
    return t


# def model_decimal_years_from_ds(ds, time_name="time"):
#     """
#     Convert model time values to decimal years.
#     Works for:
#         1) seconds since YYYY-MM-DD
#         2) already-year values (e.g. 2000, 2001, ...)
#     """

#     t = np.asarray(ds[time_name].values, dtype=float)
    
#     ##this is here cuz psu ism has a weird 2nd timestep thats messed up
#     if not np.all(np.isfinite(t)):
#         raise ValueError("Non-finite values found in model_years")

#     if np.any(np.abs(t) > 3000):
#         raise ValueError(f"Unrealistic year values found in model_years: min={np.nanmin(t)}, max={np.nanmax(t)}")

#     # Try to read units
#     unit_str = ds[time_name].attrs.get("units", None)
#     if unit_str is None:
#         unit_str = ds[time_name].attrs.get("unit", None)

#     # --------------------------------------------------
#     # CASE 1 — already years (your regridded files)
#     # --------------------------------------------------
#     if unit_str is None:
#         print("Time variable has no units — assuming values are already years.")
#         return t

#     # --------------------------------------------------
#     # CASE 2 — seconds since origin
#     # --------------------------------------------------
#     if "since" in unit_str and "second" in unit_str:

#         origin = _parse_origin_allow_day00(unit_str)

#         years = np.empty_like(t, dtype=float)

#         for i, sec in enumerate(t):

#             dt = origin + timedelta(seconds=float(sec))

#             year_start = datetime(dt.year, 1, 1)
#             frac = (dt - year_start).total_seconds() / SEC_PER_YEAR

#             years[i] = dt.year + frac

#         return years

#     # --------------------------------------------------
#     # CASE 3 — already numeric years with units
#     # --------------------------------------------------
#     print("Time units not recognized — assuming values are years.")
#     return t


def _to_2d(arr, name, fname):
    """Squeeze array to 2D."""
    arr2 = np.squeeze(arr)
    if arr2.ndim != 2:
        raise ValueError(f"{fname}: {name} not 2D after squeeze; got shape {arr2.shape}")
    return arr2


def _extract_year_from_filename(fname: str) -> int:
    """Extract year from filename."""
    m = re.search(r"(\d{4})", fname)
    if not m:
        raise ValueError(f"No year found in filename: {fname}")
    return int(m.group(1))


# def _snap_model_year_to_obs_year(t_model: float, available_years: np.ndarray) -> int | None:
#     """Map a model decimal year to the nearest available obs integer year."""
#     y = int(np.rint(t_model))
#     if y in set(available_years.tolist()):
#         return y
#     return None

def _snap_model_year_to_obs_year(t_model, available_years, tol=1.5):
    """
    Match model decimal year to nearest observation year.

    Parameters
    ----------
    t_model : float
        model time (decimal year)

    available_years : array
        integer observation years

    tol : float
        maximum allowed difference (years)

    Returns
    -------
    int or None
    """

    available_years = np.asarray(available_years)

    idx = np.argmin(np.abs(available_years - t_model))

    if np.abs(available_years[idx] - t_model) <= tol:
        return int(available_years[idx])

    return None


# ==============================================================================
# THICKNESS CHANGE (dh/dt) PROCESSING
# ==============================================================================

def compute_model_dhdt(ds, model_years, model_thickness_var="h"):
    """
    Compute dh/dt from model thickness snapshots.

    Handles:
    - reversed time axes (e.g., 2020 → 1920)
    - non-monotonic time
    - stable midpoint calculation
    """

    # --------------------------------------------------
    # Extract thickness
    # --------------------------------------------------
    h = ds[model_thickness_var].astype("float32")

    # --------------------------------------------------
    # Ensure time axis is monotonic increasing
    # --------------------------------------------------
    order = np.argsort(model_years)

    model_years = np.asarray(model_years)[order]
    h = h.isel(time=order)

    h = h.assign_coords(time=("time", model_years))

    # --------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------
    # t = model_years.astype(float)
    t = np.asarray(model_years, dtype=np.float64)

    # --------------------------------------------------
    # Compute time differences
    # --------------------------------------------------
    dt = np.diff(t)

    if np.any(dt == 0):
        raise ValueError("Duplicate model time values detected")

    # --------------------------------------------------
    # Compute midpoint times safely
    # --------------------------------------------------
    # tmid = t[:-1] + dt / 2.0
    tmid = 0.5 * (t[:-1] + t[1:])

    # --------------------------------------------------
    # Compute dh/dt
    # --------------------------------------------------
    h0 = h.isel(time=slice(0, -1)).values
    h1 = h.isel(time=slice(1, None)).values

    dhdt = (h1 - h0) / dt[:, None, None]

    # --------------------------------------------------
    # Build output DataArray
    # --------------------------------------------------
    da = xr.DataArray(
        dhdt.astype("float32"),
        dims=["time", "y", "x"],
        coords={"time": tmid},
        attrs={
            "units": "m/yr",
            "long_name": "model thickness change rate"
        }
    )

    return da, t, tmid, dt


def compute_obs_dhdt_on_model_intervals(
    obs_thickness_dir: str,
    thickness_pattern: str,
    obs_h_var: str,
    obs_rmse_var: str | None,
    model_t: np.ndarray,
    model_tmid: np.ndarray,
    model_dt: np.ndarray,
):
    """Build obs dh/dt using the SAME endpoint times as model thickness snapshots."""
    p = Path(obs_thickness_dir)
    files = sorted(p.glob(thickness_pattern))
    if len(files) == 0:
        raise ValueError(f"No thickness files found in {obs_thickness_dir}")

    years = np.array([_extract_year_from_filename(f.name) for f in files], dtype=int)
    sort_idx = np.argsort(years)
    years = years[sort_idx]
    files = [files[i] for i in sort_idx]

    print(f"Found {len(files)} thickness files")
    print(f"Obs years available: {years.min()} to {years.max()}")

    H = {}
    RMSE = {}

    print("\nLoading obs thickness yearly files:")
    for i, (f, y) in enumerate(zip(files, years), 1):
        print(f"  [{i:2d}/{len(files)}] {f.name}")
        ds = xr.open_dataset(f, engine="netcdf4")
        h2d = _to_2d(ds[obs_h_var].values, obs_h_var, f.name).astype("float32")
        H[y] = h2d

        if obs_rmse_var is not None and obs_rmse_var in ds:
            rmse2d = _to_2d(ds[obs_rmse_var].values, obs_rmse_var, f.name).astype("float32")
            RMSE[y] = rmse2d
        else:
            RMSE[y] = np.full_like(h2d, np.nan, dtype="float32")
        ds.close()

    dhdt_list = []
    unc_list = []
    used = 0

    print("\nComputing OBS dh/dt on MODEL intervals:")
    for i in range(len(model_t) - 1):
        y1 = _snap_model_year_to_obs_year(model_t[i], years)
        y2 = _snap_model_year_to_obs_year(model_t[i + 1], years)

        if (y1 is None) or (y2 is None) or (y1 not in H) or (y2 not in H):
            shape = next(iter(H.values())).shape
            dhdt_list.append(np.full(shape, np.nan, dtype="float32"))
            unc_list.append(np.full(shape, np.nan, dtype="float32"))
            continue

        dt = float(model_dt[i])
        dh = H[y2] - H[y1]
        dhdt = dh / dt

        rmse1 = RMSE[y1]
        rmse2 = RMSE[y2]
        unc = np.sqrt(rmse1**2 + rmse2**2) / dt

        dhdt_list.append(dhdt.astype("float32"))
        unc_list.append(unc.astype("float32"))
        used += 1

    obs_dhdt = xr.DataArray(
        np.stack(dhdt_list, axis=0),
        dims=["time", "y", "x"],
        coords={"time": model_tmid.astype(float)},
        attrs={"units": "m/yr", "long_name": "obs thickness change rate"}
    )

    obs_unc = xr.DataArray(
        np.stack(unc_list, axis=0),
        dims=["time", "y", "x"],
        coords={"time": model_tmid.astype(float)},
        attrs={"units": "m/yr", "long_name": "obs dh/dt uncertainty"}
    )

    print(f"✓ Built obs dh/dt. Non-missing intervals: {used}/{len(model_dt)}")
    has_unc = not np.all(np.isnan(obs_unc.values))
    
    return obs_dhdt, obs_unc, has_unc


# ==============================================================================
# VELOCITY PROCESSING
# ==============================================================================

# def compute_model_velocity_on_obs_years(
#     ds, 
#     model_years,
#     obs_years,
#     vx_var="uvelmean",
#     vy_var="vvelmean"
# ):
#     """
#     Extract model velocity at specific obs years by finding nearest model time.
#     Returns list of (vx, vy) arrays for each obs year.
#     """
#     vx = ds[vx_var].astype("float32")
#     vy = ds[vy_var].astype("float32")
    
#     vx = vx.assign_coords(time=("time", model_years))
#     vy = vy.assign_coords(time=("time", model_years))
    
#     vx_list = []
#     vy_list = []
    
#     print("\nExtracting model velocity at obs years:")
#     for obs_year in obs_years:
#         # Find nearest model time
#         idx = np.argmin(np.abs(model_years - obs_year))
#         nearest_year = model_years[idx]
        
#         if abs(nearest_year - obs_year) > 1.0:  # tolerance of 1 year
#             print(f"  Warning: obs year {obs_year} -> nearest model {nearest_year:.2f} (diff > 1yr)")
#             vx_list.append(None)
#             vy_list.append(None)
#         else:
#             print(f"  obs year {obs_year} -> model time {nearest_year:.2f}")
#             vx_list.append(vx.isel(time=idx).values)
#             vy_list.append(vy.isel(time=idx).values)
    
#     return vx_list, vy_list

def compute_model_velocity_change(ds, model_years, vx_var="ua", vy_var="va"):
    """
    Compute model velocity change rates (dvx/dt, dvy/dt)
    between model time steps.
    """

    vx = ds[vx_var].astype("float32")
    vy = ds[vy_var].astype("float32")

    # ensure time increasing
    order = np.argsort(model_years)
    model_years = np.asarray(model_years)[order]

    vx = vx.isel(time=order)
    vy = vy.isel(time=order)

    vx = vx.assign_coords(time=("time", model_years))
    vy = vy.assign_coords(time=("time", model_years))

    t = np.asarray(model_years, dtype=float)
    dt = np.diff(t)

    if np.any(dt == 0):
        raise ValueError("Duplicate velocity time values detected")

    tmid = 0.5 * (t[:-1] + t[1:])

    vx0 = vx.isel(time=slice(0, -1)).values
    vx1 = vx.isel(time=slice(1, None)).values

    vy0 = vy.isel(time=slice(0, -1)).values
    vy1 = vy.isel(time=slice(1, None)).values

    dvxdt = (vx1 - vx0) / dt[:, None, None]
    dvydt = (vy1 - vy0) / dt[:, None, None]

    dvxdt = xr.DataArray(
        dvxdt.astype("float32"),
        dims=["time","y","x"],
        coords={"time": tmid},
        attrs={"units":"m/yr²"}
    )

    dvydt = xr.DataArray(
        dvydt.astype("float32"),
        dims=["time","y","x"],
        coords={"time": tmid},
        attrs={"units":"m/yr²"}
    )

    return dvxdt, dvydt


def load_obs_velocity_yearly(
    obs_vel_dir: str,
    vel_pattern: str,
    vx_var: str,
    vy_var: str,
    vx_err_var: str = "ERRX",
    vy_err_var: str = "ERRY",
):
    """
    Load all yearly velocity observations WITH uncertainties.
    Returns dicts: year -> (vx_2d, vy_2d, vx_err_2d, vy_err_2d)
    """
    p = Path(obs_vel_dir)
    files = sorted(p.glob(vel_pattern))
    if len(files) == 0:
        raise ValueError(f"No velocity files found in {obs_vel_dir}")
    
    years = np.array([_extract_year_from_filename(f.name) for f in files], dtype=int)
    sort_idx = np.argsort(years)
    years = years[sort_idx]
    files = [files[i] for i in sort_idx]
    
    print(f"\nFound {len(files)} velocity files")
    print(f"Velocity years: {years.min()} to {years.max()}")
    
    VX = {}
    VY = {}
    VX_ERR = {}
    VY_ERR = {}
    
    print("\nLoading obs velocity files:")
    for i, (f, y) in enumerate(zip(files, years), 1):
        print(f"  [{i:2d}/{len(files)}] {f.name}")
        ds = xr.open_dataset(f, engine="netcdf4")
        
        # Load velocities
        vx_2d = _to_2d(ds[vx_var].values, vx_var, f.name).astype("float32")
        vy_2d = _to_2d(ds[vy_var].values, vy_var, f.name).astype("float32")
        VX[y] = vx_2d
        VY[y] = vy_2d
        
        # Load uncertainties
        if vx_err_var in ds:
            vx_err_2d = _to_2d(ds[vx_err_var].values, vx_err_var, f.name).astype("float32")
            VX_ERR[y] = vx_err_2d
            print(f"      {vx_err_var} range: {np.nanmin(vx_err_2d):.2f} - {np.nanmax(vx_err_2d):.2f} m/yr")
        else:
            print(f"      ⚠️  {vx_err_var} not found, using constant 10 m/yr")
            VX_ERR[y] = np.full_like(vx_2d, 10.0, dtype="float32")
        
        if vy_err_var in ds:
            vy_err_2d = _to_2d(ds[vy_err_var].values, vy_err_var, f.name).astype("float32")
            VY_ERR[y] = vy_err_2d
            print(f"      {vy_err_var} range: {np.nanmin(vy_err_2d):.2f} - {np.nanmax(vy_err_2d):.2f} m/yr")
        else:
            print(f"      ⚠️  {vy_err_var} not found, using constant 10 m/yr")
            VY_ERR[y] = np.full_like(vy_2d, 10.0, dtype="float32")
        
        ds.close()
    
    return VX, VY, VX_ERR, VY_ERR, years


def compute_obs_dvdt_on_model_intervals(
    VX,
    VY,
    VX_ERR,
    VY_ERR,
    obs_years,
    model_t,
    model_tmid,
    model_dt,
):
    """
    Compute observed velocity change rates (dvx/dt, dvy/dt)
    using the same time intervals as the model.

    Uncertainty propagation:
        σ_dv/dt = sqrt(σ1² + σ2²) / dt
    """

    dvx_list = []
    dvy_list = []
    uncx_list = []
    uncy_list = []

    used = 0

    print("\nComputing OBS dv/dt on MODEL intervals:")

    for i in range(len(model_t) - 1):

        y1 = _snap_model_year_to_obs_year(model_t[i], obs_years)
        y2 = _snap_model_year_to_obs_year(model_t[i+1], obs_years)

        if (y1 is None) or (y2 is None):
            shape = next(iter(VX.values())).shape

            dvx_list.append(np.full(shape, np.nan, dtype="float32"))
            dvy_list.append(np.full(shape, np.nan, dtype="float32"))

            uncx_list.append(np.full(shape, np.nan, dtype="float32"))
            uncy_list.append(np.full(shape, np.nan, dtype="float32"))

            continue

        dt = float(model_dt[i])

        vx1 = VX[y1]
        vx2 = VX[y2]

        vy1 = VY[y1]
        vy2 = VY[y2]

        errx1 = VX_ERR[y1]
        errx2 = VX_ERR[y2]

        erry1 = VY_ERR[y1]
        erry2 = VY_ERR[y2]

        dvx = (vx2 - vx1) / dt
        dvy = (vy2 - vy1) / dt

        uncx = np.sqrt(errx1**2 + errx2**2) / dt
        uncy = np.sqrt(erry1**2 + erry2**2) / dt

        dvx_list.append(dvx.astype("float32"))
        dvy_list.append(dvy.astype("float32"))

        uncx_list.append(uncx.astype("float32"))
        uncy_list.append(uncy.astype("float32"))

        used += 1

    obs_dvxdt = xr.DataArray(
        np.stack(dvx_list),
        dims=["time","y","x"],
        coords={"time":model_tmid},
        attrs={"units":"m/yr²"}
    )

    obs_dvydt = xr.DataArray(
        np.stack(dvy_list),
        dims=["time","y","x"],
        coords={"time":model_tmid},
        attrs={"units":"m/yr²"}
    )

    obs_uncx = xr.DataArray(
        np.stack(uncx_list),
        dims=["time","y","x"],
        coords={"time":model_tmid},
        attrs={"units":"m/yr²"}
    )

    obs_uncy = xr.DataArray(
        np.stack(uncy_list),
        dims=["time","y","x"],
        coords={"time":model_tmid},
        attrs={"units":"m/yr²"}
    )

    print(f"✓ Built obs dv/dt. Non-missing intervals: {used}/{len(model_dt)}")

    return obs_dvxdt, obs_dvydt, obs_uncx, obs_uncy
    
    
# ==============================================================================
# COMBINED DATA FLATTENING
# ==============================================================================
def flatten_and_mask_combined(
    dhdt_obs, dhdt_sigma, dhdt_models,
    dvxdt_obs, dvydt_obs,
    dvxdt_unc, dvydt_unc,
    dvxdt_models, dvydt_models,
    VX_obs_dict=None, VY_obs_dict=None, obs_vel_years=None,
):
    """
    Flatten and combine thickness change (dh/dt) and velocity change (dv/dt) data.
    Also builds a speed vector using the EXACT SAME masking/order as the likelihood.
    THIS IS MASKING OBS UNC B/C IT IS HUGEEE lol.
    """
    
    THICK_UNC_THRESHOLD = 50.0
    VEL_UNC_THRESHOLD   = 10.0
    
    M = len(dhdt_models)

    # ==========================================================
    # THICKNESS CHANGE (dh/dt)
    # ==========================================================

    y_dhdt = dhdt_obs.values.reshape(-1)
    sig_dhdt = dhdt_sigma.values.reshape(-1)
    F_dhdt = np.stack([m.values.reshape(-1) for m in dhdt_models], axis=0)

    # mask_dhdt = np.isfinite(y_dhdt) & np.isfinite(sig_dhdt) #this needs to be filtered due to large unc
    mask_dhdt = (
    np.isfinite(y_dhdt) &
    np.isfinite(sig_dhdt) &
    (sig_dhdt < THICK_UNC_THRESHOLD)
)
    for m in range(M):
        mask_dhdt &= np.isfinite(F_dhdt[m, :])

    n_dhdt_total = y_dhdt.size

    y_dhdt = y_dhdt[mask_dhdt]
    sig_dhdt = sig_dhdt[mask_dhdt]
    F_dhdt = F_dhdt[:, mask_dhdt]

    # thickness speed = 0
    speed_dhdt = np.zeros_like(y_dhdt, dtype=float)

    n_dhdt = y_dhdt.size

    print(f"\ndh/dt data:")
    print(f"  Total points: {n_dhdt_total:,}")
    print(f"  Valid points: {n_dhdt:,} ({100*n_dhdt/n_dhdt_total:.2f}%)")

    # ==========================================================
    # VELOCITY CHANGE (dv/dt)
    # ==========================================================

    y_vel_list = []
    sig_vel_list = []
    F_vel_list = []
    speed_vel_list = []

    n_intervals = dvxdt_obs.sizes["time"]

    # build mean observed speed field once
    speed_mean = None
    if VX_obs_dict is not None and VY_obs_dict is not None and obs_vel_years is not None:
        speed_fields = []
        for year in obs_vel_years:
            if year in VX_obs_dict and year in VY_obs_dict:
                vx = VX_obs_dict[year]
                vy = VY_obs_dict[year]
                speed_fields.append(np.sqrt(vx**2 + vy**2))

        if len(speed_fields) == 0:
            raise RuntimeError("No velocity fields found for speed calculation")

        speed_mean = np.nanmean(np.stack(speed_fields), axis=0)

    for i in range(n_intervals):

        dvx_obs = dvxdt_obs.isel(time=i).values.reshape(-1)
        dvy_obs = dvydt_obs.isel(time=i).values.reshape(-1)

        dvx_err = dvxdt_unc.isel(time=i).values.reshape(-1)
        dvy_err = dvydt_unc.isel(time=i).values.reshape(-1)

        dvx_models = []
        dvy_models = []

        for m in range(M):
            dvx_models.append(dvxdt_models[m].isel(time=i).values.reshape(-1))
            dvy_models.append(dvydt_models[m].isel(time=i).values.reshape(-1))

        y_vel_interval = np.concatenate([dvx_obs, dvy_obs])
        sig_vel_interval = np.concatenate([dvx_err, dvy_err])

        F_vel_interval = np.zeros((M, len(y_vel_interval)))
        for m in range(M):
            F_vel_interval[m, :] = np.concatenate([dvx_models[m], dvy_models[m]])

        # speed repeated for vx and vy blocks
        if speed_mean is not None:
            speed_flat = speed_mean.reshape(-1)
            speed_interval = np.concatenate([speed_flat, speed_flat])
        else:
            speed_interval = np.full_like(y_vel_interval, np.nan, dtype=float)

        # mask_vel = np.isfinite(y_vel_interval) & np.isfinite(sig_vel_interval) & np.isfinite(speed_interval)
        mask_vel = (
            np.isfinite(y_vel_interval) &
            np.isfinite(sig_vel_interval) &
            np.isfinite(speed_interval) &
            (sig_vel_interval < VEL_UNC_THRESHOLD)
        )
        for m in range(M):
            mask_vel &= np.isfinite(F_vel_interval[m, :])

        n_vel_total = len(y_vel_interval)
        n_vel_valid = int(mask_vel.sum())

        valid_err = sig_vel_interval[mask_vel]

        print(f"  Interval {i}: {n_vel_valid:,}/{n_vel_total:,} valid dv/dt points")
        removed = n_vel_total - n_vel_valid
        print(f"      Removed high-uncertainty pixels: {removed:,}")

        if n_vel_valid > 0:
            print(f"      Uncertainty range: {np.nanmin(valid_err):.4f} - {np.nanmax(valid_err):.4f} m/yr²")
            print(f"      Uncertainty median: {np.nanmedian(valid_err):.4f} m/yr²")

        y_vel_list.append(y_vel_interval[mask_vel])
        sig_vel_list.append(sig_vel_interval[mask_vel])
        F_vel_list.append(F_vel_interval[:, mask_vel])
        speed_vel_list.append(speed_interval[mask_vel])

    if len(y_vel_list) > 0:
        y_vel = np.concatenate(y_vel_list)
        sig_vel = np.concatenate(sig_vel_list)
        F_vel = np.concatenate(F_vel_list, axis=1)
        speed_vel = np.concatenate(speed_vel_list)

        n_vel = y_vel.size

        print(f"\nTotal velocity change data: {n_vel:,} points")
        print(f"Overall dv/dt uncertainty: {np.nanmin(sig_vel):.4f} - {np.nanmax(sig_vel):.4f} m/yr²")
    else:
        y_vel = np.array([])
        sig_vel = np.array([])
        F_vel = np.zeros((M, 0))
        speed_vel = np.array([])
        n_vel = 0
        print("\n⚠️  No valid velocity change data found")

    # ==========================================================
    # COMBINE
    # ==========================================================

    y_combined = np.concatenate([y_dhdt, y_vel])
    sigma_combined = np.concatenate([sig_dhdt, sig_vel])
    F_combined = np.concatenate([F_dhdt, F_vel], axis=1)
    speed_combined = np.concatenate([speed_dhdt, speed_vel])

    print(f"\n{'='*70}")
    print("COMBINED DATA SUMMARY")
    print(f"{'='*70}")
    print(f"dh/dt points:    {n_dhdt:,}")
    print(f"  uncertainty:   {np.nanmin(sig_dhdt):.2f} - {np.nanmax(sig_dhdt):.2f} m/yr")
    print(f"dv/dt points:    {n_vel:,}")
    if n_vel > 0:
        print(f"  uncertainty:   {np.nanmin(sig_vel):.4f} - {np.nanmax(sig_vel):.4f} m/yr²")
    print(f"Total points:    {len(y_combined):,}")
    print(f"Ensemble size:   {M}")

    if not np.all(np.isfinite(speed_combined)):
        raise RuntimeError("speed_combined still contains NaNs after masking")

    return y_combined, sigma_combined, F_combined, speed_combined, n_dhdt, n_vel
# def flatten_and_mask_combined(
#     dhdt_obs, dhdt_sigma, dhdt_models,
#     dvxdt_obs, dvydt_obs,
#     dvxdt_unc, dvydt_unc,
#     dvxdt_models, dvydt_models,
# ):
#     """
#     Flatten and combine thickness change (dh/dt) and velocity change (dv/dt) data.
#     Uses propagated uncertainties for dv/dt.
#     """

#     M = len(dhdt_models)

#     # ==========================================================
#     # THICKNESS CHANGE (dh/dt)
#     # ==========================================================

#     y_dhdt = dhdt_obs.values.reshape(-1)
#     sig_dhdt = dhdt_sigma.values.reshape(-1)

#     F_dhdt = np.stack([m.values.reshape(-1) for m in dhdt_models], axis=0)

#     mask_dhdt = np.isfinite(y_dhdt) & np.isfinite(sig_dhdt)

#     for m in range(M):
#         mask_dhdt &= np.isfinite(F_dhdt[m, :])

#     n_dhdt_total = y_dhdt.size

#     y_dhdt = y_dhdt[mask_dhdt]
#     sig_dhdt = sig_dhdt[mask_dhdt]
#     F_dhdt = F_dhdt[:, mask_dhdt]

#     n_dhdt = y_dhdt.size

#     print(f"\ndh/dt data:")
#     print(f"  Total points: {n_dhdt_total:,}")
#     print(f"  Valid points: {n_dhdt:,} ({100*n_dhdt/n_dhdt_total:.2f}%)")

#     # ==========================================================
#     # VELOCITY CHANGE (dv/dt)
#     # ==========================================================

#     y_vel_list = []
#     sig_vel_list = []
#     F_vel_list = []

#     n_intervals = dvxdt_obs.sizes["time"]

#     for i in range(n_intervals):

#         # --- observations ---
#         dvx_obs = dvxdt_obs.isel(time=i).values.reshape(-1)
#         dvy_obs = dvydt_obs.isel(time=i).values.reshape(-1)

#         dvx_err = dvxdt_unc.isel(time=i).values.reshape(-1)
#         dvy_err = dvydt_unc.isel(time=i).values.reshape(-1)

#         # --- models ---
#         dvx_models = []
#         dvy_models = []

#         for m in range(M):

#             dvx_models.append(
#                 dvxdt_models[m].isel(time=i).values.reshape(-1)
#             )

#             dvy_models.append(
#                 dvydt_models[m].isel(time=i).values.reshape(-1)
#             )

#         # --- stack vx + vy components ---
#         y_vel_interval = np.concatenate([dvx_obs, dvy_obs])
#         sig_vel_interval = np.concatenate([dvx_err, dvy_err])

#         F_vel_interval = np.zeros((M, len(y_vel_interval)))

#         for m in range(M):

#             F_vel_interval[m, :] = np.concatenate(
#                 [dvx_models[m], dvy_models[m]]
#             )

#         # --- mask invalid ---
#         mask_vel = np.isfinite(y_vel_interval) & np.isfinite(sig_vel_interval)

#         for m in range(M):
#             mask_vel &= np.isfinite(F_vel_interval[m, :])

#         n_vel_total = len(y_vel_interval)
#         n_vel_valid = int(mask_vel.sum())

#         valid_err = sig_vel_interval[mask_vel]

#         print(
#             f"  Interval {i}: {n_vel_valid:,}/{n_vel_total:,} valid dv/dt points"
#         )

#         if n_vel_valid > 0:
#             print(
#                 f"      Uncertainty range: {np.nanmin(valid_err):.4f} - {np.nanmax(valid_err):.4f} m/yr²"
#             )
#             print(
#                 f"      Uncertainty median: {np.nanmedian(valid_err):.4f} m/yr²"
#             )

#         y_vel_list.append(y_vel_interval[mask_vel])
#         sig_vel_list.append(sig_vel_interval[mask_vel])
#         F_vel_list.append(F_vel_interval[:, mask_vel])

#     # ==========================================================
#     # COMBINE ALL VELOCITY INTERVALS
#     # ==========================================================

#     if len(y_vel_list) > 0:

#         y_vel = np.concatenate(y_vel_list)
#         sig_vel = np.concatenate(sig_vel_list)
#         F_vel = np.concatenate(F_vel_list, axis=1)

#         n_vel = y_vel.size

#         print(f"\nTotal velocity change data: {n_vel:,} points")

#         print(
#             f"Overall dv/dt uncertainty: {np.nanmin(sig_vel):.4f} - {np.nanmax(sig_vel):.4f} m/yr²"
#         )

#     else:

#         y_vel = np.array([])
#         sig_vel = np.array([])
#         F_vel = np.zeros((M, 0))
#         n_vel = 0

#         print("\n⚠️  No valid velocity change data found")

#     # ==========================================================
#     # COMBINE dh/dt + dv/dt
#     # ==========================================================

#     y_combined = np.concatenate([y_dhdt, y_vel])
#     sigma_combined = np.concatenate([sig_dhdt, sig_vel])

#     F_combined = np.concatenate([F_dhdt, F_vel], axis=1)

#     print(f"\n{'='*70}")
#     print("COMBINED DATA SUMMARY")
#     print(f"{'='*70}")

#     print(f"dh/dt points:    {n_dhdt:,}")

#     print(
#         f"  uncertainty:   {np.nanmin(sig_dhdt):.2f} - {np.nanmax(sig_dhdt):.2f} m/yr"
#     )

#     print(f"dv/dt points:    {n_vel:,}")

#     if n_vel > 0:

#         print(
#             f"  uncertainty:   {np.nanmin(sig_vel):.4f} - {np.nanmax(sig_vel):.4f} m/yr²"
#         )

#     print(f"Total points:    {len(y_combined):,}")
#     print(f"Ensemble size:   {M}")

#     return y_combined, sigma_combined, F_combined, n_dhdt, n_vel

# ==============================================================================
# LOAD + PREP
# ==============================================================================

# def load_and_prepare_data():

#     # ----------------------- PATHS -----------------------
#     OBS_THICKNESS_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/761_obs/761 elev"
#     THICKNESS_PATTERN = "elev_antarctica_elevation_*.nc"
#     OBS_THICKNESS_VAR = "height"
#     OBS_THICKNESS_RMSE_VAR = "absolute_elevation_rmse"

#     OBS_VELOCITY_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/761_obs/761 veloc"
#     VELOCITY_PATTERN = "vel_Antarctica_ice_velocity_*.nc"

#     OBS_VX_VAR = "VX"
#     OBS_VY_VAR = "VY"
#     OBS_VX_ERR_VAR = "ERRX"
#     OBS_VY_ERR_VAR = "ERRY"

#     # --------------------------------------------------
#     # LOAD OBS VELOCITY
#     # --------------------------------------------------

#     VX_obs, VY_obs, VX_ERR_obs, VY_ERR_obs, obs_vel_years = load_obs_velocity_yearly(
#         OBS_VELOCITY_DIR,
#         VELOCITY_PATTERN,
#         OBS_VX_VAR,
#         OBS_VY_VAR,
#         OBS_VX_ERR_VAR,
#         OBS_VY_ERR_VAR,
#     )

#     from pathlib import Path
#     import glob

#     # ----------------------- MODEL PATHS -----------------------

#     MODEL_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/ensemble_2/regridded"
#     MODEL_PATTERN = "*_regridded.nc"

#     MODEL_PATHS = sorted(glob.glob(MODEL_DIR + "/" + MODEL_PATTERN))

#     print("Found", len(MODEL_PATHS), "model files")

#     MODEL_THICKNESS_VAR = "h"
#     MODEL_VX_VAR = "ua"
#     MODEL_VY_VAR = "va"

#     # --------------------------------------------------
#     # LOAD FIRST MODEL FOR TIME AXIS
#     # --------------------------------------------------

#     print("=" * 70)
#     print("LOADING MODEL ENSEMBLE")
#     print("=" * 70)

#     ds0 = xr.open_dataset(MODEL_PATHS[0], engine="netcdf4", decode_times=False)

#     model_years0 = model_decimal_years_from_ds(ds0, time_name="time")

#     dhdt0, t0, tmid0, dt0 = compute_model_dhdt(
#         ds0,
#         model_years0,
#         model_thickness_var=MODEL_THICKNESS_VAR,
#     )

#     print(
#         f"Canonical model years: {model_years0[0]:.3f} .. {model_years0[-1]:.3f} (n={len(model_years0)})"
#     )
#     print(
#         f"Canonical dh/dt times: {float(tmid0.min()):.3f} .. {float(tmid0.max()):.3f} (n={len(tmid0)})"
#     )

#     # --------------------------------------------------
#     # LOAD MODELS
#     # --------------------------------------------------

#     ensemble_dhdt = []
#     vx_models_by_member = []
#     vy_models_by_member = []

#     for i, fp in enumerate(MODEL_PATHS, 1):

#         print("\n" + "=" * 70)
#         print(f"MODEL {i}/{len(MODEL_PATHS)}: {Path(fp).name}")
#         print("=" * 70)

#         ds = xr.open_dataset(fp, engine="netcdf4", decode_times=False)

#         model_years = model_decimal_years_from_ds(ds, time_name="time")

#         # ------------------------------
#         # dh/dt
#         # ------------------------------

#         dhdt, t, tmid, dt = compute_model_dhdt(
#             ds,
#             model_years,
#             model_thickness_var=MODEL_THICKNESS_VAR,
#         )

#         if dhdt.sizes["time"] == len(tmid0):

#             dhdt = dhdt.assign_coords(time=tmid0.astype(float))

#         else:

#             dhdt = dhdt.reindex(
#                 time=tmid0.astype(float),
#                 method="nearest",
#                 tolerance=1e-3,
#             )

#         ensemble_dhdt.append(dhdt)

#         # ------------------------------
#         # dv/dt
#         # ------------------------------

#         dvxdt, dvydt = compute_model_velocity_change(
#             ds,
#             model_years,
#             vx_var=MODEL_VX_VAR,
#             vy_var=MODEL_VY_VAR,
#         )

#         # vx_dict = {float(t): dvxdt.sel(time=t).values for t in dvxdt.time.values}
#         # vy_dict = {float(t): dvydt.sel(time=t).values for t in dvydt.time.values}

#         # vx_models_by_member.append(vx_dict)
#         # vy_models_by_member.append(vy_dict)
#         vx_models_by_member.append(dvxdt)
#         vy_models_by_member.append(dvydt)

#         ds.close()

#     # --------------------------------------------------
#     # OBS dh/dt
#     # --------------------------------------------------

#     print("\n" + "=" * 70)
#     print("COMPUTING OBS dh/dt ON MODEL INTERVALS")
#     print("=" * 70)

#     obs_dhdt, obs_unc, has_thickness_unc = compute_obs_dhdt_on_model_intervals(
#         obs_thickness_dir=OBS_THICKNESS_DIR,
#         thickness_pattern=THICKNESS_PATTERN,
#         obs_h_var=OBS_THICKNESS_VAR,
#         obs_rmse_var=OBS_THICKNESS_RMSE_VAR,
#         model_t=t0,
#         model_tmid=tmid0,
#         model_dt=dt0,
#     )

#     # --------------------------------------------------
#     # OBS dv/dt
#     # --------------------------------------------------

#     print("\n" + "=" * 70)
#     print("COMPUTING OBS dv/dt ON MODEL INTERVALS")
#     print("=" * 70)

#     obs_dvxdt, obs_dvydt, obs_uncx, obs_uncy = compute_obs_dvdt_on_model_intervals(
#         VX_obs,
#         VY_obs,
#         VX_ERR_obs,
#         VY_ERR_obs,
#         obs_vel_years,
#         t0,
#         tmid0,
#         dt0,
#     )

#     # --------------------------------------------------
#     # RETURN DATA
#     # --------------------------------------------------

#     return {

#         "obs_dhdt": obs_dhdt,
#         "obs_unc": obs_unc,
#         "has_thickness_unc": has_thickness_unc,
#         "ensemble_dhdt": ensemble_dhdt,

#         "obs_dvxdt": obs_dvxdt,
#         "obs_dvydt": obs_dvydt,
#         "obs_uncx": obs_uncx,
#         "obs_uncy": obs_uncy,

#         "VX_obs": VX_obs,
#         "VY_obs": VY_obs,
#         "VX_ERR_obs": VX_ERR_obs,
#         "VY_ERR_obs": VY_ERR_obs,
#         "obs_vel_years": obs_vel_years,

#         "vx_models": vx_models_by_member,
#         "vy_models": vy_models_by_member,

#         "M": len(ensemble_dhdt),
#     }


def load_and_prepare_data():

    # ----------------------- PATHS -----------------------
    OBS_THICKNESS_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/761_obs/761 elev"
    THICKNESS_PATTERN = "elev_antarctica_elevation_*.nc"
    OBS_THICKNESS_VAR = "height"
    OBS_THICKNESS_RMSE_VAR = "absolute_elevation_rmse"

    OBS_VELOCITY_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/761_obs/761 veloc"
    VELOCITY_PATTERN = "vel_Antarctica_ice_velocity_*.nc"

    OBS_VX_VAR = "VX"
    OBS_VY_VAR = "VY"
    OBS_VX_ERR_VAR = "ERRX"
    OBS_VY_ERR_VAR = "ERRY"

    # --------------------------------------------------
    # LOAD OBS VELOCITY
    # --------------------------------------------------
    VX_obs, VY_obs, VX_ERR_obs, VY_ERR_obs, obs_vel_years = load_obs_velocity_yearly(
        OBS_VELOCITY_DIR,
        VELOCITY_PATTERN,
        OBS_VX_VAR,
        OBS_VY_VAR,
        OBS_VX_ERR_VAR,
        OBS_VY_ERR_VAR,
    )

    from pathlib import Path
    import glob

    # ----------------------- MODEL PATHS -----------------------
    MODEL_DIR = "/Users/sp53972/Library/CloudStorage/Box-Box/Main/Projects/Modern Data Model Scoring to SLR/inputs/ch1_ensemble/ensemble_2/regridded"
    MODEL_PATTERN = "*_regridded.nc"

    MODEL_PATHS = sorted(glob.glob(MODEL_DIR + "/" + MODEL_PATTERN))

    print("Found", len(MODEL_PATHS), "model files")

    MODEL_THICKNESS_VAR = "h"
    MODEL_VX_VAR = "ua"
    MODEL_VY_VAR = "va"

    # --------------------------------------------------
    # HELPER: clean bad time entries and mask dataset
    # --------------------------------------------------
    def _clean_and_mask_time(ds, time_name="time"):
        time_var = ds[time_name]
        t_raw = np.asarray(time_var.values, dtype=float)

        fill_value = time_var.attrs.get("_FillValue", None)
        missing_value = time_var.attrs.get("missing_value", None)

        if fill_value is not None:
            t_raw[t_raw == fill_value] = np.nan
        if missing_value is not None:
            t_raw[t_raw == missing_value] = np.nan

        # catch absurd NetCDF placeholders
        t_raw[np.abs(t_raw) > 1e20] = np.nan

        good_time = np.isfinite(t_raw)

        if not np.any(good_time):
            raise ValueError("No valid time values found in model file")

        n_bad = int(np.sum(~good_time))
        if n_bad > 0:
            print(f"  Warning: removed {n_bad} invalid time value(s)")

        ds = ds.isel(time=good_time)

        return ds

    # --------------------------------------------------
    # LOAD FIRST MODEL FOR CANONICAL TIME AXIS
    # --------------------------------------------------
    print("=" * 70)
    print("LOADING MODEL ENSEMBLE")
    print("=" * 70)

    ds0 = xr.open_dataset(MODEL_PATHS[0], engine="netcdf4", decode_times=False)
    ds0 = _clean_and_mask_time(ds0, time_name="time")

    model_years0 = model_decimal_years_from_ds(ds0, time_name="time")

    dhdt0, t0, tmid0, dt0 = compute_model_dhdt(
        ds0,
        model_years0,
        model_thickness_var=MODEL_THICKNESS_VAR,
    )

    print(
        f"Canonical model years: {model_years0[0]:.3f} .. {model_years0[-1]:.3f} (n={len(model_years0)})"
    )
    print(
        f"Canonical dh/dt times: {float(tmid0.min()):.3f} .. {float(tmid0.max()):.3f} (n={len(tmid0)})"
    )

    ds0.close()

    # --------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------
    ensemble_dhdt = []
    vx_models_by_member = []
    vy_models_by_member = []

    for i, fp in enumerate(MODEL_PATHS, 1):

        print("\n" + "=" * 70)
        print(f"MODEL {i}/{len(MODEL_PATHS)}: {Path(fp).name}")
        print("=" * 70)

        ds = xr.open_dataset(fp, engine="netcdf4", decode_times=False)
        ds = _clean_and_mask_time(ds, time_name="time")

        model_years = model_decimal_years_from_ds(ds, time_name="time")

        # ------------------------------
        # dh/dt
        # ------------------------------
        dhdt, t, tmid, dt = compute_model_dhdt(
            ds,
            model_years,
            model_thickness_var=MODEL_THICKNESS_VAR,
        )

        if dhdt.sizes["time"] == len(tmid0):
            dhdt = dhdt.assign_coords(time=tmid0.astype(float))
        else:
            dhdt = dhdt.reindex(
                time=tmid0.astype(float),
                method="nearest",
                tolerance=1e-3,
            )

        ensemble_dhdt.append(dhdt)

        # ------------------------------
        # dv/dt
        # ------------------------------
        dvxdt, dvydt = compute_model_velocity_change(
            ds,
            model_years,
            vx_var=MODEL_VX_VAR,
            vy_var=MODEL_VY_VAR,
        )

        if dvxdt.sizes["time"] == len(tmid0):
            dvxdt = dvxdt.assign_coords(time=tmid0.astype(float))
            dvydt = dvydt.assign_coords(time=tmid0.astype(float))
        else:
            dvxdt = dvxdt.reindex(
                time=tmid0.astype(float),
                method="nearest",
                tolerance=1e-3,
            )
            dvydt = dvydt.reindex(
                time=tmid0.astype(float),
                method="nearest",
                tolerance=1e-3,
            )

        vx_models_by_member.append(dvxdt)
        vy_models_by_member.append(dvydt)

        ds.close()

    # --------------------------------------------------
    # OBS dh/dt
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING OBS dh/dt ON MODEL INTERVALS")
    print("=" * 70)

    obs_dhdt, obs_unc, has_thickness_unc = compute_obs_dhdt_on_model_intervals(
        obs_thickness_dir=OBS_THICKNESS_DIR,
        thickness_pattern=THICKNESS_PATTERN,
        obs_h_var=OBS_THICKNESS_VAR,
        obs_rmse_var=OBS_THICKNESS_RMSE_VAR,
        model_t=t0,
        model_tmid=tmid0,
        model_dt=dt0,
    )

    # --------------------------------------------------
    # OBS dv/dt
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING OBS dv/dt ON MODEL INTERVALS")
    print("=" * 70)

    obs_dvxdt, obs_dvydt, obs_uncx, obs_uncy = compute_obs_dvdt_on_model_intervals(
        VX_obs,
        VY_obs,
        VX_ERR_obs,
        VY_ERR_obs,
        obs_vel_years,
        t0,
        tmid0,
        dt0,
    )

    # --------------------------------------------------
    # RETURN DATA
    # --------------------------------------------------
    return {
        "obs_dhdt": obs_dhdt,
        "obs_unc": obs_unc,
        "has_thickness_unc": has_thickness_unc,
        "ensemble_dhdt": ensemble_dhdt,

        "obs_dvxdt": obs_dvxdt,
        "obs_dvydt": obs_dvydt,
        "obs_uncx": obs_uncx,
        "obs_uncy": obs_uncy,

        "VX_obs": VX_obs,
        "VY_obs": VY_obs,
        "VX_ERR_obs": VX_ERR_obs,
        "VY_ERR_obs": VY_ERR_obs,
        "obs_vel_years": obs_vel_years,

        "vx_models": vx_models_by_member,
        "vy_models": vy_models_by_member,

        "M": len(ensemble_dhdt),
    }

def prepare_for_inference(data_dict):
    obs_dhdt = data_dict["obs_dhdt"]
    obs_unc = data_dict["obs_unc"]
    has_unc = data_dict["has_thickness_unc"]
    ens_dhdt = data_dict["ensemble_dhdt"]
    M = data_dict["M"]

    # Fill missing thickness uncertainties
    if (not has_unc) or np.all(np.isnan(obs_unc.values)):
        print("\n⚠️ obs_unc missing -> using constant 20 m/yr")
        obs_unc_filled = xr.full_like(obs_dhdt, 20.0)
    else:
        unc_vals = obs_unc.values
        finite = np.isfinite(unc_vals)
        if not np.any(finite):
            obs_unc_filled = xr.full_like(obs_dhdt, 20.0)
        else:
            fill_val = float(np.nanmedian(unc_vals))
            obs_unc_filled = obs_unc.where(np.isfinite(obs_unc), other=fill_val)

    # Remove time slices with all-NaN obs
    # good_time = np.isfinite(obs_dhdt.mean(dim=("y", "x"), skipna=True))
    # obs_dhdt = obs_dhdt.sel(time=obs_dhdt.time[good_time])
    # obs_unc_filled = obs_unc_filled.sel(time=obs_unc_filled.time[good_time])
    # ens_dhdt = [m.sel(time=obs_dhdt.time) for m in ens_dhdt]

    print(f"\nThickness time points kept: {obs_dhdt.sizes['time']}")

    # Flatten and combine
# Flatten and combine
    # y_inf, sigma_inf, F_inf, n_dhdt, n_vel = flatten_and_mask_combined(
    #     dhdt_obs=obs_dhdt,
    #     dhdt_sigma=obs_unc_filled,
    #     dhdt_models=ens_dhdt,
    
    #     dvxdt_obs=data_dict["obs_dvxdt"],
    #     dvydt_obs=data_dict["obs_dvydt"],
    
    #     dvxdt_unc=data_dict["obs_uncx"],
    #     dvydt_unc=data_dict["obs_uncy"],
    
    #     dvxdt_models=data_dict["vx_models"],
    #     dvydt_models=data_dict["vy_models"],
    # )
    y_inf, sigma_inf, F_inf, speed, n_dhdt, n_vel = flatten_and_mask_combined(
    dhdt_obs=obs_dhdt,
    dhdt_sigma=obs_unc_filled,
    dhdt_models=ens_dhdt,

    dvxdt_obs=data_dict["obs_dvxdt"],
    dvydt_obs=data_dict["obs_dvydt"],

    dvxdt_unc=data_dict["obs_uncx"],
    dvydt_unc=data_dict["obs_uncy"],

    dvxdt_models=data_dict["vx_models"],
    dvydt_models=data_dict["vy_models"],

    VX_obs_dict=data_dict["VX_obs"],
    VY_obs_dict=data_dict["VY_obs"],
    obs_vel_years=data_dict["obs_vel_years"],
    )

    # # --- BUILD SPEED VECTOR ---
    # # dh/dt points: speed = 0 (no velocity-dependent model error for thickness)
    # speed_dhdt = np.zeros(n_dhdt, dtype=float)

    # # Velocity points: |v| = sqrt(vx^2 + vy^2), same masking as flatten_and_mask_combined
    # speed_vel_list = []
    # obs_vel_years = data_dict["obs_vel_years"]

    # for year in obs_vel_years:
    #     VX_obs = data_dict["VX_obs"]
    #     VY_obs = data_dict["VY_obs"]
    #     VX_ERR = data_dict["VX_ERR_obs"]
    #     VY_ERR = data_dict["VY_ERR_obs"]

    #     if year not in VX_obs or year not in VY_obs:
    #         continue

    #     vx_obs = VX_obs[year].reshape(-1)
    #     vy_obs = VY_obs[year].reshape(-1)
    #     vx_err = VX_ERR[year].reshape(-1)
    #     vy_err = VY_ERR[year].reshape(-1)

    #     # speed repeated for vx block and vy block (same spatial location)
    #     speed_year = np.concatenate([
    #         np.sqrt(vx_obs**2 + vy_obs**2),
    #         np.sqrt(vx_obs**2 + vy_obs**2),
    #     ])

    #     y_vel_year = np.concatenate([vx_obs, vy_obs])
    #     sig_vel_year = np.concatenate([vx_err, vy_err])

    #     # Reproduce the exact same mask as in flatten_and_mask_combined
    #     mask_vel = np.isfinite(y_vel_year) & np.isfinite(sig_vel_year)
    #     skip_year = False
    #     for m in range(M):
    #         if data_dict["vx_models"][m][year] is None or data_dict["vy_models"][m][year] is None:
    #             skip_year = True
    #             break
    #         vx_m = data_dict["vx_models"][m][year].reshape(-1)
    #         vy_m = data_dict["vy_models"][m][year].reshape(-1)
    #         F_year = np.concatenate([vx_m, vy_m])
    #         mask_vel &= np.isfinite(F_year)

    #     if skip_year:
    #         continue

    #     speed_vel_list.append(speed_year[mask_vel])

    # speed_vel = np.concatenate(speed_vel_list) if speed_vel_list else np.array([])
    # speed = np.concatenate([speed_dhdt, speed_vel])

    # assert speed.size == y_inf.size, (
    #     f"speed size {speed.size} != y_obs size {y_inf.size}. "
    #     "Check that masking logic here matches flatten_and_mask_combined exactly."
    # )

    # # return {
    # #     "y_obs": y_inf,
    # #     "sigma_obs": sigma_inf,
    # #     "F": F_inf,
    # #     "speed": speed,
    # #     "M": M,
    # #     "n_obs": y_inf.size,
    # #     "n_dhdt": n_dhdt,
    # #     "n_vel": n_vel,
    # # } this is a temporary change to make it run faster
    
    # # --- BUILD SPEED VECTOR ---
    
    # # dh/dt points: speed = 0 (thickness uncertainty independent of speed)
    # speed_dhdt = np.zeros(n_dhdt, dtype=float)
    
    # # Compute mean observed speed field
    # VX_obs = data_dict["VX_obs"]
    # VY_obs = data_dict["VY_obs"]
    
    # speed_fields = []
    
    # for year in data_dict["obs_vel_years"]:
        
    #     if year not in VX_obs or year not in VY_obs:
    #         continue
        
    #     vx = VX_obs[year]
    #     vy = VY_obs[year]
    
    #     speed = np.sqrt(vx**2 + vy**2)
    
    #     speed_fields.append(speed)
    
    # # mean speed across years
    # speed_mean = np.nanmean(np.stack(speed_fields), axis=0)
    
    # # flatten
    # speed_flat = speed_mean.reshape(-1)
    
    # # repeat for vx and vy components
    # speed_vel = np.concatenate([speed_flat, speed_flat])
    
    # # tile for each time interval used in dv/dt
    # n_repeat = int(np.ceil(n_vel / speed_vel.size))
    # speed_vel = np.tile(speed_vel, n_repeat)[:n_vel]
    
    # speed = np.concatenate([speed_dhdt, speed_vel])
    
    # assert speed.size == y_inf.size
    # --- BUILD SPEED VECTOR ---
    
    # speed_dhdt = np.zeros(n_dhdt, dtype=float)
    
    # VX_obs = data_dict["VX_obs"]
    # VY_obs = data_dict["VY_obs"]
    
    # speed_fields = []
    
    # for year in VX_obs.keys():   # <-- use dictionary keys
    
    #     vx = VX_obs[year]
    #     vy = VY_obs[year]
    
    #     speed_fields.append(np.sqrt(vx**2 + vy**2))
    
    # # safety check
    # if len(speed_fields) == 0:
    #     raise RuntimeError("No velocity fields found for speed calculation")
    
    # speed_mean = np.nanmean(np.stack(speed_fields), axis=0)
    
    # speed_flat = speed_mean.reshape(-1)
    
    # speed_vel = np.concatenate([speed_flat, speed_flat])
    
    # n_repeat = int(np.ceil(n_vel / speed_vel.size))
    # speed_vel = np.tile(speed_vel, n_repeat)[:n_vel]
    
    # speed = np.concatenate([speed_dhdt, speed_vel])
        
    # --------------------------------------------------
# RANDOM SUBSAMPLING FOR MCMC SPEED
# --------------------------------------------------
# --------------------------------------------------
# RANDOM SUBSAMPLING FOR MCMC SPEED
# --------------------------------------------------

    MAX_POINTS = 20000
    n_total = y_inf.size
    
    if n_total > MAX_POINTS:
    
        print("\nSubsampling observations for faster inference")
        print(f"Original points: {n_total:,}")
    
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, MAX_POINTS, replace=False)
    
        y_inf = y_inf[idx]
        sigma_inf = sigma_inf[idx]
        F_inf = F_inf[:, idx]
        speed = speed[idx]
    
        print(f"Using {MAX_POINTS:,} randomly sampled points")
    
        is_thick = idx < n_dhdt
        n_dhdt_new = int(np.sum(is_thick))
        n_vel_new = int(MAX_POINTS - n_dhdt_new)
    
    else:
        n_dhdt_new = n_dhdt
        n_vel_new = n_vel
    
    return {
        "y_obs": y_inf,
        "sigma_obs": sigma_inf,
        "F": F_inf,
        "speed": speed,
        "M": M,
        "n_obs": y_inf.size,
        "n_dhdt": n_dhdt_new,
        "n_vel": n_vel_new,
    }

# BAYESIAN MODEL
# ==============================================================================

import numpy as np
import pymc as pm
VEL_SCALE = 1000.0
def build_model_proposal(data):
    """
    Equal weighting of thickness and velocity log-likelihoods.
    Scales likelihood by number of observations to avoid weight collapse.
    """
    y       = data["y_obs"].astype(float)
    sig_obs = data["sigma_obs"].astype(float)
    speed   = data["speed"].astype(float)
    F       = data["F"].astype(float)
    M       = int(data["M"])
    n_thick = int(data["n_dhdt"])
    n_vel   = int(data["n_vel"])

    idx_thick = slice(0, n_thick)
    idx_vel   = slice(n_thick, n_thick + n_vel)

    N = float(y.size)

    with pm.Model() as model:
        # sigma_base_thick = pm.HalfNormal("sigma_base_thick", sigma=2.0) #tighter prior on uncertainty 
        # #sigma_base_thick = pm.HalfNormal("sigma_base_thick", sigma=30.0) #really wide prior on model uncertainty
        # beta_thick       = pm.HalfNormal("beta_thick", sigma=0.5)
        # sigma_base_vel   = pm.HalfNormal("sigma_base_vel", sigma=20.0)
        # beta_vel         = pm.HalfNormal("beta_vel", sigma=0.5)
        
        ###PRIORS: the ones above were way too large lol, residual std was huge from the diagnostics at the end 
        sigma_base_thick = pm.HalfNormal("sigma_base_thick", sigma=0.5)
        sigma_base_vel   = pm.HalfNormal("sigma_base_vel", sigma=0.6)
        
        beta_thick = pm.HalfNormal("beta_thick", sigma=0.1)
        beta_vel   = pm.HalfNormal("beta_vel", sigma=0.1)

        sigma_model_thick = sigma_base_thick * pm.math.sqrt(1.0 + beta_thick * speed[idx_thick])
        sigma_model_vel   = sigma_base_vel   * pm.math.sqrt(1.0 + beta_vel   * speed[idx_vel])

        sigma_tot_thick = pm.math.sqrt(sig_obs[idx_thick]**2 + sigma_model_thick**2)
        sigma_tot_vel   = pm.math.sqrt(sig_obs[idx_vel]**2   + sigma_model_vel**2)

        logL_thick = []
        logL_vel   = []
        logL_comb  = []

        for m in range(M):
            r = y - F[m, :]

            ll_th = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_tot_thick), r[idx_thick]).sum()
            ll_v  = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_tot_vel),   r[idx_vel]).sum()

            logL_thick.append(ll_th)
            logL_vel.append(ll_v)
            logL_comb.append(ll_th + ll_v)

        logL_thick = pm.math.stack(logL_thick)
        logL_vel   = pm.math.stack(logL_vel)
        logL       = pm.math.stack(logL_comb)

        # scaled log-likelihood per observation
        logL_thick_scaled = logL_thick / N
        logL_vel_scaled   = logL_vel / N
        logL_scaled       = logL / N

        pm.Deterministic("logL_thick", logL_thick)
        pm.Deterministic("logL_vel", logL_vel)
        pm.Deterministic("logL", logL)
        pm.Deterministic("logL_scaled", logL_scaled)

        # use scaled likelihood in posterior
        pm.Potential("joint_loglik", logL_scaled.sum())

        # use scaled likelihood for weights too
        w_unnorm = pm.math.exp(logL_scaled - pm.math.max(logL_scaled))
        w = w_unnorm / pm.math.sum(w_unnorm)
        pm.Deterministic("w", w)

    return model

def run_mcmc(model, draws=500, tune=1000, chains=4, target_accept=0.95):
    print("\n" + "=" * 70)
    print("RUNNING MCMC")
    print("=" * 70)
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
        )
    return trace



def compute_model_weights(trace, data):
    """Compute weights from scaled log-likelihoods using the same sigma_model form as the Bayesian model."""
    y = data["y_obs"]
    sig_obs = data["sigma_obs"]
    F = data["F"]
    speed = data["speed"]
    M = data["M"]
    n_dhdt = data["n_dhdt"]

    N = float(y.size)

    post = trace.posterior

    sigma_base_thick = float(post["sigma_base_thick"].mean(dim=("chain", "draw")).values)
    beta_thick       = float(post["beta_thick"].mean(dim=("chain", "draw")).values)
    sigma_base_vel   = float(post["sigma_base_vel"].mean(dim=("chain", "draw")).values)
    beta_vel         = float(post["beta_vel"].mean(dim=("chain", "draw")).values)

    is_thick = np.arange(y.size) < n_dhdt

    sigma_model = np.empty_like(y, dtype=float)
    sigma_model[is_thick] = sigma_base_thick * np.sqrt(1.0 + beta_thick * speed[is_thick])
    sigma_model[~is_thick] = sigma_base_vel * np.sqrt(1.0 + beta_vel * speed[~is_thick])

    sigma_tot = np.sqrt(sig_obs**2 + sigma_model**2)

    loglik = np.zeros(M, dtype=float)
    for m in range(M):
        r = y - F[m, :]
        loglik[m] = -0.5 * np.sum((r**2 / sigma_tot**2) + np.log(2.0 * np.pi * sigma_tot**2))

    loglik_scaled = loglik / N

    w = np.exp(loglik_scaled - loglik_scaled.max())
    w = w / w.sum()

    return w, loglik_scaled


import pandas as pd

def save_model_weights(weights, loglik):

    model_ids = [f"run{i+1}" for i in range(len(weights))]

    df = pd.DataFrame({
        "model_id": model_ids,
        "weight": weights,
        "log_likelihood": loglik
    })

    df = df.sort_values("weight", ascending=False)

    df.to_csv("model_weights_table.csv", index=False)

    print("\nSaved model_weights_table.csv")

    return df
# ==============================================================================
# MAIN
# ==============================================================================



def main():
    print("=" * 70)
    print("BAYESIAN MODEL - THICKNESS + VELOCITY")
    print("=" * 70)

    raw = load_and_prepare_data()
    data = prepare_for_inference(raw)

    print("\n" + "=" * 70)
    print("READY FOR INFERENCE")
    print("=" * 70)

    model = build_model_proposal(data)
    trace = run_mcmc(model, draws=500, tune=1000, chains=4, target_accept=0.95)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS")
    print("=" * 70)
    print(az.summary(trace, hdi_prob=0.95))

    weights, loglik = compute_model_weights(trace, data)
    print("\nMODEL WEIGHTS")
    for i in np.argsort(weights)[::-1]:
        print(f"  Model {i+1}: w = {weights[i]:.4f}")

    az.plot_trace(trace, compact=True)
    plt.tight_layout()
    plt.savefig("trace_plots_combined.png", dpi=200, bbox_inches="tight")
    plt.close()

    az.plot_posterior(trace, hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig("posterior_distributions_combined.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved: trace_plots_combined.png, posterior_distributions_combined.png")
    return trace, weights, data, raw


# if __name__ == "__main__":
#     trace, weights, data, raw = main()
    
    
    
####n RESULTS 






# ==============================================================================
# RESIDUAL MAPS - Thickness and Velocity
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

    #sanity check plots for obs data coverage on the model
    # ==============================================================================
# PIXEL COVERAGE MAPS (WHERE THE MODEL IS BEING EVALUATED)
# ==============================================================================

def plot_pixel_coverage(raw):
        """
        Plot which pixels are used in the likelihood after masking.
        Shows spatial coverage of:
            1) dh/dt pixels
            2) velocity pixels for each year
        """
    
        obs_dhdt = raw["obs_dhdt"]
        VX_obs = raw["VX_obs"]
        VY_obs = raw["VY_obs"]
        VX_ERR = raw["VX_ERR_obs"]
        VY_ERR = raw["VY_ERR_obs"]
        obs_vel_years = raw["obs_vel_years"]
    
        M = raw["M"]
    
        print("\nGenerating pixel coverage maps...")
    
       
        # ==========================================================
        # THICKNESS COVERAGE BY YEAR
        # ==========================================================

        obs_dhdt = raw["obs_dhdt"]
        
        years = obs_dhdt.time.values

        # keep only times with real elevation observations
        valid_times = np.isfinite(obs_dhdt.mean(dim=("y","x"), skipna=True))
        years = obs_dhdt.time.values[valid_times]
        n_years = len(years)
        
        cols = min(4, n_years)
        rows = int(np.ceil(n_years / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, t in enumerate(years):
        
            dh = obs_dhdt.sel(time=t).values
        
            mask = np.isfinite(dh)
        
            axes[i].imshow(mask, origin="lower", cmap="viridis")
        
            axes[i].set_title(f"Thickness Pixels Used\n{float(t):.1f}")
            axes[i].axis("off")
        
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle("Thickness Observation Coverage by Time Step")
        plt.tight_layout()
        
        plt.savefig("pixel_coverage_thickness_by_year.png", dpi=300)
        
        print("✓ Saved pixel_coverage_thickness_by_year.png")
        
        plt.show()
    
     # ==========================================================
# VELOCITY CHANGE COVERAGE BY MODEL INTERVAL
# ==========================================================

        obs_dvxdt = raw["obs_dvxdt"]
        obs_dvydt = raw["obs_dvydt"]
        
        times = obs_dvxdt.time.values
        
        if len(times) == 0:
            print("No valid velocity change intervals")
            return
        
        cols = min(4, len(times))
        rows = int(np.ceil(len(times) / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, t in enumerate(times):
        
            dvx = obs_dvxdt.sel(time=t).values
            dvy = obs_dvydt.sel(time=t).values
        
            mask = np.isfinite(dvx) & np.isfinite(dvy)
        
            ax = axes[i]
            im = ax.imshow(mask, origin="lower", cmap="viridis")
        
            ax.set_title(f"Velocity Change Pixels Used\n{float(t):.1f}")
            ax.axis("off")
        
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        
        fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Used Pixel")
        
        plt.suptitle("Pixels Used for Velocity Change Likelihood")
        plt.tight_layout()
        
        plt.savefig("pixel_coverage_velocity.png", dpi=300)
        
        print("✓ Saved pixel_coverage_velocity.png")
        
        plt.show()
# ==============================================================================
# FATAL PIXEL MAPS (LIKELIHOOD OUTLIERS FOR dh/dt)
# ==============================================================================

def create_fatal_pixel_maps(raw, trace):
    """
    Fatal-pixel map for dh/dt using the same uncertainty model
    as the Bayesian likelihood.

    For dh/dt in the current HBM:
        sigma_model_thick = sigma_base_thick
    because speed_dhdt = 0 in prepare_for_inference().
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    M = raw["M"]

    post = trace.posterior
    sigma_base_thick = float(post["sigma_base_thick"].mean(dim=("chain", "draw")).values)

    print("\n" + "=" * 70)
    print("FATAL PIXEL MAPS (dh/dt)")
    print("=" * 70)
    print(f"sigma_base_thick = {sigma_base_thick:.3f}")

    obs_all = raw["obs_dhdt"]
    obs_unc_all = raw["obs_unc"]

    # Fill missing obs uncertainty with median finite value
    unc_vals = obs_unc_all.values
    finite_unc = np.isfinite(unc_vals)
    fill_unc = float(np.nanmedian(unc_vals[finite_unc])) if np.any(finite_unc) else 20.0
    obs_unc_all = obs_unc_all.where(np.isfinite(obs_unc_all), other=fill_unc)

    fig, axes = plt.subplots(1, M, figsize=(6 * M, 5), constrained_layout=True)
    if M == 1:
        axes = [axes]

    cmap = mcolors.ListedColormap(["black", "orange", "red"])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    print("\nFATAL PIXEL STATISTICS")
    print("-" * 70)

    for i in range(M):
        model_all = raw["ensemble_dhdt"][i]

        # Track worst mismatch across time at each pixel
        max_abs_z = np.full(obs_all.isel(time=0).shape, np.nan)

        for t in obs_all.time.values:
            obs = obs_all.sel(time=t).values
            obs_unc = obs_unc_all.sel(time=t).values
            model = model_all.sel(time=t).values

            sigma_tot = np.sqrt(obs_unc**2 + sigma_base_thick**2)

            valid = np.isfinite(obs) & np.isfinite(model) & np.isfinite(sigma_tot) & (sigma_tot > 0)
            z = np.full(obs.shape, np.nan)
            z[valid] = (obs[valid] - model[valid]) / sigma_tot[valid]

            if np.all(~valid):
                continue

            if np.all(np.isnan(max_abs_z)):
                max_abs_z = np.abs(z)
            else:
                max_abs_z = np.fmax(max_abs_z, np.abs(z))

        valid = np.isfinite(max_abs_z)

        fatal = np.full(max_abs_z.shape, np.nan)
        fatal[valid] = 0
        fatal[valid & (max_abs_z > 2)] = 1
        fatal[valid & (max_abs_z > 3)] = 2

        n_valid = int(np.sum(valid))
        n_bad = int(np.sum(valid & (max_abs_z > 2)))
        n_fatal = int(np.sum(valid & (max_abs_z > 3)))

        print(f"Model {i+1}: valid={n_valid:,}, |z|>2={n_bad:,}, |z|>3={n_fatal:,}")

        im = axes[i].imshow(fatal, cmap=cmap, norm=norm, origin="lower")
        axes[i].set_title(
            f"Model {i+1}\n|z|>2: {n_bad:,}\n|z|>3: {n_fatal:,}",
            fontsize=11
        )
        axes[i].axis("off")

    fig.colorbar(
        im,
        ax=axes,
        shrink=0.65,
        label="0 = OK   |   1 = 2 < |z| ≤ 3   |   2 = |z| > 3"
    )

    plt.suptitle("Fatal Pixels for dh/dt Likelihood", fontsize=14)
    plt.savefig("fatal_pixels_dhdt.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Saved: fatal_pixels_dhdt.png")

def make_dhdt_diagnostic_figure(raw, trace, model_index=0):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print("\nGenerating dh/dt diagnostic figure...")

    # ------------------------------------
    # posterior parameters
    # ------------------------------------

    post = trace.posterior
    sigma_base = float(post["sigma_base_thick"].mean(dim=("chain","draw")).values)

    # ------------------------------------
    # average fields
    # ------------------------------------

    # obs = raw["obs_dhdt"].mean(dim="time", skipna=True).values
    # model = raw["ensemble_dhdt"][model_index].mean(dim="time", skipna=True).values
    # obs_unc = raw["obs_unc"].mean(dim="time", skipna=True).values
    obs = np.nanmean(raw["obs_dhdt"].values, axis=0)
    model = np.nanmean(raw["ensemble_dhdt"][model_index].values, axis=0)
    obs_unc = np.nanmean(raw["obs_unc"].values, axis=0)
    # ------------------------------------
    # residuals
    # ------------------------------------

    residual = obs - model

    # ------------------------------------
    # sigma_total
    # ------------------------------------

    sigma_tot = np.sqrt(obs_unc**2 + sigma_base**2)

    # ------------------------------------
    # z score
    # ------------------------------------

    z = residual / sigma_tot

    # ------------------------------------
    # fatal pixel classification
    # ------------------------------------

    # fatal = np.zeros_like(z)

    # fatal[np.abs(z) > 2] = 1
    # fatal[np.abs(z) > 3] = 2
    fatal = np.full_like(z, np.nan)

    fatal[(np.abs(z) <= 2)] = 0
    fatal[(np.abs(z) > 2)] = 1
    fatal[(np.abs(z) > 3)] = 2

    # ------------------------------------
    # plotting
    # ------------------------------------

    fig, ax = plt.subplots(2,2, figsize=(12,10))

    # observed
    im0 = ax[0,0].imshow(obs, cmap="RdBu_r", vmin=-3, vmax=3)
    ax[0,0].set_title("Observed dh/dt (m/yr)")
    ax[0,0].axis("off")

    # model
    im1 = ax[0,1].imshow(model, cmap="RdBu_r", vmin=-3, vmax=3)
    ax[0,1].set_title(f"Model dh/dt (Model {model_index+1})")
    ax[0,1].axis("off")

    # residual
    im2 = ax[1,0].imshow(residual, cmap="RdBu_r", vmin=-2, vmax=2)
    ax[1,0].set_title("Residual (Obs − Model)")
    ax[1,0].axis("off")

    # fatal pixels
    cmap = mcolors.ListedColormap(["black","orange","red"])
    norm = mcolors.BoundaryNorm([0,1,2,3], cmap.N)

    im3 = ax[1,1].imshow(fatal, cmap=cmap, norm=norm)
    ax[1,1].set_title("Fatal Pixels (Likelihood)")
    ax[1,1].axis("off")

    # colorbars
    fig.colorbar(im0, ax=ax[0,0], fraction=0.046)
    fig.colorbar(im1, ax=ax[0,1], fraction=0.046)
    fig.colorbar(im2, ax=ax[1,0], fraction=0.046)

    cbar = fig.colorbar(im3, ax=ax[1,1], fraction=0.046)
    cbar.set_label("0 = OK | 1 = |z|>2 | 2 = |z|>3")

    plt.suptitle("Antarctic Ice Thickness Change Model Evaluation", fontsize=16)

    plt.tight_layout()

    plt.savefig("dhdt_model_diagnostics.png", dpi=300)

    print("✓ Saved: dhdt_model_diagnostics.png")

    plt.show()


#residual maps================================================================
def create_residual_maps(raw, weights):
    """
    Create residual maps for thickness (dh/dt) and velocity (vx, vy) 
    for each model in the ensemble.
    
    Parameters:
    -----------
    raw : dict
        Output from load_and_prepare_data() containing obs and model data
    weights : array
        Bayesian weights for each model
    """
    M = raw["M"]
    
    # =========================================================================
    # 1. THICKNESS CHANGE (dh/dt) RESIDUALS
    # =========================================================================
    
    # Compute time-averaged obs and model dh/dt
    obs_dhdt_mean = raw["obs_dhdt"].mean(dim="time", skipna=True).values
    
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5), constrained_layout=True)
    if M == 1:
        axes = [axes]
    
    # Color normalization for dh/dt residuals
    norm_dhdt = mcolors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    
    print("="*70)
    print("THICKNESS CHANGE (dh/dt) RESIDUAL STATISTICS")
    print("="*70)
    
    for i in range(M):
        model_dhdt_mean = raw["ensemble_dhdt"][i].mean(dim="time", skipna=True).values
        residual = obs_dhdt_mean - model_dhdt_mean
        
        # Statistics
        valid_resid = residual[np.isfinite(residual)]
        rmse = np.sqrt(np.mean(valid_resid**2))
        bias = np.mean(valid_resid)
        
        print(f"\nModel {i+1} (Weight: {weights[i]:.4f}):")
        print(f"  RMSE:   {rmse:.3f} m/yr")
        print(f"  Bias:   {bias:.3f} m/yr")
        print(f"  Median: {np.median(valid_resid):.3f} m/yr")
        print(f"  95th percentile: {np.percentile(np.abs(valid_resid), 95):.3f} m/yr")
        
        # Plot
        im = axes[i].imshow(residual, cmap="RdBu_r", norm=norm_dhdt, origin='lower')
        axes[i].set_title(f"Model {i+1}\nWeight: {weights[i]:.4f}\nRMSE: {rmse:.2f} m/yr", 
                         fontsize=12)
        axes[i].axis("off")
    
    # Colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, 
                 shrink=0.6, label='dh/dt Residual (Obs - Model) [m/yr]')
    plt.suptitle("Thickness Change Residuals (Time-Averaged)", fontsize=14, y=0.98)
    plt.savefig("residuals_dhdt.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: residuals_dhdt.png")
    plt.show()
    
    # =========================================================================
    # 2. VELOCITY RESIDUALS (VX and VY)
    # =========================================================================
    
    # Get velocity years that were actually used
    # used_years = []
    # for year in raw["obs_vel_years"]:
    #     # Check if this year was used (not None in models)
    #     if all(raw["vx_models"][m][year] is not None for m in range(M)):
    #         used_years.append(year)
    
    # if len(used_years) == 0:
    #     print("\n⚠️  No velocity years available for residual maps")
    #     return
    
    used_years = raw["obs_vel_years"]

    if len(used_years) == 0:
        print("\n⚠️  No velocity years available for residual maps")
        return
        
    print("\n" + "="*70)
    print("VELOCITY RESIDUAL STATISTICS (Time-Averaged)")
    print("="*70)
    
    # Compute time-averaged velocities
    # obs_vx_mean = np.nanmean(np.stack([raw["VX_obs"][y] for y in used_years]), axis=0)
    # obs_vy_mean = np.nanmean(np.stack([raw["VY_obs"][y] for y in used_years]), axis=0)
    obs_vx_mean = raw["obs_dvxdt"].mean(dim="time", skipna=True).values
    obs_vy_mean = raw["obs_dvydt"].mean(dim="time", skipna=True).values
    
    # Create figure: 2 rows (vx, vy) x M columns (models)
    fig, axes = plt.subplots(2, M, figsize=(6*M, 10), constrained_layout=True)
    if M == 1:
        axes = axes.reshape(2, 1)
    
    # Color normalization for velocity residuals
    norm_vel = mcolors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    
    for i in range(M):
        # Time-averaged model velocities
        # model_vx_mean = np.nanmean(np.stack([raw["vx_models"][i][y] for y in used_years]), axis=0)
        # model_vy_mean = np.nanmean(np.stack([raw["vy_models"][i][y] for y in used_years]), axis=0)
        
        model_vx_mean = raw["vx_models"][i].mean(dim="time", skipna=True).values
        model_vy_mean = raw["vy_models"][i].mean(dim="time", skipna=True).values
        
        
        # Residuals
        resid_vx = obs_vx_mean - model_vx_mean
        resid_vy = obs_vy_mean - model_vy_mean
        
        # Statistics for VX
        valid_vx = resid_vx[np.isfinite(resid_vx)]
        rmse_vx = np.sqrt(np.mean(valid_vx**2))
        bias_vx = np.mean(valid_vx)
        
        # Statistics for VY
        valid_vy = resid_vy[np.isfinite(resid_vy)]
        rmse_vy = np.sqrt(np.mean(valid_vy**2))
        bias_vy = np.mean(valid_vy)
        
        print(f"\nModel {i+1} (Weight: {weights[i]:.4f}):")
        print(f"  VX - RMSE: {rmse_vx:.1f} m/yr, Bias: {bias_vx:.1f} m/yr")
        print(f"  VY - RMSE: {rmse_vy:.1f} m/yr, Bias: {bias_vy:.1f} m/yr")
        
        # Plot VX residuals (top row)
        im_vx = axes[0, i].imshow(resid_vx, cmap="PiYG", norm=norm_vel, origin='lower')
        axes[0, i].set_title(f"Model {i+1} VX\nRMSE: {rmse_vx:.1f} m/yr", fontsize=11)
        axes[0, i].axis("off")
        
        # Plot VY residuals (bottom row)
        im_vy = axes[1, i].imshow(resid_vy, cmap="PiYG", norm=norm_vel, origin='lower')
        axes[1, i].set_title(f"Model {i+1} VY\nRMSE: {rmse_vy:.1f} m/yr", fontsize=11)
        axes[1, i].axis("off")
    
    # Colorbars
    fig.colorbar(im_vx, ax=axes[0, :], orientation='horizontal', pad=0.02, 
                 shrink=0.6, label='VX Residual (Obs - Model) [m/yr]')
    fig.colorbar(im_vy, ax=axes[1, :], orientation='horizontal', pad=0.02, 
                 shrink=0.6, label='VY Residual (Obs - Model) [m/yr]')
    
    plt.suptitle("Velocity Residuals (Time-Averaged)", fontsize=14, y=0.995)
    plt.savefig("residuals_velocity.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: residuals_velocity.png")
    plt.show()
    
    # =========================================================================
    # 3. SPEED RESIDUALS (Combined VX and VY)
    # =========================================================================
    
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5), constrained_layout=True)
    if M == 1:
        axes = [axes]
    
    # Calculate observed speed
    obs_speed = np.sqrt(obs_vx_mean**2 + obs_vy_mean**2)
    
    norm_speed = mcolors.TwoSlopeNorm(vcenter=0, vmin=-200, vmax=200)
    
    print("\n" + "="*70)
    print("SPEED RESIDUAL STATISTICS")
    print("="*70)
    
    for i in range(M):
        # Time-averaged model velocities
        # model_vx_mean = np.nanmean(np.stack([raw["vx_models"][i][y] for y in used_years]), axis=0)
        # model_vy_mean = np.nanmean(np.stack([raw["vy_models"][i][y] for y in used_years]), axis=0)
        model_vx_mean = raw["vx_models"][i].mean(dim="time", skipna=True).values
        model_vy_mean = raw["vy_models"][i].mean(dim="time", skipna=True).values
        # Model speed
        model_speed = np.sqrt(model_vx_mean**2 + model_vy_mean**2)
        
        # Speed residual
        resid_speed = obs_speed - model_speed
        
        # Statistics
        valid_speed = resid_speed[np.isfinite(resid_speed)]
        rmse_speed = np.sqrt(np.mean(valid_speed**2))
        bias_speed = np.mean(valid_speed)
        
        print(f"\nModel {i+1} (Weight: {weights[i]:.4f}):")
        print(f"  RMSE:   {rmse_speed:.1f} m/yr")
        print(f"  Bias:   {bias_speed:.1f} m/yr")
        print(f"  Median: {np.median(valid_speed):.1f} m/yr")
        
        # Plot
        im = axes[i].imshow(resid_speed, cmap="RdBu_r", norm=norm_speed, origin='lower')
        axes[i].set_title(f"Model {i+1}\nWeight: {weights[i]:.4f}\nRMSE: {rmse_speed:.1f} m/yr", 
                         fontsize=12)
        axes[i].axis("off")
    
    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, 
                 shrink=0.6, label='Speed Residual (Obs - Model) [m/yr]')
    plt.suptitle("Ice Speed Residuals (Time-Averaged)", fontsize=14, y=0.98)
    plt.savefig("residuals_speed.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: residuals_speed.png")
    plt.show()
    
    # =========================================================================
    # 4. SUMMARY STATISTICS TABLE
    # =========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<10} {'Weight':<12} {'dh/dt RMSE':<15} {'VX RMSE':<12} {'VY RMSE':<12} {'Speed RMSE':<12}")
    print("-"*70)
    
    for i in range(M):
        # dh/dt
        model_dhdt_mean = raw["ensemble_dhdt"][i].mean(dim="time", skipna=True).values
        residual_dhdt = obs_dhdt_mean - model_dhdt_mean
        rmse_dhdt = np.sqrt(np.nanmean(residual_dhdt**2))
        
        # Velocity
        # model_vx_mean = np.nanmean(np.stack([raw["vx_models"][i][y] for y in used_years]), axis=0)
        # model_vy_mean = np.nanmean(np.stack([raw["vy_models"][i][y] for y in used_years]), axis=0)
        
        model_vx_mean = raw["vx_models"][i].mean(dim="time", skipna=True).values
        model_vy_mean = raw["vy_models"][i].mean(dim="time", skipna=True).values
        
        resid_vx = obs_vx_mean - model_vx_mean
        resid_vy = obs_vy_mean - model_vy_mean
        
        rmse_vx = np.sqrt(np.nanmean(resid_vx**2))
        rmse_vy = np.sqrt(np.nanmean(resid_vy**2))
        
        # Speed
        model_speed = np.sqrt(model_vx_mean**2 + model_vy_mean**2)
        resid_speed = obs_speed - model_speed
        rmse_speed = np.sqrt(np.nanmean(resid_speed**2))
        
        print(f"{i+1:<10} {weights[i]:<12.4f} {rmse_dhdt:<15.2f} {rmse_vx:<12.1f} {rmse_vy:<12.1f} {rmse_speed:<12.1f}")
    
    print("\n" + "="*70)


# ==============================================================================
# RUN THE RESIDUAL ANALYSIS
# ==============================================================================

# if __name__ == "__main__":
#     # After running main() and getting results
#     # trace, weights, data, raw = main()
    
#     create_residual_maps(raw, weights)
# if __name__ == "__main__":

#     trace, weights, data, raw = main()

#     create_residual_maps(raw, weights)

#     plot_pixel_coverage(raw)

#     create_fatal_pixel_maps(raw, trace)
#     weights, loglik = compute_model_weights(trace, data)

#     weights_table = save_model_weights(weights, loglik)


#     for i in range(raw["M"]):
#         make_dhdt_diagnostic_figure(raw, trace, model_index=i)


#### calling the whole shabang 
# if __name__ == "__main__":

#     trace, weights, data, raw = main()

#    # create_residual_maps(raw, weights)

#     #plot_pixel_coverage(raw)

#     create_fatal_pixel_maps(raw, trace)
#     weights, loglik = compute_model_weights(trace, data)

#     weights_table = save_model_weights(weights, loglik)
###sanity check the model 
def residual_calibration_check(trace, data):
    """
    Sanity check: verify standardized residuals follow N(0,1)

    If uncertainty model is calibrated:
        mean(z) ≈ 0
        std(z) ≈ 1
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    print("\n======================================================================")
    print("RESIDUAL CALIBRATION CHECK")
    print("======================================================================")

    # --------------------------------------------------
    # Posterior mean parameters
    # --------------------------------------------------

    sigma_base_thick = trace.posterior["sigma_base_thick"].mean().values
    beta_thick       = trace.posterior["beta_thick"].mean().values
    sigma_base_vel   = trace.posterior["sigma_base_vel"].mean().values
    beta_vel         = trace.posterior["beta_vel"].mean().values

    print("Posterior parameters:")
    print("sigma_base_thick =", sigma_base_thick)
    print("beta_thick       =", beta_thick)
    print("sigma_base_vel   =", sigma_base_vel)
    print("beta_vel         =", beta_vel)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    y       = data["y_obs"]
    F       = data["F"]
    speed   = data["speed"]
    sig_obs = data["sigma_obs"]

    n_thick = data["n_dhdt"]
    n_vel   = data["n_vel"]

    idx_thick = slice(0, n_thick)
    idx_vel   = slice(n_thick, n_thick+n_vel)

    # --------------------------------------------------
    # Rebuild model uncertainty (MATCHES YOUR MODEL)
    # --------------------------------------------------

    sigma_model_thick = sigma_base_thick * np.sqrt(1 + beta_thick * speed[idx_thick])
    sigma_model_vel   = sigma_base_vel   * np.sqrt(1 + beta_vel   * speed[idx_vel])

    sigma_tot_thick = np.sqrt(sig_obs[idx_thick]**2 + sigma_model_thick**2)
    sigma_tot_vel   = np.sqrt(sig_obs[idx_vel]**2   + sigma_model_vel**2)

    # --------------------------------------------------
    # Best model (highest weight)
    # --------------------------------------------------

    weights = trace.posterior["w"].mean(("chain","draw")).values
    best_model = np.argmax(weights)

    print("\nUsing best model:", best_model+1)

    residuals = y - F[best_model]

    # --------------------------------------------------
    # Standardized residuals
    # --------------------------------------------------

    z_thick = residuals[idx_thick] / sigma_tot_thick
    z_vel   = residuals[idx_vel]   / sigma_tot_vel

    z = np.concatenate([z_thick, z_vel])
    z = z[np.isfinite(z)]

    print("\nResidual statistics")
    print("mean =", np.mean(z))
    print("std  =", np.std(z))

    # --------------------------------------------------
    # Plot histogram
    # --------------------------------------------------

    plt.figure(figsize=(6,5))

    plt.hist(z, bins=80, density=True, alpha=0.7, label="Standardized residuals")

    x = np.linspace(-5,5,500)
    plt.plot(x, norm.pdf(x,0,1), 'r', lw=2, label="Normal(0,1)")

    plt.xlabel("z-score")
    plt.ylabel("Density")
    plt.title("Residual Calibration Check")
    plt.legend()

    plt.tight_layout()
    plt.savefig("residual_calibration.png", dpi=300)

    print("✓ Saved residual_calibration.png")

    plt.show()
    
# ==============================================================================
# SPATIAL AUTOCORRELATION TEST (Correlation Length + Effective Sample Size)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def estimate_spatial_correlation_length(field, grid_km=5, plot=True):
    """
    Estimate spatial correlation length scale from a 2D field.

    Parameters
    ----------
    field : 2D array
        residual field (obs - model)

    grid_km : float
        grid spacing in km

    plot : bool
        show correlation decay plot

    Returns
    -------
    L : float
        estimated correlation length (km)
    """

    field = field.copy()

    # --------------------------------------------------
    # Remove NaNs
    # --------------------------------------------------

    mask = np.isfinite(field)

    if np.sum(mask) < 100:
        raise ValueError("Too few valid pixels to estimate correlation")

    field = field - np.nanmean(field)

    field[~mask] = 0

    # --------------------------------------------------
    # 2D autocorrelation using FFT
    # --------------------------------------------------

    corr = fftconvolve(field, field[::-1, ::-1], mode="same")

    corr /= np.nanmax(corr)

    # --------------------------------------------------
    # radial averaging
    # --------------------------------------------------

    y, x = np.indices(corr.shape)

    center = np.array(corr.shape) // 2

    r = np.sqrt((x-center[1])**2 + (y-center[0])**2)

    r = r.astype(int)

    tbin = np.bincount(r.ravel(), corr.ravel())
    nr = np.bincount(r.ravel())

    radial_corr = tbin / nr

    distance = np.arange(len(radial_corr)) * grid_km

    # --------------------------------------------------
    # find correlation length (1/e decay)
    # --------------------------------------------------

    idx = np.where(radial_corr < 1/np.e)[0]

    if len(idx) == 0:
        L = np.nan
    else:
        L = distance[idx[0]]

    # --------------------------------------------------
    # plot
    # --------------------------------------------------

    if plot:

        plt.figure(figsize=(6,4))

        plt.plot(distance, radial_corr)

        plt.axhline(1/np.e, linestyle="--", color="red", label="1/e threshold")
        plt.axvline(L, linestyle="--", color="black", label=f"L ≈ {L:.1f} km")

        plt.xlabel("Distance (km)")
        plt.ylabel("Correlation")

        plt.title("Spatial Correlation Length")

        plt.legend()

        plt.tight_layout()

        plt.show()

    return L


def effective_sample_size(field, L, grid_km=5):
    """
    Estimate effective number of independent pixels.
    """

    ny, nx = field.shape

    area = nx * ny * grid_km**2

    Neff = area / (np.pi * L**2)

    return int(Neff)


def spatial_autocorrelation_diagnostic(raw, grid_km=8):

    """
    Run spatial autocorrelation diagnostics for all models using dh/dt residuals.
    Computes:
        - Correlation length (L)
        - Lag-1 spatial correlation
        - Integrated autocorrelation strength
        - Effective sample size
    """

    import numpy as np
    from scipy.signal import fftconvolve

    print("\n" + "="*70)
    print("SPATIAL AUTOCORRELATION DIAGNOSTIC")
    print("="*70)

    obs = raw["obs_dhdt"].mean(dim="time", skipna=True).values

    M = raw["M"]

    results = []

    for i in range(M):

        model = raw["ensemble_dhdt"][i].mean(dim="time", skipna=True).values

        residual = obs - model

        print(f"\nModel {i+1}")

        # --------------------------------------------------
        # Remove NaNs
        # --------------------------------------------------

        field = residual.copy()
        mask = np.isfinite(field)

        field = field - np.nanmean(field)
        field[~mask] = 0

        # --------------------------------------------------
        # 2D autocorrelation (FFT)
        # --------------------------------------------------

        corr = fftconvolve(field, field[::-1, ::-1], mode="same")
        corr /= np.nanmax(corr)

        # --------------------------------------------------
        # radial averaging
        # --------------------------------------------------

        y, x = np.indices(corr.shape)

        center = np.array(corr.shape) // 2

        r = np.sqrt((x-center[1])**2 + (y-center[0])**2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), corr.ravel())
        nr = np.bincount(r.ravel())

        radial_corr = tbin / nr

        distance = np.arange(len(radial_corr)) * grid_km

        # --------------------------------------------------
        # correlation length (1/e crossing)
        # --------------------------------------------------

        idx = np.where(radial_corr < 1/np.e)[0]

        if len(idx) > 0:
            L = distance[idx[0]]
        else:
            L = np.nan

        # --------------------------------------------------
        # SAC strength metrics
        # --------------------------------------------------

        lag1_corr = radial_corr[1]

        integrated_corr = np.trapz(radial_corr, distance)

        # --------------------------------------------------
        # effective sample size
        # --------------------------------------------------

        ny, nx = residual.shape

        area = nx * ny * grid_km**2

        Neff = int(area / (np.pi * L**2)) if np.isfinite(L) else np.nan

        Npix = np.sum(np.isfinite(residual))

        # --------------------------------------------------
        # print results
        # --------------------------------------------------

        print(f"Correlation length:       {L:.1f} km")
        print(f"Lag-1 correlation:       {lag1_corr:.3f}")
        print(f"Integrated correlation:  {integrated_corr:.1f} km")
        print(f"Total pixels:            {Npix:,}")
        print(f"Effective samples:       {Neff:,}")

        results.append((L, lag1_corr, integrated_corr, Neff))

    print("\n" + "="*70)

    
    return results

def plot_pixels_used_after_uncertainty_filter(raw):
    """
    Simple diagnostic plot showing which pixels remain after applying
    uncertainty thresholds used in flatten_and_mask_combined().

    Shows:
        1) Thickness pixels used
        2) Velocity change pixels used
    """

    import numpy as np
    import matplotlib.pyplot as plt

    THICK_UNC_THRESHOLD = 50.0
    VEL_UNC_THRESHOLD = 10.0

    print("\nPlotting pixels used after uncertainty filtering...")

    # ==========================================================
    # THICKNESS
    # ==========================================================

    obs = raw["obs_dhdt"].mean(dim="time", skipna=True).values
    unc = raw["obs_unc"].mean(dim="time", skipna=True).values

    mask_thick = (
        np.isfinite(obs)
        & np.isfinite(unc)
        & (unc < THICK_UNC_THRESHOLD)
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im0 = ax[0].imshow(mask_thick, origin="lower", cmap="viridis")
    ax[0].set_title("Thickness Pixels Used\n(after uncertainty filter)")
    ax[0].axis("off")

    plt.colorbar(im0, ax=ax[0], label="1 = used")

    print(f"Thickness pixels used: {np.sum(mask_thick):,}")

    # ==========================================================
    # VELOCITY
    # ==========================================================

    dvx = raw["obs_dvxdt"].mean(dim="time", skipna=True).values
    dvy = raw["obs_dvydt"].mean(dim="time", skipna=True).values

    uncx = raw["obs_uncx"].mean(dim="time", skipna=True).values
    uncy = raw["obs_uncy"].mean(dim="time", skipna=True).values

    mask_vel = (
        np.isfinite(dvx)
        & np.isfinite(dvy)
        & np.isfinite(uncx)
        & np.isfinite(uncy)
        & (uncx < VEL_UNC_THRESHOLD)
        & (uncy < VEL_UNC_THRESHOLD)
    )

    im1 = ax[1].imshow(mask_vel, origin="lower", cmap="viridis")
    ax[1].set_title("Velocity Pixels Used\n(after uncertainty filter)")
    ax[1].axis("off")

    plt.colorbar(im1, ax=ax[1], label="1 = used")

    print(f"Velocity pixels used: {np.sum(mask_vel):,}")

    plt.tight_layout()

    plt.savefig("pixels_used_after_uncertainty_filter.png", dpi=300)

    print("✓ Saved pixels_used_after_uncertainty_filter.png")

    plt.show()

if __name__ == "__main__":

    trace, weights, data, raw = main()

    create_fatal_pixel_maps(raw, trace)

    weights, loglik = compute_model_weights(trace, data)

    weights_table = save_model_weights(weights, loglik)

    # --------------------------------------------------
    # Sanity check
    # --------------------------------------------------

    residual_calibration_check(trace, data)
    spatial_autocorrelation_diagnostic(raw, grid_km=8)
    plot_pixels_used_after_uncertainty_filter(raw)


    # for i in range(raw["M"]):
    #     make_dhdt_diagnostic_figure(raw, trace, model_index=i)
    
    

