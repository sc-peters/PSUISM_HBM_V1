#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:15:06 2026

@author: sp53972
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

# DATA UTILITIES 
    #making everything annual and removing any days/seconds 
    #making decimals uniform 
    #nan to any missing years from data
    #removing any weird data years from ism (psu-ism has a weird 2nd time index)
    #squeeze into 2d arrays 
    #identify year from file name - right now the observational data has to have the year in the filename 
    #compute modeled and observational dhdt/dvdt
    #flatten and mask all the data 

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

    # some netcdfs have missing values/nans 
    fill_value = time_var.attrs.get("_FillValue", None)
    missing_value = time_var.attrs.get("missing_value", None)

    if fill_value is not None:
        t[t == fill_value] = np.nan
    if missing_value is not None:
        t[t == missing_value] = np.nan

    # also catch absurd placeholder values
    t[np.abs(t) > 1e20] = np.nan

    #dropping the psu-ism 2nd time step 
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

    # has years
    if unit_str is None:
        print("Time variable has no units — assuming values are already years.")
        return t

    # time reported in seconds
    if "since" in unit_str and "second" in unit_str:
        origin = _parse_origin_allow_day00(unit_str)

        years = np.empty_like(t, dtype=float)

        for i, sec in enumerate(t):
            dt = origin + timedelta(seconds=float(sec))
            year_start = datetime(dt.year, 1, 1)
            frac = (dt - year_start).total_seconds() / SEC_PER_YEAR
            years[i] = dt.year + frac

        return years

    #already in years/has units
    print("Time units not recognized — assuming values are years.")
    return t


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

#matching closest model years to obs years 
def _snap_model_year_to_obs_year(t_model, available_years, tol=1.5):

    available_years = np.asarray(available_years)

    idx = np.argmin(np.abs(available_years - t_model))

    if np.abs(available_years[idx] - t_model) <= tol:
        return int(available_years[idx])

    return None


#compute ISM dhdt

def compute_model_dhdt(ds, model_years, model_thickness_var="h"):
    #model thickness is specific to psu-ism (h var) need to add list of variables for ismip 
    h = ds[model_thickness_var].astype("float32")

    order = np.argsort(model_years)

    model_years = np.asarray(model_years)[order]
    h = h.isel(time=order)

    h = h.assign_coords(time=("time", model_years))

    t = np.asarray(model_years, dtype=np.float64)

    dt = np.diff(t)

    if np.any(dt == 0):
        raise ValueError("Duplicate model time values detected")

    #midpoint time 
    # tmid = t[:-1] + dt / 2.0
    tmid = 0.5 * (t[:-1] + t[1:])

    # dhdt
    h0 = h.isel(time=slice(0, -1)).values
    h1 = h.isel(time=slice(1, None)).values

    dhdt = (h1 - h0) / dt[:, None, None]

    # array 
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
    
    
#   Flatten and combine thickness change (dh/dt) and velocity change (dv/dt) data. Also builds a speed vector using the EXACT SAME masking/order as the likelihood. limiting large obs unc

   
def flatten_and_mask_combined(
    dhdt_obs, dhdt_sigma, dhdt_models,
    dvxdt_obs, dvydt_obs,
    dvxdt_unc, dvydt_unc,
    dvxdt_models, dvydt_models,
    VX_obs_dict=None, VY_obs_dict=None, obs_vel_years=None,
):
    
    THICK_UNC_THRESHOLD = 50.0
    VEL_UNC_THRESHOLD   = 10.0
    
    M = len(dhdt_models)

    # dhdt

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

 #velocity change 

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

    #combine dhdt and dvdt 

    y_combined = np.concatenate([y_dhdt, y_vel])
    
    
    print("n_thickness:", n_dhdt)
    print("n_velocity:", n_vel)
    print("ratio vel/thick:", n_vel / n_dhdt)
    
    
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
