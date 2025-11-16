import pandas as pd
import numpy as np
import os
import pickle
import itertools
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf


def normalize_id(x):
    """
    function to normalize every cell to consistent ID format
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # strip trailing ".0" if it came from Excel as a float-looking code
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()


def to_month_period(c):
    """
    convert columns to month periods; leave any non-date columns untouched if present
    """

    try:
        return pd.to_datetime(c).to_period('M')
    except Exception:
        return c


def load_price_panel(excel_path, sheet_name=None):
    """
    Reads a monthly price file from a given sheet in an Excel workbook.
    First column = ID, others = dates.

    Returns:
        prices: DataFrame (index = Date as Period[M], columns = asset IDs)
        returns: DataFrame (same shape, monthly returns)
    """

    df = pd.read_excel(excel_path,sheet_name=sheet_name)

    # first column is ID
    df = df.rename(columns={df.columns[0]: 'id'})
    df['id'] = df['id'].map(normalize_id)
    df = df.set_index('id')

    # columns -> monthly PeriodIndex
    df.columns = pd.to_datetime(df.columns).to_period('M')

    # prices with Date as index
    prices = df.T
    prices.index.name = 'Date'

    # compute returns: r_t = P_{t+1}/P_t - 1
    returns = prices.shift(-1).divide(prices) - 1
    returns = returns.iloc[:-1]

    return prices, returns


def load_composition_panel(excel_path, sheet_name=None):
    """
    Reads a composition file (or sheet) with columns = months, rows = assets.
    Each column lists the tickers held in that month.

    excel_path : path to Excel file
    sheet_name : sheet name inside the workbook (None = first sheet)
    """

    comp = pd.read_excel(excel_path, sheet_name=sheet_name)

    # convert columns to Period[M]
    comp.columns = [to_month_period(c) for c in comp.columns]
    comp.columns = pd.PeriodIndex(comp.columns, freq='M')

    # normalize IDs in every column
    for c in comp.columns:
        comp[c] = comp[c].map(normalize_id)

    return comp


def load_metadata_panel(excel_path, sheet_name=None):
    """
    Load metadata (ID, NAME, ISIN, TICKER, SECTOR) for one universe.

    excel_path : path to metadata Excel file
    sheet_name : sheet name ('S&P500', 'MSCI', etc.)

    Returns:
        DataFrame indexed by normalized ID with columns:
        ['NAME', 'ISIN', 'TICKER', 'SECTOR', ...]
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Assume first column is 'Type' = internal ID
    if 'Type' in df.columns:
        df = df.rename(columns={'Type': 'id'})
    else:
        df = df.rename(columns={df.columns[0]: 'id'})

    # Normalize IDs
    df['id'] = df['id'].map(normalize_id)

    # Set ID as index
    df = df.set_index('id')

    # Standardize ASSET CLASS column name if present
    # e.g. "ASSET CLASS" -> "ASSET_CLASS"
    if 'ASSET CLASS' in df.columns:
        df = df.rename(columns={'ASSET CLASS': 'ASSET_CLASS'})

    # Ensure SECTOR column exists (for equities); if missing, fill with NaN
    if 'SECTOR' in df.columns:
        df['SECTOR'] = df['SECTOR'].astype(str).str.strip()
    else:
        df['SECTOR'] = np.nan

    return df


def load_esg_scores(excel_path, sheet_name=None):
    """
    Load ESG panel where rows = dates, columns = company IDs, values = ESG numeric score.
    Converts index to Period[M] and normalizes tickers.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M')
    df = df.rename(columns={date_col: 'Date'})
    df = df.set_index('Date')
    df.columns = [normalize_id(c) for c in df.columns]

    return df

def classify_esg(df):
    """
    Input:
        df: DataFrame indexed by Period[M], columns = asset IDs,
            values = numeric ESG scores.
    Output:
        DataFrame indexed by Period[M], same columns, values = 'L','M','H'
    """

    def classify_row(row):
        # row: ESG scores for one Date, index = asset IDs
        s = row.dropna()

        if s.empty:
            # no data for this date
            return pd.Series(index=row.index, dtype=object)

        # quantiles on available scores
        Q1 = np.nanpercentile(s, 25)
        Q3 = np.nanpercentile(s, 75)

        # labels defined only on non-NaN assets
        labels = pd.Series(index=s.index, dtype=object)
        labels[s < Q1] = "L"
        labels[(s >= Q1) & (s < Q3)] = "M"
        labels[s >= Q3] = "H"

        # reindex back to the full row index (all assets),
        # missing ones become NaN labels
        return labels.reindex(row.index)

    labels_df = df.apply(classify_row, axis=1)
    return labels_df


def filter_equity_candidates(raw_candidates,
                             candidates_period,
                             metadata_equity,
                             esg_equity,
                             keep_sectors=None,
                             keep_esg=None):
    """
    raw_candidates : list of IDs from composition (already normalized ideally)
    candidates_period : Period[M] of the rebalance/candidate date
    metadata_equity : DataFrame indexed by ID, with at least column 'SECTOR'
    esg_equity : DataFrame indexed by Period[M], columns = IDs, values = 'L','M','H'
    keep_sectors : list of sector names to KEEP (None = keep all)
    keep_esg : list of ESG labels to KEEP, e.g. ['M','H'] (None = keep all)

    Returns:
        filtered_candidates : list of IDs that pass all filters
    """

    # Start from a clean Index of IDs
    ids = pd.Index(raw_candidates).dropna()

    # ---------- Sector filter ----------
    if keep_sectors is not None and len(keep_sectors) > 0:
        # Get sectors for these IDs
        sectors = metadata_equity.reindex(ids)['SECTOR']
        mask = sectors.isin(keep_sectors)

        filtered_ids = sectors.index[mask]
        # (optional) debug prints:
        # dropped = sectors.index[~mask | sectors.isna()]
        # print(f"Dropped {len(dropped)} by sector filter.")
    else:
        filtered_ids = ids

    # ---------- ESG filter ----------
    if keep_esg is not None and len(keep_esg) > 0:
        if candidates_period in esg_equity.index:
            esg_row = esg_equity.loc[candidates_period]
            esg_for_ids = esg_row.reindex(filtered_ids)

            mask_esg = esg_for_ids.isin(keep_esg)
            filtered_ids = esg_for_ids.index[mask_esg]
            # (optional) debug:
            # dropped_esg = esg_for_ids.index[~mask_esg | esg_for_ids.isna()]
            # print(f"Dropped {len(dropped_esg)} by ESG filter.")
        else:
            # No ESG data for this month → skip ESG filter
            # print(f"No ESG data for {candidates_period}, skipping ESG filter.")
            pass

    return list(filtered_ids)


def markowitz_long_only(estimation_window,
                        gamma=None,
                        max_weight_per_asset=0.05,
                        asset_class_for_assets = None,
                        sector_for_assets=None,
                        sector_constraints=None,
                        esg_for_assets=None,
                        esg_constraints=None,
                        asset_class_constraints=None):
    """
    estimation_window : DataFrame of returns, columns = assets, rows = months
    gamma : risk aversion parameter (must be > 0)
    max_weight_per_asset : upper bound per asset (e.g. 0.10 for 10%)
    asset_class_for_assets : pd.Series indexed by asset ID, giving asset class
    sector_for_assets : pd.Series indexed by asset ID, giving sector name
    sector_constraints : dict, e.g.
        {
            'Information Technology': {'max': 0.20},
            'Health Care': {'min': 0.10},
        }
    esg_for_assets : pd.Series indexed by asset ID, giving 'L','M','H' or NaN
    esg_constraints : dict, same style as sector_constraints
    asset_class_constraints : dict, e.g.
        {
            'Equity': {'min': 0.7},
            'Fixed Income': {'max': 0.2},
        }
    """

    # ------------------ Basic sanity checks ------------------
    if estimation_window is None or estimation_window.shape[1] == 0:
        raise ValueError("markowitz_long_only: estimation_window has no assets (0 columns).")

    if gamma is None or gamma <= 0:
        raise ValueError(f"markowitz_long_only: gamma must be positive, got {gamma}.")

    assets = list(estimation_window.columns)
    n = len(assets)

    if max_weight_per_asset <= 0 or max_weight_per_asset > 1:
        raise ValueError(
            f"markowitz_long_only: max_weight_per_asset should be in (0,1], got {max_weight_per_asset}."
        )

    # ------------------ Estimate mu and Sigma ------------------
    mu_hat = estimation_window.mean(axis=0).values.astype(float)

    X = estimation_window.values  # rows=months, cols=assets
    lw = LedoitWolf().fit(X)
    sigma_hat = lw.covariance_.astype(float)

    # Safety: enforce positive definiteness by clipping eigenvalues
    eigvals, eigvecs = np.linalg.eigh(sigma_hat)
    eps = 1e-8
    eigvals_clipped = np.clip(eigvals, eps, None)
    sigma_hat = (eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T).astype(float)

    # ------------------ Initial guess ------------------
    x0 = np.ones(n, dtype=float) / n

    # ------------------ Equity mask (for relative constraints) ------------------
    if asset_class_for_assets is not None:
        asset_class_for_assets = asset_class_for_assets.reindex(assets)
        equity_mask = (asset_class_for_assets == 'Equity').astype(float).values
    else:
        # fallback: treat all assets as "equity"
        equity_mask = np.ones(n, dtype=float)

    # To avoid divide-by-zero behavior in constraints when equity weight ~ 0,
    # we can assert that equity has some minimal feasible weight if
    # sector/esg constraints are used.
    if (sector_constraints or esg_constraints) and equity_mask.sum() == 0:
        raise ValueError(
            "Sector/ESG constraints specified but no asset is labeled as 'Equity' "
            "in asset_class_for_assets."
        )

    def obj_f(w, Sigma, mu, gamma):
        return 0.5 * (w @ Sigma @ w) - gamma * (mu @ w)

    def f_obj_grad(w, Sigma, mu, gamma):
        return Sigma @ w - gamma * mu

    # ------------------ Constraints list ------------------
    constraints_list = []

    # Budget constraint: sum w_i = 1
    constraints_list.append({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0,
        'jac': lambda w: np.ones_like(w, dtype=float)
    })

    # ---------- Sector constraints (optional) ----------
    if (sector_for_assets is not None) and (sector_constraints is not None):
        sector_for_assets = sector_for_assets.reindex(assets)

        for sector_name, cons in sector_constraints.items():
            if cons is None:
                continue

            mask = (sector_for_assets == sector_name).astype(float).values
            if mask.sum() == 0:
                # no asset currently in this sector
                continue

            # sector_weight = sum(mask * w)
            # equity_weight = sum(equity_mask * w)

            # Max: sector_weight / equity_weight <= cap
            #  => cap * equity_weight - sector_weight >= 0
            if 'max' in cons and cons['max'] is not None:
                cap = float(cons['max'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, ms=mask, me=equity_mask, c=cap: c * np.dot(me, w) - np.dot(ms, w),
                    'jac': lambda w, ms=mask, me=equity_mask, c=cap: c * me - ms,
                })

            # Min: sector_weight / equity_weight >= floor
            #  => sector_weight - floor * equity_weight >= 0
            if 'min' in cons and cons['min'] is not None:
                floor = float(cons['min'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, ms=mask, me=equity_mask, f=floor: np.dot(ms, w) - f * np.dot(me, w),
                    'jac': lambda w, ms=mask, me=equity_mask, f=floor: ms - f * me,
                })

    # ---------- ESG constraints (optional) ----------
    if (esg_for_assets is not None) and (esg_constraints is not None):
        esg_for_assets = esg_for_assets.reindex(assets)

        for label, cons in esg_constraints.items():
            if cons is None:
                continue

            mask = (esg_for_assets == label).astype(float).values
            if mask.sum() == 0:
                continue

            # esg_label_weight = sum(mask * w)
            # equity_weight    = sum(equity_mask * w)

            # Max: esg_weight / equity_weight <= cap
            if 'max' in cons and cons['max'] is not None:
                cap = float(cons['max'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, ms=mask, me=equity_mask, c=cap: c * np.dot(me, w) - np.dot(ms, w),
                    'jac': lambda w, ms=mask, me=equity_mask, c=cap: c * me - ms,
                })

            # Min: esg_weight / equity_weight >= floor
            if 'min' in cons and cons['min'] is not None:
                floor = float(cons['min'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, ms=mask, me=equity_mask, f=floor: np.dot(ms, w) - f * np.dot(me, w),
                    'jac': lambda w, ms=mask, me=equity_mask, f=floor: ms - f * me,
                })

    # ---------- Asset-class constraints (optional, absolute on total portfolio) ----------
    if (asset_class_for_assets is not None) and (asset_class_constraints is not None):
        asset_class_for_assets = asset_class_for_assets.reindex(assets)

        for ac_name, cons in asset_class_constraints.items():
            if cons is None:
                continue

            mask = (asset_class_for_assets == ac_name).astype(float).values
            if mask.sum() == 0:
                # no asset in this asset class in the current window
                continue

            # Max: sum_{i: class_i = ac_name} w_i <= cap
            if 'max' in cons and cons['max'] is not None:
                cap = float(cons['max'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, m=mask, c=cap: c - np.dot(m, w),
                    'jac': lambda w, m=mask: -m,
                })

            # Min: sum_{i: class_i = ac_name} w_i >= floor
            if 'min' in cons and cons['min'] is not None:
                floor = float(cons['min'])
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, m=mask, f=floor: np.dot(m, w) - f,
                    'jac': lambda w, m=mask: m,
                })

    # ------------------ Bounds ------------------
    bounds = [(0.0, max_weight_per_asset)] * n

    res = minimize(
        obj_f,
        x0,
        args=(sigma_hat, mu_hat, gamma),
        method='SLSQP',
        jac=f_obj_grad,
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )

    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    w_opt = res.x.astype(float)

    # Numerical cleanup: clip tiny negatives to 0, renormalize
    w_opt = np.where(w_opt < 0, 0.0, w_opt)
    s = w_opt.sum()
    if s <= 0:
        raise ValueError("Optimization returned non-positive total weight.")
    w_opt /= s

    return pd.Series(w_opt, index=assets, name="weights_opt")

def check_sector_constraints_feasibility(assets, metadata_equity, sector_constraints):
    if sector_constraints is None:
        return

    # mapping: asset -> sector
    sector_for_assets = metadata_equity['SECTOR'].reindex(assets)

    # 1) total min cannot exceed 1
    total_min = sum(cons.get('min', 0) for cons in sector_constraints.values())
    if total_min > 1.0:
        raise ValueError(f"Total minimum sector weights {total_min:.2f} exceed 1.0")

    # 2) each sector with a min must appear in assets
    for sector_name, cons in sector_constraints.items():
        if 'min' in cons:
            if not any(sector_for_assets == sector_name):
                raise ValueError(
                    f"Sector '{sector_name}' has a min constraint but does not appear in the universe!"
                )

    # 3) consistency: min <= max
    for sector_name, cons in sector_constraints.items():
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"In sector '{sector_name}', min={cons['min']} > max={cons['max']}"
                )


def check_esg_constraints_feasibility(esg_for_assets, esg_constraints):
    if esg_constraints is None:
        return

    # 1) total min cannot exceed 1
    total_min = sum(cons.get('min', 0) for cons in esg_constraints.values())
    if total_min > 1.0:
        raise ValueError(f"Total minimum ESG weights {total_min:.2f} exceed 1.0")

    # 2) each ESG label with a min must exist in current assets
    for label, cons in esg_constraints.items():
        if 'min' in cons:
            if not any(esg_for_assets == label):
                raise ValueError(
                    f"ESG label '{label}' has a min constraint but no asset has this label in this universe/period!"
                )

    # 3) consistency: min <= max (if both present)
    for label, cons in esg_constraints.items():
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"For ESG '{label}', min={cons['min']} > max={cons['max']}"
                )


def select_other_assets(metadata_other,
                        selected_asset_classes=None,
                        keep_ids_by_class=None):
    """
    Select non-equity assets (Other Class) based on client choices.

    metadata_other : DataFrame indexed by ID, with column 'ASSET_CLASS'
    selected_asset_classes : list of asset class names to INCLUDE
        e.g. ['Commodities', 'Fixed Income']
        None or empty -> include all asset classes.
    keep_ids_by_class : optional dict mapping asset class -> list of IDs to KEEP within that class.
        Example:
            {
                'Alternative Instruments': ['BTC', 'ETH'],  # only BTC & ETH in that class
                'Commodities': None,                        # keep all commodities
            }

        If a class is in selected_asset_classes but not in keep_ids_by_class,
        we keep all IDs in that class.

    Returns:
        Index (or list) of IDs to include from metadata_other.
    """

    df = metadata_other.copy()

    # 1) Filter by asset class
    if selected_asset_classes is not None and len(selected_asset_classes) > 0:
        df = df[df['ASSET_CLASS'].isin(selected_asset_classes)]

    # 2) If no per-class ID filter is provided -> keep all
    if keep_ids_by_class is None:
        return df.index

    # 3) Apply per-class ID filters (if any)
    selected_ids = []

    for asset_class in df['ASSET_CLASS'].unique():
        df_class = df[df['ASSET_CLASS'] == asset_class]

        ids_class = df_class.index

        # If user provided a list of IDs for this class, intersect
        ids_to_keep = keep_ids_by_class.get(asset_class, None)
        if ids_to_keep is not None:
            ids_class = ids_class.intersection(pd.Index(ids_to_keep))

        selected_ids.extend(list(ids_class))

    return pd.Index(selected_ids)


def check_asset_class_constraints_feasibility(assets, metadata_all, asset_class_constraints):
    if asset_class_constraints is None:
        return

    asset_class_for_assets = metadata_all['ASSET_CLASS'].reindex(assets)

    # 1) total min cannot exceed 1
    total_min = 0.0
    for cons in asset_class_constraints.values():
        if cons is None:
            continue
        total_min += cons.get('min', 0)

    if total_min > 1.0:
        raise ValueError(f"Total minimum asset-class weights {total_min:.2f} exceed 1.0")

    # 2) each asset class with a min must appear in assets
    for ac_name, cons in asset_class_constraints.items():
        if cons is None:
            continue
        if 'min' in cons:
            if not any(asset_class_for_assets == ac_name):
                raise ValueError(
                    f"Asset class '{ac_name}' has a min constraint but does not appear in the universe!"
                )

    # 3) consistency: min <= max
    for ac_name, cons in asset_class_constraints.items():
        if cons is None:
            continue
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"In asset class '{ac_name}', min={cons['min']} > max={cons['max']}"
                )
