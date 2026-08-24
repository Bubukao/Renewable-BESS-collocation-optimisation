"""
Phase 2 Investment and Operational Optimisation Model
=====================================================

This script implements the Phase 2 model described in Section 3.5.2 of the
dissertation methodology.

Methodology alignment
---------------------
- Endogenous investment capacities: K_PV, K_Wind, and K_BESS.
- Seven asset configurations are retained.
- Each enabled technology has a minimum active capacity of 10 MW.
- Renewable-containing configurations are evaluated at grid connection ratios
  of 60%, 70%, 80%, 90%, and 100% of optimised renewable capacity.
- Standalone BESS has a grid connection equal to its optimised power rating and
  is evaluated under Merchant only.
- Grid import is prohibited except for standalone BESS.
- Merchant exports use hourly 2024 N2EX day-ahead prices.
- Fixed-price PPA exports use £86.21/MWh.
- Project horizon: 25 years; discount rate: 7%.
- The operating year uses 16 cluster-weighted representative days
  (4 per season × 24 hours = 384 optimisation hours).
- Representative-day BESS SOC starts at 50% and is cyclic within each
  non-consecutive representative day.
- BESS duration is fixed at 4 hours, so energy capacity = 4 × K_BESS.
- Phase 1 linear degradation cost is excluded in Phase 2 to avoid
  double-counting because annual BESS OPEX is 2.5% of CAPEX and includes
  degradation-related augmentation.
- The objective maximises 25-year project NPV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


# ============================================================
# PHASE 2: INVESTMENT + OPERATION OPTIMISATION
# 25-year NPV model using representative seasonal days
# ============================================================


# ============================================================
# 1. File paths
# ============================================================

PRICE_PATH = Path("data/raw/GB_price.csv")

PV_PATH = Path("data/raw/ninja-pv-GB.csv")

WIND_PATH = Path("data/raw/ninja-wind-GB.csv")

OUTPUT_DIRECTORY = Path("results/phase2")

OUTPUT_FILE = OUTPUT_DIRECTORY / "Phase2_Investment_Results.xlsx"


# ============================================================
# 2. General settings
# ============================================================

START_DATE = "2024-01-01 00:00:00+00:00"
END_DATE = "2025-01-01 00:00:00+00:00"

WIND_COLUMN = None
MULTI_ZONE_WIND_METHOD = "mean"

PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.07

# Fixed-price pay-as-produced PPA
PPA_PRICE_GBP_PER_MWH = 86.21

# Phase 1 degradation cost is excluded in Phase 2 to avoid double-counting,
# because BESS OPEX includes augmentation associated with degradation.
DEGRADATION_COST_GBP_PER_MWH = 0.0

# Battery technical assumptions
BATTERY_DURATION_HOURS = 4.0
SOC_INITIAL = 0.50
SOC_MIN = 0.10
SOC_MAX = 0.90
ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95

# Representative-day model:
# 4 representative days per season = 16 x 24 = 384 hours.
# Representative days are selected as Typical / High PV / High Wind /
# High Price, then cluster-weighted by the number of actual days assigned
# to each representative day.
# Hourly resolution is retained because a 4-hour BESS is poorly represented
# by coarse 3- or 4-hour time steps.
REPRESENTATIVE_DAYS_PER_SEASON = 4
DELTA_T_HOURS = 1.0

# Investment budget
# Derived from the CAPEX of the Phase 1 configuration with the highest
# revenue per MWh exported: PV-BESS at the Phase 1 reference sizes.
# 160 MW PV + 100 MW BESS = £257.14 million.
TOTAL_BUDGET_GBP = 257_140_000.0

PHASE1_REFERENCE_PV_MW = 160.0
PHASE1_REFERENCE_WIND_MW = 0.0
PHASE1_REFERENCE_BESS_MW = 100.0

# IMPORTANT:
# A positive minimum is used so a nominal PV-Wind-BESS case cannot
# mathematically collapse into PV-only, Wind-only, etc.
# Set to 0.0 if you explicitly want the optimiser to be allowed
# to drop an enabled technology.
MIN_ACTIVE_CAPACITY_MW = 10.0

# Grid connection ratios used for renewable-containing scenarios.
# Grid capacity itself is NOT an investment decision variable.
GRID_LEVELS = [0.60, 0.70, 0.80, 0.90, 1.00]

# Standalone BESS:
# P_grid = K_BESS (battery power rating).
# Hybrid / renewable scenarios:
# P_grid = grid_level * (K_PV + K_Wind)

SOLVER_NAME = "highs"


# ============================================================
# 3. Cost assumptions
# ============================================================

# Wind
WIND_CAPEX_GBP_PER_MW = 1_588_000.0
WIND_OPEX_GBP_PER_MW_YEAR = 40_100.0

# Solar PV
PV_CAPEX_GBP_PER_MW = 659_000.0
PV_OPEX_GBP_PER_MW_YEAR = 9_300.0

# BESS
# CAPEX: £1,517/kW = £1,517,000/MW
BESS_CAPEX_GBP_PER_MW = 1_517_000.0

# Annual BESS OPEX = 2.5% of CAPEX/year.
# Phase 2 degradation cost is set to zero to avoid double-counting.
BESS_OPEX_SHARE_OF_CAPEX = 0.025
BESS_OPEX_GBP_PER_MW_YEAR = (
    BESS_OPEX_SHARE_OF_CAPEX
    * BESS_CAPEX_GBP_PER_MW
)


REFERENCE_PHASE1_CAPEX_GBP = (
    PHASE1_REFERENCE_PV_MW * PV_CAPEX_GBP_PER_MW
    + PHASE1_REFERENCE_WIND_MW * WIND_CAPEX_GBP_PER_MW
    + PHASE1_REFERENCE_BESS_MW * BESS_CAPEX_GBP_PER_MW
)

if not np.isclose(
    TOTAL_BUDGET_GBP,
    REFERENCE_PHASE1_CAPEX_GBP,
):
    raise ValueError(
        "TOTAL_BUDGET_GBP does not match the CAPEX of the "
        "Phase 1 reference PV-Wind-BESS configuration."
    )

# Maximum possible BESS capacity implied by the entire budget.
# Used as a valid Big-M for charge/discharge mutual exclusivity.
BESS_BIG_M_MW = TOTAL_BUDGET_GBP / BESS_CAPEX_GBP_PER_MW

# 25-year present-value factor for a constant annual cash flow.
ANNUITY_FACTOR = sum(
    1.0 / ((1.0 + DISCOUNT_RATE) ** year)
    for year in range(1, PROJECT_LIFETIME_YEARS + 1)
)


# ============================================================
# 4. Scenario definitions
# ============================================================

CONFIGURATIONS = {
    "PV": {
        "pv": 1,
        "wind": 0,
        "bess": 0,
        "grid_import": 0,
    },
    "Wind": {
        "pv": 0,
        "wind": 1,
        "bess": 0,
        "grid_import": 0,
    },
    "PV_Wind": {
        "pv": 1,
        "wind": 1,
        "bess": 0,
        "grid_import": 0,
    },
    "BESS": {
        "pv": 0,
        "wind": 0,
        "bess": 1,
        "grid_import": 1,
    },
    "PV_BESS": {
        "pv": 1,
        "wind": 0,
        "bess": 1,
        "grid_import": 0,
    },
    "Wind_BESS": {
        "pv": 0,
        "wind": 1,
        "bess": 1,
        "grid_import": 0,
    },
    "PV_Wind_BESS": {
        "pv": 1,
        "wind": 1,
        "bess": 1,
        "grid_import": 0,
    },
}

CONTRACTS = ["Merchant", "PPA"]


# ============================================================
# 5. Data import
# ============================================================

def load_price_data():
    df = pd.read_csv(PRICE_PATH)

    df["time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    )

    df = (
        df[["time", "price"]]
        .dropna()
        .drop_duplicates(subset="time", keep="first")
        .sort_values("time")
        .set_index("time")
    )

    return df


def load_pv_data():
    df = pd.read_csv(
        PV_PATH,
        skiprows=3,
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if "time" not in df.columns:
        raise ValueError(
            "PV file does not contain a 'time' column."
        )

    if "NATIONAL" not in df.columns:
        raise ValueError(
            "PV file does not contain a 'NATIONAL' column."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df["pv_cf"] = pd.to_numeric(
        df["NATIONAL"],
        errors="coerce",
    )

    df = (
        df[["time", "pv_cf"]]
        .dropna()
        .drop_duplicates(subset="time", keep="first")
        .sort_values("time")
        .set_index("time")
    )

    return df


def load_wind_data():
    df = pd.read_csv(
        WIND_PATH,
        skiprows=3,
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if "time" not in df.columns:
        raise ValueError(
            "Wind file does not contain a 'time' column."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    non_time_columns = [
        c for c in df.columns
        if c != "time"
    ]

    preferred_names = [
        "NATIONAL",
        "GB",
        "United Kingdom",
        "electricity",
    ]

    selected_column = None

    if WIND_COLUMN is not None:
        if WIND_COLUMN not in df.columns:
            raise ValueError(
                f"WIND_COLUMN='{WIND_COLUMN}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        selected_column = WIND_COLUMN

    else:
        for c in preferred_names:
            if c in df.columns:
                selected_column = c
                break

    if selected_column is not None:
        df["wind_cf"] = pd.to_numeric(
            df[selected_column],
            errors="coerce",
        )

    else:
        numeric = df[non_time_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        usable = [
            c for c in numeric.columns
            if numeric[c].notna().any()
        ]

        if len(usable) == 1:
            df["wind_cf"] = numeric[usable[0]]

        elif len(usable) > 1:
            if MULTI_ZONE_WIND_METHOD == "mean":
                print(
                    "Warning: multiple wind columns found. "
                    "Using unweighted regional mean."
                )
                df["wind_cf"] = numeric[usable].mean(axis=1)

            else:
                raise ValueError(
                    "Multiple wind columns found. "
                    "Set WIND_COLUMN explicitly."
                )

        else:
            raise ValueError(
                "No usable wind capacity-factor column found."
            )

    df = (
        df[["time", "wind_cf"]]
        .dropna()
        .drop_duplicates(subset="time", keep="first")
        .sort_values("time")
        .set_index("time")
    )

    return df


def prepare_hourly_data():
    price = load_price_data()
    pv = load_pv_data()
    wind = load_wind_data()

    data = (
        price
        .join(pv, how="inner")
        .join(wind, how="inner")
    )

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    data = data.loc[
        (data.index >= start)
        & (data.index < end)
    ].copy()

    data = data[
        ["price", "pv_cf", "wind_cf"]
    ].dropna()

    data["pv_cf"] = data["pv_cf"].clip(
        lower=0.0,
        upper=1.0,
    )

    data["wind_cf"] = data["wind_cf"].clip(
        lower=0.0,
        upper=1.0,
    )

    print(
        f"Prepared {len(data):,} hourly observations "
        f"from {data.index.min()} to {data.index.max()}."
    )

    return data


# ============================================================
# 6. Representative seasonal days
# ============================================================

def season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def select_representative_days(hourly_data):
    """
    Select four representative 24-hour days per season:

    1. Typical:
       Actual day whose combined hourly price / PV CF / wind CF profile
       is closest to the seasonal average daily profile.

    2. High_PV:
       Actual day with the highest daily average PV capacity factor.

    3. High_Wind:
       Actual day with the highest daily average wind capacity factor.

    4. High_Price:
       Actual day with the highest daily average wholesale price.

    Total optimisation horizon:
        4 seasons x 4 representative days x 24 hours = 384 hours.

    Representative days are NOT given equal weights. After selection,
    every complete actual day in the season is assigned to the nearest
    representative day using a 72-dimensional standardised hourly vector
    (24 price + 24 PV CF + 24 wind CF values). The weight of each
    representative day equals the number of actual days in its cluster.
    """

    data = hourly_data.copy()

    data["date"] = data.index.date
    data["hour"] = data.index.hour
    data["season"] = [
        season_from_month(m)
        for m in data.index.month
    ]

    # Keep only complete 24-hour days.
    day_counts = data.groupby("date").size()
    complete_dates = day_counts[
        day_counts == 24
    ].index

    data = data[
        data["date"].isin(complete_dates)
    ].copy()

    representative_rows = []
    representative_metadata = []

    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Autumn",
    ]

    rep_day_id = 0

    for season in season_order:

        season_df = data[
            data["season"] == season
        ].copy()

        dates = sorted(
            season_df["date"].unique()
        )

        if len(dates) < REPRESENTATIVE_DAYS_PER_SEASON:
            raise ValueError(
                f"Not enough complete days available for {season}."
            )

        # Standardise WITHIN each season so price, PV CF and wind CF
        # contribute comparably to the distance metric.
        z = season_df[
            ["price", "pv_cf", "wind_cf"]
        ].copy()

        for column in z.columns:

            mean_value = z[column].mean()
            std_value = z[column].std()

            if std_value <= 1e-12:
                z[column] = 0.0
            else:
                z[column] = (
                    z[column] - mean_value
                ) / std_value

        season_df[
            ["z_price", "z_pv_cf", "z_wind_cf"]
        ] = z[
            ["price", "pv_cf", "wind_cf"]
        ]

        # Build one 72-dimensional vector per actual day.
        day_vectors = {}

        for d in dates:

            day_df = (
                season_df[
                    season_df["date"] == d
                ]
                .sort_index()
            )

            day_vectors[d] = np.concatenate([
                day_df["z_price"].to_numpy(),
                day_df["z_pv_cf"].to_numpy(),
                day_df["z_wind_cf"].to_numpy(),
            ])

        all_vectors = np.vstack(
            [day_vectors[d] for d in dates]
        )

        seasonal_mean_vector = np.mean(
            all_vectors,
            axis=0,
        )

        daily_stats = (
            season_df
            .groupby("date")
            .agg(
                mean_price=("price", "mean"),
                mean_pv_cf=("pv_cf", "mean"),
                mean_wind_cf=("wind_cf", "mean"),
            )
        )

        # 1. Typical day.
        distance_to_mean = {
            d: float(
                np.linalg.norm(
                    day_vectors[d]
                    - seasonal_mean_vector
                )
            )
            for d in dates
        }

        typical_date = min(
            distance_to_mean,
            key=distance_to_mean.get,
        )

        # 2. High PV day.
        high_pv_date = (
            daily_stats["mean_pv_cf"]
            .idxmax()
        )

        # 3. High Wind day.
        high_wind_date = (
            daily_stats["mean_wind_cf"]
            .idxmax()
        )

        # 4. High Price day.
        high_price_date = (
            daily_stats["mean_price"]
            .idxmax()
        )

        selected_candidates = [
            ("Typical", typical_date),
            ("High_PV", high_pv_date),
            ("High_Wind", high_wind_date),
            ("High_Price", high_price_date),
        ]

        # Ensure four unique representative dates.
        selected = []
        used_dates = set()

        for label, selected_date in selected_candidates:

            if selected_date not in used_dates:
                selected.append(
                    (label, selected_date)
                )
                used_dates.add(
                    selected_date
                )

        # If two categories select the same actual date, fill the empty
        # slot with the next closest unused typical day.
        if len(selected) < REPRESENTATIVE_DAYS_PER_SEASON:

            fallback_dates = sorted(
                dates,
                key=lambda d: distance_to_mean[d],
            )

            for d in fallback_dates:

                if d not in used_dates:
                    selected.append(
                        ("Additional_Typical", d)
                    )
                    used_dates.add(d)

                if (
                    len(selected)
                    == REPRESENTATIVE_DAYS_PER_SEASON
                ):
                    break

        if (
            len(selected)
            != REPRESENTATIVE_DAYS_PER_SEASON
        ):
            raise RuntimeError(
                f"Could not select "
                f"{REPRESENTATIVE_DAYS_PER_SEASON} "
                f"unique representative days for {season}."
            )

        rep_dates = [
            d
            for _, d in selected
        ]

        # Assign every actual day to its nearest selected representative.
        cluster_members = {
            d: []
            for d in rep_dates
        }

        for actual_date in dates:

            distances_to_reps = {
                rep_date: float(
                    np.linalg.norm(
                        day_vectors[actual_date]
                        - day_vectors[rep_date]
                    )
                )
                for rep_date in rep_dates
            }

            nearest_rep_date = min(
                distances_to_reps,
                key=distances_to_reps.get,
            )

            cluster_members[
                nearest_rep_date
            ].append(
                actual_date
            )

        # Store selected representative days with cluster-derived weights.
        for day_type, selected_date in selected:

            selected_df = (
                season_df[
                    season_df["date"] == selected_date
                ]
                .sort_index()
                .copy()
            )

            represented_days = float(
                len(
                    cluster_members[
                        selected_date
                    ]
                )
            )

            if represented_days <= 0:
                raise RuntimeError(
                    f"Representative day {selected_date} "
                    f"in {season} received zero cluster weight."
                )

            selected_df["rep_day_id"] = (
                rep_day_id
            )

            selected_df["rep_hour"] = (
                np.arange(24)
            )

            selected_df["weight_days"] = (
                represented_days
            )

            selected_df["day_type"] = (
                day_type
            )

            representative_rows.append(
                selected_df
            )

            representative_metadata.append({
                "rep_day_id": rep_day_id,
                "season": season,
                "day_type": day_type,
                "selected_date": str(
                    selected_date
                ),
                "represented_days": represented_days,
                "cluster_share_of_season_pct": (
                    100.0
                    * represented_days
                    / len(dates)
                ),
                "mean_price_GBP_per_MWh": (
                    selected_df["price"].mean()
                ),
                "mean_pv_cf": (
                    selected_df["pv_cf"].mean()
                ),
                "mean_wind_cf": (
                    selected_df["wind_cf"].mean()
                ),
                "distance_to_seasonal_mean": (
                    distance_to_mean[
                        selected_date
                    ]
                ),
            })

            rep_day_id += 1

        # Safety check: seasonal cluster weights must recover all days.
        season_weight_sum = sum(
            len(
                cluster_members[d]
            )
            for d in rep_dates
        )

        if season_weight_sum != len(dates):
            raise RuntimeError(
                f"Cluster weights for {season} sum to "
                f"{season_weight_sum}, expected {len(dates)}."
            )

        print(
            f"\n{season}: "
            f"{len(dates)} complete days assigned "
            f"across {len(rep_dates)} representative days."
        )

        for label, rep_date in selected:
            print(
                f"  {label:<18} "
                f"{rep_date}  "
                f"weight={len(cluster_members[rep_date]):.0f} days"
            )

    rep = pd.concat(
        representative_rows,
        axis=0,
    )

    rep = rep[
        [
            "price",
            "pv_cf",
            "wind_cf",
            "season",
            "date",
            "day_type",
            "rep_day_id",
            "rep_hour",
            "weight_days",
        ]
    ].copy()

    metadata = pd.DataFrame(
        representative_metadata
    )

    total_days = metadata[
        "represented_days"
    ].sum()

    print(
        "\nRepresentative days selected "
        "with cluster-based annualisation weights:"
    )

    print(
        metadata.to_string(
            index=False
        )
    )

    print(
        f"\nRepresentative-day weights sum to "
        f"{total_days:.0f} days."
    )

    print(
        f"Total optimisation hours = "
        f"{len(rep):,}"
    )

    expected_hours = (
        4
        * REPRESENTATIVE_DAYS_PER_SEASON
        * 24
    )

    if len(rep) != expected_hours:
        raise RuntimeError(
            f"Representative horizon has {len(rep)} hours, "
            f"expected {expected_hours}."
        )

    return rep, metadata



# ============================================================
# 6B. Curtailment decomposition
# ============================================================

def split_curtailment(
    res_available_mw,
    grid_connection_mw,
    charge_mw,
    discharge_mw,
    grid_import_mw,
    total_curtailment_mw,
):
    """
    Split total renewable curtailment into:

    1. Physical curtailment:
       The minimum curtailment forced by the grid export limit,
       conditional on the battery charge/discharge decision actually
       chosen by the optimiser.

       physical = max(
           0,
           RES + grid import + battery discharge
           - battery charge - grid export limit
       )

    2. Economic curtailment:
       Any additional curtailment beyond the physically unavoidable
       amount. This captures voluntary non-export, for example during
       negative-price periods in merchant scenarios.

    The decomposition is performed after optimisation, so it does NOT
    change the NPV-maximising dispatch. It only improves interpretation
    of the reported curtailment results.
    """

    forced = max(
        0.0,
        float(res_available_mw)
        + float(grid_import_mw)
        + float(discharge_mw)
        - float(charge_mw)
        - float(grid_connection_mw),
    )

    # Numerical safeguard: physical curtailment cannot exceed the
    # optimiser's actual total curtailment.
    physical = min(
        max(float(total_curtailment_mw), 0.0),
        forced,
    )

    economic = max(
        0.0,
        float(total_curtailment_mw)
        - physical,
    )

    return physical, economic


# ============================================================
# 7. Scenario optimisation
# ============================================================

def optimise_scenario(
    rep_data,
    configuration_name,
    contract,
    grid_level=None,
):
    flags = CONFIGURATIONS[
        configuration_name
    ]

    pv_enabled = flags["pv"]
    wind_enabled = flags["wind"]
    bess_enabled = flags["bess"]
    grid_import_enabled = flags[
        "grid_import"
    ]

    # Standalone BESS is merchant only.
    if (
        configuration_name == "BESS"
        and contract != "Merchant"
    ):
        return None, None

    if (
        configuration_name != "BESS"
        and grid_level is None
    ):
        raise ValueError(
            "Renewable-containing scenarios require "
            "a grid_level."
        )

    rows = rep_data.reset_index(
        names="timestamp"
    ).copy()

    n_periods = len(rows)

    # Index lookup for representative-day/hour combinations.
    period_lookup = {
        (
            int(row.rep_day_id),
            int(row.rep_hour),
        ): i
        for i, row in rows.iterrows()
    }

    rep_day_ids = sorted(
        rows["rep_day_id"]
        .astype(int)
        .unique()
    )

    m = pyo.ConcreteModel(
        name=(
            f"Phase2_{configuration_name}_"
            f"{contract}_"
            f"{grid_level}"
        )
    )

    m.T = pyo.RangeSet(
        0,
        n_periods - 1,
    )

    # --------------------------------------------------------
    # 7.1 Investment capacity variables
    # --------------------------------------------------------

    m.K_pv = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    m.K_wind = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    m.K_bess = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    # Enforce technology availability and minimum size.
    if pv_enabled:
        m.pv_min = pyo.Constraint(
            expr=m.K_pv
            >= MIN_ACTIVE_CAPACITY_MW
        )
    else:
        m.pv_zero = pyo.Constraint(
            expr=m.K_pv == 0.0
        )

    if wind_enabled:
        m.wind_min = pyo.Constraint(
            expr=m.K_wind
            >= MIN_ACTIVE_CAPACITY_MW
        )
    else:
        m.wind_zero = pyo.Constraint(
            expr=m.K_wind == 0.0
        )

    if bess_enabled:
        m.bess_min = pyo.Constraint(
            expr=m.K_bess
            >= MIN_ACTIVE_CAPACITY_MW
        )
    else:
        m.bess_zero = pyo.Constraint(
            expr=m.K_bess == 0.0
        )

    # --------------------------------------------------------
    # 7.2 CAPEX and budget
    # --------------------------------------------------------

    m.total_capex = pyo.Expression(
        expr=(
            PV_CAPEX_GBP_PER_MW
            * m.K_pv

            + WIND_CAPEX_GBP_PER_MW
            * m.K_wind

            + BESS_CAPEX_GBP_PER_MW
            * m.K_bess
        )
    )

    m.budget_constraint = pyo.Constraint(
        expr=(
            m.total_capex
            <= TOTAL_BUDGET_GBP
        )
    )

    m.annual_opex = pyo.Expression(
        expr=(
            PV_OPEX_GBP_PER_MW_YEAR
            * m.K_pv

            + WIND_OPEX_GBP_PER_MW_YEAR
            * m.K_wind

            + BESS_OPEX_GBP_PER_MW_YEAR
            * m.K_bess
        )
    )

    # --------------------------------------------------------
    # 7.3 Operating variables
    # --------------------------------------------------------

    m.P_out = pyo.Var(
        m.T,
        domain=pyo.NonNegativeReals,
    )

    m.P_in = pyo.Var(
        m.T,
        domain=pyo.NonNegativeReals,
    )

    m.C = pyo.Var(
        m.T,
        domain=pyo.NonNegativeReals,
    )

    m.D = pyo.Var(
        m.T,
        domain=pyo.NonNegativeReals,
    )

    m.Curt = pyo.Var(
        m.T,
        domain=pyo.NonNegativeReals,
    )

    # Battery mode:
    # 1 = charge
    # 0 = discharge / idle
    m.battery_mode = pyo.Var(
        m.T,
        domain=pyo.Binary,
    )

    # Energy states indexed by representative day and
    # hour boundary 0..24.
    m.R = pyo.Set(
        initialize=rep_day_ids
    )

    m.H_STATE = pyo.RangeSet(
        0,
        24,
    )

    m.E = pyo.Var(
        m.R,
        m.H_STATE,
        domain=pyo.NonNegativeReals,
    )

    # --------------------------------------------------------
    # 7.4 Available renewable generation
    # --------------------------------------------------------

    def pv_available_rule(m, t):
        cf = float(
            rows.loc[t, "pv_cf"]
        )
        return cf * m.K_pv

    def wind_available_rule(m, t):
        cf = float(
            rows.loc[t, "wind_cf"]
        )
        return cf * m.K_wind

    m.PV_available = pyo.Expression(
        m.T,
        rule=pv_available_rule,
    )

    m.Wind_available = pyo.Expression(
        m.T,
        rule=wind_available_rule,
    )

    m.RES_available = pyo.Expression(
        m.T,
        rule=lambda m, t: (
            m.PV_available[t]
            + m.Wind_available[t]
        ),
    )

    # --------------------------------------------------------
    # 7.5 Grid connection expression
    # --------------------------------------------------------

    if configuration_name == "BESS":
        # Standalone battery:
        # grid connection equals battery power rating.
        m.P_grid = pyo.Expression(
            expr=m.K_bess
        )
    else:
        # Renewable-containing scenarios:
        # grid is a fixed ratio of installed renewable capacity.
        m.P_grid = pyo.Expression(
            expr=(
                float(grid_level)
                * (
                    m.K_pv
                    + m.K_wind
                )
            )
        )

    # --------------------------------------------------------
    # 7.6 Grid limits
    # --------------------------------------------------------

    m.export_limit = pyo.Constraint(
        m.T,
        rule=lambda m, t: (
            m.P_out[t]
            <= m.P_grid
        ),
    )

    def import_limit_rule(m, t):
        if grid_import_enabled:
            return (
                m.P_in[t]
                <= m.P_grid
            )
        return m.P_in[t] == 0.0

    m.import_limit = pyo.Constraint(
        m.T,
        rule=import_limit_rule,
    )

    # --------------------------------------------------------
    # 7.7 Power balance
    # --------------------------------------------------------

    def power_balance_rule(m, t):
        return (
            m.RES_available[t]
            + m.P_in[t]
            + m.D[t]
            ==
            m.P_out[t]
            + m.C[t]
            + m.Curt[t]
        )

    m.power_balance = pyo.Constraint(
        m.T,
        rule=power_balance_rule,
    )

    # Curtailment cannot exceed available renewable generation.
    m.curtailment_limit = pyo.Constraint(
        m.T,
        rule=lambda m, t: (
            m.Curt[t]
            <= m.RES_available[t]
        ),
    )

    # --------------------------------------------------------
    # 7.8 Battery power constraints
    # --------------------------------------------------------

    if bess_enabled:

        # Actual battery power cannot exceed installed K_BESS.
        m.charge_capacity = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.C[t]
                <= m.K_bess
            ),
        )

        m.discharge_capacity = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.D[t]
                <= m.K_bess
            ),
        )

        # Big-M formulation avoids the nonlinear product:
        # K_BESS * binary_mode.
        m.charge_mode = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.C[t]
                <= BESS_BIG_M_MW
                * m.battery_mode[t]
            ),
        )

        m.discharge_mode = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.D[t]
                <= BESS_BIG_M_MW
                * (
                    1
                    - m.battery_mode[t]
                )
            ),
        )

    else:
        m.charge_zero = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.C[t] == 0.0
            ),
        )

        m.discharge_zero = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.D[t] == 0.0
            ),
        )

        m.mode_zero = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.battery_mode[t]
                == 0
            ),
        )

    # --------------------------------------------------------
    # 7.9 Battery SOC constraints
    # --------------------------------------------------------

    if bess_enabled:

        # State bounds depend linearly on installed BESS power:
        # E_capacity = 4 h * K_BESS.
        def soc_lower_rule(m, r, h):
            return (
                m.E[r, h]
                >= SOC_MIN
                * BATTERY_DURATION_HOURS
                * m.K_bess
            )

        def soc_upper_rule(m, r, h):
            return (
                m.E[r, h]
                <= SOC_MAX
                * BATTERY_DURATION_HOURS
                * m.K_bess
            )

        m.soc_lower = pyo.Constraint(
            m.R,
            m.H_STATE,
            rule=soc_lower_rule,
        )

        m.soc_upper = pyo.Constraint(
            m.R,
            m.H_STATE,
            rule=soc_upper_rule,
        )

        # Each representative day begins at 50% SOC.
        # Because the selected seasonal days are not consecutive
        # calendar days, energy cannot be transferred artificially
        # from one representative season into another.
        m.initial_soc = pyo.Constraint(
            m.R,
            rule=lambda m, r: (
                m.E[r, 0]
                ==
                SOC_INITIAL
                * BATTERY_DURATION_HOURS
                * m.K_bess
            ),
        )

        # Hourly state transition within each representative day.
        def soc_evolution_rule(m, r, h):
            t = period_lookup[
                (int(r), int(h))
            ]

            return (
                m.E[r, h + 1]
                ==
                m.E[r, h]
                + ETA_CHARGE
                * m.C[t]
                * DELTA_T_HOURS
                - (
                    m.D[t]
                    / ETA_DISCHARGE
                )
                * DELTA_T_HOURS
            )

        m.soc_evolution = pyo.Constraint(
            m.R,
            pyo.RangeSet(0, 23),
            rule=soc_evolution_rule,
        )

        # Cyclic representative-day SOC:
        # prevents a representative day from creating or destroying
        # stored energy when annualised by its seasonal weight.
        m.cyclic_soc = pyo.Constraint(
            m.R,
            rule=lambda m, r: (
                m.E[r, 24]
                == m.E[r, 0]
            ),
        )

    else:
        m.energy_zero = pyo.Constraint(
            m.R,
            m.H_STATE,
            rule=lambda m, r, h: (
                m.E[r, h] == 0.0
            ),
        )

    # --------------------------------------------------------
    # 7.10 Annual operating cash flow
    # --------------------------------------------------------

    def export_price_for_period(t):
        if contract == "PPA":
            return PPA_PRICE_GBP_PER_MWH
        return float(
            rows.loc[t, "price"]
        )

    annual_export_revenue = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * export_price_for_period(t)
        * m.P_out[t]
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    annual_import_cost = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * float(
            rows.loc[t, "price"]
        )
        * m.P_in[t]
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    annual_degradation_cost = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * DEGRADATION_COST_GBP_PER_MWH
        * m.D[t]
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    m.annual_export_revenue = pyo.Expression(
        expr=annual_export_revenue
    )

    m.annual_import_cost = pyo.Expression(
        expr=annual_import_cost
    )

    m.annual_degradation_cost = pyo.Expression(
        expr=annual_degradation_cost
    )

    m.annual_operating_cashflow = pyo.Expression(
        expr=(
            m.annual_export_revenue
            - m.annual_import_cost
            - m.annual_degradation_cost
            - m.annual_opex
        )
    )

    # --------------------------------------------------------
    # 7.11 NPV objective
    # --------------------------------------------------------

    m.project_npv = pyo.Expression(
        expr=(
            -m.total_capex
            + ANNUITY_FACTOR
            * m.annual_operating_cashflow
        )
    )

    m.objective = pyo.Objective(
        expr=m.project_npv,
        sense=pyo.maximize,
    )

    # --------------------------------------------------------
    # 7.12 Solve
    # --------------------------------------------------------

    solver = pyo.SolverFactory(
        SOLVER_NAME
    )

    result = solver.solve(
        m,
        tee=False,
    )

    termination = str(
        result.solver.termination_condition
    )

    if termination.lower() not in {
        "optimal",
        "locallyoptimal",
        "globallyoptimal",
    }:
        raise RuntimeError(
            f"Scenario {configuration_name}, {contract}, "
            f"grid={grid_level}: solver terminated with "
            f"{termination}"
        )

    # --------------------------------------------------------
    # 7.13 Extract summary
    # --------------------------------------------------------

    k_pv = pyo.value(m.K_pv)
    k_wind = pyo.value(m.K_wind)
    k_bess = pyo.value(m.K_bess)
    p_grid = pyo.value(m.P_grid)

    total_capex = pyo.value(
        m.total_capex
    )

    annual_opex_value = pyo.value(
        m.annual_opex
    )

    annual_export_revenue_value = pyo.value(
        m.annual_export_revenue
    )

    annual_import_cost_value = pyo.value(
        m.annual_import_cost
    )

    annual_degradation_cost_value = pyo.value(
        m.annual_degradation_cost
    )

    annual_cashflow = pyo.value(
        m.annual_operating_cashflow
    )

    npv = pyo.value(
        m.project_npv
    )

    annual_export_mwh = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * pyo.value(
            m.P_out[t]
        )
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    revenue_per_mwh_export = (
        annual_export_revenue_value / annual_export_mwh
        if annual_export_mwh > 1e-9
        else np.nan
    )

    annual_import_mwh = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * pyo.value(
            m.P_in[t]
        )
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    annual_charge_mwh = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * pyo.value(
            m.C[t]
        )
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    annual_discharge_mwh = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * pyo.value(
            m.D[t]
        )
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    # --------------------------------------------------------
    # Curtailment decomposition:
    # total = physical + economic
    # --------------------------------------------------------

    annual_total_curtailment_mwh = 0.0
    annual_physical_curtailment_mwh = 0.0
    annual_economic_curtailment_mwh = 0.0

    for t in range(n_periods):

        weight = float(
            rows.loc[t, "weight_days"]
        )

        res_available_t = pyo.value(
            m.RES_available[t]
        )

        total_curt_t = pyo.value(
            m.Curt[t]
        )

        charge_t = pyo.value(
            m.C[t]
        )

        discharge_t = pyo.value(
            m.D[t]
        )

        import_t = pyo.value(
            m.P_in[t]
        )

        physical_curt_t, economic_curt_t = (
            split_curtailment(
                res_available_mw=res_available_t,
                grid_connection_mw=p_grid,
                charge_mw=charge_t,
                discharge_mw=discharge_t,
                grid_import_mw=import_t,
                total_curtailment_mw=total_curt_t,
            )
        )

        annual_total_curtailment_mwh += (
            weight
            * total_curt_t
            * DELTA_T_HOURS
        )

        annual_physical_curtailment_mwh += (
            weight
            * physical_curt_t
            * DELTA_T_HOURS
        )

        annual_economic_curtailment_mwh += (
            weight
            * economic_curt_t
            * DELTA_T_HOURS
        )

    annual_res_generation_mwh = sum(
        float(
            rows.loc[t, "weight_days"]
        )
        * pyo.value(
            m.RES_available[t]
        )
        * DELTA_T_HOURS
        for t in range(n_periods)
    )

    total_curtailment_rate = (
        annual_total_curtailment_mwh
        / annual_res_generation_mwh
        if annual_res_generation_mwh > 1e-9
        else 0.0
    )

    physical_curtailment_rate = (
        annual_physical_curtailment_mwh
        / annual_res_generation_mwh
        if annual_res_generation_mwh > 1e-9
        else 0.0
    )

    economic_curtailment_rate = (
        annual_economic_curtailment_mwh
        / annual_res_generation_mwh
        if annual_res_generation_mwh > 1e-9
        else 0.0
    )

    budget_used_pct = (
        total_capex
        / TOTAL_BUDGET_GBP
        * 100.0
    )

    summary = {
        "Configuration": configuration_name,
        "Contract": contract,
        "Grid_level": (
            grid_level
            if grid_level is not None
            else np.nan
        ),
        "K_PV_MW": k_pv,
        "K_Wind_MW": k_wind,
        "K_BESS_MW": k_bess,
        "BESS_energy_MWh": (
            BATTERY_DURATION_HOURS
            * k_bess
        ),
        "Grid_connection_MW": p_grid,
        "CAPEX_GBP": total_capex,
        "Budget_used_pct": budget_used_pct,
        "Annual_OPEX_GBP": annual_opex_value,
        "Annual_export_revenue_GBP": (
            annual_export_revenue_value
        ),
        "Annual_import_cost_GBP": (
            annual_import_cost_value
        ),
        "Annual_degradation_cost_GBP": (
            annual_degradation_cost_value
        ),
        "Annual_operating_cashflow_GBP": (
            annual_cashflow
        ),
        "NPV_25yr_GBP": npv,
        "Annual_RES_generation_MWh": (
            annual_res_generation_mwh
        ),
        "Annual_export_MWh": annual_export_mwh,
        "Revenue_per_MWh_export_GBP": revenue_per_mwh_export,
        "Annual_import_MWh": annual_import_mwh,
        "Annual_charge_MWh": annual_charge_mwh,
        "Annual_discharge_MWh": (
            annual_discharge_mwh
        ),
        # Total curtailment is retained for backward compatibility.
        "Annual_curtailment_MWh": (
            annual_total_curtailment_mwh
        ),
        "Curtailment_rate_pct": (
            total_curtailment_rate
            * 100.0
        ),

        # New decomposition for interpretation.
        "Annual_physical_curtailment_MWh": (
            annual_physical_curtailment_mwh
        ),
        "Physical_curtailment_rate_pct": (
            physical_curtailment_rate
            * 100.0
        ),
        "Annual_economic_curtailment_MWh": (
            annual_economic_curtailment_mwh
        ),
        "Economic_curtailment_rate_pct": (
            economic_curtailment_rate
            * 100.0
        ),
        "Solver_termination": termination,
    }

    # --------------------------------------------------------
    # 7.14 Extract representative dispatch
    # --------------------------------------------------------

    dispatch_rows = []

    for t in range(n_periods):
        r = int(
            rows.loc[t, "rep_day_id"]
        )
        h = int(
            rows.loc[t, "rep_hour"]
        )

        pv_available = pyo.value(
            m.PV_available[t]
        )

        wind_available = pyo.value(
            m.Wind_available[t]
        )

        res_available = (
            pv_available
            + wind_available
        )

        p_out = pyo.value(
            m.P_out[t]
        )

        p_in = pyo.value(
            m.P_in[t]
        )

        charge = pyo.value(
            m.C[t]
        )

        discharge = pyo.value(
            m.D[t]
        )

        curt = pyo.value(
            m.Curt[t]
        )

        physical_curt, economic_curt = (
            split_curtailment(
                res_available_mw=res_available,
                grid_connection_mw=p_grid,
                charge_mw=charge,
                discharge_mw=discharge,
                grid_import_mw=p_in,
                total_curtailment_mw=curt,
            )
        )

        soc_start = (
            pyo.value(
                m.E[r, h]
            )
            if bess_enabled
            else 0.0
        )

        soc_end = (
            pyo.value(
                m.E[r, h + 1]
            )
            if bess_enabled
            else 0.0
        )

        bess_energy_capacity = (
            BATTERY_DURATION_HOURS
            * k_bess
        )

        soc_start_pct = (
            100.0
            * soc_start
            / bess_energy_capacity
            if bess_energy_capacity > 1e-9
            else 0.0
        )

        soc_end_pct = (
            100.0
            * soc_end
            / bess_energy_capacity
            if bess_energy_capacity > 1e-9
            else 0.0
        )

        dispatch_rows.append({
            "Configuration": configuration_name,
            "Contract": contract,
            "Grid_level": (
                grid_level
                if grid_level is not None
                else np.nan
            ),
            "Season": rows.loc[t, "season"],
            "Representative_date": str(
                rows.loc[t, "date"]
            ),
            "Hour": h,
            "Weight_days": float(
                rows.loc[t, "weight_days"]
            ),
            "Price_GBP_per_MWh": float(
                rows.loc[t, "price"]
            ),
            "PV_CF": float(
                rows.loc[t, "pv_cf"]
            ),
            "Wind_CF": float(
                rows.loc[t, "wind_cf"]
            ),
            "PV_available_MW": pv_available,
            "Wind_available_MW": wind_available,
            "RES_available_MW": res_available,
            "P_out_MW": p_out,
            "P_in_MW": p_in,
            "Charge_MW": charge,
            "Discharge_MW": discharge,
            # Total curtailment retained for backward compatibility.
            "Curtailment_MW": curt,
            "Physical_curtailment_MW": physical_curt,
            "Economic_curtailment_MW": economic_curt,
            "SOC_start_MWh": soc_start,
            "SOC_end_MWh": soc_end,
            "SOC_start_pct": soc_start_pct,
            "SOC_end_pct": soc_end_pct,
        })

    dispatch = pd.DataFrame(
        dispatch_rows
    )

    return summary, dispatch


# ============================================================
# 8. Run all scenarios
# ============================================================

def run_all_scenarios(rep_data):
    summaries = []
    dispatch_frames = []

    # Renewable-containing configurations:
    # 6 configurations x 5 grid levels x 2 contracts = 60.
    renewable_configurations = [
        name
        for name in CONFIGURATIONS
        if name != "BESS"
    ]

    total_runs = (
        len(renewable_configurations)
        * len(GRID_LEVELS)
        * len(CONTRACTS)
        + 1
    )

    run_number = 0

    for config_name in renewable_configurations:

        for grid_level in GRID_LEVELS:

            for contract in CONTRACTS:
                run_number += 1

                print(
                    f"\nRunning Phase 2 scenario "
                    f"{run_number}/{total_runs}: "
                    f"{config_name}, "
                    f"{contract}, "
                    f"grid={grid_level:.0%}"
                )

                summary, dispatch = (
                    optimise_scenario(
                        rep_data=rep_data,
                        configuration_name=config_name,
                        contract=contract,
                        grid_level=grid_level,
                    )
                )

                summaries.append(
                    summary
                )

                dispatch_frames.append(
                    dispatch
                )

    # Standalone BESS merchant scenario.
    run_number += 1

    print(
        f"\nRunning Phase 2 scenario "
        f"{run_number}/{total_runs}: "
        f"BESS, Merchant"
    )

    summary, dispatch = (
        optimise_scenario(
            rep_data=rep_data,
            configuration_name="BESS",
            contract="Merchant",
            grid_level=None,
        )
    )

    summaries.append(
        summary
    )

    dispatch_frames.append(
        dispatch
    )

    summary_df = pd.DataFrame(
        summaries
    )

    dispatch_df = pd.concat(
        dispatch_frames,
        ignore_index=True,
    )

    return summary_df, dispatch_df


# ============================================================
# 9. Excel export
# ============================================================

def export_results(
    summary_df,
    dispatch_df,
    rep_metadata,
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = pd.DataFrame({
        "Parameter": [
            "Project lifetime",
            "Discount rate",
            "Investment budget",
            "Minimum active capacity",
            "PV CAPEX",
            "PV OPEX",
            "Wind CAPEX",
            "Wind OPEX",
            "BESS CAPEX",
            "BESS OPEX share",
            "BESS OPEX",
            "Battery duration",
            "Initial SOC",
            "Minimum SOC",
            "Maximum SOC",
            "Charge efficiency",
            "Discharge efficiency",
            "Phase 2 degradation cost",
            "PPA price",
            "Representative days",
            "Representative hours",
            "Representative-day selection",
            "Representative-day weighting",
            "Annual cash-flow assumption",
            "Grid levels",
            "Standalone BESS grid rule",
            "Curtailment reporting",
        ],
        "Value": [
            PROJECT_LIFETIME_YEARS,
            DISCOUNT_RATE,
            TOTAL_BUDGET_GBP,
            MIN_ACTIVE_CAPACITY_MW,
            PV_CAPEX_GBP_PER_MW,
            PV_OPEX_GBP_PER_MW_YEAR,
            WIND_CAPEX_GBP_PER_MW,
            WIND_OPEX_GBP_PER_MW_YEAR,
            BESS_CAPEX_GBP_PER_MW,
            BESS_OPEX_SHARE_OF_CAPEX,
            BESS_OPEX_GBP_PER_MW_YEAR,
            BATTERY_DURATION_HOURS,
            SOC_INITIAL,
            SOC_MIN,
            SOC_MAX,
            ETA_CHARGE,
            ETA_DISCHARGE,
            DEGRADATION_COST_GBP_PER_MWH,
            PPA_PRICE_GBP_PER_MWH,
            16,
            384,
            "Typical + High PV + High Wind + High Price per season",
            "Nearest-profile cluster size within each season",
            (
                "Representative 2024 operating year "
                "repeated as constant real annual cash flow "
                "and discounted over 25 years"
            ),
            ", ".join(
                f"{x:.0%}"
                for x in GRID_LEVELS
            ),
            (
                "Grid connection equals optimised "
                "BESS power rating"
            ),
            (
                "Total curtailment is decomposed after optimisation "
                "into physically unavoidable grid-limited curtailment "
                "and additional economic curtailment"
            ),
        ],
        "Unit_or_note": [
            "years",
            "fraction",
            "GBP",
            "MW",
            "GBP/MW",
            "GBP/MW-year",
            "GBP/MW",
            "GBP/MW-year",
            "GBP/MW",
            "fraction of CAPEX/year",
            "GBP/MW-year",
            "hours",
            "fraction",
            "fraction",
            "fraction",
            "fraction",
            "fraction",
            "GBP/MWh discharged",
            "GBP/MWh",
            "4 selected days per season; cluster-weighted",
            "hours/year before weighting",
            "selection rule",
            "number of actual days assigned to each representative day",
            "No escalation / no interannual variation",
            "renewable grid ratio",
            "P_grid = K_BESS",
            "Post-processing decomposition; objective function unchanged",
        ],
    })

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Scenario_Summary",
            index=False,
        )

        dispatch_df.to_excel(
            writer,
            sheet_name="Representative_Dispatch",
            index=False,
        )

        rep_metadata.to_excel(
            writer,
            sheet_name="Representative_Days",
            index=False,
        )

        parameters.to_excel(
            writer,
            sheet_name="Model_Parameters",
            index=False,
        )

    print(
        f"\nResults exported to:\n{OUTPUT_FILE}"
    )


# ============================================================
# 10. Main
# ============================================================

def main():
    print(
        "\n"
        "============================================================\n"
        "PHASE 2 INVESTMENT + OPERATION OPTIMISATION\n"
        "============================================================"
    )

    print(
        f"\n25-year annuity factor at "
        f"{DISCOUNT_RATE:.1%}: "
        f"{ANNUITY_FACTOR:.6f}"
    )

    print(
        f"BESS CAPEX: "
        f"£{BESS_CAPEX_GBP_PER_MW:,.0f}/MW "
        f"for a {BATTERY_DURATION_HOURS:.0f}-hour system"
    )

    print(
        f"BESS OPEX: "
        f"£{BESS_OPEX_GBP_PER_MW_YEAR:,.0f}/MW-year "
        f"({BESS_OPEX_SHARE_OF_CAPEX:.2%} of CAPEX)"
    )

    hourly_data = (
        prepare_hourly_data()
    )

    rep_data, rep_metadata = (
        select_representative_days(
            hourly_data
        )
    )

    summary_df, dispatch_df = (
        run_all_scenarios(
            rep_data
        )
    )

    # Rank best scenarios by NPV.
    summary_df = summary_df.sort_values(
        "NPV_25yr_GBP",
        ascending=False,
    ).reset_index(drop=True)

    print(
        "\nTop 10 scenarios by NPV:\n"
    )

    display_columns = [
        "Configuration",
        "Contract",
        "Grid_level",
        "K_PV_MW",
        "K_Wind_MW",
        "K_BESS_MW",
        "CAPEX_GBP",
        "Annual_operating_cashflow_GBP",
        "NPV_25yr_GBP",
    ]

    print(
        summary_df[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    export_results(
        summary_df=summary_df,
        dispatch_df=dispatch_df,
        rep_metadata=rep_metadata,
    )


if __name__ == "__main__":
    main()
