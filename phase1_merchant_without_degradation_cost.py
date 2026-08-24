"""
Phase 1 Sensitivity Analysis: Merchant without Battery Degradation Cost
======================================================================

Purpose
-------
This script reproduces the Phase 1 Merchant operational model while setting
the linear battery degradation cost (c_deg) to £0/MWh discharged.

The sensitivity case is identical to the Phase 1 methodology in all other
respects:
- PV capacity: 160 MW
- Wind capacity: 130 MW
- BESS: 100 MW / 400 MWh (4 h)
- Initial SOC: 50%
- SOC bounds: 10%-90%
- Charge/discharge efficiency: 95%
- Hourly 2024 Great Britain data
- Grid connection for renewable configurations: 60%-100% of installed RES
- Standalone BESS grid connection: fixed at 100 MW
- Grid import: prohibited except for standalone BESS
- Contract: Merchant only
- No curtailment penalty
- Terminal SOC returns to the initial SOC at the end of the modelling horizon

The only intentional change from the Phase 1 baseline is:
    c_deg = £42.80/MWh  ->  £0/MWh
"""

```
import re
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. File paths
# ============================================================

PRICE_PATH = Path("data/raw/GB_price.csv")

PV_PATH = Path("data/raw/ninja-pv-GB.csv")

WIND_PATH = Path("data/raw/ninja-wind-GB.csv")

OUTPUT_DIRECTORY = Path("results/phase1/sensitivity_no_degradation")


# ============================================================
# 2. User settings
# ============================================================

START_DATE = "2024-01-01 00:00:00+00:00"
END_DATE = "2025-01-01 00:00:00+00:00"

# If the parsed wind file contains one national column such as GB,
# United Kingdom, or electricity, the script selects it automatically.
#
# If it contains many regional columns, set the exact desired column name
# here after checking the printed list.
WIND_COLUMN = None

# If WIND_COLUMN remains None and multiple regional columns are found:
#   "mean"  -> use the unweighted average as a temporary proxy
#   "error" -> stop and ask you to specify WIND_COLUMN
MULTI_ZONE_WIND_METHOD = "mean"

# ============================================================
# 3. Model parameters
# ============================================================

PV_CAPACITY_MW = 160.0
WIND_CAPACITY_MW = 130.0

BATTERY_ENERGY_MWH = 400.0
BATTERY_POWER_MW = 100.0

SOC_INITIAL = 0.50
SOC_MIN = 0.10
SOC_MAX = 0.90

ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95

DEGRADATION_COST_GBP_PER_MWH = 42.80

DELTA_T = 1.0

# ============================================================
# 4. Data import and preprocessing
# ============================================================

print("\nLoading input data...")

# ------------------------------------------------------------
# 4.1 Day-ahead electricity prices
# ------------------------------------------------------------

price_data = pd.read_csv(PRICE_PATH)

# Parse timestamps as UTC
price_data["time"] = pd.to_datetime(
    price_data["start_time"],
    utc=True,
    errors="coerce"
)

price_data["price"] = pd.to_numeric(
    price_data["price"],
    errors="coerce"
)

# Keep only required columns
price_data = price_data[
    ["time", "price"]
].copy()

# Remove invalid observations
price_data = price_data.dropna(
    subset=["time", "price"]
)

# Remove duplicate timestamps if any
price_data = price_data.drop_duplicates(
    subset="time",
    keep="first"
)

# Sort chronologically because the original price file
# is ordered from newest to oldest
price_data = price_data.sort_values(
    "time"
)

price_data = price_data.set_index("time")


# ------------------------------------------------------------
# 4.2 PV capacity factor
# ------------------------------------------------------------
#
# Renewables.ninja files contain three metadata rows before
# the actual column names:
#
# row 1 = dataset description
# row 2 = units
# row 3 = metadata dictionary
# row 4 = actual header ("time", "NATIONAL", ...)
#
# Therefore skiprows=3.
# ------------------------------------------------------------

pv_data = pd.read_csv(
    PV_PATH,
    skiprows=3
)

# Clean column names
pv_data.columns = [
    str(col).strip()
    for col in pv_data.columns
]

print("\nPV columns:")
print(pv_data.columns.tolist())

if "time" not in pv_data.columns:
    raise ValueError(
        "PV file does not contain a 'time' column "
        "after skipping the metadata rows."
    )

if "NATIONAL" not in pv_data.columns:
    raise ValueError(
        "PV file does not contain a 'NATIONAL' column."
    )

pv_data["time"] = pd.to_datetime(
    pv_data["time"],
    utc=True,
    errors="coerce"
)

pv_data["pv_cf"] = pd.to_numeric(
    pv_data["NATIONAL"],
    errors="coerce"
)

pv_data = pv_data[
    ["time", "pv_cf"]
].copy()

pv_data = pv_data.dropna(
    subset=["time", "pv_cf"]
)

pv_data = pv_data.drop_duplicates(
    subset="time",
    keep="first"
)

pv_data = pv_data.sort_values(
    "time"
)

pv_data = pv_data.set_index("time")


# ------------------------------------------------------------
# 4.3 Wind capacity factor
# ------------------------------------------------------------

wind_data = pd.read_csv(
    WIND_PATH,
    skiprows=3
)

wind_data.columns = [
    str(col).strip()
    for col in wind_data.columns
]

print("\nWind columns:")
print(wind_data.columns.tolist())

if "time" not in wind_data.columns:
    raise ValueError(
        "Wind file does not contain a 'time' column "
        "after skipping the metadata rows."
    )

wind_data["time"] = pd.to_datetime(
    wind_data["time"],
    utc=True,
    errors="coerce"
)

# ------------------------------------------------------------
# Select wind column
#
# Priority:
# 1. User explicitly specifies WIND_COLUMN
# 2. NATIONAL
# 3. GB
# 4. United Kingdom
# 5. electricity
# 6. Regional mean, only if requested
# ------------------------------------------------------------

if WIND_COLUMN is not None:

    if WIND_COLUMN not in wind_data.columns:
        raise ValueError(
            f"WIND_COLUMN='{WIND_COLUMN}' was not found.\n"
            f"Available wind columns:\n"
            f"{wind_data.columns.tolist()}"
        )

    selected_wind_column = WIND_COLUMN

else:

    national_candidates = [
        "NATIONAL",
        "GB",
        "United Kingdom",
        "electricity",
    ]

    selected_wind_column = None

    for candidate in national_candidates:

        if candidate in wind_data.columns:
            selected_wind_column = candidate
            break

    # If no national column can be found:
    if selected_wind_column is None:

        regional_columns = [
            col
            for col in wind_data.columns
            if col != "time"
        ]

        if MULTI_ZONE_WIND_METHOD == "mean":

            print(
                "\nWarning: No national wind column found. "
                "Using the unweighted mean of regional columns."
            )

            for col in regional_columns:
                wind_data[col] = pd.to_numeric(
                    wind_data[col],
                    errors="coerce"
                )

            wind_data["wind_cf"] = (
                wind_data[
                    regional_columns
                ].mean(axis=1)
            )

        else:

            raise ValueError(
                "Multiple regional wind columns were found "
                "but no NATIONAL column could be identified. "
                "Please specify WIND_COLUMN."
            )


# If a national/specified column was found
if selected_wind_column is not None:

    print(
        f"\nSelected wind column: "
        f"{selected_wind_column}"
    )

    wind_data["wind_cf"] = pd.to_numeric(
        wind_data[selected_wind_column],
        errors="coerce"
    )


wind_data = wind_data[
    ["time", "wind_cf"]
].copy()

wind_data = wind_data.dropna(
    subset=["time", "wind_cf"]
)

wind_data = wind_data.drop_duplicates(
    subset="time",
    keep="first"
)

wind_data = wind_data.sort_values(
    "time"
)

wind_data = wind_data.set_index("time")


# ------------------------------------------------------------
# 4.4 Restrict all datasets to modelling period
# ------------------------------------------------------------

start_time = pd.Timestamp(START_DATE)
end_time = pd.Timestamp(END_DATE)

price_data = price_data.loc[
    (price_data.index >= start_time)
    & (price_data.index < end_time)
].copy()

pv_data = pv_data.loc[
    (pv_data.index >= start_time)
    & (pv_data.index < end_time)
].copy()

wind_data = wind_data.loc[
    (wind_data.index >= start_time)
    & (wind_data.index < end_time)
].copy()


# ------------------------------------------------------------
# 4.5 Merge price, PV and wind data
# ------------------------------------------------------------

model_data = (
    price_data
    .join(
        pv_data,
        how="inner"
    )
    .join(
        wind_data,
        how="inner"
    )
)

model_data = model_data[
    ["price", "pv_cf", "wind_cf"]
].copy()

model_data = model_data.dropna()

model_data = model_data.sort_index()


# ------------------------------------------------------------
# 4.6 Data validation
# ------------------------------------------------------------

if model_data.empty:
    raise ValueError(
        "model_data is empty after merging. "
        "Check the dates and timestamps."
    )

# Capacity factors should normally lie between 0 and 1.
# Slight numerical deviations are clipped.
model_data["pv_cf"] = model_data[
    "pv_cf"
].clip(
    lower=0.0,
    upper=1.0
)

model_data["wind_cf"] = model_data[
    "wind_cf"
].clip(
    lower=0.0,
    upper=1.0
)


# ------------------------------------------------------------
# 4.7 Check temporal continuity
# ------------------------------------------------------------

expected_index = pd.date_range(
    start=start_time,
    end=end_time,
    freq="h",
    inclusive="left"
)

missing_hours = expected_index.difference(
    model_data.index
)

print("\n================================================")
print("MODEL DATA SUMMARY")
print("================================================")

print(
    f"Start: {model_data.index.min()}"
)

print(
    f"End:   {model_data.index.max()}"
)

print(
    f"Number of observations: "
    f"{len(model_data)}"
)

print(
    f"Expected observations: "
    f"{len(expected_index)}"
)

print(
    f"Missing hourly observations: "
    f"{len(missing_hours)}"
)

if len(missing_hours) > 0:

    print("\nFirst missing timestamps:")

    print(
        missing_hours[:10]
    )


print("\nSummary statistics:")
print(
    model_data[
        ["price", "pv_cf", "wind_cf"]
    ].describe()
)


print("\nFirst five observations:")
print(
    model_data.head()
)


print("\nLast five observations:")
print(
    model_data.tail()
)

# ============================================================
# 4. Scenario settings
# ============================================================

# Merchant-only sensitivity analysis
MERCHANT_ALPHA = 0.0

GRID_LEVELS = [0.60, 0.70, 0.80, 0.90, 1.00]

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
        "grid_import": 1,   # Only standalone BESS can import
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

# ============================================================
# 5. Phase 1 sensitivity setting: Merchant without c_deg
# ============================================================
#
# This sensitivity case retains the Phase 1 methodology exactly
# except that the linear battery degradation cost is set to zero.
#
# Baseline Phase 1 c_deg: £42.80/MWh discharged
# Sensitivity case c_deg: £0/MWh discharged
#
# All technical capacities, grid rules, import rules, SOC rules,
# efficiencies, and hourly 2024 input data remain unchanged.
# ============================================================

DEGRADATION_COST_GBP_PER_MWH = 0.0

OUTPUT_DIRECTORY = Path(
    "results/phase1/sensitivity_no_degradation"
)
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

EXCEL_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "Phase1_Merchant_Without_Cdeg_Results.xlsx"
)

# ============================================================
# 6. Input validation
# ============================================================

# ============================================================

required_columns = {'price', 'pv_cf', 'wind_cf'}
missing_columns = required_columns.difference(model_data.columns)
if missing_columns:
    raise ValueError(f'model_data is missing required columns: {missing_columns}')

model_data = model_data.copy().sort_index().dropna(subset=['price', 'pv_cf', 'wind_cf'])
if model_data.empty:
    raise ValueError('model_data is empty.')

# ============================================================
# 7. Helper functions
# ============================================================

def get_installed_res_capacity(config_name):
    flags = CONFIGURATIONS[config_name]
    return float(flags['pv'] * PV_CAPACITY_MW + flags['wind'] * WIND_CAPACITY_MW)


def get_grid_connection_capacity(config_name, grid_level):
    if config_name == 'BESS':
        return float(BATTERY_POWER_MW)
    return float(grid_level * get_installed_res_capacity(config_name))


def build_generation_profiles(config_name, data):
    flags = CONFIGURATIONS[config_name]
    pv_generation = flags['pv'] * PV_CAPACITY_MW * data['pv_cf'].to_numpy()
    wind_generation = flags['wind'] * WIND_CAPACITY_MW * data['wind_cf'].to_numpy()
    return pv_generation, wind_generation, pv_generation + wind_generation


def add_curtailment_breakdown(hourly, grid_capacity_mw):
    hourly = hourly.copy()
    hourly['Renewable_export_MW'] = np.maximum(
        0.0, hourly['P_out_MW'] - hourly['Discharge_MW']
    )
    hourly['Grid_headroom_for_RES_MW'] = np.maximum(
        0.0, grid_capacity_mw - hourly['Discharge_MW']
    )
    hourly['Total_curtailment_MW'] = np.maximum(
        0.0, hourly['Curtailment_MW']
    )
    hourly['Physical_curtailment_MW'] = np.maximum(
        0.0,
        hourly['RES_generation_MW']
        - hourly['Charge_MW']
        - hourly['Grid_headroom_for_RES_MW']
    )
    hourly['Physical_curtailment_MW'] = np.minimum(
        hourly['Physical_curtailment_MW'], hourly['Total_curtailment_MW']
    )
    hourly['Economic_curtailment_MW'] = np.maximum(
        0.0,
        hourly['Total_curtailment_MW'] - hourly['Physical_curtailment_MW']
    )
    return hourly

# ============================================================
# 8. Merchant optimisation model
# ============================================================

def run_scenario(config_name, grid_level, solver_name='highs'):
    flags = CONFIGURATIONS[config_name]
    pv_enabled = flags['pv']
    wind_enabled = flags['wind']
    bess_enabled = flags['bess']
    grid_import_enabled = flags['grid_import']

    p_res_installed = get_installed_res_capacity(config_name)
    p_grid = get_grid_connection_capacity(config_name, grid_level)

    pv_generation, wind_generation, res_generation = build_generation_profiles(
        config_name, model_data
    )
    prices = model_data['price'].to_numpy()
    n_hours = len(model_data)

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, n_hours - 1)

    m.price = pyo.Param(
        m.T,
        initialize={t: float(prices[t]) for t in range(n_hours)}
    )
    m.res_generation = pyo.Param(
        m.T,
        initialize={t: float(res_generation[t]) for t in range(n_hours)}
    )

    m.P_out = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.P_in = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.C = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.D = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.Curt = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.E = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.battery_mode = pyo.Var(m.T, domain=pyo.Binary)

    # Merchant objective, with c_deg excluded (= 0)
    def objective_rule(m):
        return sum(
            (
                m.price[t] * m.P_out[t]
                - m.price[t] * m.P_in[t]
                - DEGRADATION_COST_GBP_PER_MWH * m.D[t]
            ) * DELTA_T
            for t in m.T
        )

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def export_limit_rule(m, t):
        return m.P_out[t] <= p_grid
    m.export_limit = pyo.Constraint(m.T, rule=export_limit_rule)

    def import_limit_rule(m, t):
        if grid_import_enabled == 1:
            return m.P_in[t] <= p_grid
        return m.P_in[t] == 0
    m.import_limit = pyo.Constraint(m.T, rule=import_limit_rule)

    def curtailment_limit_rule(m, t):
        return m.Curt[t] <= m.res_generation[t]
    m.curtailment_limit = pyo.Constraint(m.T, rule=curtailment_limit_rule)

    def charge_limit_rule(m, t):
        if bess_enabled == 0:
            return m.C[t] == 0
        return m.C[t] <= BATTERY_POWER_MW * m.battery_mode[t]
    m.charge_limit = pyo.Constraint(m.T, rule=charge_limit_rule)

    def discharge_limit_rule(m, t):
        if bess_enabled == 0:
            return m.D[t] == 0
        return m.D[t] <= BATTERY_POWER_MW * (1 - m.battery_mode[t])
    m.discharge_limit = pyo.Constraint(m.T, rule=discharge_limit_rule)

    def bess_import_definition_rule(m, t):
        if config_name == 'BESS':
            return m.P_in[t] == m.C[t]
        return pyo.Constraint.Skip
    m.bess_import_definition = pyo.Constraint(m.T, rule=bess_import_definition_rule)

    def energy_balance_rule(m, t):
        return (
            m.res_generation[t] + m.P_in[t] + m.D[t]
            == m.P_out[t] + m.C[t] + m.Curt[t]
        )
    m.energy_balance = pyo.Constraint(m.T, rule=energy_balance_rule)

    E_MIN = SOC_MIN * BATTERY_ENERGY_MWH
    E_MAX = SOC_MAX * BATTERY_ENERGY_MWH
    E_INITIAL = SOC_INITIAL * BATTERY_ENERGY_MWH

    def energy_min_rule(m, t):
        if bess_enabled == 0:
            return m.E[t] == 0
        return m.E[t] >= E_MIN
    m.energy_min = pyo.Constraint(m.T, rule=energy_min_rule)

    def energy_max_rule(m, t):
        if bess_enabled == 0:
            return pyo.Constraint.Skip
        return m.E[t] <= E_MAX
    m.energy_max = pyo.Constraint(m.T, rule=energy_max_rule)

    def soc_balance_rule(m, t):
        if bess_enabled == 0:
            return pyo.Constraint.Skip
        if t == 0:
            return (
                m.E[t]
                == E_INITIAL
                + ETA_CHARGE * m.C[t] * DELTA_T
                - (m.D[t] / ETA_DISCHARGE) * DELTA_T
            )
        return (
            m.E[t]
            == m.E[t - 1]
            + ETA_CHARGE * m.C[t] * DELTA_T
            - (m.D[t] / ETA_DISCHARGE) * DELTA_T
        )
    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance_rule)

    if bess_enabled == 1:
        def terminal_soc_rule(m):
            return m.E[n_hours - 1] == E_INITIAL

        m.terminal_soc = pyo.Constraint(
            rule=terminal_soc_rule
        )

    solver = pyo.SolverFactory(solver_name)
    results = solver.solve(m, tee=False)
    termination = results.solver.termination_condition

    if termination != pyo.TerminationCondition.optimal:
        print(f'Warning: {config_name}, {grid_level:.0%}: {termination}')

    hourly = pd.DataFrame(index=model_data.index)
    hourly['price'] = prices
    hourly['PV_generation_MW'] = pv_generation
    hourly['Wind_generation_MW'] = wind_generation
    hourly['RES_generation_MW'] = res_generation
    hourly['P_out_MW'] = [pyo.value(m.P_out[t]) for t in m.T]
    hourly['P_in_MW'] = [pyo.value(m.P_in[t]) for t in m.T]
    hourly['Charge_MW'] = [pyo.value(m.C[t]) for t in m.T]
    hourly['Discharge_MW'] = [pyo.value(m.D[t]) for t in m.T]
    hourly['Curtailment_MW'] = [pyo.value(m.Curt[t]) for t in m.T]
    hourly['Battery_energy_MWh'] = [pyo.value(m.E[t]) for t in m.T]

    if bess_enabled == 1:
        hourly['SOC'] = hourly['Battery_energy_MWh'] / BATTERY_ENERGY_MWH
    else:
        hourly['SOC'] = 0.0

    hourly = add_curtailment_breakdown(hourly, p_grid)

    hourly['Export_revenue_GBP'] = (
        hourly['price'] * hourly['P_out_MW'] * DELTA_T
    )
    hourly['Grid_import_cost_GBP'] = (
        hourly['price'] * hourly['P_in_MW'] * DELTA_T
    )
    hourly['Degradation_cost_GBP'] = (
        DEGRADATION_COST_GBP_PER_MWH * hourly['Discharge_MW'] * DELTA_T
    )
    hourly['Net_revenue_GBP'] = (
        hourly['Export_revenue_GBP']
        - hourly['Grid_import_cost_GBP']
        - hourly['Degradation_cost_GBP']
    )

    total_res_generation = hourly['RES_generation_MW'].sum() * DELTA_T
    total_export = hourly['P_out_MW'].sum() * DELTA_T
    total_import = hourly['P_in_MW'].sum() * DELTA_T
    total_charge = hourly['Charge_MW'].sum() * DELTA_T
    total_discharge = hourly['Discharge_MW'].sum() * DELTA_T

    physical_curtailment_mwh = hourly['Physical_curtailment_MW'].sum() * DELTA_T
    economic_curtailment_mwh = hourly['Economic_curtailment_MW'].sum() * DELTA_T
    total_curtailment_mwh = hourly['Total_curtailment_MW'].sum() * DELTA_T
    total_revenue = hourly['Net_revenue_GBP'].sum()

    if total_res_generation > 0:
        physical_curtailment_pct = physical_curtailment_mwh / total_res_generation * 100
        economic_curtailment_pct = economic_curtailment_mwh / total_res_generation * 100
        total_curtailment_pct = total_curtailment_mwh / total_res_generation * 100
    else:
        physical_curtailment_pct = np.nan
        economic_curtailment_pct = np.nan
        total_curtailment_pct = np.nan

    grid_utilisation_rate = (
        total_export / (p_grid * n_hours) if p_grid > 0 else np.nan
    )
    battery_throughput = total_charge + total_discharge
    equivalent_discharge_cycles = (
        total_discharge / BATTERY_ENERGY_MWH if bess_enabled == 1 else np.nan
    )
    revenue_per_mwh_res = (
        total_revenue / total_res_generation if total_res_generation > 0 else np.nan
    )
    revenue_per_mwh_export = (
        total_revenue / total_export if total_export > 0 else np.nan
    )

    summary = {
        'degradation_case': 'Exclude c_deg',
        'c_deg_GBP_per_MWh': DEGRADATION_COST_GBP_PER_MWH,
        'contract': 'Merchant',
        'configuration': config_name,
        'grid_level': grid_level,
        'PV_capacity_MW': PV_CAPACITY_MW if pv_enabled else 0.0,
        'Wind_capacity_MW': WIND_CAPACITY_MW if wind_enabled else 0.0,
        'RES_capacity_MW': p_res_installed,
        'BESS_power_MW': BATTERY_POWER_MW if bess_enabled else 0.0,
        'BESS_energy_MWh': BATTERY_ENERGY_MWH if bess_enabled else 0.0,
        'Grid_connection_MW': p_grid,
        'RES_generation_MWh': total_res_generation,
        'Grid_export_MWh': total_export,
        'Grid_import_MWh': total_import,
        'Physical_curtailment_MWh': physical_curtailment_mwh,
        'Economic_curtailment_MWh': economic_curtailment_mwh,
        'Total_curtailment_MWh': total_curtailment_mwh,
        'Physical_curtailment_pct': physical_curtailment_pct,
        'Economic_curtailment_pct': economic_curtailment_pct,
        'Total_curtailment_pct': total_curtailment_pct,
        'Grid_utilisation_rate': grid_utilisation_rate,
        'Battery_charge_MWh': total_charge,
        'Battery_discharge_MWh': total_discharge,
        'Battery_throughput_MWh': battery_throughput,
        'Equivalent_discharge_cycles': equivalent_discharge_cycles,
        'Net_revenue_GBP': total_revenue,
        'Revenue_per_MWh_RES_GBP': revenue_per_mwh_res,
        'Revenue_per_MWh_export_GBP': revenue_per_mwh_export,
        'Solver_status': str(termination),
    }

    return summary, hourly

# ============================================================
# 9. Run Merchant without-c_deg scenarios
# ============================================================

summary_rows = []

merchant_scenarios = []

for config_name in CONFIGURATIONS:

    if config_name == "BESS":
        # Methodology: standalone BESS has one fixed grid connection
        # equal to its battery power rating.
        merchant_scenarios.append(
            (config_name, 1.00)
        )

    else:
        merchant_scenarios.extend(
            (config_name, grid_level)
            for grid_level in GRID_LEVELS
        )

total_scenarios = len(merchant_scenarios)

for scenario_number, (config_name, grid_level) in enumerate(
    merchant_scenarios,
    start=1,
):

    print(
        f"Running Merchant scenario "
        f"{scenario_number}/{total_scenarios}: "
        f"{config_name}, "
        f"grid = {grid_level:.0%}, "
        f"c_deg = £0/MWh"
    )

    summary, hourly = run_scenario(
        config_name=config_name,
        grid_level=grid_level,
        solver_name="highs",
    )

    summary_rows.append(summary)

    # Uncomment if hourly CSV files are needed:
    # scenario_key = (
    #     f"WithoutCdeg_Merchant_"
    #     f"{config_name}_"
    #     f"{int(grid_level * 100)}pct"
    # )
    # hourly.to_csv(
    #     OUTPUT_DIRECTORY / f"{scenario_key}.csv"
    # )

merchant_results = pd.DataFrame(
    summary_rows
)

merchant_results = (
    merchant_results
    .sort_values(
        by=["configuration", "grid_level"]
    )
    .reset_index(drop=True)
)

assert len(merchant_results) == 31, (
    f"Expected 31 scenarios, "
    f"but got {len(merchant_results)}."
)


# ============================================================
# 10. Analysis tables
# ============================================================

battery_analysis = merchant_results[
    merchant_results['BESS_power_MW'] > 0
][[
    'configuration',
    'grid_level',
    'Grid_connection_MW',
    'BESS_power_MW',
    'BESS_energy_MWh',
    'Battery_charge_MWh',
    'Battery_discharge_MWh',
    'Battery_throughput_MWh',
    'Equivalent_discharge_cycles',
    'Net_revenue_GBP',
]].copy()

curtailment_analysis = merchant_results[[
    'configuration',
    'grid_level',
    'Grid_connection_MW',
    'RES_capacity_MW',
    'RES_generation_MWh',
    'Grid_export_MWh',
    'Physical_curtailment_MWh',
    'Economic_curtailment_MWh',
    'Total_curtailment_MWh',
    'Physical_curtailment_pct',
    'Economic_curtailment_pct',
    'Total_curtailment_pct',
    'Grid_utilisation_rate',
]].copy()

grid_sensitivity = merchant_results[[
    'configuration',
    'grid_level',
    'Grid_connection_MW',
    'Physical_curtailment_pct',
    'Economic_curtailment_pct',
    'Total_curtailment_pct',
    'Battery_discharge_MWh',
    'Equivalent_discharge_cycles',
    'Net_revenue_GBP',
]].copy()

# ============================================================
# 11. Export Excel
# ============================================================

with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine='openpyxl') as writer:
    merchant_results.to_excel(
        writer,
        sheet_name='Merchant Summary',
        index=False
    )
    battery_analysis.to_excel(
        writer,
        sheet_name='Battery Analysis',
        index=False
    )
    curtailment_analysis.to_excel(
        writer,
        sheet_name='Curtailment Analysis',
        index=False
    )
    grid_sensitivity.to_excel(
        writer,
        sheet_name='Grid Sensitivity',
        index=False
    )

print('\n==============================================')
print('MERCHANT EXCLUDE c_deg SENSITIVITY COMPLETED')
print('==============================================')
print(f'Scenarios completed: {len(merchant_results)}')
print(f'Output file: {EXCEL_OUTPUT_PATH}')
