# Renewable Energy Collocation Optimisation in Great Britain

This repository contains the optimisation models and supporting outputs developed for a Master's dissertation investigating the operational and investment performance of utility-scale renewable energy collocation projects in Great Britain.

The study evaluates different configurations of solar PV, wind and battery energy storage systems (BESS) under alternative grid connection capacities and electricity contracting structures. A two-phase optimisation framework is used to distinguish short-term operational performance from long-term investment value.

## Research Framework

The analysis is structured into two phases.

### Phase 1 – Operational Optimisation

Phase 1 evaluates system operation under fixed renewable generation and BESS capacities.

The model compares standalone and collocated configurations across different grid connection levels under:

- Merchant operation
- Fixed-price PPA operation

Key outputs include renewable curtailment, electricity exports, battery operation and project revenue.

### Phase 2 – Investment Optimisation

Phase 2 extends the operational framework by allowing renewable and BESS capacities to be optimised subject to investment constraints.

The model evaluates long-term project performance using a 25-year investment horizon and net present value (NPV).

This phase examines whether the operational benefits identified in Phase 1 translate into long-term investment value.

## Repository Structure

| File | Description |
|---|---|
| `phase1_operational_model.py` | Main Phase 1 operational optimisation model |
| `phase1_merchant_without_degradation.py` | Additional Phase 1 merchant scenario excluding battery degradation cost |
| `phase1_results.xlsx` | Phase 1 optimisation results |
| `degradation_cost_analysis.xlsx` | Analysis of the effect of battery degradation cost |
| `phase2_investment_model.py` | Phase 2 investment and capacity optimisation model |
| `phase2_results.xlsx` | Phase 2 optimisation results |
| `README.md` | Repository documentation |
| `LICENSE` | Repository licence |

## Model Scope

The models represent utility-scale renewable projects in Great Britain using hourly generation and electricity price data.

The configurations considered are:

1. PV
2. Wind
3. PV–Wind
4. BESS
5. PV–BESS
6. Wind–BESS
7. PV–Wind–BESS

For renewable configurations, grid connection capacity is evaluated from 60% to 100% of installed renewable capacity.

Two electricity contracting structures are considered:

- Merchant electricity sales based on day-ahead market prices
- Fixed-price Power Purchase Agreement (PPA)

## Optimisation Approach

The models are formulated as linear optimisation problems and implemented in Python using Pyomo.

Phase 1 maximises operational revenue subject to renewable generation, grid connection and battery operating constraints.

Phase 2 extends the framework to long-term capacity optimisation and evaluates investment performance through NPV.

Battery degradation is represented through a linear cost applied to discharged energy.

## Notes

The repository contains the computational implementation and model outputs used in the dissertation. Detailed mathematical formulations, parameter assumptions, data sources and methodological justification are provided in the dissertation itself.
