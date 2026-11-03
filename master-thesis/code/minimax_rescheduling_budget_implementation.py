"""
Budget-aware minimax rescheduling implementation for synthetic B2B truck-demand profiles.

This script implements the deterministic thesis model with disruption-budget experiments.
It uses a single weighted lexicographic MILP per daily scenario:
    1. minimize peak truck demand Z,
    2. among peak-best solutions, minimize the number of changed customers,
    3. among those, minimize total absolute shift magnitude.

The weighting constants are chosen dynamically so that a one-unit improvement in the
higher-priority objective dominates the largest possible change in all lower-priority
terms. This reproduces lexicographic logic much faster than solving three MILPs per
scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
import argparse
import csv
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# -----------------------------
# Configuration
# -----------------------------
INTERVAL_MINUTES = 15
START_HOUR = 8
END_HOUR = 18
# The data generator uses pd.date_range(start=08:00, end=18:00, freq=15min), inclusive.
# Therefore the planning horizon has 41 slot-start positions: 08:00, ..., 18:00.
NUM_SLOTS = int((END_HOUR - START_HOUR) * 60 / INTERVAL_MINUTES) + 1

DEFAULT_MAX_SHIFT_SLOTS = 4  # +/- 60 minutes
FLEXIBILITY_LEVELS = [0, 1, 2, 4]  # 0, 15, 30, 60 minutes
BUDGET_SHARES = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
SOLVER_OPTIONS = {"time_limit": 10, "mip_rel_gap": 0.0, "presolve": True}
TOL = 1e-6


@dataclass(frozen=True)
class CustomerProfile:
    day_index: int
    customer: str
    start_idx: int
    profile: np.ndarray

    @property
    def duration(self) -> int:
        return int(len(self.profile))


@dataclass
class SolveResult:
    experiment: str
    day_index: int
    max_shift_slots: int
    budget_share_requested: Optional[float]
    budget_customers: Optional[int]
    status: str
    baseline_peak: int
    optimized_peak: float
    changed_customers: int
    total_shift_slots: int
    selected_shifts: dict[str, int]
    baseline_load: np.ndarray
    optimized_load: np.ndarray
    objective_value: float


def parse_time_to_slot(time_text: str) -> int:
    hour, minute = map(int, str(time_text).split(":"))
    return int(((hour - START_HOUR) * 60 + minute) / INTERVAL_MINUTES)


def slot_index_to_time(slot_idx: int) -> str:
    total_min = START_HOUR * 60 + slot_idx * INTERVAL_MINUTES
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def read_customer_profiles(csv_path: str | Path) -> dict[int, list[CustomerProfile]]:
    df = pd.read_csv(csv_path)
    slot_cols = [c for c in df.columns if c.startswith("slot_") and c.endswith("_trucks")]
    day_profiles: dict[int, list[CustomerProfile]] = {}

    for _, row in df.iterrows():
        duration = int(row["duration_slots"])
        values = pd.to_numeric(row[slot_cols[:duration]], errors="coerce").fillna(0).astype(int).to_numpy()
        profile = np.asarray(values, dtype=int)
        start_idx = parse_time_to_slot(row["start_time"])
        customer = str(row["customer"])
        day = int(row["day_index"])
        day_profiles.setdefault(day, []).append(CustomerProfile(day, customer, start_idx, profile))

    return dict(sorted(day_profiles.items()))


def reconstruct_load(profiles: Iterable[CustomerProfile], shifts: Optional[dict[str, int]] = None, num_slots: int = NUM_SLOTS) -> np.ndarray:
    load = np.zeros(num_slots, dtype=int)
    shifts = shifts or {}
    for p in profiles:
        k = int(shifts.get(p.customer, 0))
        start = p.start_idx + k
        end = start + p.duration
        if start < 0 or end > num_slots:
            raise ValueError(f"Profile {p.customer} falls outside horizon after shift {k}.")
        load[start:end] += p.profile
    return load


def feasible_shifts(profile: CustomerProfile, max_shift_slots: int, num_slots: int = NUM_SLOTS) -> list[int]:
    shifts = []
    for k in range(-max_shift_slots, max_shift_slots + 1):
        start = profile.start_idx + k
        end = start + profile.duration
        if 0 <= start and end <= num_slots:
            shifts.append(k)
    if 0 not in shifts:
        raise ValueError(f"Original schedule infeasible for {profile.customer} on day {profile.day_index}.")
    return shifts


def build_variable_index(profiles: list[CustomerProfile], max_shift_slots: int, num_slots: int = NUM_SLOTS):
    options = []
    for i, p in enumerate(profiles):
        for k in feasible_shifts(p, max_shift_slots, num_slots):
            options.append((i, p.customer, k))
    z_index = len(options)
    return options, z_index


def build_constraints(
    profiles: list[CustomerProfile],
    options: list[tuple[int, str, int]],
    z_index: int,
    budget_customers: Optional[int] = None,
    num_slots: int = NUM_SLOTS,
):
    n_vars = z_index + 1
    n_customers = len(profiles)
    constraints = []

    # Exactly one shift option per customer.
    A_assign = lil_matrix((n_customers, n_vars), dtype=float)
    for var_idx, (i, _, _) in enumerate(options):
        A_assign[i, var_idx] = 1.0
    constraints.append(LinearConstraint(A_assign.tocsr(), np.ones(n_customers), np.ones(n_customers)))

    # Aggregate load in each slot may not exceed Z.
    A_peak = lil_matrix((num_slots, n_vars), dtype=float)
    for var_idx, (i, _, k) in enumerate(options):
        p = profiles[i]
        start = p.start_idx + k
        for u, val in enumerate(p.profile):
            if val:
                A_peak[start + u, var_idx] += float(val)
    A_peak[:, z_index] = -1.0
    constraints.append(LinearConstraint(A_peak.tocsr(), -np.inf * np.ones(num_slots), np.zeros(num_slots)))

    # Optional disruption budget: at most B customers may receive non-zero shifts.
    if budget_customers is not None:
        A_budget = lil_matrix((1, n_vars), dtype=float)
        for var_idx, (_, _, k) in enumerate(options):
            if k != 0:
                A_budget[0, var_idx] = 1.0
        constraints.append(LinearConstraint(A_budget.tocsr(), -np.inf, float(budget_customers)))

    return constraints


def solve_day_weighted_lexicographic(
    profiles: list[CustomerProfile],
    max_shift_slots: int = DEFAULT_MAX_SHIFT_SLOTS,
    budget_share_requested: Optional[float] = None,
    budget_customers: Optional[int] = None,
    experiment: str = "default",
    num_slots: int = NUM_SLOTS,
) -> SolveResult:
    day_index = profiles[0].day_index
    baseline_load = reconstruct_load(profiles, num_slots=num_slots)
    baseline_peak = int(baseline_load.max())

    options, z_index = build_variable_index(profiles, max_shift_slots, num_slots)
    n_vars = z_index + 1
    n = len(profiles)

    # Dynamic lexicographic weights.
    # A one-unit reduction in changed_customers dominates any possible difference in shift magnitude.
    # A one-unit reduction in Z dominates any possible difference in both lower-priority terms.
    max_possible_shift_magnitude = n * max_shift_slots
    w_shift = 1.0
    w_changed = float(max_possible_shift_magnitude + 1)
    w_peak = float(n * w_changed + max_possible_shift_magnitude + 1)

    c = np.zeros(n_vars, dtype=float)
    c[z_index] = w_peak
    for var_idx, (_, _, k) in enumerate(options):
        if k != 0:
            c[var_idx] += w_changed
        c[var_idx] += abs(k) * w_shift

    constraints = build_constraints(profiles, options, z_index, budget_customers=budget_customers, num_slots=num_slots)

    integrality = np.ones(n_vars, dtype=int)
    integrality[z_index] = 0
    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    upper[z_index] = sum(int(np.sum(p.profile)) for p in profiles)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options=SOLVER_OPTIONS,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed for day {day_index}, experiment={experiment}: {result.message}")

    selected = {}
    for var_idx, (_, customer, k) in enumerate(options):
        if result.x[var_idx] > 0.5:
            selected[customer] = int(k)

    missing = [p.customer for p in profiles if p.customer not in selected]
    if missing:
        raise RuntimeError(f"Missing selected shifts for day {day_index}: {missing}")

    optimized_load = reconstruct_load(profiles, shifts=selected, num_slots=num_slots)
    changed = sum(1 for k in selected.values() if k != 0)
    total_shift_slots = sum(abs(k) for k in selected.values())

    return SolveResult(
        experiment=experiment,
        day_index=day_index,
        max_shift_slots=max_shift_slots,
        budget_share_requested=budget_share_requested,
        budget_customers=budget_customers,
        status="optimal",
        baseline_peak=baseline_peak,
        optimized_peak=float(optimized_load.max()),
        changed_customers=int(changed),
        total_shift_slots=int(total_shift_slots),
        selected_shifts=selected,
        baseline_load=baseline_load,
        optimized_load=optimized_load,
        objective_value=float(result.fun),
    )


def result_to_summary_row(result: SolveResult, profiles: list[CustomerProfile]) -> dict:
    n = len(profiles)
    reduction_abs = result.baseline_peak - result.optimized_peak
    reduction_pct = reduction_abs / result.baseline_peak if result.baseline_peak else 0.0
    return {
        "experiment": result.experiment,
        "day_index": result.day_index,
        "num_customers": n,
        "max_shift_slots": result.max_shift_slots,
        "max_shift_minutes": result.max_shift_slots * INTERVAL_MINUTES,
        "budget_share_requested": result.budget_share_requested,
        "budget_customers": result.budget_customers,
        "actual_budget_share": (result.budget_customers / n if result.budget_customers is not None and n else None),
        "baseline_peak": result.baseline_peak,
        "optimized_peak": result.optimized_peak,
        "peak_reduction_abs": reduction_abs,
        "peak_reduction_pct": reduction_pct,
        "changed_customers": result.changed_customers,
        "changed_share": result.changed_customers / n if n else 0.0,
        "unused_budget_customers": (result.budget_customers - result.changed_customers if result.budget_customers is not None else None),
        "total_shift_slots": result.total_shift_slots,
        "total_shift_minutes": result.total_shift_slots * INTERVAL_MINUTES,
        "mean_shift_minutes_all_customers": result.total_shift_slots * INTERVAL_MINUTES / n if n else 0.0,
        "mean_shift_minutes_changed_customers": (
            result.total_shift_slots * INTERVAL_MINUTES / result.changed_customers if result.changed_customers else 0.0
        ),
        "baseline_peak_time": slot_index_to_time(int(result.baseline_load.argmax())),
        "optimized_peak_time": slot_index_to_time(int(result.optimized_load.argmax())),
    }


def build_shift_rows(result: SolveResult, profiles: list[CustomerProfile]) -> list[dict]:
    rows = []
    by_customer = {p.customer: p for p in profiles}
    for customer in sorted(result.selected_shifts, key=lambda c: int(c.split()[-1]) if c.split()[-1].isdigit() else c):
        shift = result.selected_shifts[customer]
        p = by_customer[customer]
        rows.append({
            "experiment": result.experiment,
            "day_index": result.day_index,
            "budget_share_requested": result.budget_share_requested,
            "budget_customers": result.budget_customers,
            "customer": customer,
            "original_start_slot": p.start_idx,
            "original_start_time": slot_index_to_time(p.start_idx),
            "duration_slots": p.duration,
            "duration_minutes": p.duration * INTERVAL_MINUTES,
            "selected_shift_slots": shift,
            "selected_shift_minutes": shift * INTERVAL_MINUTES,
            "new_start_slot": p.start_idx + shift,
            "new_start_time": slot_index_to_time(p.start_idx + shift),
            "changed": int(shift != 0),
        })
    return rows


def build_load_rows(result: SolveResult) -> list[dict]:
    rows = []
    for t, (base, opt) in enumerate(zip(result.baseline_load, result.optimized_load)):
        rows.append({
            "experiment": result.experiment,
            "day_index": result.day_index,
            "budget_share_requested": result.budget_share_requested,
            "budget_customers": result.budget_customers,
            "slot_index": t,
            "time": slot_index_to_time(t),
            "baseline_load": int(base),
            "optimized_load": int(opt),
        })
    return rows


def run_experiments(csv_path: str | Path, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_day = read_customer_profiles(csv_path)

    baseline_rows = []
    for day, profiles in profiles_by_day.items():
        load = reconstruct_load(profiles)
        baseline_rows.append({
            "day_index": day,
            "num_customers": len(profiles),
            "baseline_peak": int(load.max()),
            "baseline_total_truck_slots": int(load.sum()),
            "baseline_mean_load": float(load.mean()),
            "baseline_peak_time": slot_index_to_time(int(load.argmax())),
            "horizon_slots": NUM_SLOTS,
            "slot_length_minutes": INTERVAL_MINUTES,
        })
    baseline_df = pd.DataFrame(baseline_rows)

    # Experiment 1: flexibility sensitivity, no explicit movement budget.
    flexibility_rows = []
    default_results: list[SolveResult] = []
    for max_shift in FLEXIBILITY_LEVELS:
        for day, profiles in profiles_by_day.items():
            res = solve_day_weighted_lexicographic(
                profiles,
                max_shift_slots=max_shift,
                budget_share_requested=None,
                budget_customers=None,
                experiment="flexibility_sensitivity",
            )
            flexibility_rows.append(result_to_summary_row(res, profiles))
            if max_shift == DEFAULT_MAX_SHIFT_SLOTS:
                default_results.append(res)
    flexibility_df = pd.DataFrame(flexibility_rows)

    # Experiment 2: disruption-budget sensitivity under default +/-60 minute flexibility.
    budget_rows = []
    budget_results: list[SolveResult] = []
    for share in BUDGET_SHARES:
        for day, profiles in profiles_by_day.items():
            n = len(profiles)
            B = int(math.floor(share * n)) if share < 1.0 else n
            res = solve_day_weighted_lexicographic(
                profiles,
                max_shift_slots=DEFAULT_MAX_SHIFT_SLOTS,
                budget_share_requested=share,
                budget_customers=B,
                experiment="budget_sensitivity",
            )
            budget_rows.append(result_to_summary_row(res, profiles))
            budget_results.append(res)
    budget_df = pd.DataFrame(budget_rows)

    # Detailed default unrestricted +/-60 min solution.
    default_shift_rows = []
    default_load_rows = []
    for res in default_results:
        profiles = profiles_by_day[res.day_index]
        default_shift_rows.extend(build_shift_rows(res, profiles))
        default_load_rows.extend(build_load_rows(res))
    default_shifts_df = pd.DataFrame(default_shift_rows)
    default_load_df = pd.DataFrame(default_load_rows)

    # Detailed budget solution tables for all budget scenarios.
    budget_shift_rows = []
    budget_load_rows = []
    for res in budget_results:
        profiles = profiles_by_day[res.day_index]
        budget_shift_rows.extend(build_shift_rows(res, profiles))
        budget_load_rows.extend(build_load_rows(res))
    budget_shifts_df = pd.DataFrame(budget_shift_rows)
    budget_load_df = pd.DataFrame(budget_load_rows)

    # Aggregated summary tables for direct thesis use.
    flex_agg_df = flexibility_df.groupby("max_shift_minutes", as_index=False).agg(
        avg_baseline_peak=("baseline_peak", "mean"),
        avg_optimized_peak=("optimized_peak", "mean"),
        avg_peak_reduction_abs=("peak_reduction_abs", "mean"),
        avg_peak_reduction_pct=("peak_reduction_pct", "mean"),
        avg_changed_customers=("changed_customers", "mean"),
        avg_changed_share=("changed_share", "mean"),
        avg_total_shift_minutes=("total_shift_minutes", "mean"),
    )
    budget_agg_df = budget_df.groupby("budget_share_requested", as_index=False).agg(
        avg_actual_budget_share=("actual_budget_share", "mean"),
        avg_baseline_peak=("baseline_peak", "mean"),
        avg_optimized_peak=("optimized_peak", "mean"),
        avg_peak_reduction_abs=("peak_reduction_abs", "mean"),
        avg_peak_reduction_pct=("peak_reduction_pct", "mean"),
        avg_changed_customers=("changed_customers", "mean"),
        avg_changed_share=("changed_share", "mean"),
        avg_unused_budget_customers=("unused_budget_customers", "mean"),
        avg_total_shift_minutes=("total_shift_minutes", "mean"),
        avg_shift_minutes_per_changed_customer=("mean_shift_minutes_changed_customers", "mean"),
    )

    # Capture of unrestricted benefit by budget share.
    unrestricted = budget_agg_df.loc[budget_agg_df["budget_share_requested"] == 1.0, "avg_peak_reduction_abs"].iloc[0]
    budget_agg_df["captured_unrestricted_reduction_pct"] = np.where(
        unrestricted > 0,
        budget_agg_df["avg_peak_reduction_abs"] / unrestricted,
        0.0,
    )
    budget_agg_df["marginal_reduction_pct_points"] = budget_agg_df["avg_peak_reduction_pct"].diff().fillna(0.0)

    outputs = {
        "baseline_summary": baseline_df,
        "flexibility_sensitivity": flexibility_df,
        "flexibility_aggregated": flex_agg_df,
        "budget_sensitivity": budget_df,
        "budget_aggregated": budget_agg_df,
        "default_customer_shifts": default_shifts_df,
        "default_load_curves": default_load_df,
        "budget_customer_shifts": budget_shifts_df,
        "budget_load_curves": budget_load_df,
    }

    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    metadata = {
        "input_csv": str(csv_path),
        "num_days": int(len(profiles_by_day)),
        "num_customers_total": int(sum(len(v) for v in profiles_by_day.values())),
        "horizon_slots": NUM_SLOTS,
        "slot_length_minutes": INTERVAL_MINUTES,
        "default_max_shift_slots": DEFAULT_MAX_SHIFT_SLOTS,
        "default_max_shift_minutes": DEFAULT_MAX_SHIFT_SLOTS * INTERVAL_MINUTES,
        "flexibility_levels_minutes": [x * INTERVAL_MINUTES for x in FLEXIBILITY_LEVELS],
        "budget_shares": BUDGET_SHARES,
        "solver": "scipy.optimize.milp / HiGHS",
        "lexicographic_objective": "single weighted MILP: peak, changed customers, total shift magnitude",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    create_plots(outputs, out_dir)
    return outputs


def create_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    flex_agg = outputs["flexibility_aggregated"].copy()
    budget_agg = outputs["budget_aggregated"].copy()
    budget_load = outputs["budget_load_curves"].copy()

    plt.figure(figsize=(8, 5))
    plt.plot(flex_agg["max_shift_minutes"], flex_agg["avg_peak_reduction_pct"] * 100, marker="o")
    plt.xlabel("Maximum allowed shift in either direction (minutes)")
    plt.ylabel("Average peak reduction (%)")
    plt.title("Peak reduction by customer flexibility")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_peak_reduction_by_flexibility.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(budget_agg["budget_share_requested"] * 100, budget_agg["avg_peak_reduction_pct"] * 100, marker="o")
    plt.xlabel("Allowed changed customers (% of daily customers)")
    plt.ylabel("Average peak reduction (%)")
    plt.title("Peak reduction by disruption budget (+/-60 min flexibility)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_peak_reduction_by_budget.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(budget_agg["avg_changed_share"] * 100, budget_agg["avg_peak_reduction_pct"] * 100, marker="o")
    plt.xlabel("Actual changed customers (% of daily customers)")
    plt.ylabel("Average peak reduction (%)")
    plt.title("Efficiency-disruption frontier (+/-60 min flexibility)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_efficiency_disruption_frontier.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        budget_agg["budget_share_requested"] * 100,
        budget_agg["captured_unrestricted_reduction_pct"] * 100,
        marker="o",
    )
    plt.xlabel("Allowed changed customers (% of daily customers)")
    plt.ylabel("Captured unrestricted peak reduction (%)")
    plt.title("How much of the unrestricted benefit is captured?")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_captured_unrestricted_benefit.png", dpi=180)
    plt.close()

    # Average load curves for selected budget levels.
    selected_shares = [0.0, 0.2, 0.5, 1.0]
    avg_load = budget_load.groupby(["budget_share_requested", "slot_index", "time"], as_index=False).agg(
        baseline_load=("baseline_load", "mean"),
        optimized_load=("optimized_load", "mean"),
    )
    baseline = avg_load[avg_load["budget_share_requested"] == 0.0][["time", "baseline_load"]].drop_duplicates()
    plt.figure(figsize=(10, 5))
    plt.plot(baseline["time"], baseline["baseline_load"], label="Baseline / 0% budget", marker="o", markersize=3)
    for share in selected_shares[1:]:
        subset = avg_load[avg_load["budget_share_requested"] == share]
        plt.plot(subset["time"], subset["optimized_load"], label=f"Budget {int(share*100)}%", marker="o", markersize=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time")
    plt.ylabel("Average aggregate truck demand")
    plt.title("Average load curve under selected disruption budgets")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_average_load_by_budget.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Run budget-aware minimax truck-demand rescheduling experiments.")
    parser.add_argument("--input", default="/mnt/data/customer_requests.csv", help="Path to customer_requests.csv")
    parser.add_argument("--out", default="/mnt/data/minimax_budget_outputs", help="Output directory")
    args = parser.parse_args()

    outputs = run_experiments(args.input, args.out)
    budget_agg = outputs["budget_aggregated"].copy()
    budget_agg["avg_peak_reduction_pct"] *= 100
    budget_agg["captured_unrestricted_reduction_pct"] *= 100
    print("Budget sensitivity (+/-60 min flexibility):")
    print(budget_agg[[
        "budget_share_requested",
        "avg_peak_reduction_pct",
        "avg_changed_share",
        "avg_total_shift_minutes",
        "captured_unrestricted_reduction_pct",
    ]].round(3).to_string(index=False))
    print(f"\nOutputs written to: {args.out}")
