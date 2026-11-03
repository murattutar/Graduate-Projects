"""
Recoverable rescheduling extension for the minimax B2B truck-demand model.

This script evaluates a two-stage operational recovery logic after customer rejection:
1. A deterministic budget-constrained minimax model proposes shifts.
2. Shifted customers accept independently with probability p.
3. Accepted shifts are fixed. Rejected proposed shifts return to the original schedule.
4. A second-stage recovery optimization is allowed to move a limited number of customers
   that were not part of the first-stage proposal.
5. The realized peak is compared with the no-recovery acceptance outcome.

The recovery model remains a minimax profile-placement MILP. It is intentionally a
recoverable rescheduling analysis rather than a fully conservative robust optimization model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
import argparse
import json
import math
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

INTERVAL_MINUTES = 15
START_HOUR = 8
END_HOUR = 18
NUM_SLOTS = int((END_HOUR - START_HOUR) * 60 / INTERVAL_MINUTES) + 1
DEFAULT_MAX_SHIFT_SLOTS = 4
DEFAULT_INITIAL_BUDGET_SHARE = 0.5
DEFAULT_RECOVERY_BUDGET_SHARES = [0.0, 0.1, 0.2]
DEFAULT_ACCEPTANCE_PROBABILITIES = [0.8, 0.6, 0.4]
DEFAULT_NUM_SCENARIOS = 100
DEFAULT_SEED = 20260503
SOLVER_OPTIONS = {"time_limit": 0.5, "mip_rel_gap": 0.01, "presolve": True}


@dataclass(frozen=True)
class CustomerProfile:
    day_index: int
    customer: str
    start_idx: int
    profile: np.ndarray

    @property
    def duration(self) -> int:
        return int(len(self.profile))


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
        day = int(row["day_index"])
        customer = str(row["customer"])
        start_idx = parse_time_to_slot(row["start_time"])
        day_profiles.setdefault(day, []).append(CustomerProfile(day, customer, start_idx, np.asarray(values, dtype=int)))
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


def load_first_stage_plan(shifts_df: pd.DataFrame, day_index: int, budget_share: float) -> dict[str, int]:
    subset = shifts_df[
        (shifts_df["day_index"].astype(int) == int(day_index))
        & (np.isclose(shifts_df["budget_share_requested"].astype(float), float(budget_share)))
    ]
    if subset.empty:
        raise ValueError(f"No first-stage plan for day={day_index}, budget_share={budget_share}")
    return dict(zip(subset["customer"].astype(str), subset["selected_shift_slots"].astype(int)))


def solve_recovery_milp(
    profiles: list[CustomerProfile],
    fixed_shifts: dict[str, int],
    eligible_customers: set[str],
    recovery_budget_customers: int,
    max_shift_slots: int = DEFAULT_MAX_SHIFT_SLOTS,
    num_slots: int = NUM_SLOTS,
) -> tuple[dict[str, int], np.ndarray, str, float]:
    """Solve a recovery minimax MILP with some customers fixed and only eligible customers adjustable.

    Customers not in eligible_customers are fixed to fixed_shifts (default 0). Eligible customers can choose one
    feasible shift, but the number of nonzero recovery shifts is bounded by recovery_budget_customers.
    """
    if recovery_budget_customers <= 0 or not eligible_customers:
        final_shifts = {p.customer: int(fixed_shifts.get(p.customer, 0)) for p in profiles}
        load = reconstruct_load(profiles, final_shifts, num_slots)
        return final_shifts, load, "no_recovery_budget", 0.0

    # Variable options only for eligible customers. Fixed customers are converted to a constant load contribution.
    fixed_load = np.zeros(num_slots, dtype=float)
    adjustable: list[CustomerProfile] = []
    for p in profiles:
        if p.customer in eligible_customers:
            adjustable.append(p)
        else:
            k = int(fixed_shifts.get(p.customer, 0))
            start = p.start_idx + k
            fixed_load[start : start + p.duration] += p.profile

    if not adjustable:
        final_shifts = {p.customer: int(fixed_shifts.get(p.customer, 0)) for p in profiles}
        load = reconstruct_load(profiles, final_shifts, num_slots)
        return final_shifts, load, "no_eligible_customers", 0.0

    options: list[tuple[int, str, int]] = []
    for i, p in enumerate(adjustable):
        for k in feasible_shifts(p, max_shift_slots, num_slots):
            options.append((i, p.customer, k))

    z_index = len(options)
    n_vars = z_index + 1
    n_adj = len(adjustable)

    # Lexicographic objective: realized peak first, recovery changes second, recovery shift magnitude third.
    max_possible_shift = n_adj * max_shift_slots
    w_shift = 1.0
    w_changed = float(max_possible_shift + 1)
    w_peak = float(n_adj * w_changed + max_possible_shift + 1)
    c = np.zeros(n_vars)
    c[z_index] = w_peak
    for var_idx, (_, _, k) in enumerate(options):
        if k != 0:
            c[var_idx] += w_changed
        c[var_idx] += abs(k) * w_shift

    constraints = []
    # exactly one shift per adjustable customer
    A_assign = lil_matrix((n_adj, n_vars), dtype=float)
    for var_idx, (i, _, _) in enumerate(options):
        A_assign[i, var_idx] = 1.0
    constraints.append(LinearConstraint(A_assign.tocsr(), np.ones(n_adj), np.ones(n_adj)))

    # peak constraints: fixed load + adjustable selected load <= Z
    A_peak = lil_matrix((num_slots, n_vars), dtype=float)
    for var_idx, (i, _, k) in enumerate(options):
        p = adjustable[i]
        start = p.start_idx + k
        for u, val in enumerate(p.profile):
            if val:
                A_peak[start + u, var_idx] += float(val)
    A_peak[:, z_index] = -1.0
    constraints.append(LinearConstraint(A_peak.tocsr(), -np.inf * np.ones(num_slots), -fixed_load))

    # recovery budget
    A_budget = lil_matrix((1, n_vars), dtype=float)
    for var_idx, (_, _, k) in enumerate(options):
        if k != 0:
            A_budget[0, var_idx] = 1.0
    constraints.append(LinearConstraint(A_budget.tocsr(), -np.inf, float(recovery_budget_customers)))

    integrality = np.ones(n_vars, dtype=int)
    integrality[z_index] = 0
    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    upper[z_index] = sum(int(np.sum(p.profile)) for p in profiles)

    result = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints, options=SOLVER_OPTIONS)
    if not result.success or result.x is None:
        # Conservative fallback: no recovery instead of failing the whole experiment.
        final_shifts = {p.customer: int(fixed_shifts.get(p.customer, 0)) for p in profiles}
        load = reconstruct_load(profiles, final_shifts, num_slots)
        return final_shifts, load, f"fallback_no_recovery: {result.message}", float("nan")

    final_shifts = {p.customer: int(fixed_shifts.get(p.customer, 0)) for p in profiles}
    for var_idx, (_, customer, k) in enumerate(options):
        if result.x[var_idx] > 0.5:
            final_shifts[customer] = int(k)
    load = reconstruct_load(profiles, final_shifts, num_slots)
    return final_shifts, load, "optimal", float(result.fun)


def run_recoverable_experiments(
    input_csv: str | Path,
    budget_output_dir: str | Path,
    out_dir: str | Path,
    initial_budget_share: float = DEFAULT_INITIAL_BUDGET_SHARE,
    recovery_budget_shares: list[float] = DEFAULT_RECOVERY_BUDGET_SHARES,
    acceptance_probabilities: list[float] = DEFAULT_ACCEPTANCE_PROBABILITIES,
    num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_by_day = read_customer_profiles(input_csv)
    shifts_df = pd.read_csv(Path(budget_output_dir) / "budget_customer_shifts.csv")
    budget_summary = pd.read_csv(Path(budget_output_dir) / "budget_sensitivity.csv")

    rng = np.random.default_rng(seed)
    scenario_rows = []
    load_rows = []
    recovery_shift_rows = []

    for day_index, profiles in profiles_by_day.items():
        n = len(profiles)
        baseline_load = reconstruct_load(profiles)
        baseline_peak = int(baseline_load.max())
        first_stage_shifts = load_first_stage_plan(shifts_df, day_index, initial_budget_share)
        planned_load = reconstruct_load(profiles, first_stage_shifts)
        planned_peak = int(planned_load.max())
        planned_reduction = baseline_peak - planned_peak
        proposed_changed = {p.customer for p in profiles if int(first_stage_shifts.get(p.customer, 0)) != 0}
        unproposed_customers = {p.customer for p in profiles if p.customer not in proposed_changed}

        for p_accept in acceptance_probabilities:
            for scenario_id in range(1, num_scenarios + 1):
                accepted = set()
                rejected = set()
                fixed_after_acceptance = {}
                for p in profiles:
                    proposed_shift = int(first_stage_shifts.get(p.customer, 0))
                    if proposed_shift == 0:
                        fixed_after_acceptance[p.customer] = 0
                    else:
                        if rng.random() < p_accept:
                            accepted.add(p.customer)
                            fixed_after_acceptance[p.customer] = proposed_shift
                        else:
                            rejected.add(p.customer)
                            fixed_after_acceptance[p.customer] = 0

                no_recovery_load = reconstruct_load(profiles, fixed_after_acceptance)
                no_recovery_peak = int(no_recovery_load.max())
                no_recovery_reduction = baseline_peak - no_recovery_peak

                for rec_share in recovery_budget_shares:
                    rec_budget = int(math.floor(rec_share * n))
                    # If rec_share=1 is ever used, cap to all unproposed customers.
                    rec_budget = min(rec_budget, len(unproposed_customers))
                    final_shifts, recovery_load, status, obj = solve_recovery_milp(
                        profiles=profiles,
                        fixed_shifts=fixed_after_acceptance,
                        eligible_customers=unproposed_customers,
                        recovery_budget_customers=rec_budget,
                    )
                    recovery_peak = int(recovery_load.max())
                    recovery_reduction = baseline_peak - recovery_peak
                    recovery_changed = [c for c in unproposed_customers if int(final_shifts.get(c, 0)) != 0]
                    recovery_changed_count = len(recovery_changed)
                    recovery_shift_minutes = sum(abs(int(final_shifts.get(c, 0))) * INTERVAL_MINUTES for c in recovery_changed)
                    total_implemented_changed = len(accepted) + recovery_changed_count
                    total_contacted = len(proposed_changed) + recovery_changed_count
                    lost_without_recovery = planned_reduction - no_recovery_reduction
                    recovered_loss = recovery_reduction - no_recovery_reduction
                    recovered_lost_benefit_share = recovered_loss / lost_without_recovery if lost_without_recovery > 0 else np.nan
                    retained_planned_without_recovery = no_recovery_reduction / planned_reduction if planned_reduction > 0 else np.nan
                    retained_planned_with_recovery = recovery_reduction / planned_reduction if planned_reduction > 0 else np.nan

                    scenario_rows.append({
                        "day_index": day_index,
                        "num_customers": n,
                        "initial_budget_share": initial_budget_share,
                        "acceptance_probability": p_accept,
                        "scenario_id": scenario_id,
                        "recovery_budget_share": rec_share,
                        "recovery_budget_customers": rec_budget,
                        "baseline_peak": baseline_peak,
                        "planned_peak_all_accept": planned_peak,
                        "no_recovery_peak": no_recovery_peak,
                        "recovery_peak": recovery_peak,
                        "planned_peak_reduction_abs": planned_reduction,
                        "no_recovery_reduction_abs": no_recovery_reduction,
                        "recovery_reduction_abs": recovery_reduction,
                        "planned_peak_reduction_pct": planned_reduction / baseline_peak if baseline_peak else 0.0,
                        "no_recovery_reduction_pct": no_recovery_reduction / baseline_peak if baseline_peak else 0.0,
                        "recovery_reduction_pct": recovery_reduction / baseline_peak if baseline_peak else 0.0,
                        "retained_planned_without_recovery": retained_planned_without_recovery,
                        "retained_planned_with_recovery": retained_planned_with_recovery,
                        "lost_without_recovery_abs": lost_without_recovery,
                        "recovered_loss_abs": recovered_loss,
                        "recovered_lost_benefit_share": recovered_lost_benefit_share,
                        "first_stage_proposed_changed": len(proposed_changed),
                        "accepted_first_stage_changes": len(accepted),
                        "rejected_first_stage_changes": len(rejected),
                        "realized_acceptance_rate": len(accepted) / len(proposed_changed) if proposed_changed else 1.0,
                        "recovery_changed_customers": recovery_changed_count,
                        "recovery_shift_minutes": recovery_shift_minutes,
                        "total_implemented_changed_customers": total_implemented_changed,
                        "total_contacted_customers": total_contacted,
                        "recovery_status": status,
                    })

                    # Store average-relevant load curves for selected scenario-level analysis; not every scenario to keep file smaller.
                    for t in range(NUM_SLOTS):
                        load_rows.append({
                            "day_index": day_index,
                            "acceptance_probability": p_accept,
                            "scenario_id": scenario_id,
                            "recovery_budget_share": rec_share,
                            "slot_index": t,
                            "time": slot_index_to_time(t),
                            "baseline_load": int(baseline_load[t]),
                            "planned_load_all_accept": int(planned_load[t]),
                            "no_recovery_load": int(no_recovery_load[t]),
                            "recovery_load": int(recovery_load[t]),
                        })

                    for c in recovery_changed:
                        recovery_shift_rows.append({
                            "day_index": day_index,
                            "acceptance_probability": p_accept,
                            "scenario_id": scenario_id,
                            "recovery_budget_share": rec_share,
                            "customer": c,
                            "recovery_shift_slots": int(final_shifts.get(c, 0)),
                            "recovery_shift_minutes": int(final_shifts.get(c, 0)) * INTERVAL_MINUTES,
                        })

    scenario_df = pd.DataFrame(scenario_rows)
    load_df = pd.DataFrame(load_rows)
    recovery_shifts_df = pd.DataFrame(recovery_shift_rows)

    agg_df = scenario_df.groupby(["acceptance_probability", "recovery_budget_share"], as_index=False).agg(
        avg_baseline_peak=("baseline_peak", "mean"),
        avg_planned_peak_all_accept=("planned_peak_all_accept", "mean"),
        avg_no_recovery_peak=("no_recovery_peak", "mean"),
        avg_recovery_peak=("recovery_peak", "mean"),
        avg_planned_peak_reduction_pct=("planned_peak_reduction_pct", "mean"),
        avg_no_recovery_reduction_pct=("no_recovery_reduction_pct", "mean"),
        avg_recovery_reduction_pct=("recovery_reduction_pct", "mean"),
        avg_retained_planned_without_recovery=("retained_planned_without_recovery", "mean"),
        avg_retained_planned_with_recovery=("retained_planned_with_recovery", "mean"),
        avg_lost_without_recovery_abs=("lost_without_recovery_abs", "mean"),
        avg_recovered_loss_abs=("recovered_loss_abs", "mean"),
        avg_recovered_lost_benefit_share=("recovered_lost_benefit_share", "mean"),
        avg_first_stage_proposed_changed=("first_stage_proposed_changed", "mean"),
        avg_accepted_first_stage_changes=("accepted_first_stage_changes", "mean"),
        avg_rejected_first_stage_changes=("rejected_first_stage_changes", "mean"),
        avg_recovery_changed_customers=("recovery_changed_customers", "mean"),
        avg_total_implemented_changed_customers=("total_implemented_changed_customers", "mean"),
        avg_total_contacted_customers=("total_contacted_customers", "mean"),
    )
    agg_df["incremental_recovery_reduction_pct_points"] = (
        agg_df["avg_recovery_reduction_pct"] - agg_df["avg_no_recovery_reduction_pct"]
    )

    day_agg_df = scenario_df.groupby(["day_index", "acceptance_probability", "recovery_budget_share"], as_index=False).agg(
        baseline_peak=("baseline_peak", "first"),
        planned_peak_all_accept=("planned_peak_all_accept", "first"),
        avg_no_recovery_peak=("no_recovery_peak", "mean"),
        avg_recovery_peak=("recovery_peak", "mean"),
        avg_no_recovery_reduction_pct=("no_recovery_reduction_pct", "mean"),
        avg_recovery_reduction_pct=("recovery_reduction_pct", "mean"),
        avg_recovered_lost_benefit_share=("recovered_lost_benefit_share", "mean"),
        avg_recovery_changed_customers=("recovery_changed_customers", "mean"),
    )

    avg_load_df = load_df.groupby(["acceptance_probability", "recovery_budget_share", "slot_index", "time"], as_index=False).agg(
        baseline_load=("baseline_load", "mean"),
        planned_load_all_accept=("planned_load_all_accept", "mean"),
        no_recovery_load=("no_recovery_load", "mean"),
        recovery_load=("recovery_load", "mean"),
    )

    outputs = {
        "recoverable_scenario_results": scenario_df,
        "recoverable_aggregated": agg_df,
        "recoverable_day_summary": day_agg_df,
        "recoverable_load_curves": avg_load_df,
        "recoverable_recovery_shifts": recovery_shifts_df,
    }
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    metadata = {
        "input_csv": str(input_csv),
        "budget_output_dir": str(budget_output_dir),
        "num_days": int(len(profiles_by_day)),
        "num_customers_total": int(sum(len(v) for v in profiles_by_day.values())),
        "initial_budget_share": initial_budget_share,
        "recovery_budget_shares": recovery_budget_shares,
        "acceptance_probabilities": acceptance_probabilities,
        "num_scenarios_per_day_probability": num_scenarios,
        "seed": seed,
        "max_shift_minutes": DEFAULT_MAX_SHIFT_SLOTS * INTERVAL_MINUTES,
        "recovery_eligible_customers": "customers not proposed for a first-stage shift",
        "fixed_customers": "accepted first-stage shifts remain shifted; rejected first-stage shifts return to original",
        "solver": "scipy.optimize.milp / HiGHS",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    create_recovery_plots(outputs, out_dir)
    return outputs


def create_recovery_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    agg = outputs["recoverable_aggregated"].copy()
    load = outputs["recoverable_load_curves"].copy()

    plt.figure(figsize=(8, 5))
    for rec_share in sorted(agg["recovery_budget_share"].unique()):
        subset = agg[agg["recovery_budget_share"] == rec_share].sort_values("acceptance_probability")
        plt.plot(
            subset["acceptance_probability"] * 100,
            subset["avg_recovery_reduction_pct"] * 100,
            marker="o",
            label=f"Recovery budget {int(rec_share*100)}%",
        )
    plt.xlabel("Acceptance probability (%)")
    plt.ylabel("Average realized peak reduction (%)")
    plt.title("Recoverable rescheduling under customer rejection")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_recovery_peak_reduction_by_acceptance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for p_accept in sorted(agg["acceptance_probability"].unique()):
        subset = agg[agg["acceptance_probability"] == p_accept].sort_values("recovery_budget_share")
        plt.plot(
            subset["recovery_budget_share"] * 100,
            subset["incremental_recovery_reduction_pct_points"] * 100,
            marker="o",
            label=f"Acceptance {int(p_accept*100)}%",
        )
    plt.xlabel("Additional recovery budget (% of daily customers)")
    plt.ylabel("Incremental peak reduction from recovery (percentage points)")
    plt.title("Marginal value of second-stage recovery")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_recovery_incremental_value.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for p_accept in sorted(agg["acceptance_probability"].unique()):
        subset = agg[agg["acceptance_probability"] == p_accept].sort_values("recovery_budget_share")
        plt.plot(
            subset["recovery_budget_share"] * 100,
            subset["avg_recovered_lost_benefit_share"] * 100,
            marker="o",
            label=f"Acceptance {int(p_accept*100)}%",
        )
    plt.xlabel("Additional recovery budget (% of daily customers)")
    plt.ylabel("Recovered lost benefit (%)")
    plt.title("How much rejection loss can recovery restore?")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_recovered_lost_benefit.png", dpi=180)
    plt.close()

    # Average load curve for central p=60% setting.
    central = load[np.isclose(load["acceptance_probability"], 0.6)]
    if not central.empty:
        plt.figure(figsize=(10, 5))
        base = central[central["recovery_budget_share"] == 0.0].sort_values("slot_index")
        plt.plot(base["time"], base["baseline_load"], label="Baseline", marker="o", markersize=3)
        plt.plot(base["time"], base["planned_load_all_accept"], label="Planned, full acceptance", marker="o", markersize=3)
        plt.plot(base["time"], base["no_recovery_load"], label="Acceptance 60%, no recovery", marker="o", markersize=3)
        rec = central[central["recovery_budget_share"] == central["recovery_budget_share"].max()].sort_values("slot_index")
        plt.plot(rec["time"], rec["recovery_load"], label=f"Acceptance 60%, recovery {int(rec['recovery_budget_share'].max()*100)}%", marker="o", markersize=3)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Time")
        plt.ylabel("Average aggregate truck demand")
        plt.title("Average load curve with and without recovery (60% acceptance)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plot_recovery_average_load_curve_p60.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Run recoverable rescheduling experiments after customer rejection.")
    parser.add_argument("--input", default="/mnt/data/customer_requests.csv", help="Path to customer_requests.csv")
    parser.add_argument("--budget-out", default="/mnt/data/budget_package_extract/minimax_budget_outputs", help="Directory with budget_customer_shifts.csv")
    parser.add_argument("--out", default="/mnt/data/minimax_recoverable_outputs", help="Output directory")
    parser.add_argument("--scenarios", type=int, default=DEFAULT_NUM_SCENARIOS, help="Monte Carlo scenarios per day and probability")
    args = parser.parse_args()

    outputs = run_recoverable_experiments(args.input, args.budget_out, args.out, num_scenarios=args.scenarios)
    agg = outputs["recoverable_aggregated"].copy()
    for col in ["avg_planned_peak_reduction_pct", "avg_no_recovery_reduction_pct", "avg_recovery_reduction_pct", "avg_recovered_lost_benefit_share"]:
        agg[col] = agg[col] * 100
    print("Recoverable rescheduling summary:")
    print(agg[[
        "acceptance_probability",
        "recovery_budget_share",
        "avg_no_recovery_reduction_pct",
        "avg_recovery_reduction_pct",
        "incremental_recovery_reduction_pct_points",
        "avg_recovered_lost_benefit_share",
        "avg_recovery_changed_customers",
    ]].round(3).to_string(index=False))
    print(f"\nOutputs written to: {args.out}")
