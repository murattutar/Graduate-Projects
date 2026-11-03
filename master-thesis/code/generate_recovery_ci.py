#!/usr/bin/env python3
"""Create reproducible Table 15 values and confidence-band figures.

This script is deliberately separate from the user's original thesis package. It
reads the scenario-level output of the 100-scenario rerun and writes only into
the dedicated audit directory supplied with --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z_975 = 1.96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_design(data: pd.DataFrame, expected_scenarios: int) -> None:
    required = {
        "day_index",
        "acceptance_probability",
        "scenario_id",
        "recovery_budget_share",
        "no_recovery_reduction_pct",
        "recovery_reduction_pct",
        "recovered_lost_benefit_share",
        "recovery_changed_customers",
        "recovery_status",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Scenario file is missing columns: {sorted(missing)}")

    counts = (
        data.groupby(
            ["day_index", "acceptance_probability", "recovery_budget_share"],
            observed=True,
        )["scenario_id"]
        .nunique()
    )
    if not (counts == expected_scenarios).all():
        bad = counts[counts != expected_scenarios]
        raise ValueError(
            f"Expected exactly {expected_scenarios} scenarios in every cell; "
            f"found mismatches: {bad.to_dict()}"
        )


def confidence_bands(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["acceptance_probability", "recovery_budget_share"]
    rows: list[dict[str, float | int]] = []

    for key, cell in data.groupby(keys, sort=True, observed=True):
        p_accept, rec_budget = key
        values = cell["recovery_reduction_pct"].astype(float)

        day_stats = (
            cell.groupby("day_index", observed=True)["recovery_reduction_pct"]
            .agg(["mean", "var", "count"])
            .reset_index()
        )
        num_days = int(day_stats.shape[0])
        mean = float(day_stats["mean"].mean())

        # Conditional Monte Carlo SE: days are the fixed 30-instance test set;
        # uncertainty comes only from finite scenario sampling within each day.
        conditional_mc_variance = float(
            ((day_stats["var"] / day_stats["count"]).sum()) / (num_days**2)
        )
        conditional_mc_se = float(np.sqrt(conditional_mc_variance))

        # Day-cluster SE: first average simulations within each day, then treat
        # the 30 daily instances as the independent units. This quantifies
        # variation in mean performance across days and is the recommended band
        # when the figure is interpreted beyond this fixed instance set.
        day_cluster_se = float(day_stats["mean"].std(ddof=1) / np.sqrt(num_days))

        # Pooled SE is supplied for audit comparison only. It assumes every
        # scenario-day observation is iid and therefore ignores nesting by day.
        pooled_se = float(values.std(ddof=1) / np.sqrt(values.shape[0]))

        row: dict[str, float | int] = {
            "acceptance_probability": float(p_accept),
            "recovery_budget_share": float(rec_budget),
            "mean_recovery_reduction_pct": mean,
            "num_days": num_days,
            "scenarios_per_day": int(day_stats["count"].min()),
            "scenario_day_observations": int(values.shape[0]),
            "conditional_mc_se_pct": conditional_mc_se,
            "pooled_se_pct": pooled_se,
            "day_cluster_se_pct": day_cluster_se,
        }
        for method, se in (
            ("conditional_mc", conditional_mc_se),
            ("pooled", pooled_se),
            ("day_cluster", day_cluster_se),
        ):
            row[f"{method}_ci95_lower_pct"] = mean - Z_975 * se
            row[f"{method}_ci95_upper_pct"] = mean + Z_975 * se
        rows.append(row)

    result = pd.DataFrame(rows)
    pct_columns = [
        col
        for col in result.columns
        if col.endswith("_pct") and col not in {"acceptance_probability", "recovery_budget_share"}
    ]
    result[pct_columns] = result[pct_columns] * 100.0
    return result


def table_15(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["acceptance_probability", "recovery_budget_share"]
    table = (
        data.groupby(keys, as_index=False, sort=True, observed=True)
        .agg(
            before_recovery_reduction=("no_recovery_reduction_pct", "mean"),
            after_recovery_reduction=("recovery_reduction_pct", "mean"),
            recovered_lost_benefit=("recovered_lost_benefit_share", "mean"),
            recovery_changes_customers_per_day=("recovery_changed_customers", "mean"),
            scenario_day_observations=("scenario_id", "size"),
        )
    )
    table["incremental_value_pp"] = (
        table["after_recovery_reduction"] - table["before_recovery_reduction"]
    ) * 100.0
    for column in [
        "before_recovery_reduction",
        "after_recovery_reduction",
        "recovered_lost_benefit",
    ]:
        table[column] = table[column] * 100.0

    table["implementation_probability_pct"] = table["acceptance_probability"] * 100.0
    table["recovery_budget_share_pct"] = table["recovery_budget_share"] * 100.0
    return table[
        [
            "implementation_probability_pct",
            "recovery_budget_share_pct",
            "before_recovery_reduction",
            "after_recovery_reduction",
            "incremental_value_pp",
            "recovered_lost_benefit",
            "recovery_changes_customers_per_day",
            "scenario_day_observations",
        ]
    ]


def render_ci_figure(bands: pd.DataFrame, method: str, output_path: Path) -> None:
    lower = f"{method}_ci95_lower_pct"
    upper = f"{method}_ci95_upper_pct"
    if lower not in bands or upper not in bands:
        raise ValueError(f"Unknown CI method: {method}")

    fig, ax = plt.subplots(figsize=(8.0, 5.7))
    for rec_share in sorted(bands["recovery_budget_share"].unique()):
        subset = bands[np.isclose(bands["recovery_budget_share"], rec_share)].sort_values(
            "acceptance_probability"
        )
        x = subset["acceptance_probability"].to_numpy() * 100.0
        y = subset["mean_recovery_reduction_pct"].to_numpy()
        (line,) = ax.plot(
            x,
            y,
            marker="o",
            label=f"Recovery budget {int(round(rec_share * 100))}%",
        )
        ax.fill_between(
            x,
            subset[lower].to_numpy(),
            subset[upper].to_numpy(),
            color=line.get_color(),
            alpha=0.16,
        )

    ax.set_xlabel("First-stage implementation probability (%)")
    ax.set_ylabel("Average peak reduction after recovery (%)")
    ax.set_title("Recoverable rescheduling under partial implementation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=100)
    parser.add_argument("--source-script", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--first-stage-shifts", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.scenario_csv)
    validate_design(data, args.expected_scenarios)

    bands = confidence_bands(data)
    table = table_15(data)
    bands.to_csv(args.output_dir / "recovery_confidence_bands_all_methods.csv", index=False)
    table.to_csv(args.output_dir / "table_15_100_scenarios.csv", index=False)

    render_ci_figure(
        bands,
        "day_cluster",
        args.output_dir / "plot_recovery_peak_reduction_100_day_cluster_ci95.png",
    )
    render_ci_figure(
        bands,
        "conditional_mc",
        args.output_dir / "plot_recovery_peak_reduction_100_conditional_mc_ci95.png",
    )

    status_counts = data["recovery_status"].value_counts(dropna=False).to_dict()
    provenance: dict[str, object] = {
        "scenario_csv": str(args.scenario_csv),
        "scenario_csv_sha256": sha256(args.scenario_csv),
        "expected_scenarios_per_day_probability": args.expected_scenarios,
        "validated_num_days": int(data["day_index"].nunique()),
        "validated_probabilities": sorted(data["acceptance_probability"].unique().tolist()),
        "validated_recovery_budgets": sorted(data["recovery_budget_share"].unique().tolist()),
        "scenario_rows": int(data.shape[0]),
        "recovery_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "ci_multiplier": Z_975,
        "recommended_ci_for_cross_day_interpretation": "day_cluster",
        "recommended_ci_for_monte_carlo_error_conditional_on_fixed_days": "conditional_mc",
    }
    for label, path in (
        ("source_script", args.source_script),
        ("input_csv", args.input_csv),
        ("first_stage_shifts", args.first_stage_shifts),
    ):
        if path is not None:
            provenance[label] = str(path)
            provenance[f"{label}_sha256"] = sha256(path)
    (args.output_dir / "ci_and_table_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(table.round(4).to_string(index=False))
    print("\nConfidence-band outputs:")
    print((args.output_dir / "recovery_confidence_bands_all_methods.csv").resolve())


if __name__ == "__main__":
    main()
