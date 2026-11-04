"""Persistence and plotting helpers for reproducible experiment results."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .knapsack import CountSummary


def _summary_to_dict(summary: CountSummary) -> dict:
    return {
        "feasible_by_value": {str(k): int(v) for k, v in summary.feasible_by_value.items()},
        "infeasible_by_value": {str(k): int(v) for k, v in summary.infeasible_by_value.items()},
        "feasible_probability": float(summary.feasible_probability),
        "conditional_mean_value": float(summary.conditional_mean_value),
        "best_feasible_value": summary.best_feasible_value,
    }


def _summary_from_dict(data: Mapping) -> CountSummary:
    return CountSummary(
        feasible_by_value={int(k): int(v) for k, v in data["feasible_by_value"].items()},
        infeasible_by_value={int(k): int(v) for k, v in data["infeasible_by_value"].items()},
        feasible_probability=float(data["feasible_probability"]),
        conditional_mean_value=float(data["conditional_mean_value"]),
        best_feasible_value=data["best_feasible_value"],
    )


def summarize_sample_values(values, feasible) -> CountSummary:
    """Aggregate sample values into the same representation used for counts."""
    values = np.asarray(values, dtype=np.int64)
    feasible = np.asarray(feasible, dtype=bool)
    feasible_counts = Counter(map(int, values[feasible]))
    infeasible_counts = Counter(map(int, values[~feasible]))
    n_feasible = int(np.sum(feasible))
    return CountSummary(
        feasible_by_value=dict(feasible_counts),
        infeasible_by_value=dict(infeasible_counts),
        feasible_probability=n_feasible / len(values) if len(values) else 0.0,
        conditional_mean_value=float(np.mean(values[feasible])) if n_feasible else 0.0,
        best_feasible_value=int(np.max(values[feasible])) if n_feasible else None,
    )


def save_evaluation_results(
    directory: str | Path,
    evaluation: Mapping,
    *,
    reference: Mapping | None = None,
) -> Path:
    """Save final-shot counts and numerical summaries for every QAOA depth."""
    directory = Path(directory)
    if directory.name != "simulator":
        directory = directory / "simulator"
    directory.mkdir(parents=True, exist_ok=True)
    summary_data = {"reference": None, "depths": {}}
    if reference is not None:
        summary_data["reference"] = {
            key: value for key, value in reference.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    for depth, result in evaluation.items():
        counts_path = directory / f"evaluation_counts_p{depth}.json"
        counts_path.write_text(json.dumps(result["counts"], indent=2))
        summary_data["depths"][str(depth)] = {
            "cvar": float(result["cvar"]),
            "summary": _summary_to_dict(result["summary"]),
            "counts_file": counts_path.name,
        }
    output = directory / "evaluation_summary.json"
    output.write_text(json.dumps(summary_data, indent=2))
    return output


def save_postprocessing_comparison(
    path: str | Path,
    distributions: Mapping[str, CountSummary],
    *,
    final_depth: int,
    reference_value: int | None,
) -> Path:
    """Save every aggregate distribution required by the 3-by-3 plot."""
    path = Path(path)
    data = {
        "final_depth": int(final_depth),
        "reference_value": reference_value,
        "distributions": {
            name: _summary_to_dict(summary) for name, summary in distributions.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_postprocessing_comparison(path: str | Path) -> dict:
    """Load a saved comparison without requiring notebook variables."""
    data = json.loads(Path(path).read_text())
    data["distributions"] = {
        name: _summary_from_dict(summary)
        for name, summary in data["distributions"].items()
    }
    return data


def latest_result_directory(results_root: str | Path) -> Path:
    """Return the newest completed run containing post-processing results."""
    candidates = [
        path.parent
        for path in Path(results_root).glob("*/postprocessing_comparison.json")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No postprocessing_comparison.json found below {results_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def plot_saved_postprocessing(path: str | Path):
    """Recreate the full comparison plot using only a saved JSON artifact."""
    import matplotlib.pyplot as plt

    data = load_postprocessing_comparison(path)
    d = data["distributions"]
    depth = data["final_depth"]
    reference_value = data["reference_value"]
    layout = [
        ("random_before", "Uniform random before repair"),
        ("warm_before", "Warm start before repair"),
        ("qaoa_before", f"QAOA before repair, p={depth}"),
        ("random_repair", "Random: repair only"),
        ("warm_repair", "Warm start: repair only"),
        ("qaoa_repair", f"QAOA: repair only, p={depth}"),
        ("random_complete", "Random: repair + greedy completion"),
        ("warm_complete", "Warm start: repair + greedy completion"),
        ("qaoa_complete", f"QAOA: repair + greedy completion, p={depth}"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (key, title) in zip(axes.flat, layout):
        summary = d[key]
        ax.hist(
            [list(summary.feasible_by_value), list(summary.infeasible_by_value)],
            weights=[list(summary.feasible_by_value.values()),
                     list(summary.infeasible_by_value.values())],
            bins=50, stacked=True, density=False,
            color=["tab:blue", "tab:red"], label=["Feasible", "Infeasible"],
        )
        if reference_value is not None:
            ax.axvline(reference_value, color="black", linestyle="--", label="Gurobi")
        ax.set(title=title, xlabel="Knapsack value", ylabel="Counts")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig, axes
