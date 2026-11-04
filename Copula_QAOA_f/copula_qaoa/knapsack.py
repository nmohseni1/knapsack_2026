"""Knapsack instances, warm starts, evaluation, and classical solvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class KnapsackInstance:
    values: NDArray[np.int64]
    weights: NDArray[np.int64]
    capacity: int
    name: str = "knapsack"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.int64)
        if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
            raise ValueError("values and weights must be equally sized 1D arrays")
        if len(values) == 0 or np.any(values <= 0) or np.any(weights <= 0):
            raise ValueError("values and weights must be nonempty and positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "capacity", int(self.capacity))

    @property
    def n_items(self) -> int:
        return len(self.values)

    @property
    def ratios(self) -> NDArray[np.float64]:
        return self.values / self.weights


@dataclass(frozen=True)
class WarmStart:
    probabilities: NDArray[np.float64]
    scores: NDArray[np.float64]
    greedy_solution: NDArray[np.bool_]
    cutoff_score: float
    order: NDArray[np.int64]

    @property
    def ordered_probabilities(self) -> NDArray[np.float64]:
        return self.probabilities[self.order]


@dataclass(frozen=True)
class CountSummary:
    feasible_by_value: dict[int, int]
    infeasible_by_value: dict[int, int]
    feasible_probability: float
    conditional_mean_value: float
    best_feasible_value: int | None


def load_knapsack(path: str | Path) -> KnapsackInstance:
    """Load the repository's n/items/capacity `.knap` format."""
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    n_items = int(lines[0])
    if len(lines) != n_items + 2:
        raise ValueError(f"expected {n_items} items, found {len(lines) - 2}")
    items = np.asarray([tuple(map(int, line.split()[-2:])) for line in lines[1:-1]])
    return KnapsackInstance(items[:, 0], items[:, 1], int(lines[-1]), path.stem)


def build_warm_start(
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    k: float,
    lambda_weight: float = 0.0,
    epsilon: float = 1e-6,
) -> WarmStart:
    """Build the clipped, smoothed, weight-aware lazy-greedy warm start."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be equally sized 1D arrays")
    if capacity <= 0 or np.any(weights <= 0):
        raise ValueError("capacity and weights must be positive")
    scores = values / weights - lambda_weight * weights / capacity
    order = np.lexsort((weights, -scores)).astype(np.int64)
    greedy = np.zeros(len(weights), dtype=bool)
    total_weight = 0.0
    for index in order:
        if total_weight + weights[index] > capacity:
            break
        greedy[index] = True
        total_weight += weights[index]
    if not np.any(greedy):
        raise ValueError("lazy greedy selected no items")
    cutoff = float(np.min(scores[greedy]))
    occupancy_scale = float(np.sum(weights) / capacity)
    exponent = np.clip(-k * (scores - cutoff), -700.0, 700.0)
    probabilities = 1.0 / (1.0 + (occupancy_scale - 1.0) * np.exp(exponent))
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return WarmStart(probabilities, scores, greedy, cutoff, order)


def sample_warm_start(probabilities: ArrayLike, n_samples: int, seed: int | None = None):
    """Draw independent bitstrings from the product warm-start distribution."""
    probabilities = np.asarray(probabilities, dtype=float)
    return np.random.default_rng(seed).random((n_samples, len(probabilities))) < probabilities


def evaluate_samples(samples, values, weights, capacity):
    """Return total values, total weights, and feasibility for sample rows."""
    samples = np.asarray(samples, dtype=bool)
    total_values = samples @ np.asarray(values)
    total_weights = samples @ np.asarray(weights)
    return total_values, total_weights, total_weights <= capacity


def evaluate_counts(
    counts: Mapping[str, int],
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    reverse_bitstrings: bool = True,
) -> CountSummary:
    """Summarize Qiskit-order measurement counts by objective and feasibility."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    feasible: dict[int, int] = {}
    infeasible: dict[int, int] = {}
    feasible_count = 0
    feasible_value_sum = 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1] if reverse_bitstrings else bitstring
        selected = np.fromiter((bit == "1" for bit in bits), dtype=bool)
        value = int(np.sum(values[selected]))
        weight = int(np.sum(weights[selected]))
        target = feasible if weight <= capacity else infeasible
        target[value] = target.get(value, 0) + int(count)
        if weight <= capacity:
            feasible_count += int(count)
            feasible_value_sum += value * int(count)
    total_count = sum(counts.values())
    return CountSummary(
        feasible,
        infeasible,
        feasible_count / total_count if total_count else 0.0,
        feasible_value_sum / feasible_count if feasible_count else 0.0,
        max(feasible) if feasible else None,
    )


def solve_with_gurobi(
    values: ArrayLike,
    weights: ArrayLike,
    capacity: int,
    *,
    time_limit: float | None = None,
    optimality_gap: float = 1e-12,
    feasibility_tolerance: float = 1e-9,
    integer_tolerance: float = 1e-9,
    presolve: int = 2,
    threads: int | None = None,
    method: int | None = None,
    #cuts: int = 2,
    #heuristics: float = 0.7,
    node_method: int | None = None,
    scaling: int = 0,
    branching: int | None = None,
    solution_limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve the binary knapsack model with configurable Gurobi tolerances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise ImportError("Gurobi and a valid license are required") from error
    # Preserve the original knapsack coefficients.  Values and weights are not
    # necessarily integers, so converting them to int64 would truncate them and
    # make Gurobi solve a different problem.
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    capacity = float(capacity)
    model = gp.Model("knapsack")
    model.Params.OutputFlag = int(verbose)
    model.Params.MIPGap = optimality_gap
    model.Params.FeasibilityTol = feasibility_tolerance
    model.Params.IntFeasTol = integer_tolerance
    model.Params.Presolve = presolve
    # model.Params.Cuts = cuts
    # model.Params.Heuristics = heuristics
    # model.Params.ScaleFlag = scaling
    for name, value in (
        ("Threads", threads), ("Method", method), ("NodeMethod", node_method),
        ("BranchDir", branching), ("SolutionLimit", solution_limit),
        ("TimeLimit", time_limit),
    ):
        if value is not None:
            setattr(model.Params, name, value)
    x = model.addVars(len(values), vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(float(values[i]) * x[i] for i in range(len(values))),
        GRB.MAXIMIZE,
    )
    model.addConstr(
        gp.quicksum(float(weights[i]) * x[i] for i in range(len(weights)))
        <= capacity
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("Gurobi did not find a feasible solution")
    selected = np.array([x[i].X > 0.5 for i in range(len(values))], dtype=bool)
    total_value = float(np.sum(values[selected]))
    total_weight = float(np.sum(weights[selected]))
    achieved_gap = float(model.MIPGap)
    return {
        "value": total_value,
        "weight": total_weight,
        "total_value": total_value,
        "total_weight": total_weight,
        "feasible": total_weight <= capacity,
        "bitstring": "".join("1" if bit else "0" for bit in selected),
        "runtime": float(model.Runtime),
        "gap": achieved_gap,
        "optimality_gap": achieved_gap,
        "proven_optimal": model.Status == GRB.OPTIMAL,
        "status": int(model.Status),
        "bound": float(model.ObjBound),
        "nodes": float(model.NodeCount),
        "configured_optimality_gap": float(optimality_gap),
        "feasibility_tolerance": float(feasibility_tolerance),
        "integer_tolerance": float(integer_tolerance),
    }
