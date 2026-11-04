"""Feasibility repair and greedy completion for knapsack samples."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _arrays(solution, values, weights, scores=None):
    bits = np.asarray(solution, dtype=bool).copy()
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if bits.ndim != 1 or bits.shape != values.shape or weights.shape != values.shape:
        raise ValueError("solution, values, and weights must have the same 1D shape")
    if np.any(weights <= 0):
        raise ValueError("all weights must be positive")
    scores = values / weights if scores is None else np.asarray(scores, dtype=float)
    if scores.shape != values.shape:
        raise ValueError("scores must have one entry per item")
    return bits, values, weights, scores


def repair_solution(
    solution: ArrayLike,
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    scores: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Remove low-score selected items one at a time until feasible."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    repaired, _, weights, scores = _arrays(solution, values, weights, scores)
    total_weight = float(np.sum(weights[repaired]))
    # Lowest score first; remove the heavier item first for an exact tie.
    for index in np.lexsort((-weights, scores)):
        if total_weight <= capacity:
            break
        if repaired[index]:
            repaired[index] = False
            total_weight -= weights[index]
    return repaired


def repair_and_complete_solution(
    solution: ArrayLike,
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    scores: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Repair feasibility, then add high-score items one at a time when they fit."""
    repaired = repair_solution(solution, values, weights, capacity, scores)
    _, _, weights, scores = _arrays(repaired, values, weights, scores)
    total_weight = float(np.sum(weights[repaired]))
    # Highest score first; try the lighter item first for an exact tie.
    for index in np.lexsort((weights, -scores)):
        if not repaired[index] and total_weight + weights[index] <= capacity:
            repaired[index] = True
            total_weight += weights[index]
    return repaired


def _process_samples(samples, operation, values, weights, capacity, scores=None):
    samples = np.asarray(samples, dtype=bool)
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional Boolean array")
    return np.asarray(
        [operation(sample, values, weights, capacity, scores) for sample in samples],
        dtype=bool,
    )


def repair_samples(samples, values, weights, capacity, scores=None):
    """Apply removal-only feasibility repair independently to every sample."""
    return _process_samples(samples, repair_solution, values, weights, capacity, scores)


def repair_and_complete_samples(samples, values, weights, capacity, scores=None):
    """Apply repair and greedy completion independently to every sample."""
    return _process_samples(
        samples, repair_and_complete_solution, values, weights, capacity, scores
    )


def _process_counts(counts, operation, values, weights, capacity, scores=None):
    output: dict[str, int] = {}
    for measured, count in counts.items():
        logical = measured[::-1]
        solution = np.fromiter(
            (bit == "1" for bit in logical), dtype=bool, count=len(logical)
        )
        processed = operation(solution, values, weights, capacity, scores)
        processed_measured = "".join("1" if bit else "0" for bit in processed)[::-1]
        output[processed_measured] = output.get(processed_measured, 0) + int(count)
    if sum(output.values()) != sum(counts.values()):
        raise RuntimeError("post-processing changed the total number of shots")
    return output


def repair_counts(
    counts: Mapping[str, int], values, weights, capacity, scores=None
) -> dict[str, int]:
    """Apply removal-only repair to Qiskit-order strings and preserve shots."""
    return _process_counts(counts, repair_solution, values, weights, capacity, scores)


def repair_and_complete_counts(
    counts: Mapping[str, int], values, weights, capacity, scores=None
) -> dict[str, int]:
    """Apply repair and completion to Qiskit-order strings and preserve shots."""
    return _process_counts(
        counts, repair_and_complete_solution, values, weights, capacity, scores
    )
