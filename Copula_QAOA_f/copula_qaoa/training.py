"""CVaR training and evaluation for warm-started Copula-QAOA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qiskit import transpile
from scipy.optimize import minimize

from .circuits import (
    bind_qaoa_parameters,
    parameterized_copula_qaoa_circuit,
)


def extract_counts(pub_result: Any) -> dict[str, int]:
    """Extract measurement counts from an Aer or Runtime SamplerV2 result."""

    for register_name in ("measure", "meas", "c"):
        register = getattr(pub_result.data, register_name, None)

        if register is not None and hasattr(register, "get_counts"):
            return register.get_counts()

    raise TypeError(
        "The sampler result does not contain a measurement-count register."
    )


def sample_circuit(
    circuit,
    sampler_mps,
    shots: int,
    basis_gates=None,
) -> dict[str, int]:
    """Transpile, measure, and sample one QAOA circuit.

    This follows the simulation procedure from the original notebook.
    """

    sampled_circuit = circuit.copy()

    # Training is an ideal Aer-MPS simulation and must not inherit the IBM
    # backend's coupling map.  A topology-free universal basis supports any
    # simulated qubit count and decomposes the custom two-qubit copula gates.
    simulation_basis = ["u", "cx"] if basis_gates is None else list(basis_gates)
    sampled_circuit = transpile(
        sampled_circuit,
        basis_gates=simulation_basis,
        optimization_level=0,
    )

    sampled_circuit.measure_active()

    job = sampler_mps.run(
        [sampled_circuit],
        shots=shots,
    )

    pub_result = job.result()[0]

    return extract_counts(pub_result)


def feasible_cvar(
    counts: Mapping[str, int],
    values,
    weights,
    capacity,
    alpha: float = 0.1,
):
    """Calculate the original all-shot feasible CVaR objective.

    Probabilities are calculated relative to all shots. Infeasible outcomes
    therefore receive zero reward rather than being removed by renormalizing
    the feasible distribution.
    """

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")

    values = np.asarray(values)
    weights = np.asarray(weights)

    total_shots = sum(counts.values())

    if total_shots == 0:
        return 0.0, 0.0, None

    feasible_shots = 0
    feasible_outcomes = []

    for measured_bitstring, count in counts.items():
        # Qiskit returns the classical bitstring in little-endian order.
        bitstring = measured_bitstring[::-1]

        selected = np.fromiter(
            (bit == "1" for bit in bitstring),
            dtype=bool,
            count=len(bitstring),
        )

        total_weight = int(np.sum(weights[selected]))

        if total_weight <= capacity:
            total_value = int(np.sum(values[selected]))
            probability = count / total_shots

            feasible_outcomes.append(
                (total_value, probability)
            )

            feasible_shots += count

    if not feasible_outcomes:
        return 0.0, 0.0, None

    # Highest-value feasible solutions first.
    feasible_outcomes.sort(
        key=lambda outcome: outcome[0],
        reverse=True,
    )

    remaining_probability = alpha
    weighted_tail_value = 0.0

    for value, probability in feasible_outcomes:
        probability_to_take = min(
            probability,
            remaining_probability,
        )

        weighted_tail_value += probability_to_take * value
        remaining_probability -= probability_to_take

        if remaining_probability <= 0:
            break

    cvar = weighted_tail_value / alpha
    feasible_probability = feasible_shots / total_shots
    best_feasible_value = feasible_outcomes[0][0]

    return (
        cvar,
        feasible_probability,
        best_feasible_value,
    )


def interpolate_initialize(
    optimized_gamma,
    optimized_beta,
    new_depth: int,
):
    """Interpolate optimized angles to initialize the next QAOA depth."""

    optimized_gamma = np.asarray(
        optimized_gamma,
        dtype=float,
    )

    optimized_beta = np.asarray(
        optimized_beta,
        dtype=float,
    )

    old_points = np.linspace(
        0,
        1,
        len(optimized_gamma),
    )

    new_points = np.linspace(
        0,
        1,
        new_depth,
    )

    new_gamma = np.interp(
        new_points,
        old_points,
        optimized_gamma,
    )

    new_beta = np.interp(
        new_points,
        old_points,
        optimized_beta,
    )

    return new_gamma, new_beta


@dataclass
class TrainingResult:
    """Training result for one Copula-QAOA depth."""

    depth: int
    beta: np.ndarray
    gamma: np.ndarray
    cvar: float
    nfev: int
    circuit: Any
    feasible_probability: float
    best_feasible_value: int | None
    success: bool
    message: str
    counts: dict[str, int] | None = None
    parameterized_circuit: Any | None = None
    circuit_report: dict[str, Any] | None = None

    @property
    def objective(self):
        return self.cvar

    @property
    def evaluations(self):
        return self.nfev


def train_copula_qaoa(
    values,
    weights,
    capacity,
    probabilities,
    sampler_mps,
    *,
    p_values=(1, 2, 3, 4),
    depths=None,
    shots=15_000,
    alpha=0.1,
    theta=-1.0,
    initial_beta=0.78,
    initial_gamma=0.2,
    tolerance=1e-2,
    maxiter=30_000,
    basis_gates=None,
    conditional_on_feasible=False,
    method="COBYLA",
    artifact_directory=None,
    analysis_pass_manager=None,
    compilation_metadata=None,
):
    """Train Copula-QAOA using the original progressive COBYLA procedure."""

    if conditional_on_feasible:
        raise ValueError(
            "The original notebook uses all-shot CVaR. "
            "Set conditional_on_feasible=False."
        )

    if depths is not None:
        p_values = tuple(depths)
    else:
        p_values = tuple(p_values)

    if not p_values:
        raise ValueError("At least one QAOA depth must be provided.")

    results = {}

    optimized_beta = np.array(
        [initial_beta],
        dtype=float,
    )

    optimized_gamma = np.array(
        [initial_gamma],
        dtype=float,
    )

    for position, depth in enumerate(p_values):
        print(f"\nTraining Copula-QAOA at p={depth}")

        circuit, beta_parameters, gamma_parameters = (
            parameterized_copula_qaoa_circuit(
                values,
                probabilities,
                depth,
                theta,
            )
        )

        def objective(parameters):
            beta_values = parameters[:depth]
            gamma_values = parameters[depth:]

            bound_circuit = bind_qaoa_parameters(
                circuit,
                beta_parameters,
                gamma_parameters,
                beta_values,
                gamma_values,
            )

            counts = sample_circuit(
                bound_circuit,
                sampler_mps,
                shots,
                basis_gates,
            )

            cvar, feasible_probability, best_value = feasible_cvar(
                counts,
                values,
                weights,
                capacity,
                alpha,
            )

            print(
                f"CVaR={cvar:,.4f}, "
                f"feasible={feasible_probability:.4f}, "
                f"best={best_value}"
            )

            return -cvar

        initial_parameters = np.concatenate(
            [
                optimized_beta,
                optimized_gamma,
            ]
        )

        bounds = (
            [(0, np.pi)] * depth
            + [(0, 2 * np.pi)] * depth
        )

        optimization_result = minimize(
            objective,
            initial_parameters,
            method=method,
            bounds=bounds,
            tol=tolerance,
            options={
                "maxiter": maxiter,
            },
        )

        optimized_beta = optimization_result.x[:depth]
        optimized_gamma = optimization_result.x[depth:]

        trained_circuit = bind_qaoa_parameters(
            circuit,
            beta_parameters,
            gamma_parameters,
            optimized_beta,
            optimized_gamma,
        )

        final_counts = sample_circuit(
            trained_circuit,
            sampler_mps,
            shots,
            basis_gates,
        )

        (
            final_cvar,
            feasible_probability,
            best_feasible_value,
        ) = feasible_cvar(
            final_counts,
            values,
            weights,
            capacity,
            alpha,
        )

        circuit_report = None
        if artifact_directory is not None:
            if analysis_pass_manager is None:
                raise ValueError(
                    "analysis_pass_manager is required when artifact_directory is set"
                )
            from .circuit_analysis import analyze_and_save_circuits

            circuit_report = analyze_and_save_circuits(
                parameterized_circuit=circuit,
                trained_circuit=trained_circuit,
                pass_manager=analysis_pass_manager,
                method="Copula-QAOA",
                qaoa_depth=depth,
                output_directory=artifact_directory,
                compilation_metadata=compilation_metadata,
            )

        results[depth] = TrainingResult(
            depth=depth,
            beta=optimized_beta.copy(),
            gamma=optimized_gamma.copy(),
            cvar=final_cvar,
            nfev=int(optimization_result.nfev),
            circuit=trained_circuit,
            feasible_probability=feasible_probability,
            best_feasible_value=best_feasible_value,
            success=bool(optimization_result.success),
            message=str(optimization_result.message),
            counts=final_counts,
            parameterized_circuit=circuit,
            circuit_report=circuit_report,
        )

        print("Optimized beta:", optimized_beta)
        print("Optimized gamma:", optimized_gamma)
        print("Final CVaR:", final_cvar)

        if position + 1 < len(p_values):
            next_depth = p_values[position + 1]

            optimized_gamma, optimized_beta = interpolate_initialize(
                optimized_gamma,
                optimized_beta,
                next_depth,
            )

    if artifact_directory is not None:
        artifact_directory = Path(artifact_directory)
        simulator_directory = artifact_directory / "simulator"
        simulator_directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "configuration": {
                "depths": list(p_values), "shots": int(shots), "alpha": float(alpha),
                "theta": float(theta), "initial_beta": float(initial_beta),
                "initial_gamma": float(initial_gamma), "tolerance": float(tolerance),
                "maxiter": int(maxiter), "optimizer": method,
            },
            "results": {},
        }
        for depth, result in results.items():
            summary["results"][str(depth)] = {
                "beta": result.beta.tolist(), "gamma": result.gamma.tolist(),
                "cvar": float(result.cvar), "evaluations": int(result.nfev),
                "feasible_probability": float(result.feasible_probability),
                "best_feasible_value": result.best_feasible_value,
                "success": bool(result.success), "message": result.message,
            }
            (simulator_directory / f"training_counts_p{depth}.json").write_text(
                json.dumps(result.counts, indent=2)
            )
        (simulator_directory / "training_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    return results


def run_sampler_counts(
    circuit,
    sampler_mps,
    shots: int,
    basis_gates=None,
):
    """Sample a trained circuit outside the optimization loop."""

    return sample_circuit(
        circuit,
        sampler_mps,
        shots,
        basis_gates,
    )


def cvar_from_counts(
    counts,
    values,
    weights,
    capacity,
    alpha=0.1,
    conditional_on_feasible=False,
    **_,
):
    """Compatibility wrapper for evaluating previously obtained counts."""

    if conditional_on_feasible:
        raise ValueError(
            "The original notebook uses all-shot CVaR. "
            "Set conditional_on_feasible=False."
        )

    return feasible_cvar(
        counts,
        values,
        weights,
        capacity,
        alpha,
    )


def take_prefix_sum(
    value_counts: Mapping[int, int],
    max_samples: int | None = None,
) -> dict[int, int]:
    """Keep at most ``max_samples`` shots, preserving value order."""

    if max_samples is None:
        return dict(value_counts)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None.")

    selected_counts: dict[int, int] = {}
    used_samples = 0
    for value, count in value_counts.items():
        remaining = max_samples - used_samples
        if remaining <= 0:
            break
        count_to_use = min(int(count), remaining)
        selected_counts[int(value)] = count_to_use
        used_samples += count_to_use
    return selected_counts


def approximation_ratio(
    feasible_by_value: Mapping[int, int],
    optimal_value: int,
    max_samples: int | None = None,
) -> tuple[float, int]:
    """Calculate the approximation ratio using up to max_samples valid shots.

    Returns:
        approximation_ratio
        number_of_valid_shots_used
    """

    if optimal_value <= 0:
        raise ValueError("optimal_value must be positive.")

    # Highest-value feasible solutions first.
    sorted_feasible = dict(
        sorted(
            feasible_by_value.items(),
            key=lambda item: item[0],
            reverse=True,
        )
    )

    selected_counts = take_prefix_sum(
        sorted_feasible,
        max_samples=max_samples,
    )

    valid_shots_used = sum(selected_counts.values())

    if valid_shots_used == 0:
        return 0.0, 0

    weighted_value = sum(
        value * count
        for value, count in selected_counts.items()
    )

    ratio = weighted_value / (
        valid_shots_used * optimal_value
    )

    return ratio, valid_shots_used
