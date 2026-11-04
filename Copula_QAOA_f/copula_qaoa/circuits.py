"""High-level circuit construction for warm-started Copula-QAOA."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def initialize_product_state(circuit: QuantumCircuit, probabilities: ArrayLike) -> QuantumCircuit:
    """Prepare independent Bernoulli marginals using RY rotations."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != circuit.num_qubits:
        raise ValueError("one probability is required per qubit")
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("probabilities must lie strictly between zero and one")
    for qubit, probability in enumerate(probabilities):
        circuit.ry(2.0 * np.arcsin(np.sqrt(probability)), qubit)
    return circuit


def copula_pair_mixer(p_first: float, p_second: float, beta, theta: float = -1.0) -> QuantumCircuit:
    """Construct the two-qubit Copula mixer used by the original notebook."""
    if not 0.0 < p_first < 1.0 or not 0.0 < p_second < 1.0:
        raise ValueError("copula probabilities must lie strictly between zero and one")
    if not -1.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [-1, 1]")
    shift = theta * p_first * p_second * (1 - p_first) * (1 - p_second)
    p_second_given_first = np.clip((p_first * p_second + shift) / p_first, 0.0, 1.0)
    p_second_given_not_first = np.clip(
        ((1 - p_first) * p_second - shift) / (1 - p_first), 0.0, 1.0
    )
    rotation = QuantumCircuit(2, name="copula_rotation")
    rotation.ry(2.0 * np.arcsin(np.sqrt(p_first)), 0)
    rotation.cry(2.0 * np.arcsin(np.sqrt(p_second_given_first)), 0, 1)
    rotation.x(0)
    rotation.cry(2.0 * np.arcsin(np.sqrt(p_second_given_not_first)), 0, 1)
    rotation.x(0)
    mixer = QuantumCircuit(2, name="copula_pair")
    mixer.compose(rotation.inverse(), inplace=True)
    mixer.rz(-2.0 * beta, 0)
    mixer.rz(-2.0 * beta, 1)
    mixer.compose(rotation, inplace=True)
    return mixer


def ring_edges(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Return the alternating-edge schedule used by the ring mixer."""
    if n_qubits < 2:
        raise ValueError("the ring mixer requires at least two qubits")
    return tuple(
        (first, (first + 1) % n_qubits)
        for start in (1, 0)
        for first in range(start, n_qubits, 2)
    )


def add_ring_copula(circuit: QuantumCircuit, probabilities: ArrayLike, beta, theta: float = -1.0) -> QuantumCircuit:
    """Apply Copula pair mixers to alternating edges of a ring."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != circuit.num_qubits:
        raise ValueError("one probability is required per qubit")
    for first, second in ring_edges(circuit.num_qubits):
        mixer = copula_pair_mixer(probabilities[first], probabilities[second], beta, theta)
        circuit.compose(mixer, [first, second], inplace=True)
    return circuit


def add_value_phase(circuit: QuantumCircuit, values: ArrayLike, gamma) -> QuantumCircuit:
    """Apply the original maximum-value-normalized phase separator."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != circuit.num_qubits:
        raise ValueError("one value is required per qubit")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        raise ValueError("at least one value must be nonzero")
    for qubit, value in enumerate(values):
        circuit.p(gamma * value / scale, qubit)
    return circuit


def build_copula_qaoa_circuit(
    values: ArrayLike,
    probabilities: ArrayLike,
    gammas: Sequence,
    betas: Sequence,
    theta: float = -1.0,
) -> QuantumCircuit:
    """Build an unmeasured, untranspiled high-level Copula-QAOA circuit."""
    values = np.asarray(values)
    probabilities = np.asarray(probabilities)
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have the same length")
    circuit = QuantumCircuit(len(values), name=f"copula_qaoa_p{len(gammas)}")
    initialize_product_state(circuit, probabilities)
    for gamma, beta in zip(gammas, betas):
        add_value_phase(circuit, values, gamma)
        add_ring_copula(circuit, probabilities, beta, theta)
    return circuit


def parameterized_copula_qaoa_circuit(
    values: ArrayLike, probabilities: ArrayLike, depth: int, theta: float = -1.0
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build a high-level parameterized circuit and its beta/gamma vectors."""
    if depth < 1:
        raise ValueError("depth must be at least one")
    betas = ParameterVector("beta", depth)
    gammas = ParameterVector("gamma", depth)
    circuit = build_copula_qaoa_circuit(values, probabilities, gammas, betas, theta)
    return circuit, betas, gammas


def bind_qaoa_parameters(
    circuit: QuantumCircuit,
    betas: Sequence,
    gammas: Sequence,
    beta_values: ArrayLike,
    gamma_values: ArrayLike,
) -> QuantumCircuit:
    """Bind beta and gamma explicitly, avoiding parameter-order ambiguity."""
    beta_values = np.asarray(beta_values, dtype=float)
    gamma_values = np.asarray(gamma_values, dtype=float)
    if len(betas) != len(beta_values) or len(gammas) != len(gamma_values):
        raise ValueError("parameter vectors and values must have matching lengths")
    mapping = dict(zip(betas, beta_values))
    mapping.update(dict(zip(gammas, gamma_values)))
    return circuit.assign_parameters(mapping, inplace=False)
