"""Method-independent circuit compilation, persistence, and resource metrics."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from qiskit import QuantumCircuit, qpy


_NON_GATE_OPERATIONS = {"barrier", "delay", "measure", "reset"}


@dataclass(frozen=True)
class CircuitMetrics:
    method: str
    qaoa_depth: int
    representation: str
    qubits: int
    classical_bits: int
    free_parameters: int
    total_depth: int
    total_operations: int
    one_qubit_depth: int
    two_qubit_depth: int
    one_qubit_gates: int
    two_qubit_gates: int
    three_plus_qubit_gates: int
    measurements: int
    swaps: int
    operation_counts: dict[str, int]


def _is_gate_of_arity(instruction, arity: int) -> bool:
    operation = instruction.operation
    return operation.name not in _NON_GATE_OPERATIONS and operation.num_qubits == arity


def circuit_metrics(
    circuit: QuantumCircuit,
    *,
    method: str,
    qaoa_depth: int,
    representation: str,
) -> CircuitMetrics:
    """Calculate standardized logical or compiled circuit metrics."""
    operation_counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    one_qubit_gates = sum(
        1 for instruction in circuit.data if _is_gate_of_arity(instruction, 1)
    )
    two_qubit_gates = sum(
        1 for instruction in circuit.data if _is_gate_of_arity(instruction, 2)
    )
    three_plus = sum(
        1
        for instruction in circuit.data
        if instruction.operation.name not in _NON_GATE_OPERATIONS
        and instruction.operation.num_qubits >= 3
    )
    return CircuitMetrics(
        method=method,
        qaoa_depth=int(qaoa_depth),
        representation=representation,
        qubits=circuit.num_qubits,
        classical_bits=circuit.num_clbits,
        free_parameters=len(circuit.parameters),
        total_depth=int(circuit.depth()),
        total_operations=int(circuit.size()),
        one_qubit_depth=int(circuit.depth(lambda item: _is_gate_of_arity(item, 1))),
        two_qubit_depth=int(circuit.depth(lambda item: _is_gate_of_arity(item, 2))),
        one_qubit_gates=one_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        three_plus_qubit_gates=three_plus,
        measurements=operation_counts.get("measure", 0),
        swaps=operation_counts.get("swap", 0),
        operation_counts=operation_counts,
    )


def save_qpy(circuit: QuantumCircuit, path: str | Path) -> Path:
    """Save one circuit in QPY format and return the resolved output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        qpy.dump(circuit, output)
    return path.resolve()


def load_qpy(path: str | Path) -> QuantumCircuit:
    """Load exactly one circuit from a QPY file."""
    with Path(path).open("rb") as source:
        circuits = qpy.load(source)
    if len(circuits) != 1:
        raise ValueError(f"expected one circuit in {path}, found {len(circuits)}")
    return circuits[0]


def _append_metrics_csv(path: Path, rows: list[CircuitMetrics]) -> None:
    scalar_fields = [
        name for name in CircuitMetrics.__dataclass_fields__ if name != "operation_counts"
    ]
    exists = path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=scalar_fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("operation_counts")
            writer.writerow(data)


def analyze_and_save_circuits(
    *,
    parameterized_circuit: QuantumCircuit,
    trained_circuit: QuantumCircuit,
    pass_manager,
    method: str,
    qaoa_depth: int,
    output_directory: str | Path,
    compilation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save high-level circuits, compile the trained circuit, and report metrics."""
    output_directory = Path(output_directory).resolve()
    circuit_directory = output_directory / "circuits"
    backend_name = str((compilation_metadata or {}).get("backend") or "backend_unspecified")
    safe_backend = "".join(c if c.isalnum() or c in "_-" else "_" for c in backend_name)
    compiled_directory = circuit_directory / safe_backend
    compiled_directory.mkdir(parents=True, exist_ok=True)
    safe_method = method.lower().replace(" ", "_").replace("-", "_")
    stem = f"{safe_method}_p{qaoa_depth}"

    parameterized_path = save_qpy(
        parameterized_circuit, circuit_directory / f"{stem}_high_level_parameterized.qpy"
    )
    trained_path = save_qpy(
        trained_circuit, circuit_directory / f"{stem}_high_level_trained.qpy"
    )
    start = time.perf_counter()
    compiled_circuit = pass_manager.run(trained_circuit)
    compilation_seconds = time.perf_counter() - start
    compiled_path = save_qpy(
        compiled_circuit, compiled_directory / f"{stem}_compiled.qpy"
    )

    rows = [
        circuit_metrics(
            trained_circuit, method=method, qaoa_depth=qaoa_depth,
            representation="high_level_trained",
        ),
        circuit_metrics(
            compiled_circuit, method=method, qaoa_depth=qaoa_depth,
            representation="compiled",
        ),
    ]
    _append_metrics_csv(compiled_directory / "circuit_metrics.csv", rows)
    report = {
        "method": method,
        "qaoa_depth": int(qaoa_depth),
        "compilation_seconds": compilation_seconds,
        "compilation": dict(compilation_metadata or {}),
        "files": {
            "high_level_parameterized": str(parameterized_path.relative_to(output_directory)),
            "high_level_trained": str(trained_path.relative_to(output_directory)),
            "compiled": str(compiled_path.relative_to(output_directory)),
        },
        "file_paths_relative_to": "experiment_directory",
        "metrics": [asdict(row) for row in rows],
    }
    report_path = compiled_directory / f"{stem}_circuit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_file"] = str(report_path.resolve())
    report["compiled_circuit"] = compiled_circuit
    return report


def comparison_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return scalar metric rows suitable for a DataFrame or printed table."""
    return [
        {key: value for key, value in row.items() if key != "operation_counts"}
        for row in report["metrics"]
    ]
