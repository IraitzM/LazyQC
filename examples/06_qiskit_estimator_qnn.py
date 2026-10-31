"""Train a Qiskit Machine Learning ``EstimatorQNN`` through LazyQC's Estimator.

``EstimatorQNN`` sends its circuit, the feature batch and every shifted circuit of the
gradient through ``estimator.run``. Passing ``lazyqc.frontends.qiskit.Estimator`` instead of
Aer's Estimator is the only change; LazyQC then plans each job as one batch.

Two observable families are available. ``--observables local`` reads one ``Z`` per qubit, so
each output depends on a narrow causal cone that LazyQC shares across outputs and rows.
``--observables parity`` reads ``Z...Z`` on every qubit. Timings depend on the ansatz, the
observables and the host, so measure your own workload.

LazyQC limits Qulacs' OpenMP team to the task widths it runs (``native_threads="auto"``), so no
environment variables are needed. Aer keeps its own threading.

Run with the optional packages added on the fly:

    uv run --extra cutting --with qiskit-machine-learning --with qiskit-aer \
        python examples/06_qiskit_estimator_qnn.py --qubits 14 --observables local
"""

import argparse
from time import perf_counter

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

from lazyqc.frontends.qiskit import Estimator as LazyEstimator


def build_qnn(estimator, qubits, reps, observables):
    """Angle-encoding feature map followed by a linearly entangled RealAmplitudes ansatz."""
    features = QuantumCircuit(qubits, name="features")
    inputs = []
    for q in range(qubits):
        from qiskit.circuit import Parameter

        inputs.append(Parameter(f"x{q}"))
        features.ry(inputs[-1], q)
    ansatz = RealAmplitudes(qubits, reps=reps, entanglement="linear")
    # Aer's Estimator needs basis gates, so flatten once for both estimators.
    circuit = features.compose(ansatz).decompose(reps=4)
    if observables == "local":
        ops = [SparsePauliOp("I" * (qubits - 1 - q) + "Z" + "I" * q) for q in range(qubits)]
    else:
        ops = [SparsePauliOp("Z" * qubits)]
    return EstimatorQNN(
        circuit=circuit,
        observables=ops,
        input_params=inputs,
        weight_params=ansatz.parameters,
        estimator=estimator,
        default_precision=0.0,
    )


def timed(function, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        best = min(best, perf_counter() - start)
    return result, best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, default=14)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--observables", choices=("local", "parity"), default="local")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    networks = {
        "aer": build_qnn(AerEstimator(), args.qubits, args.reps, args.observables),
        "lazyqc": build_qnn(LazyEstimator(), args.qubits, args.reps, args.observables),
    }
    qnn = networks["lazyqc"]
    inputs = rng.uniform(0, np.pi, (args.batch, qnn.num_inputs))
    weights = rng.uniform(-1, 1, qnn.num_weights)
    targets = (
        0.5 * np.sin(inputs[:, : qnn.output_shape[0]])
        if args.observables == "local"
        else (np.sign(np.sin(inputs.sum(axis=1)))[:, None])
    )

    print(
        f"{args.qubits} qubits, {args.reps} reps, {qnn.num_weights} weights, "
        f"{qnn.output_shape[0]} outputs, batch {args.batch}, observables {args.observables}"
    )
    forward, backward = {}, {}
    for name, network in networks.items():
        network.forward(inputs, weights)  # warm-up: imports, native libraries, translation
        forward[name], t_forward = timed(lambda n=network: n.forward(inputs, weights), args.repeats)
        backward[name], t_backward = timed(
            lambda n=network: n.backward(inputs, weights), args.repeats
        )
        print(f"  {name:7} forward {t_forward * 1e3:9.1f} ms   backward {t_backward * 1e3:9.1f} ms")
    print("  max |forward difference| :", float(np.abs(forward["aer"] - forward["lazyqc"]).max()))
    print(
        "  max |gradient difference|:",
        float(np.abs(backward["aer"][1] - backward["lazyqc"][1]).max()),
    )

    start = perf_counter()
    for step in range(args.steps):
        output = qnn.forward(inputs, weights)
        _, weight_gradient = qnn.backward(inputs, weights)
        error = output - targets
        loss = float(np.mean(error**2))
        weights = weights - 0.5 * np.mean(2 * error[:, :, None] * weight_gradient, axis=(0, 1))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:2} loss {loss:.4f}")
    print(f"  trained {args.steps} steps through LazyQC in {perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
