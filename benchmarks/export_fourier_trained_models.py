#!/usr/bin/env python3
"""Export the trained angle classifiers of the Fourier geometry-selection study.

Run inside the Fourier project's environment, which provides torch and its data
pipeline; nothing in that project is modified:

    uv run --project ../fourierquantumencoder --offline \\
        python benchmarks/export_fourier_trained_models.py

For every completed (dataset, seed) run with an ``angle.pt`` checkpoint, this
rebuilds the test split with the study's own ``data_for``, checks its SHA-256
against the recorded ``inputs.json``, and writes the 8 encoded test angles,
trained weights, linear read-out, labels and the study's saved predictions to
``results/qml_workloads/fourier_trained/<dataset>-<seed>.npz``.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOURIER = ROOT.parent / "fourierquantumencoder"
STUDY = FOURIER / "results/geometry-selection"
OUTPUT = ROOT / "results/qml_workloads/fourier_trained"
DATASETS = ("ECG", "MNIST1D", "EEG", "PneumoniaMNIST")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier-root", type=Path, default=FOURIER)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="export a subset; the index keeps earlier entries for the others",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.fourier_root))
    import torch
    from experiments import run_geometry_selection as study

    protocol = json.loads((STUDY / "protocol.json").read_text())
    commit = subprocess.run(
        ["git", "-C", str(args.fourier_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT / "index.json"
    index = [
        run
        for run in (json.loads(index_path.read_text())["runs"] if index_path.exists() else [])
        if run["dataset"] not in args.datasets
    ]
    for dataset in args.datasets:
        for seed in protocol["seeds"]:
            folder = STUDY / f"{dataset}-{seed}"
            checkpoint = folder / "angle.pt"
            if not (folder / "complete.json").exists() or not checkpoint.exists():
                print(f"skip {folder.name}: incomplete or no angle checkpoint", flush=True)
                continue
            recorded = json.loads((folder / "inputs.json").read_text())
            data = study.data_for(dataset, seed)
            if study.digest(data["xt"]) != recorded["test_sha256"]:
                raise SystemExit(f"{folder.name}: rebuilt test inputs differ from the study")
            spec = study.candidates(data["shape"])[0]
            if spec["candidate"] != "angle":
                raise SystemExit(f"{folder.name}: first candidate is {spec['candidate']}")
            angles = study.model_input(data["xt"], data["shape"], spec).numpy()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            predictions = np.load(folder / "angle-test.npy")
            target = OUTPUT / f"{dataset}-{seed}.npz"
            np.savez_compressed(
                target,
                angles=angles.astype(np.float64),
                labels=np.asarray(data["yt"], dtype=np.int64),
                weights=state["weights"].double().numpy(),
                readout_weight=state["readout.weight"].double().numpy(),
                readout_bias=state["readout.bias"].double().numpy(),
                study_predictions=predictions.astype(np.int64),
            )
            index.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "file": target.name,
                    "grid": list(spec["grid"]),
                    "test_examples": int(len(angles)),
                    "test_sha256": recorded["test_sha256"],
                    "checkpoint_sha256": sha256(checkpoint),
                    "predictions_sha256": sha256(folder / "angle-test.npy"),
                }
            )
            print(f"exported {target.name} ({len(angles)} test examples)", flush=True)
    index.sort(key=lambda run: (DATASETS.index(run["dataset"]), run["seed"]))
    index_path.write_text(
        json.dumps(
            {
                "source": "fourierquantumencoder results/geometry-selection "
                "(geometry-validation-v2)",
                "source_commit": commit,
                "protocol_sha256": sha256(STUDY / "protocol.json"),
                "model": "code.qnn.AngleEncodingClassifier(seq_len=8, n_qubits=8, n_reuploads=4, "
                "entangler_layers=1, use_entangling_ansatz=True, use_entangling_measurement=True)",
                "circuit": "4 x [AngleEmbedding(RY) on 8 wires; StronglyEntanglingLayers(1 layer, "
                "range 1)]; readout Z_i and Z_i Z_{i+1 mod 8}; linear head on the 16 values",
                "runs": index,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
