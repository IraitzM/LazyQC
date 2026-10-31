# Generated-table provenance audit

Status: **PASS**. 13 generated tables regenerated and compared byte for byte. No benchmark was rerun and no raw result was edited.

| Label | Generated file | Generator | Raw source(s) | Byte-identical |
|---|---|---|---|---|
| `tab:rq1` | `paper/sections/generated/rq1_ratios.tex` | `benchmarks/summarize_direct.py` | `results/direct/scaling_1t.json`<br>`results/direct/scaling_mt.json` | True |
| `tab:backends` | `paper/sections/generated/backend_matrix.tex` | `benchmarks/summarize_direct.py` | `results/direct/backend_matrix.json`<br>`results/direct/scaling_1t.json` | True |
| `artifact-only:remote-backend` | `paper/sections/generated/backend_matrix_nvidia-h100-pcie.tex` | `benchmarks/summarize_direct.py` | `results/direct/remote/nvidia-h100-pcie/backend_matrix.json` | True |
| `tab:memory-wall` | `paper/sections/generated/memory_wall.tex` | `benchmarks/summarize_direct.py` | `results/direct/memory_wall.json` | True |
| `tab:intermediate-cones` | `paper/sections/generated/intermediate_cones.tex` | `benchmarks/summarize_direct.py` | `results/direct/memory_wall.json` | True |
| `tab:jacobian` | `paper/sections/generated/jacobian.tex` | `benchmarks/summarize_direct.py` | `results/direct/jacobian_1t.json`<br>`results/direct/jacobian_mt.json`<br>`results/direct/jacobian_mt_chunked24.json` | True |
| `tab:training` | `paper/sections/generated/training.tex` | `benchmarks/summarize_direct.py` | `results/direct/training_1t.json`<br>`results/direct/training_mt.json` | True |
| `tab:remote` | `paper/sections/generated/remote_ratios_nvidia-h100-pcie.tex` | `benchmarks/summarize_direct.py` | `results/direct/scaling_1t.json`<br>`results/direct/scaling_mt.json`<br>`results/direct/remote/nvidia-h100-pcie/scaling_1t.json`<br>`results/direct/remote/nvidia-h100-pcie/scaling_mt.json` | True |
| `tab:heldout` | `paper/sections/generated/heldout_families.tex` | `benchmarks/paper_generalization_tables.py` | `results/qml_workloads/qml_heldout_family_summary.json` | True |
| `tab:regret` | `paper/sections/generated/regret_decomposition.tex` | `benchmarks/paper_generalization_tables.py` | `results/qml_workloads/grouping_oracle.json`<br>`results/qml_workloads/grouping_oracle_heldout.json` | True |
| `tab:trained` | `paper/sections/generated/trained_models.tex` | `benchmarks/paper_generalization_tables.py` | `results/qml_workloads/trained_iris_qml_v3.json`<br>`results/qml_workloads/fourier_trained_qml.json` | True |
| `tab:policy-ablations` | `paper/sections/generated/policy_ablations.tex` | `benchmarks/summarize_policy_ablations.py` | `results/qml_workloads/qml_heldout_family_session0.json`<br>`results/qml_workloads/qml_heldout_family_session1.json` | True |
| `tab:lightcone-ablation` | `paper/sections/generated/lightcone_ablation.tex` | `benchmarks/summarize_lightcone_ablation.py` | `results/direct/lightcone_ablation.json`<br>`results/direct/lightcone_jacobian_ablation.json` | True |

The JSON companion records SHA-256 hashes for every generated table and raw source, the regeneration commands, and the raw-artifact count/failure checks.
