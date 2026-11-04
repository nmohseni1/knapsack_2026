# Knapsack warm-start and Copula-QAOA

This project uses six focused Python modules:

- `copula_qaoa/knapsack.py`: instance loading, the weight-aware warm start, sampling, evaluation, and optional Gurobi reference.
- `copula_qaoa/circuits.py`: state preparation, cost layers, copula mixers, and parameterized circuit construction.
- `copula_qaoa/training.py`: sampling, all-shot CVaR, COBYLA training, progressive-depth initialization, and evaluation helpers.
- `copula_qaoa/postprocessing.py`: removal-only repair and optional greedy completion.
- `copula_qaoa/circuit_analysis.py`: QPY persistence, compilation, and standardized logical/compiled resource metrics.
- `copula_qaoa/artifacts.py`: saved evaluation counts, post-processing distributions, and plots reconstructed from saved runs.
- `copula_qaoa_exploration.ipynb`: the complete experiment, including plots, Aer MPS simulation, CVaR training, circuit reporting, depth comparison, and optional IBM hardware execution.

Plots remain in the notebook because they are part of interactive exploration rather than reusable algorithmic logic.

The notebook starts with `QUICK_MODE=True`, which performs a small but real end-to-end training run. Set it to `False` for depths 1–4 and the original research-scale shot and optimizer budgets. Hardware and Gurobi cells are deliberately opt-in.

Each training run creates a timestamped `results/` directory containing parameterized high-level, trained high-level, and compiled QPY circuits; optimized parameters; training and final-evaluation counts; warm-start data; experiment metadata; post-processing distributions; and JSON/CSV circuit reports. The saved-results cell can recreate the full comparison plot after a kernel restart without repeating training, sampling, or repair. Use the `copula_qaoa_clean` environment.

The parameterized high-level QPY circuit and its metadata are compatible in purpose with a circuit-generation repository. Training, simulation, hardware submission, evaluation, and post-processing remain separate experiment stages rather than being hidden inside the circuit generator.


## Saved result folders

Each experiment under `results/` uses this layout:

```text
<experiment>/
  experiment_configuration.json
  warm_start.npz
  postprocessing_comparison.json
  simulator_hardware_best_values.json
  circuits/
  simulator/
    training_counts_p*.json
    training_summary.json
    evaluation_counts_p*.json
    evaluation_summary.json
  hardware/
    ibm_without_qctrl_<runtime-job-id>/
    ibm_with_qctrl_<function-job-id>/
    ibm_with_qctrl_raw_<runtime-job-id>/
    ibm_with_qctrl_phoenix_raw_p1_to_p5/
```

Only files available for that experiment are present. Shared baselines and cross-method comparisons remain at the experiment root.
Circuit reports are grouped by compilation backend under `circuits/`. Simulator
PDFs belong in `simulator/`; hardware PDFs belong in their hardware run folder.
`raw` means underlying Runtime measurements, not final Q-CTRL-mitigated results.
The combined five-depth Q-CTRL folder has a `manifest.json` recording the source
job and shot count of every depth; it combines two execution jobs. Job IDs used
with IBM APIs are unchanged and do not include the folder prefixes.

Historical notebook outputs are preserved and may still display old paths.
Reload notebooks from disk after this layout update; do not run all cells just
to load plots. Use saved-result visualization cells only.


### Backend-specific circuit artifacts

Reusable, untranspiled circuits stay in `circuits/*.qpy`. Compiled circuits and
their before/after metric reports are grouped by the recorded compilation backend:

```text
circuits/
  copula_qaoa_p1_high_level_parameterized.qpy
  copula_qaoa_p1_high_level_trained.qpy
  ibm_marrakesh/
    copula_qaoa_p1_compiled.qpy
    copula_qaoa_p1_circuit_report.json
    circuit_metrics.csv
```

A compilation target does not imply that a hardware job ran on that backend.
Report `files` entries are relative to the experiment directory, so they continue
to work if the repository is moved. Existing metric numbers are unchanged.
Future circuit saving uses the supplied compilation metadata backend, or the
explicit `backend_unspecified` folder when no backend was provided.
Each notebook ends with a standalone saved-results inventory and circuit-metrics
cell. It reads local files only and does not require training or an IBM login.


## Final experiment packages

`results/final_100_items/` and `results/final_150_items/` are the two final saved
experiments. Older runs and duplicate source folders were moved outside this
repository to a dated `Copula_QAOA_archive_*` folder; nothing was deleted.
Each final folder has a manifest with job IDs, exact input and circuit locations,
cached best-value and AR summaries, and an explicit list of unavailable metrics.
Use the last notebook cell for instant saved-only plots. Do not run all cells.
