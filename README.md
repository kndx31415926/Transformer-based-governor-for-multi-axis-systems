# Transformer-based-governor-for-multi-axis-systems
Transformer-based governor for multi-axis systems
# Deployable Power-Aware Time Scaling (DC-Bus Power Budgeting) — MATLAB Reference Implementation

This repository contains the MATLAB reference implementation accompanying the manuscript:

> **Deployable power-aware time scaling with a look-ahead Transformer-based governor for DC-bus input power budgeting and smoothing in multi-axis systems**  
> Submitted to *Control Engineering Practice*

## What’s included

- Two-layer control architecture:
  - Inner-loop computed-torque tracking controller
  - Outer-loop progress-rate (time-scaling) governor
- DC-bus input power budgeting with regeneration sharing / dump accounting
- Feasibility-search teacher for governor-consistent supervision
- Look-ahead Transformer encoder (MATLAB `dlnetwork`) predicting `a_des`
- Paper-oriented reproduction scripts (representative trial + Monte-Carlo evaluation)

---

## Requirements

- MATLAB (R2021a+ recommended)
- Robotics System Toolbox
- Deep Learning Toolbox

All default parameters in `spinn3d_params_block.m`.

---

## Quick start

1. Clone or download this repository.
2. Open MATLAB and `cd` to the repository root.
3. Add the repository to your MATLAB path:

```matlab
addpath(genpath(pwd));
```

4. Ensure the folder `data_alpha/` exists at the repository root.

This repository is expected to include a pre-generated `data_alpha/` directory produced by running
`spinn3d_run_alpha_gov_budget_trf()` with the default parameter block. If `data_alpha/` is missing,
see **Training from scratch** below.

---

## Run the online demos

### Proposed method (Transformer + governor)

```matlab
spinn3d_demo_ctcgov_online_nn
```

### Baselines

```matlab
spinn3d_demo_ctcgov_online_nonn
spinn3d_demo_ctcgov_online_nonn_nolimit
```

Tip: Each script provides usage details via MATLAB help, e.g.:

```matlab
help spinn3d_demo_ctcgov_online_nn
```

---

## Reproduce paper results

### Representative trial figure

```matlab
paper_fig3_representative_trial();
```

This generates a paper-style comparison figure (Baseline vs TRF), optionally including end-effector
trajectory and robot skeleton overlays.

### Monte-Carlo evaluation (tables + paired statistics)

```matlab
R = paper_power_eval();
```

Key outputs:
- `R.trials_table`
- `R.summary_table`
- `R.paired_stats_table`

By default, exports are written to:

```
paper_mc_payload_only_nn_vs_nolimit_vs_nonn/
```

Optional configuration example:

```matlab
opts = struct();
opts.include_nonn_feasible = false;  % compare only TRF vs baseline
opts.power_eval_mode = 'raw';        % evaluate using raw bus power (see script header)
R = paper_power_eval(opts);
```

---

## Training from scratch (dataset → Transformer → model)

All parameters are defined in `spinn3d_params_block.m`. To regenerate the dataset and train the
Transformer model, run:

```matlab
out = spinn3d_run_alpha_gov_budget_trf();
disp(out);
```

This script generates and/or updates:

```
data_alpha/alpha_gov_seq_ds_chunk_*.mat
data_alpha/spinn3d_alpha_gov_seq_dataset_master.mat
data_alpha/model_alpha_gov_trf.mat
```

The model path is defined in `spinn3d_params_block.m` via `PB.paths.model_alpha`.

---

## Paper ↔ code naming

The paper’s progress rate \(a(t)\) corresponds to `alpha` in code.

| Paper | Code |
|------:|:-----|
| \(a(t)\) | `alpha` |
| \(a_{des}\) | `a_des` |
| \(a_{post}\) | `a_post` |
| \(P_{grid}\) | `Pgrid` |
| \(P_{dump}\) | `Pdump` |

---

## Notes on large files

The `data_alpha/` directory may contain large `.mat` files (dataset chunks and model checkpoints).
If you hit GitHub file-size limits, consider:
- Git LFS for large artifacts, and/or
- uploading `data_alpha/` as a GitHub Release asset, and/or
- archiving a frozen snapshot via a DOI-backed repository (e.g., Zenodo) for long-term reproducibility.

---

## License

Add a `LICENSE` file before making the repository public (e.g., MIT, BSD-3-Clause, GPL-3.0).

---

## Citation

If you use this code, please cite the accompanying manuscript.

```bibtex
@article{li2026dc_bus_timescaling,
  title   = {Deployable power-aware time scaling with a look-ahead Transformer-based governor for DC-bus input power budgeting and smoothing in multi-axis systems},
  author  = {Li, Peizhong and Li, Zhenglin and Zhu, Ting and Fang, Zhijie and Fang, Hao},
  journal = {Control Engineering Practice},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Contact

Please open a GitHub issue for questions and bug reports.
