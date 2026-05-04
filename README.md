# Does Steepest Selection Help? An Empirical Study of SPGD

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)

First systematic empirical study of **Steepest Perturbed Gradient Descent
(SPGD)** [Vahedi & Ilies, 2024] on machine-learning loss landscapes.

This repository contains the full code, configurations, figures, and
result artefacts for the WPI MA 551 (Computational Statistics) final
project, *"Does Steepest Selection Help? An Empirical Study of
Saddle-Point-Escaping Optimizers on Non-Convex Machine-Learning Loss
Landscapes"* by **Shyam Patadia** (Spring 2026).

---

## TL;DR

SPGD is a candidate-selection rule on top of gradient descent: every
*IterP* steps, sample N<sub>P</sub> uniform candidates around the
current iterate and commit the one that yields the steepest loss
decrease. The original paper proposed it but never tested it on neural
networks and never compared against a same-compute random-selection
control. This work does both.

**Headline result.** On CIFAR-10/ResNet-18 with three paired seeds,
SPGD beats its random-perturbation control (RPGD) on **every seed**
with a paired-mean test-accuracy gap of **+3.78 percentage points**.
On a non-convex matrix-completion benchmark with documented saddle
structure (MovieLens-100K, Burer–Monteiro), SPGD's selection rule
fires at **30%** acceptance vs **9%** for the random control —
direct evidence that the mechanism is empirically active on a real
ML problem.

> **It's not the noise. It's the choice.**

---

## Repository layout

```
.
├── src/spgd_study/          # Importable package
│   ├── optimizers/          # SPGD, RPGD, PGD as torch.optim.Optimizer subclasses
│   ├── benchmarks.py        # Rastrigin / Ackley / Rosenbrock
│   ├── data.py              # OpenML / CIFAR-10 / MovieLens loaders
│   ├── diagnostics.py       # Stagnation episodes + escape-time logger
│   ├── models.py            # MLP / ResNet-18 (BN-disabled in early layers)
│   ├── runner.py            # Synthetic / Two Moons trainer
│   ├── nn_runner.py         # Mini-batch trainer (OpenML, CIFAR, etc.)
│   └── utils.py             # Seeding, paired-protocol helpers
├── experiments/             # One script per experiment + plotting + configs
│   ├── configs/             # YAML hyperparameters per experiment
│   ├── exp1_benchmarks.py   # Synthetic non-convex benchmarks
│   ├── exp2_two_moons.py    # Two Moons MLP (visualisation)
│   ├── exp3_openml.py       # OpenML-CC18 tabular MLP
│   ├── exp4_cifar10.py      # ResNet-18 anchor experiment
│   ├── exp5_ablation.py     # N_P × IterP grid
│   ├── exp6_matrix_completion.py
│   ├── compute_exp6_escape_time.py
│   └── plot_exp{1..6}.py    # All plotting scripts (PNG out)
├── tests/                   # Smoke tests for optimizers, OpenML loader
├── turing/                  # WPI Turing cluster Slurm scripts
├── figures/                 # All PNG figures used in the report
├── results/                 # CSVs and JSON summaries used by the report
├── report/                  # LaTeX sources for report + presentation
├── pyproject.toml           # uv-managed project
└── uv.lock                  # Pinned dependency graph
```

---

## Setup with `uv`

This project is managed end-to-end with [`uv`](https://docs.astral.sh/uv/).
Install uv first (one-time):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the project's pinned environment:

```bash
uv sync --python 3.11
```

This creates `.venv/` and installs PyTorch (CUDA 12.1 wheels), torchvision,
NumPy, scikit-learn, OpenML, pandas, matplotlib, etc., exactly as
specified in `uv.lock`.

Verify the install:

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## Reproducing the experiments

All scripts are run via `uv run` so they pick up the project's locked
environment automatically. Each experiment writes results into
`results/` and figures into `figures/`.

### Experiment 1 — Synthetic non-convex benchmarks
```bash
uv run python experiments/exp1_benchmarks.py --config experiments/configs/exp1.yaml
uv run python experiments/plot_exp1.py
```

### Experiment 2 — Two Moons MLP (Appendix; visualisation diagnostic)
```bash
uv run python experiments/exp2_two_moons.py --config experiments/configs/exp2.yaml
uv run python experiments/plot_exp2.py
```

### Experiment 3 — OpenML-CC18 tabular
```bash
uv run python experiments/exp3_openml.py --config experiments/configs/exp3.yaml
uv run python experiments/plot_exp3.py
```

### Experiment 4 — CIFAR-10 / ResNet-18 (anchor experiment)
Local single-seed (any CUDA GPU):
```bash
uv run python experiments/exp4_cifar10.py --config experiments/configs/exp4.yaml --seed 0 --optimizer spgd
```
On the WPI Turing cluster (Slurm array, all 5 optimizers × 3 seeds):
```bash
sbatch turing/slurm_exp4.sbatch
```
Plot:
```bash
uv run python experiments/plot_exp4.py
```

### Experiment 5 — N<sub>P</sub> × *IterP* ablation (Appendix)
```bash
uv run python experiments/exp5_ablation.py --config experiments/configs/exp5.yaml
uv run python experiments/plot_exp5.py
```

### Experiment 6 — MovieLens-100K matrix completion
```bash
uv run python experiments/exp6_matrix_completion.py --config experiments/configs/exp6.yaml
uv run python experiments/compute_exp6_escape_time.py
uv run python experiments/plot_exp6.py
```

---

## The five optimizers compared

| Name     | Description                                                                                          |
|:---------|:-----------------------------------------------------------------------------------------------------|
| **SGD**  | Vanilla stochastic gradient descent. First-order baseline.                                           |
| **Adam** | Per-parameter adaptive scaling. Modern ML default.                                                   |
| **PGD**  | Gradient descent with one Gaussian perturbation when ‖∇f‖ is small (Jin et al., 2017).               |
| **RPGD** | *(novel control)* Same multi-candidate perturbation as SPGD, but commits a **random** candidate.    |
| **SPGD** | Steepest of N<sub>P</sub> uniform candidates committed; paper's ≤ acceptance rule.                   |

The **SPGD-vs-RPGD** contrast is the cleanest test of SPGD's central
claim: both algorithms perturb the same way and pay the same compute
cost, so any difference is attributable to the selection rule.
Implementations live in [`src/spgd_study/optimizers/`](src/spgd_study/optimizers/).

---

## Compute environment

| | |
|:---|:---|
| **Local** | Windows 11, RTX 4050 — used for Experiments 1, 2, 3, 5, 6 |
| **Cluster** | WPI Turing, NVIDIA A30 GPU — used for Experiment 4 (CIFAR-10/ResNet-18) |
| **Package manager** | [`uv`](https://github.com/astral-sh/uv) |
| **Cluster transfer** | `scp` |

---

## Report

LaTeX sources for both the written report and the presentation live in
[`report/`](report/):

- [`report.tex`](report/report.tex) — main report (arXiv-style)
- [`presentation.tex`](report/presentation.tex) — Beamer presentation
- [`references.bib`](report/references.bib) — bibliography

Build with:

```bash
cd report
pdflatex report && bibtex report && pdflatex report && pdflatex report
pdflatex presentation && pdflatex presentation
```

---

## Citation

If you reference this work:

```bibtex
@misc{patadia2026spgd,
  author       = {Shyam Patadia},
  title        = {Does Steepest Selection Help? An Empirical Study of
                  Saddle-Point-Escaping Optimizers on Non-Convex Machine-Learning
                  Loss Landscapes},
  year         = {2026},
  howpublished = {WPI MA 551 Computational Statistics, Final Project},
  url          = {https://github.com/shyampatadia/spgd_research}
}
```

The original SPGD paper:

```bibtex
@article{vahedi2024spgd,
  title   = {SPGD: Steepest Perturbed Gradient Descent Optimization},
  author  = {Vahedi, Amir M. and Ilies, Horea T.},
  journal = {arXiv preprint arXiv:2411.04946},
  year    = {2024}
}
```

---

## Acknowledgments

The author thanks the WPI Academic Computing team for access to the
**Turing** cluster, on which the CIFAR-10 anchor experiment was carried
out.

## License

Code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE) if present, otherwise add one before distributing).
The third-party SPGD paper sources under `spgd_paper/` are excluded from
the repository for copyright reasons.
