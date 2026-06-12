# JensenShannonDivergence

Python package for Jensen-Shannon divergence estimation on tabular data.

The core API works with NumPy arrays, pandas DataFrames, PyTorch tensors, or numeric array-like values.

## Installation

### From GitHub

If the package is only published on GitHub, install it directly from the repository:

```bash
pip install "git+https://github.com/AlbaGarridoLopezz/jensenshannondivergence.git"
```

To install a specific branch or tag:

```bash
pip install "git+https://github.com/AlbaGarridoLopezz/jensenshannondivergence.git@main"
```

### From PyPI

After publishing the package to PyPI, users will be able to install it with:

```bash
pip install jensenshannondivergence
```

Optional extras can be installed with:

```bash
pip install "jensenshannondivergence[all]"
```

### Local Development

Clone the repository and install it in editable mode:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

If you want the optional extras:

```bash
pip install -e .[all]
```

## What Users Should Import

The library is *custom-first*: call it with your own real/reference samples and generated/synthetic samples.

### Simple API

`estimate_jensen_shannon` returns a `float` with the estimated Jensen-Shannon divergence.

```python
import numpy as np
from jensenshannondivergence import estimate_jensen_shannon

x_reference = np.random.normal(size=(1000, 10))
x_synthetic = np.random.normal(loc=0.2, size=(1000, 10))

js = estimate_jensen_shannon(
    x_reference,
    x_synthetic,
    discriminator_type="MLP",  # MLP, RF, XGBoost, LogReg, LogRegPol, TabPFN
    n_iter=30,                 # used by RF/XGBoost/LogReg optimizers
    seed=0,
)

print(js)
```

Use `return_result=True` if you need the full evaluator and output path:

```python
from jensenshannondivergence import estimate_jensen_shannon

result = estimate_jensen_shannon(
    x_reference,
    x_synthetic,
    discriminator_type="RF",
    m=500,
    l=250,
    n_iter=20,
    return_result=True,
)

print(result.evaluator.disc_js)
print(result.results_path)
```

### Predefined experiments (repository)

Predefined experiment loaders live under the `experiments/` folder and are intended
for repository-based development. The library package no longer exposes a
`use_predefined` runtime option. To run a predefined use case from the repo,
load the experiment data and call the library API with the returned tensors:

```python
# run from the repository root (so `experiments` is importable)
from experiments import data as exp_data
from jensenshannondivergence import estimate_jensen_shannon

# load tensors for a use case
x_r, x_s, dist_r, dist_s = exp_data.load_data('use_case_7', n=10, m=2000, l=2000, seed=0)

js = estimate_jensen_shannon(
    x_r,
    x_s,
    discriminator_type='MLP',
    m=2000,
    l=2000,
    seed=0,
)
```

If you prefer orchestration, use `main_experiments.py` which loads experiment
data and calls the library for you (see the `Experiments CLI` section above).

## CLI For Your Own Data

After installation, `jsd-estimate` estimates JS directly from two CSV files:

```bash
jsd-estimate --x-p real_data.csv --x-q gen_data.csv --discriminator MLP --epochs 100
```

Useful arguments:

- `--discriminator` / `--classifier`
- `--m`, `--l`
- `--n-iter`
- `--epochs`
- `--ratio-correction-mode`
- `--results-root`
- `--save-plots`

## Experiments CLI

`main_experiments.py` is only for predefined experiments and is intended for repository development runs.

List available use cases:

```bash
python main_experiments.py list
```

Train selected experiments:

```bash
python main_experiments.py train --discriminators MLP RF --experiments use_case_1 use_case_3 --n-iter 30
```

Run discriminator tests across classifiers:

```bash
python main_experiments.py test-discriminators --experiments use_case_1 --n-iter 20
```

Use `--models`, `--datasets-use-case-4` for the real-data use cases.

## Interactive Tutorial Notebook

Use the tutorial notebook for a minimal end-to-end run:

- `Tutorial.ipynb`

The tutorial saves run outputs to:

- `experiments/tutorials_outputs/`

If you edit code under `src/`, restart the notebook kernel before re-running cells.

## Experiments Path Convention

All read/write experiment paths are centralized under `experiments/`, including:

- `experiments/data/`
- `experiments/results_MLP/`
- `experiments/results_RF/`
- `experiments/results_XGBoost/`
- `experiments/results_LogReg/`
- `experiments/results_LogRegPol/`
- `experiments/results_TabPFN/`
- `experiments/results_discriminators/`
- `experiments/calibration_audit_results/`
- `experiments/tutorials_outputs/`

You can override this root with the environment variable `JSD_EXPERIMENTS_ROOT`.

Notes on results and plotting behavior: library functions do not produce plots by default — plotting is performed by the experiments scripts and notebooks. Experiment outputs (tables, CSVs, and optional plots) are written under the experiments root; control plot saving with the `save_plots` argument in the experiments CLI or by setting `save_plots=True` in the notebooks.

## Notes

- For best performance, run with GPU when available.
- Some baselines require optional dependencies (`syndat`, `synthcity`, `tabpfn`).
- If using TabPFN, make sure your PyTorch version is compatible.
