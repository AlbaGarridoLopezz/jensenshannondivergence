# Author: UPM
# Email: alba.garrido.lopez@upm.es
# Date: 25/05/2026

from __future__ import annotations
import argparse
import csv
import random
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import os
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from jensenshannondivergence import JSDEstimator
from jensenshannondivergence.estimators import normalize_discriminator_type, supported_discriminators

def _load_experiments_data_module():
    data_path = EXPERIMENTS_ROOT / "data.py"
    spec = importlib.util.spec_from_file_location("jensenshannondivergence_experiments_data", data_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiments data module from {data_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    N: Sequence[Any] = field(default_factory=list)
    M: Sequence[int] = field(default_factory=list)
    L: Sequence[int] = field(default_factory=list)
    models: Sequence[str | None] = field(default_factory=lambda: (None,))
    datasets: Sequence[str | None] = field(default_factory=lambda: (None,))
    gap_magnitudes: Sequence[float] = field(default_factory=lambda: (0.7,))
    ratio_modes: Sequence[str] = field(default_factory=lambda: ("auto",))


def uc1() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_1", N=[None], M=[20, 200, 2000], L=[20, 200, 2000])


def uc2() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_2", N=[None], M=[20, 200, 2000], L=[20, 200, 2000])


def uc3() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_3", N=list(range(10, 151, 10)), M=[2000], L=[2000])


def uc4(models: Sequence[str] = ("vae", "ctgan"), datasets: Sequence[str] = ("adult", "intrusion")) -> ExperimentSpec:
    return ExperimentSpec(name="use_case_4", N=[10000], M=[7500], L=[1000], models=tuple(models), datasets=tuple(datasets))


def uc5() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_5", N=[round(i / 10, 2) for i in range(10)], M=[2000], L=[2000])


def uc6() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_6", N=[0.1, 0.5, 1.0], M=[2000], L=[2000], gap_magnitudes=(0.3, 0.7, 1.0), ratio_modes=("on", "off"))


def uc7() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_7", N=[2, 10, 25, 40, 50], M=[2000], L=[2000])


def uc8() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_8", N=[0.1, 0.5, 1.0], M=[2000], L=[2000], gap_magnitudes=(0.3, 0.7, 1.0), ratio_modes=("auto", "on", "off"))


def uc11(models: Sequence[str] = ("vae", "ctgan"), datasets: Sequence[str] = ("adult", "intrusion")) -> ExperimentSpec:
    return ExperimentSpec(name="use_case_11", N=["train_100", "train_50", "train_25", "train_12_5"], M=[400], L=[112], models=tuple(models), datasets=tuple(datasets))


def uc12(models: Sequence[str] = ("vae", "ctgan"), datasets: Sequence[str] = ("adult", "intrusion")) -> ExperimentSpec:
    return ExperimentSpec(name="use_case_12", N=[10000], M=[3750], L=[500], models=tuple(models), datasets=tuple(datasets))


def uc13() -> ExperimentSpec:
    return ExperimentSpec(name="use_case_13", N=[0.1, 0.5, 1.0], M=[2000], L=[2000], gap_magnitudes=(0.3, 0.7, 1.0), ratio_modes=("on", "off"))


def uc14(models: Sequence[str] = ("vae", "ctgan"), datasets: Sequence[str] = ("adult", "intrusion")) -> ExperimentSpec:
    return ExperimentSpec(name="use_case_14", N=[1000], M=[750], L=[100], models=tuple(models), datasets=tuple(datasets))


EXPERIMENT_BUILDERS = {
    "use_case_1": uc1,
    "use_case_2": uc2,
    "use_case_3": uc3,
    "use_case_4": uc4,
    "use_case_5": uc5,
    "use_case_6": uc6,
    "use_case_7": uc7,
    "use_case_8": uc8,
    "use_case_11": uc11,
    "use_case_12": uc12,
    "use_case_13": uc13,
    "use_case_14": uc14,
}


DEFAULT_EXPERIMENTS = tuple(EXPERIMENT_BUILDERS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

_EXPERIMENTS_DATA = None

def _load_data(*args, **kwargs):
    global _EXPERIMENTS_DATA
    if _EXPERIMENTS_DATA is None:
        _EXPERIMENTS_DATA = _load_experiments_data_module()
    return _EXPERIMENTS_DATA.load_data(*args, **kwargs)

def plot_discriminators_js_mean_error_boxplot(results_discriminators_dir=None, exclude_use_cases=('use_case_4',), filename='js_mean_error_boxplot.png'):
    """Generates and saves absolute error vs GT boxplots for each use case.

    This function is placed in `main_experiments.py` because it belongs to the
    experimentation layer (not the library itself). It saves one file per use case
    in its corresponding folder.
    """
    if results_discriminators_dir is None:
        results_discriminators_dir = EXPERIMENTS_ROOT / 'results_discriminators'
    else:
        results_discriminators_dir = Path(results_discriminators_dir)

    method_order = ['MLP', 'RF', 'XGBoost', 'LogReg', 'LogRegPol', 'TabPFN', 'Syndat', 'Synthcity']
    method_to_col = {
        'MLP': 'Discriminator MLP',
        'RF': 'Discriminator RF',
        'XGBoost': 'Discriminator XGBoost',
        'LogReg': 'Discriminator LogReg',
        'LogRegPol': 'Discriminator LogRegPol',
        'TabPFN': 'Discriminator TabPFN',
        'Syndat': 'Syndat',
        'Synthcity': 'Synthcity',
    }

    if not results_discriminators_dir.is_dir():
        print(Fore.YELLOW + f'Boxplot skipped: results dir not found: {results_discriminators_dir}' + Style.RESET_ALL)
        return None

    use_case_dirs = sorted(p for p in results_discriminators_dir.iterdir() if p.is_dir() and p.name.startswith('use_case_'))
    outputs = {}
    for use_case_path in use_case_dirs:
        use_case = use_case_path.name
        if use_case in exclude_use_cases:
            continue
        uc_dir = use_case_path
        csv_files = []
        for root, _, files in os.walk(uc_dir):
            for f in files:
                if f.startswith('comparison_js') and f.endswith('.csv'):
                    csv_files.append(Path(root) / f)
        csv_files = sorted(csv_files)
        if not csv_files:
            continue

        method_errs = {m: [] for m in method_order}
        had_any_gt = False
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'GT MC JS' not in df.columns:
                continue
            gt = pd.to_numeric(df['GT MC JS'], errors='coerce')
            if gt.notna().sum() == 0:
                continue
            had_any_gt = True
            for method, col in method_to_col.items():
                if col not in df.columns:
                    continue
                vals = pd.to_numeric(df[col], errors='coerce')
                err = (vals - gt).abs()
                method_errs[method].extend(pd.to_numeric(err, errors='coerce').dropna().tolist())

        if not had_any_gt:
            continue
        data_ = {m: v for m, v in method_errs.items() if len(v) > 0}
        if len(data_) < 2:
            continue
        methods = [m for m in method_order if m in data_]
        values = [data_[m] for m in methods]

        # Dibujar y guardar
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(values, labels=methods, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if prop_cycle:
            for i, patch in enumerate(bp.get('boxes', [])):
                color = prop_cycle[i % len(prop_cycle)]
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor(color)
        ax.set_ylabel('Absolute Error vs GT (JS)')
        ax.set_xlabel('Method')
        ax.set_title(f'Discriminators: JS Error distribution ({use_case})')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-8)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out_path = uc_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        outputs[use_case] = str(out_path)

    if not outputs:
        print(Fore.YELLOW + 'Boxplot skipped: no use cases produced a plot (missing GT or data).' + Style.RESET_ALL)
        return None
    return outputs

def build_experiment_spec(
    experiment: str,
    *,
    models: Sequence[str],
    datasets_use_case_4: Sequence[str],
    datasets_use_case_11: Sequence[str],
) -> ExperimentSpec:
    if experiment not in EXPERIMENT_BUILDERS:
        valid = ", ".join(EXPERIMENT_BUILDERS)
        raise ValueError(f"Unknown experiment {experiment!r}. Use one of: {valid}.")
    spec_builder = EXPERIMENT_BUILDERS[experiment]
    if experiment == "use_case_4":
        return spec_builder(models=models, datasets=datasets_use_case_4)
    if experiment in {"use_case_11", "use_case_12", "use_case_14"}:
        return spec_builder(models=models, datasets=datasets_use_case_11)
    return spec_builder()


def _data_path_for(experiment: str, dataset: str | None, model: str | None, results_root: Path) -> str | None:
    if dataset is None or model is None:
        return None
    if experiment not in {"use_case_4", "use_case_11", "use_case_12", "use_case_14"}:
        return None
    return str(results_root / "data" / dataset / model)


def _append_js_row(result, *, n: Any, m: int, l: int, model: str | None, dataset: str | None, gap_magnitude: float | None) -> None:
    out_path = result.results_path / "js.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "N": n,
        "M": m,
        "L": l,
        "seed": result.seed,
        "model": model,
        "dataset": dataset,
        "gap_magnitude": gap_magnitude,
        "GT MC JS": result.evaluator.mc_gt_js.item() if result.evaluator.mc_gt_js is not None else None,
        "MC JS": result.evaluator.mc_js.item() if result.evaluator.mc_js is not None else None,
        f"Discriminator {result.evaluator.discriminator_type}": result.evaluator.disc_js.item(),
        "computation_time": result.evaluator.computation_time,
        "Syndat": result.evaluator.syndat_js,
        "Synthcity": result.evaluator.synthcity_js,
    }
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiments(
    *,
    discriminator_types: Sequence[str],
    experiments: Sequence[str],
    n_iter: int | None = None,
    seeds: Sequence[int] = (0,),
    ratio_correction_mode: str = "on",
    ratio_correction_threshold: float = 0.1,
    models: Sequence[str] = ("vae", "ctgan"),
    datasets_use_case_4: Sequence[str] = ("adult", "intrusion"),
    datasets_use_case_11: Sequence[str] = ("adult", "intrusion"),
    results_root: str | Path = EXPERIMENTS_ROOT,
    epochs: int = 10_000,
    include_optional_baselines: bool = False,
    continue_on_error: bool = False,
    verbose: bool = True,
) -> list[Any]:
    results_root = Path(results_root)
    results = []
    for discriminator_type in discriminator_types:
        discriminator_type = normalize_discriminator_type(discriminator_type)
        estimator = JSDEstimator(
            discriminator_type=discriminator_type,
            results_root=results_root,
            n_iter=n_iter,
            verbose=verbose,
            ratio_correction_mode=ratio_correction_mode,
            ratio_correction_threshold=ratio_correction_threshold,
            include_optional_baselines=include_optional_baselines,
        )
        for experiment in experiments:
            spec = build_experiment_spec(
                experiment,
                models=models,
                datasets_use_case_4=datasets_use_case_4,
                datasets_use_case_11=datasets_use_case_11,
            )
            ratio_modes = spec.ratio_modes if experiment in {"use_case_6", "use_case_8", "use_case_13"} else (ratio_correction_mode,)
            gap_values = spec.gap_magnitudes if experiment in {"use_case_6", "use_case_8", "use_case_13"} else (None,)
            for model in spec.models:
                for dataset in spec.datasets:
                    data_path = _data_path_for(experiment, dataset, model, results_root)
                    for ratio_mode in ratio_modes:
                        estimator.ratio_correction_mode = ratio_mode
                        for gap_magnitude in gap_values:
                            for n in spec.N:
                                for m in spec.M:
                                    for l in spec.L:
                                        for seed in seeds:
                                            set_seed(seed)
                                            try:
                                                x_r, x_s, dist_r, dist_s = _load_data(
                                                    experiment,
                                                    n,
                                                    m,
                                                    l,
                                                    seed,
                                                    data_path,
                                                    gap_magnitude=gap_magnitude,
                                                )
                                                result = estimator.run(
                                                    experiment=experiment,
                                                    m=m,
                                                    l=l,
                                                    seed=seed,
                                                    model=model,
                                                    x_p=x_r,
                                                    x_q=x_s,
                                                    dist_p=dist_r,
                                                    dist_q=dist_s,
                                                    epochs=epochs,
                                                    include_optional_baselines=include_optional_baselines,
                                                )
                                            except Exception as exc:
                                                if not continue_on_error:
                                                    raise
                                                print(
                                                    "Skipped failed run: "
                                                    f"discriminator={discriminator_type}, experiment={experiment}, "
                                                    f"N={n}, M={m}, L={l}, seed={seed}: {exc}"
                                                )
                                                continue
                                            _append_js_row(result, n=n, m=m, l=l, model=model, dataset=dataset, gap_magnitude=gap_magnitude)
                                            results.append(result)
    return results


def run_discriminator_tests(
    *,
    discriminator_types: Sequence[str] = supported_discriminators(),
    experiments: Sequence[str] = ("use_case_1",),
    make_boxplots: bool = True,
    **kwargs,
) -> list[Any]:
    kwargs.setdefault("continue_on_error", True)
    results = run_experiments(
        discriminator_types=discriminator_types,
        experiments=experiments,
        **kwargs,
    )
    if make_boxplots:
        results_root = Path(kwargs.get('results_root', EXPERIMENTS_ROOT))
        plot_discriminators_js_mean_error_boxplot(results_root / 'results_discriminators')
    return results


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--discriminators",
        nargs="+",
        default=["MLP"],
        choices=supported_discriminators(),
        help="Discriminators: MLP RF LogReg LogRegPol XGBoost TabPFN.",
    )
    parser.add_argument("--experiments", nargs="+", default=["use_case_1"], help="Experiment use cases to run.")
    parser.add_argument("--n-iter", type=int, default=None, help="Optimization iterations for RF/XGBoost/LogReg.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds to run.")
    parser.add_argument("--epochs", type=int, default=10_000, help="Training epochs for MLP.")
    parser.add_argument("--results-root", default=str(EXPERIMENTS_ROOT), help="Root folder for experiment data/results.")
    parser.add_argument("--ratio-correction-mode", choices=["auto", "on", "off"], default="on", help="Ratio correction mode.")
    parser.add_argument("--ratio-correction-threshold", type=float, default=0.1, help="Threshold for auto ratio correction.")
    parser.add_argument("--models", nargs="+", default=["vae", "ctgan"], help="Models for real-data use cases.")
    parser.add_argument("--datasets-use-case-4", nargs="+", default=["adult", "intrusion"], help="Datasets for use_case_4.")
    parser.add_argument("--datasets-use-case-11", nargs="+", default=["adult", "intrusion"], help="Datasets for use_case_11/12/14.")
    parser.add_argument("--include-optional-baselines", action="store_true", help="Run optional syndat/synthcity baselines.")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed runs and continue with the next configuration.")
    parser.add_argument("--quiet", action="store_true", help="Reduce estimator logging.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jensen-Shannon divergence experiments.")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available experiments and discriminators.")
    list_parser.set_defaults(command="list")

    train_parser = subparsers.add_parser("train", help="Train estimators on predefined experiments.")
    _add_run_arguments(train_parser)

    test_parser = subparsers.add_parser("test-discriminators", help="Run the same experiments across discriminators.")
    _add_run_arguments(test_parser)
    test_parser.set_defaults(discriminators=list(supported_discriminators()), continue_on_error=True)
    test_parser.add_argument("--no-boxplots", action="store_true", help="Skip discriminator error boxplots.")

    return parser.parse_args()


def _run_from_args(args: argparse.Namespace) -> None:
    common = {
        "discriminator_types": args.discriminators,
        "experiments": args.experiments,
        "n_iter": args.n_iter,
        "seeds": args.seeds,
        "ratio_correction_mode": args.ratio_correction_mode,
        "ratio_correction_threshold": args.ratio_correction_threshold,
        "models": args.models,
        "datasets_use_case_4": args.datasets_use_case_4,
        "datasets_use_case_11": args.datasets_use_case_11,
        "results_root": args.results_root,
        "epochs": args.epochs,
        "include_optional_baselines": args.include_optional_baselines,
        "continue_on_error": args.continue_on_error,
        "verbose": not args.quiet,
    }
    if args.command == "test-discriminators":
        results = run_discriminator_tests(make_boxplots=not args.no_boxplots, **common)
    else:
        results = run_experiments(**common)
    print(f"Finished {len(results)} run(s).")


def main() -> None:
    args = _parse_args()
    if args.command in {None, "list"}:
        print("Experiments:", ", ".join(DEFAULT_EXPERIMENTS))
        print("Discriminators:", ", ".join(supported_discriminators()))
        return
    _run_from_args(args)


if __name__ == "__main__":
    main()
