# Author: UPM
# Email: alba.garrido.lopez@upm.es
# Date: 25/05/2026

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jensenshannondivergence import estimate_jensen_shannon, estimate_js, supported_discriminators


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Jensen-Shannon divergence between two tabular CSV files."
    )
    parser.add_argument("--x-p", "--reference", required=True, help="CSV file with reference/real samples.")
    parser.add_argument("--x-q", "--synthetic", required=True, help="CSV file with synthetic/generated samples.")
    parser.add_argument(
        "--discriminator",
        "--classifier",
        default="MLP",
        help="Discriminator/classifier used to estimate the density ratio. Canonical values: "
        + ", ".join(supported_discriminators()),
    )
    parser.add_argument("--m", type=int, default=None, help="Number of training samples from each distribution.")
    parser.add_argument("--l", type=int, default=None, help="Number of validation and test samples per split.")
    parser.add_argument("--n-iter", type=int, default=None, help="Optimization iterations for RF/XGBoost/LogReg.")
    parser.add_argument("--epochs", type=int, default=10_000, help="Training epochs for MLP.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--results-root", default=None, help="Directory where estimation artifacts are saved.")
    parser.add_argument(
        "--ratio-correction-mode",
        choices=["auto", "on", "off"],
        default="on",
        help="Correction mode for unequal class priors in discriminator training.",
    )
    parser.add_argument(
        "--ratio-correction-threshold",
        type=float,
        default=0.1,
        help="Threshold used when ratio correction mode is 'auto'.",
    )
    parser.add_argument(
        "--include-optional-baselines",
        action="store_true",
        help="Also run optional syndat/synthcity JSD baselines when installed.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce estimator logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x_p = pd.read_csv(args.x_p)
    x_q = pd.read_csv(args.x_q)
    result = estimate_jensen_shannon(
        x_p,
        x_q,
        discriminator_type=args.discriminator,
        m=args.m,
        l=args.l,
        n_iter=args.n_iter,
        seed=args.seed,
        epochs=args.epochs,
        results_root=args.results_root,
        ratio_correction_mode=args.ratio_correction_mode,
        ratio_correction_threshold=args.ratio_correction_threshold,
        include_optional_baselines=args.include_optional_baselines,
        verbose=not args.quiet,
        return_result=True,
    )
    print(f"Estimated JS divergence: {float(result.evaluator.disc_js.detach().cpu().item())}")
    print(f"Results saved in: {result.results_path}")


if __name__ == "__main__":
    main()
