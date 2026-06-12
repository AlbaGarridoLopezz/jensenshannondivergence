from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


SUPPORTED_DISCRIMINATORS = ("MLP", "RF", "XGBoost", "LogReg", "LogRegPol", "TabPFN")


@dataclass(frozen=True)
class JSDEstimatorResult:
    evaluator: Any
    experiment: str
    seed: int
    results_path: Path


def normalize_discriminator_type(discriminator_type: str) -> str:
    """Return the canonical discriminator name used internally."""
    if discriminator_type not in SUPPORTED_DISCRIMINATORS:
        valid = ", ".join(supported_discriminators())
        raise ValueError(f"Unknown discriminator type: {discriminator_type!r}. Use one of: {valid}.")
    return discriminator_type


def _as_tensor(data: Any, *, name: str) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.detach().clone() if data.requires_grad else data
        return tensor.to(dtype=torch.float32)
    if pd is not None and isinstance(data, (pd.DataFrame, pd.Series)):
        array = data.to_numpy()
    else:
        array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like object with shape (n_samples, n_features).")
    return torch.tensor(array, dtype=torch.float32)


class JSDEstimator:
    def __init__(
        self,
        *,
        discriminator_type: str = "MLP",
        results_root: str | Path | None = None,
        n_iter: Optional[int] = None,
        verbose: bool = True,
        ratio_correction_mode: str = "on",
        ratio_correction_threshold: float = 0.1,
        include_optional_baselines: bool = False,
        save_plots: bool = False,
    ) -> None:
        self.discriminator_type = normalize_discriminator_type(discriminator_type)
        self.results_root = Path(results_root) if results_root is not None else Path(__file__).resolve().parents[2] / "experiments"
        self.n_iter = n_iter
        self.verbose = verbose
        self.ratio_correction_mode = ratio_correction_mode
        self.ratio_correction_threshold = ratio_correction_threshold
        self.include_optional_baselines = include_optional_baselines
        self.save_plots = save_plots

    def run(
        self,
        *,
        experiment: str | None = None,
        n: Any = None,
        m: int | None = None,
        l: int | None = None,
        seed: int = 0,
        data_path: str | None = None,
        model: str | None = None,
        gap_magnitude: float | None = None,
        n_iter: int | None = None,
        x_p: Any = None,
        x_q: Any = None,
        dist_p: Any = None,
        dist_q: Any = None,
        discriminator_type: str | None = None,
        epochs: int = 10_000,
        include_optional_baselines: bool | None = None,
        save_plots: bool | None = None,
    ) -> JSDEstimatorResult:
        DivergenceEvaluator = importlib.import_module(
            "jensenshannondivergence.estimator.evaluator"
        ).DivergenceEvaluator

        if x_p is None or x_q is None:
            raise ValueError(
                "run() requires passing both 'x_p' and 'x_q'. Load predefined experiment data from `experiments/` first, then pass the tensors here."
            )
        # enforce discriminator provided either in run() or as instance default
        if discriminator_type is None and (not hasattr(self, 'discriminator_type') or self.discriminator_type is None):
            raise ValueError(
                "You must provide a discriminator type. Pass 'discriminator_type' to run() or set it when creating JSDEstimator(discriminator_type=...)."
            )
        x_r, x_s = _as_tensor(x_p, name="x_p"), _as_tensor(x_q, name="x_q")
        if x_r.shape[1] != x_s.shape[1]:
            raise ValueError(
                f"x_p and x_q must have the same number of features; got {x_r.shape[1]} and {x_s.shape[1]}."
            )
        if x_r.shape[0] != x_s.shape[0]:
            raise ValueError(
                f"x_p and x_q must have the same number of samples; got {x_r.shape[0]} and {x_s.shape[0]}."
            )
        dist_r, dist_s = dist_p, dist_q
        exp_name = experiment or "custom_data"
        # If m/l not provided or do not match the data length expectations,
        # derive sensible defaults from the provided tensors so splits succeed.
        total_p = len(x_r)
        if m is None or l is None:
            l_derived = total_p // 4
            m_derived = total_p - 2 * l_derived
            m = m_derived
            l = l_derived
            if self.verbose:
                print(f"Adjusted m,l to fit custom data: m={m}, l={l}")
        else:
            if (m + 2 * l) != total_p:
                raise ValueError(
                    f"Provided m,l do not match data length (expected m+2l={total_p}); received m={m}, l={l}."
                )

        # use discriminator_type provided to run() in custom mode, otherwise fall back to instance value
        active_discriminator = normalize_discriminator_type(
            discriminator_type if discriminator_type is not None else self.discriminator_type
        )

        results_path = self.results_root / f"results_{active_discriminator}" / exp_name
        if model:
            results_path = results_path / model
        results_path.mkdir(parents=True, exist_ok=True)

        evaluator = DivergenceEvaluator(
            x_r,
            x_s,
            n,
            m,
            l,
            dist_r,
            dist_s,
            seed,
            exp_name,
            self.verbose,
            active_discriminator,
            experiment_results_path=str(results_path) + "/",
            n_iter=n_iter if n_iter is not None else self.n_iter,
            gap_magnitude=gap_magnitude,
            ratio_correction_mode=self.ratio_correction_mode,
            ratio_correction_threshold=self.ratio_correction_threshold,
        )
        # Only run Monte Carlo ground-truth estimation when analytic/sampleable
        # distributions are provided (i.e., not in custom data mode).
        if dist_r is not None and dist_s is not None:
            evaluator.monte_carlo_gt_estimation()

        # Split data for training/validation/test. Requires m and l to be set.
        train_m = m / (m + 2 * l)
        evaluator.split_estimation_data(train_m)

        # Monte Carlo estimation of JS using the distributions is only available
        # when distributions were provided. Skip in custom-data mode.
        if dist_r is not None and dist_s is not None:
            evaluator.monte_carlo_estimation()
        evaluator.probabilistic_classifier_estimation(
            str(results_path) + "/",
            epochs=epochs,
            save_plots=self.save_plots if save_plots is None else save_plots,
        )
        run_optional_baselines = self.include_optional_baselines if include_optional_baselines is None else include_optional_baselines
        if run_optional_baselines:
            try:
                evaluator.syndat_js_estimation()
            except Exception as _e:  # optional dependency may be missing
                if self.verbose:
                    print(f"Skipped syndat_js_estimation: {_e}")
                evaluator.syndat_js = None
                evaluator.syndat_time = None
            try:
                evaluator.synthcity_js_estimation()
            except Exception as _e:  # optional dependency may be missing
                if self.verbose:
                    print(f"Skipped synthcity_js_estimation: {_e}")
                evaluator.synthcity_js = None
                evaluator.synthcity_time = None
        return JSDEstimatorResult(
            evaluator=evaluator,
            experiment=exp_name,
            seed=seed,
            results_path=results_path,
        )


def supported_discriminators() -> Sequence[str]:
    return SUPPORTED_DISCRIMINATORS


def estimate_jensen_shannon(
    x_p: Any,
    x_q: Any,
    *,
    discriminator_type: str = "MLP",
    m: int | None = None,
    l: int | None = None,
    n_iter: int | None = None,
    seed: int = 0,
    epochs: int = 10_000,
    results_root: str | Path | None = None,
    ratio_correction_mode: str = "on",
    ratio_correction_threshold: float = 0.1,
    include_optional_baselines: bool = False,
    save_plots: bool = False,
    verbose: bool = True,
    return_result: bool = False,
) -> float | JSDEstimatorResult:
    """Estimate Jensen-Shannon divergence between two tabular samples."""
    estimator = JSDEstimator(
        discriminator_type=discriminator_type,
        results_root=results_root,
        n_iter=n_iter,
        verbose=verbose,
        ratio_correction_mode=ratio_correction_mode,
        ratio_correction_threshold=ratio_correction_threshold,
        include_optional_baselines=include_optional_baselines,
        save_plots=save_plots,
    )
    result = estimator.run(
        x_p=x_p,
        x_q=x_q,
        m=m,
        l=l,
        seed=seed,
        epochs=epochs,
        include_optional_baselines=include_optional_baselines,
        save_plots=save_plots,
    )
    if return_result:
        return result
    return float(result.evaluator.disc_js.detach().cpu().item())
