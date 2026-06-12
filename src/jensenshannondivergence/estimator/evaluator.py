import torch
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon
from .divergence import JS

try:
    from synthcity.metrics.eval_statistical import JensenShannonDistance as _SynthcityJensenShannonDistance
except Exception:  # pragma: no cover
    class _SynthcityJensenShannonDistance:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *args, **kwargs):
            raise ImportError('synthcity is required for synthcity_js_estimation')



class JensenShannonDistanceFixed(_SynthcityJensenShannonDistance):
    def _evaluate_stats(self, X_gt, X_syn) -> Tuple[Dict, Dict, Dict]:
        stats_gt = {}
        stats_syn = {}
        stats_ = {}
        for col in X_gt.columns:
            real_col = X_gt[col].dropna()
            syn_col = X_syn[col].dropna()
            combined = np.concatenate([real_col.values, syn_col.values])
            bins = np.histogram_bin_edges(combined, bins='auto')
            real_counts, _ = np.histogram(real_col.values, bins=bins)
            syn_counts, _ = np.histogram(syn_col.values, bins=bins)
            real_counts = real_counts.astype(float) + 1
            syn_counts = syn_counts.astype(float) + 1
            real_prob = real_counts / real_counts.sum()
            syn_prob = syn_counts / syn_counts.sum()
            stats_gt[col] = real_prob
            stats_syn[col] = syn_prob
            stats_[col] = float(jensenshannon(real_prob, syn_prob, base=2) ** 2)
            if np.isnan(stats_[col]):
                raise RuntimeError(f"NaN in JSD for column '{col}'")
        return stats_, stats_gt, stats_syn


class DivergenceEvaluator:
    def __init__(self, x_p, x_q, n, m, l, dist_p, dist_q, seed, experiment, verbose,
                 discriminator_type='MLP', experiment_results_path=None, n_iter=None,
                 gap_magnitude=None, ratio_correction_mode='auto', ratio_correction_threshold=0.1):
        self.x_p = x_p
        self.x_q = x_q
        self.n = n
        self.m = m
        self.l = l
        self.dist_p = dist_p
        self.dist_q = dist_q
        self.experiment = experiment
        self.seed = seed
        self.verbose = verbose
        self.discriminator_type = discriminator_type
        self.experiment_results_path = experiment_results_path
        self.n_iter = n_iter
        self.gap_magnitude = gap_magnitude
        self.ratio_correction_mode = ratio_correction_mode
        self.ratio_correction_threshold = ratio_correction_threshold

        self.js = None
        self.mc_gt_js = None
        self.mc_js = None
        self.disc_js = None
        self.syndat_js = None
        self.syndat_time = None
        self.synthcity_js = None
        self.synthcity_time = None
        self.computation_time = None

    def monte_carlo_gt_estimation(self, iterations=100, gt_samples=5000):
        self.js = JS(self.dist_p, self.dist_q, self.x_p, self.x_q, div='JS', discriminator_type=self.discriminator_type,
            experiment=self.experiment, ratio_correction_mode=self.ratio_correction_mode,
            ratio_correction_threshold=self.ratio_correction_threshold)
        if self.dist_p is None or self.dist_q is None:
            if self.verbose:
                print('MC ground truth divergence not available for this experiment')
        else:
            mc_gt_js = torch.tensor(0)
            for i in range(1, iterations + 1):
                p_test = self.dist_p.sample((gt_samples,))
                if self.experiment == 'use_case_3':
                    q_test, _ = self.dist_q.sample(gt_samples)
                    q_test = torch.tensor(q_test, dtype=torch.float32)
                else:
                    q_test = self.dist_q.sample((gt_samples,))
                js1 = self.js.mc(p_test, q_test)
                mc_gt_js = (mc_gt_js * (i - 1) + js1) / i
            self.mc_gt_js = mc_gt_js

    def split_estimation_data(self, train_m):
        if self.js is None:
            self.js = JS(self.dist_p, self.dist_q, self.x_p, self.x_q, div='JS', discriminator_type=self.discriminator_type,
                experiment=self.experiment, ratio_correction_mode=self.ratio_correction_mode,
                ratio_correction_threshold=self.ratio_correction_threshold)
        self.js.split_data(train_m=train_m, n=self.n, m=self.m, l=self.l)

    def monte_carlo_estimation(self):
        if self.dist_p is None or self.dist_q is None:
            if self.verbose:
                print('MC ground truth divergence not available for this experiment')
        else:
            self.mc_js = self.js.mc()

    def probabilistic_classifier_estimation(self, path, epochs=10000, save_plots=False):
        if self.js is None:
            self.js = JS(self.dist_p, self.dist_q, self.x_p, self.x_q, div='JS', discriminator_type=self.discriminator_type,
                experiment=self.experiment, ratio_correction_mode=self.ratio_correction_mode,
                ratio_correction_threshold=self.ratio_correction_threshold)
        js_training_results, js_estimates = self.js.forward(
            epochs,
            path,
            experiment_results_path=self.experiment_results_path,
            n_iter=self.n_iter,
            save_plots=save_plots,
        )
        self.computation_time = js_training_results['total_time']
        self.disc_js = js_estimates[2]

    def syndat_js_estimation(self):
        t0 = time.time()
        from syndat.metrics import jensen_shannon_distance
        train_m = self.m / (self.m + 2 * self.l)
        split_p = int(len(self.x_p) * train_m)
        split_q = int(len(self.x_q) * train_m)
        p_data = self.x_p[:split_p].cpu().numpy()
        q_data = self.x_q[:split_q].cpu().numpy()
        df_p = pd.DataFrame(p_data, columns=[f'f{i}' for i in range(p_data.shape[1])])
        df_q = pd.DataFrame(q_data, columns=[f'f{i}' for i in range(q_data.shape[1])])
        jsd_dict = jensen_shannon_distance(df_p, df_q)
        self.syndat_js = float(np.mean([v ** 2 for v in jsd_dict.values()]))
        self.syndat_time = time.time() - t0

    def synthcity_js_estimation(self):
        t0 = time.time()
        from synthcity.plugins.core.dataloader import GenericDataLoader
        train_m = self.m / (self.m + 2 * self.l)
        split_p = int(len(self.x_p) * train_m)
        split_q = int(len(self.x_q) * train_m)
        p_data = self.x_p[:split_p].cpu().numpy()
        q_data = self.x_q[:split_q].cpu().numpy()
        cols = [f'f{i}' for i in range(p_data.shape[1])]
        df_p = pd.DataFrame(p_data, columns=cols)
        df_q = pd.DataFrame(q_data, columns=cols)
        loader_p = GenericDataLoader(df_p)
        loader_q = GenericDataLoader(df_q)
        evaluator = JensenShannonDistanceFixed(use_cache=False)
        result = evaluator.evaluate(loader_p, loader_q)
        self.synthcity_js = float(result['marginal'])
        self.synthcity_time = time.time() - t0

    def get_info(self):
        estimates = (self.mc_gt_js.item() if self.mc_gt_js is not None else -1,
                     self.mc_js.item() if self.mc_js is not None else -1,
                     self.disc_js.item() if self.disc_js is not None else -1,
                     self.computation_time if self.computation_time is not None else -1,
                     self.syndat_js if self.syndat_js is not None else -1,
                     self.syndat_time if self.syndat_time is not None else -1,
                     self.synthcity_js if self.synthcity_js is not None else -1,
                     self.synthcity_time if self.synthcity_time is not None else -1)
        return self.n, self.m, self.l, self.seed, estimates
