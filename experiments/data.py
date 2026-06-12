# Author: UPM
# Email: alba.garrido.lopez@upm.es
# Date: 25/05/2026

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder

_DATA_UTILS_PATH = Path(__file__).resolve().parents[1] / 'src' / 'jensenshannondivergence' / 'data_utils.py'
_DATA_UTILS_SPEC = importlib.util.spec_from_file_location('jensenshannondivergence_data_utils', _DATA_UTILS_PATH)
if _DATA_UTILS_SPEC is None or _DATA_UTILS_SPEC.loader is None:
    raise ImportError(f'Could not load data_utils helpers from {_DATA_UTILS_PATH}')
_DATA_UTILS_MODULE = importlib.util.module_from_spec(_DATA_UTILS_SPEC)
_DATA_UTILS_SPEC.loader.exec_module(_DATA_UTILS_MODULE)

GMM = _DATA_UTILS_MODULE.GMM
create_corr_bimodal_gm = _DATA_UTILS_MODULE.create_corr_bimodal_gm
create_corr_mvn = _DATA_UTILS_MODULE.create_corr_mvn
create_independent_gm = _DATA_UTILS_MODULE.create_independent_gm
load_mvn = _DATA_UTILS_MODULE.load_mvn

from jensenshannondivergence.utils import set_seed


def _data_file(path, filename):
    if path is None:
        return None
    return Path(path).expanduser() / filename


def _load_real_gen_csv_samples(m, l, seed, path, *, use_case_name):
    if path is None:
        raise ValueError(f'{use_case_name} requires data_path with real_data.csv/gen_data.csv')
    real_path = _data_file(path, 'real_data.csv')
    gen_path = _data_file(path, 'gen_data.csv')
    if not real_path.is_file():
        raise FileNotFoundError(f'Missing real data file: {real_path}')
    if not gen_path.is_file():
        raise FileNotFoundError(f'Missing synthetic data file: {gen_path}')

    real_df = pd.read_csv(real_path)
    gen_df = pd.read_csv(gen_path)
    n_samples_available = min(real_df.shape[0], gen_df.shape[0])
    n_samples = m + (2 * l)
    if n_samples_available < n_samples:
        print(f'Warning: only {n_samples_available} samples available.')
        n_samples = n_samples_available

    real_df = real_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    gen_df = gen_df.sample(frac=1, random_state=int(seed * 7 + 1)).reset_index(drop=True)
    real_df = real_df.iloc[:n_samples]
    gen_df = gen_df.iloc[:n_samples]

    x = pd.concat([real_df, gen_df], axis=0)
    y = np.concatenate([np.ones((real_df.shape[0],)), np.zeros((gen_df.shape[0],))])
    x = preprocess_data(x)
    real_df = x[y == 1].values
    synthetic_df = x[y == 0].values
    df_r = torch.tensor(real_df, dtype=torch.float32)
    df_s = torch.tensor(synthetic_df, dtype=torch.float32)
    return df_r, df_s, None, None

def str2num(df):
    df_copy = df.copy()
    cols = df_copy.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if not cols:
        return df_copy, cols
    label_encoder = OrdinalEncoder()
    df_copy[cols] = label_encoder.fit_transform(df_copy[cols])
    return df_copy, cols


def get_feat_distributions(df, cols=[]):
    n_feat = df.shape[1]
    feat_dist = []
    for i in range(n_feat):
        col = df.columns[i]
        values = df.iloc[:, i].unique()
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)]
        if col in cols:
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
        if col == 'native-country':
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
        elif 'soil_type' in col and all(np.sort(no_nan_values) == np.array(range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
        else:
            if no_nan_values.size <= 2 and all(np.sort(no_nan_values) == np.array(range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                feat_dist.append(('bernoulli', 1))
            elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
                if no_nan_values.dtype == 'float64':
                    no_nan_values = no_nan_values.astype(int)
                if np.unique(no_nan_values).size < 20 and np.amin(no_nan_values) == 0 and all(np.sort(no_nan_values) == np.array(range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                    feat_dist.append(('categorical', np.max(no_nan_values) + 1))
                else:
                    feat_dist.append(('gaussian', 2))
            else:
                feat_dist.append(('gaussian', 2))
    return feat_dist


def normalize_data(raw_df, feat_distributions):
    num_patient, num_feature = raw_df.shape
    norm_df = raw_df.copy()
    for i in range(num_feature):
        values = raw_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~np.isnan(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            if len(np.unique(no_nan_values)) == 1:
                continue
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1
        elif feat_distributions[i][0] == 'weibull':
            loc = -1 if 0 in no_nan_values else 0
            scale = np.amax(no_nan_values) - loc
        else:
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        if scale == 0:
            scale = 1
        norm_df.iloc[:, i] = (values - loc) / scale
    return norm_df


def impute_data(df, gen_mask=False, feat_distributions=None):
    imp = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=10, sample_posterior=False)
    arr = imp.fit_transform(df)
    imp_df = pd.DataFrame(arr, columns=df.columns, index=df.index)
    mask = np.isnan(df.values)
    return imp_df, mask, imp


def preprocess_data(data):
    data, cols = str2num(data)
    feat_distributions = get_feat_distributions(data, cols)
    norm_df = normalize_data(data, feat_distributions)
    imp_norm_df, mask, _ = impute_data(norm_df, gen_mask=False, feat_distributions=feat_distributions)
    return imp_norm_df

def load_multivariate_gaussian_distributions(m, l, seed, n_dims=10):
    set_seed(seed)
    dist_r = load_mvn(n_dims=n_dims, dist=1)
    dist_s = load_mvn(n_dims=n_dims, dist=0)
    set_seed(int(seed * 2 + 1))
    x_r = dist_r.sample(torch.Size((m + 2 * l,)))
    x_s = dist_s.sample(torch.Size((m + 2 * l,)))
    return x_r, x_s, dist_r, dist_s


def load_gaussian_mixture_distributions(m, l, seed):
    set_seed(seed)
    dist_r = create_independent_gm()
    dist_s = create_independent_gm(component_probs=torch.tensor([0.5, 0.5]), loc=torch.tensor([[0., 0.], [-1., -1.]]), scale=torch.tensor([.5, .5]))
    set_seed(int(seed * 2 + 1))
    x_r_tr = dist_r.sample(torch.Size((m,)))
    x_r_ev = dist_r.sample(torch.Size((l * 2,)))
    x_r = torch.cat((x_r_tr, x_r_ev), dim=0)
    x_s_tr = dist_s.sample(torch.Size((m,)))
    x_s_ev = dist_s.sample(torch.Size((l * 2,)))
    x_s = torch.cat((x_s_tr, x_s_ev), dim=0)
    return x_r, x_s, dist_r, dist_s


def load_gaussian_mixtures_distributions_generative_process(n, m, l, seed):
    set_seed(seed)
    dist_r = create_corr_bimodal_gm()
    x_r = dist_r.sample(torch.Size((n,)))
    dist_s = GMM(n_components=2, random_state=23)
    dist_s.fit(x_r.numpy())
    set_seed(int(seed * 2 + 1))
    x_r_tr = dist_r.sample(torch.Size((m,)))
    x_r_ev = dist_r.sample(torch.Size((l * 2,)))
    x_r = torch.cat((x_r_tr, x_r_ev), dim=0)
    x_s_tr = torch.tensor(dist_s.sample(m)[0][0:m, :], dtype=torch.float32)
    x_s_ev = torch.tensor(dist_s.sample(l * 2)[0], dtype=torch.float32)
    x_s = torch.cat((x_s_tr, x_s_ev), dim=0)
    return x_r, x_s, dist_r, dist_s


def load_real_data_distributions_generative_process(m, l, seed, path):
    return _load_real_gen_csv_samples(m, l, seed, path, use_case_name='use_case_4')


def load_reduced_real_data_distributions(m, l, seed, path):
    return _load_real_gen_csv_samples(m, l, seed, path, use_case_name='use_case_12')


def load_low_data_real_data_distributions(m, l, seed, path):
    return _load_real_gen_csv_samples(m, l, seed, path, use_case_name='use_case_14')


def load_sample_regime_distributions(train_regime, m, l, seed, path):
    if path is None:
        raise ValueError('use_case_11 requires data_path with real_data.csv/gen_data.csv')
    real_path = _data_file(path, 'real_data.csv')
    gen_path = _data_file(path, 'gen_data.csv')
    if not real_path.is_file():
        raise FileNotFoundError(f'Missing real data file: {real_path}')
    if not gen_path.is_file():
        raise FileNotFoundError(f'Missing synthetic data file: {gen_path}')
    real_df = pd.read_csv(real_path)
    gen_df = pd.read_csv(gen_path)
    real_df = real_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    gen_df = gen_df.sample(frac=1, random_state=int(seed * 7 + 1)).reset_index(drop=True)
    if len(real_df) < 10000 or len(gen_df) < 10000:
        raise ValueError(f'use_case_11 requires >=10000 rows in real/gen data. Found real={len(real_df)}, gen={len(gen_df)}')
    regime_alias = {'100': 'train_100', '1': 'train_100', '1.0': 'train_100', 'full': 'train_100', 'train_100': 'train_100', '50': 'train_50', '0.5': 'train_50', '1/2': 'train_50', 'half': 'train_50', 'train_50': 'train_50', '25': 'train_25', '0.25': 'train_25', '1/4': 'train_25', 'quarter': 'train_25', 'train_25': 'train_25', '12.5': 'train_12_5', '0.125': 'train_12_5', '1/8': 'train_12_5', 'eighth': 'train_12_5', 'train_12_5': 'train_12_5'}
    regime_norm = regime_alias.get(str(train_regime).lower(), str(train_regime).lower())
    frac_map = {'train_100': 1.0, 'train_50': 0.5, 'train_25': 0.25, 'train_12_5': 0.125}
    if regime_norm not in frac_map:
        raise ValueError(f'Unknown use_case_11 train regime: {train_regime}')
    train_pool = real_df.iloc[:5000].reset_index(drop=True)
    test_pool = real_df.iloc[5000:10000].reset_index(drop=True)
    target_size = m + (2 * l)
    if target_size <= 0:
        raise ValueError('m + 2*l must be positive in use_case_11')
    if target_size > len(test_pool):
        raise ValueError(f'use_case_11 target_size={target_size} exceeds fixed real test size={len(test_pool)}')
    frac = frac_map[regime_norm]
    n_train = int(len(train_pool) * frac)
    if n_train < target_size:
        raise ValueError(f'use_case_11 regime {regime_norm} has n_train={n_train}, but requires at least target_size={target_size} without replacement.')
    p_raw = test_pool.iloc[:target_size].reset_index(drop=True)
    syn_pool = gen_df.iloc[:n_train].reset_index(drop=True)
    q_raw = syn_pool.iloc[:target_size].reset_index(drop=True)
    x = pd.concat([p_raw, q_raw], axis=0)
    y = np.concatenate([np.ones((p_raw.shape[0],)), np.zeros((q_raw.shape[0],))])
    x = preprocess_data(x)
    p_df = x[y == 1]
    q_df = x[y == 0]
    x_p = torch.tensor(p_df.values, dtype=torch.float32)
    x_q = torch.tensor(q_df.values, dtype=torch.float32)
    return x_p, x_q, None, None


def load_corr_mvn_distributions(rho, m, l, seed, n_dims=2):
    set_seed(int(seed * 2 + 1))
    dist_r = create_corr_mvn(n_dims=n_dims, rho=0.0)
    dist_s = create_corr_mvn(n_dims=n_dims, rho=rho)
    x_r = dist_r.sample(torch.Size((m + 2 * l,)))
    x_s = dist_s.sample(torch.Size((m + 2 * l,)))
    return x_r, x_s, dist_r, dist_s


def load_imbalanced_bivariate_gaussian_distributions(ratio, m, l, seed, gap_magnitude=0.7):
    set_seed(seed)
    loc_p = torch.tensor([0.0, 0.0], dtype=torch.float32)
    cov_p = torch.tensor([[1.0, 0.25], [0.25, 1.0]], dtype=torch.float32)
    base_direction = torch.tensor([1.0, -1.0], dtype=torch.float32)
    loc_q = loc_p + gap_magnitude * base_direction
    cov_q = torch.tensor([[1.6, -0.35], [-0.35, 0.9]], dtype=torch.float32)
    dist_r = torch.distributions.MultivariateNormal(loc_p, covariance_matrix=cov_p)
    dist_s = torch.distributions.MultivariateNormal(loc_q, covariance_matrix=cov_q)
    set_seed(int(seed * 2 + 1))
    total_p = m + 2 * l
    total_q = max(int(ratio * total_p), 4)
    x_r = dist_r.sample(torch.Size((total_p,)))
    x_s = dist_s.sample(torch.Size((total_q,)))
    return x_r, x_s, dist_r, dist_s


def load_data(experiment, n, m, l, seed, data_path=None, gap_magnitude=None):
    if experiment == 'use_case_1':
        return load_multivariate_gaussian_distributions(m, l, seed)
    if experiment == 'use_case_2':
        return load_gaussian_mixture_distributions(m, l, seed)
    if experiment == 'use_case_3':
        return load_gaussian_mixtures_distributions_generative_process(n, m, l, seed)
    if experiment == 'use_case_4':
        return load_real_data_distributions_generative_process(m, l, seed, data_path)
    if experiment == 'use_case_5':
        return load_corr_mvn_distributions(n, m, l, seed)
    if experiment in ('use_case_6', 'use_case_8', 'use_case_13'):
        k = gap_magnitude if gap_magnitude is not None else 0.7
        return load_imbalanced_bivariate_gaussian_distributions(n, m, l, seed, gap_magnitude=k)
    if experiment == 'use_case_7':
        return load_multivariate_gaussian_distributions(m, l, seed, n_dims=n)
    if experiment == 'use_case_11':
        return load_sample_regime_distributions(n, m, l, seed, data_path)
    if experiment == 'use_case_12':
        return load_reduced_real_data_distributions(m, l, seed, data_path)
    if experiment == 'use_case_14':
        return load_low_data_real_data_distributions(m, l, seed, data_path)
    raise RuntimeError('Experiment not recognized')
