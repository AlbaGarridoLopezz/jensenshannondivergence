import os
import torch
from pathlib import Path

from colorama import Fore, Style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiments_root() -> Path:
    override = os.getenv("JSD_EXPERIMENTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "experiments"


def experiments_path(*parts: str) -> Path:
    return experiments_root().joinpath(*parts)


def plot_discriminators_js_mean_error_boxplot(results_discriminators_dir=None, exclude_use_cases=('use_case_4',), filename='js_mean_error_boxplot.png'):
    if results_discriminators_dir is None:
        results_discriminators_dir = experiments_root() / 'results_discriminators'
    else:
        results_discriminators_dir = Path(results_discriminators_dir).expanduser()

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

    outputs = {}
    if not results_discriminators_dir.is_dir():
        print(Fore.YELLOW + f'Boxplot skipped: results dir not found: {results_discriminators_dir}' + Style.RESET_ALL)
        return None

    use_case_dirs = sorted(p for p in results_discriminators_dir.iterdir() if p.is_dir() and p.name.startswith('use_case_'))
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
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(values, labels=methods, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if prop_cycle:
            for i, patch in enumerate(bp.get('boxes', [])):
                color = prop_cycle[i % len(prop_cycle)]
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor(color)
        ax.set_ylabel('Absolute error vs GT (JS)')
        ax.set_xlabel('Method')
        ax.set_title(f'Discriminators: JS error distribution ({use_case})')
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def model_suffix(experiment, model):
    if experiment in ('use_case_4', 'use_case_11', 'use_case_14') and model:
        return f'_{model}'
    return ''


def build_paths(experiment, model=None, results_subdir=None):
    root = experiments_root()
    base_mlp = root / 'results_MLP' / experiment
    base_rf = root / 'results_RF' / experiment
    base_logreg = root / 'results_LogReg' / experiment
    base_logregpol = root / 'results_LogRegPol' / experiment
    output = root / 'results_discriminators' / experiment
    if results_subdir:
        base_mlp = base_mlp / results_subdir
        base_rf = base_rf / results_subdir
        base_logreg = base_logreg / results_subdir
        base_logregpol = base_logregpol / results_subdir
        output = output / results_subdir
    os.makedirs(output, exist_ok=True)
    if experiment in ('use_case_4', 'use_case_11', 'use_case_14') and model:
        return (str(base_mlp / model), str(base_rf / model),
                str(base_logreg / model), str(base_logregpol / model),
                str(output))
    return str(base_mlp), str(base_rf), str(base_logreg), str(base_logregpol), str(output)


def load_csv_results(mlp_path, rf_path, logreg_path, logregpol_path):
    mlp_path = Path(mlp_path)
    rf_path = Path(rf_path)
    logreg_path = Path(logreg_path)
    logregpol_path = Path(logregpol_path)
    js_mlp = pd.read_csv(mlp_path / 'js.csv')
    js_rf = pd.read_csv(rf_path / 'js.csv')
    js_logreg = pd.read_csv(logreg_path / 'js.csv') if (logreg_path / 'js.csv').is_file() else None
    js_logregpol = pd.read_csv(logregpol_path / 'js.csv') if (logregpol_path / 'js.csv').is_file() else None
    return js_mlp, js_rf, js_logreg, js_logregpol


def load_optional_tabpfn_results(tabpfn_path):
    js_path = Path(tabpfn_path) / 'js.csv'
    if js_path.is_file():
        return pd.read_csv(js_path)
    return None


def load_optional_discriminator_results(base_path):
    js_path = Path(base_path) / 'js.csv'
    if js_path.is_file():
        return pd.read_csv(js_path)
    return None


def _parse_off_values_from_tex(table_dir, gap_dir):
    table_dir = Path(table_dir)
    if not table_dir.is_dir():
        return {}
    import re
    gap_str = gap_dir.replace('gap_', '')
    try:
        target_gap = float(gap_str)
    except Exception:
        target_gap = None

    def _parse_value(cell):
        cell = cell.replace('\\', '').strip()
        m = re.search(r"([0-9]+\.[0-9]+|[0-9]+|--)", cell)
        if not m:
            return None
        tok = m.group(1)
        if tok == '--':
            return float('nan')
        try:
            return float(tok)
        except Exception:
            return None

    results = {}
    for f in table_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix != '.tex':
            continue
        try:
            with f.open('r', encoding='utf-8') as fh:
                for line in fh:
                    if 'OFF' not in line:
                        continue
                    parts = [p.strip() for p in line.replace('\\', '').split('&')]
                    if len(parts) < 3:
                        continue
                    n_val = None
                    gap_val = None
                    start_idx = None
                    if len(parts) >= 4 and parts[2].upper() == 'OFF':
                        try:
                            n_val = float(parts[0])
                            gap_val = float(parts[1])
                            start_idx = 3
                        except Exception:
                            n_val, gap_val, start_idx = None, None, None
                    elif parts[0].upper().startswith('CORRECTION OFF'):
                        try:
                            n_val = float(parts[1])
                            start_idx = 2
                        except Exception:
                            n_val, start_idx = None, None
                    if n_val is None or start_idx is None:
                        continue
                    if target_gap is not None and gap_val is not None and abs(gap_val - target_gap) > 1e-9:
                        continue
                    nums = []
                    for token in parts[start_idx:]:
                        val = _parse_value(token)
                        if val is not None:
                            nums.append(val)
                    if nums:
                        results[float(n_val)] = nums[:8]
        except Exception:
            continue
    return results


def build_x_labels(df):
    if 'N' in df.columns:
        labels = [f"N={n}, M={m}, L={l}" for n, m, l in zip(df['N'], df['M'], df['L'])]
    else:
        labels = [f"M={m}, L={l}" for m, l in zip(df['M'], df['L'])]
    return range(len(labels)), labels


def _df_x(df, label_to_x):
    if 'N' in df.columns:
        row_labels = [f"N={n}, M={m}, L={l}" for n, m, l in zip(df['N'], df['M'], df['L'])]
    else:
        row_labels = [f"M={m}, L={l}" for m, l in zip(df['M'], df['L'])]
    return np.array([label_to_x[lbl] for lbl in row_labels if lbl in label_to_x])
