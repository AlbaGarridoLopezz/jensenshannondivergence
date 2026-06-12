import numpy as np
import pandas as pd
import sys
from pathlib import Path
import importlib.util
import pytest

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import jensenshannondivergence
from jensenshannondivergence import estimate_jensen_shannon


def test_predefined_experiments_module_is_available_in_repo():
    predefined_path = project_root / 'experiments' / 'data.py'
    spec = importlib.util.spec_from_file_location('experiments_data_test', predefined_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_data = module.load_data
    assert callable(load_data)

def test_import_package():
    assert hasattr(jensenshannondivergence, 'estimate_jensen_shannon')


def test_version_is_defined():
    assert isinstance(jensenshannondivergence.__version__, str)
    assert jensenshannondivergence.__version__

def test_small_estimate_runs():
    x_p = np.random.normal(size=(20, 3))
    x_q = np.random.normal(loc=0.1, size=(20, 3))
    js = estimate_jensen_shannon(x_p, x_q, discriminator_type='MLP', epochs=1, n_iter=None, seed=0, verbose=False)
    # ensure it can be converted to a float and is finite
    val = float(js)
    assert val >= 0.0


def test_return_result_exposes_paths_and_evaluator():
    x_p = np.random.normal(size=(20, 3))
    x_q = np.random.normal(loc=0.1, size=(20, 3))
    result = estimate_jensen_shannon(x_p, x_q, discriminator_type='MLP', epochs=1, n_iter=None, seed=0, verbose=False, return_result=True)
    assert hasattr(result, 'evaluator')
    assert hasattr(result, 'results_path')
    assert result.results_path.exists()
    assert hasattr(result.evaluator, 'disc_js')


def test_str2num_accepts_numeric_only_dataframe():
    df = pd.DataFrame({'a': [1, 2], 'b': [3.5, 4.5]})
    module_path = project_root / 'experiments' / 'data.py'
    spec = importlib.util.spec_from_file_location('experiments_data_for_str2num', module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out, cols = module.str2num(df)
    assert cols == []
    pd.testing.assert_frame_equal(out, df)


def test_custom_mode_rejects_mismatched_sample_counts():
    x_p = np.random.normal(size=(20, 3))
    x_q = np.random.normal(size=(18, 3))
    with pytest.raises(ValueError, match='same number of samples'):
        estimate_jensen_shannon(x_p, x_q, discriminator_type='MLP', epochs=1, verbose=False)