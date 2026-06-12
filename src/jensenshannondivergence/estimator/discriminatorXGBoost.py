import time
import os
import json
import warnings

import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings('ignore')


class DiscriminatorXGBoost:
    def __init__(self, random_state=0, hyperparams_path=None, n_iter=None, experiment=None):
        self.random_state = random_state
        self.hyperparams_path = hyperparams_path
        self.n_iter = n_iter
        self.experiment = experiment
        self.best_params = None
        self.model = None
        self.is_fitted = False

    def forward(self, data, sigmoid=False):
        x = data.detach().cpu().numpy()
        probs = self.model.predict_proba(x)[:, 1]
        if sigmoid:
            return torch.tensor(probs, dtype=torch.float32)
        eps = 1e-6
        probs = np.clip(probs, eps, 1 - eps)
        log_ratio = np.log(probs / (1 - probs))
        return torch.tensor(log_ratio, dtype=torch.float32)

    def predict(self, data, sigmoid=True):
        out = []
        for batch in data:
            if len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y)
        return torch.cat(out)

    def _base_estimator(self, **params):
        base = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': self.random_state, 'n_jobs': 1, 'verbosity': 0, 'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 1e-6, 'reg_lambda': 1.0, 'scale_pos_weight': 1.0}
        base.update(params)
        return XGBClassifier(**base)

    def optimize_and_save_hyperparams(self, x_train, y_train, x_val, y_val, n_iter=50):
        start_time = time.time()
        param_space = {'n_estimators': Integer(100, 600), 'max_depth': Integer(3, 10), 'learning_rate': Real(1e-3, 3e-1, prior='log-uniform'), 'subsample': Real(0.5, 1.0), 'colsample_bytree': Real(0.5, 1.0), 'min_child_weight': Integer(1, 20), 'reg_alpha': Real(1e-8, 10.0, prior='log-uniform'), 'reg_lambda': Real(1e-8, 10.0, prior='log-uniform')}
        if self.experiment != 'use_case_13':
            param_space['scale_pos_weight'] = Real(0.5, 2.5)
        xgb = self._base_estimator()
        print(f"BayesSearchCV (XGBoost): testing {n_iter} parameter combinations (Bayesian optimization)...")
        bayes_search = BayesSearchCV(xgb, param_space, n_iter=n_iter, cv=5, scoring='neg_log_loss', n_jobs=1, verbose=0, random_state=self.random_state, return_train_score=True)
        bayes_search.fit(x_train, y_train)
        self.best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in bayes_search.best_params_.items()}
        if self.experiment == 'use_case_13':
            self.best_params['scale_pos_weight'] = 1.0
        if self.hyperparams_path:
            os.makedirs(os.path.dirname(self.hyperparams_path), exist_ok=True)
            with open(self.hyperparams_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
        elapsed_time = time.time() - start_time
        return self.best_params, elapsed_time

    def train_loop(self, train_dl, val_dl, epochs, lr=1e-3):
        del epochs, lr
        x_train_list = []
        y_train_list = []
        for x, y in train_dl:
            x_train_list.append(x.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())
        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list)
        self.N_p = np.sum(y_train == 1)
        self.N_q = np.sum(y_train == 0)
        x_val_list = []
        y_val_list = []
        for x, y in val_dl:
            x_val_list.append(x.detach().cpu().numpy())
            y_val_list.append(y.detach().cpu().numpy())
        x_val = np.vstack(x_val_list)
        y_val = np.concatenate(y_val_list)
        n_iter_val = self.n_iter if self.n_iter is not None else 50
        self.best_params, optimization_time = self.optimize_and_save_hyperparams(x_train, y_train, x_val, y_val, n_iter=n_iter_val)
        xgb_params = {}
        if self.best_params:
            xgb_params.update(self.best_params)
        if self.experiment == 'use_case_13':
            xgb_params['scale_pos_weight'] = 1.0
        base_xgb = self._base_estimator(**xgb_params)
        base_xgb.fit(x_train, y_train)
        method = 'isotonic' if len(y_val) > 1000 else 'sigmoid'
        self.model = CalibratedClassifierCV(base_xgb, method=method, cv='prefit')
        self.model.fit(x_val, y_val)
        self.is_fitted = True
        y_train_pred = self.model.predict_proba(x_train)
        train_loss = log_loss(y_train, y_train_pred)
        y_val_pred = self.model.predict_proba(x_val)
        val_loss = log_loss(y_val, y_val_pred)
        return {'tr_loss': [train_loss], 'eval_loss': [val_loss], 'total_time': optimization_time}
