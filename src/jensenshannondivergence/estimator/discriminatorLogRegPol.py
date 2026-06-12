import time
import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from skopt.callbacks import DeadlineStopper

import warnings
warnings.filterwarnings('ignore')


class DiscriminatorLogRegPol:
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
        if sigmoid:
            probs = self.model.predict_proba(x)[:, 1]
            return torch.tensor(probs, dtype=torch.float32)
        else:
            probs = self.model.predict_proba(x)[:, 1]
            eps = 1e-6
            probs = np.clip(probs, eps, 1 - eps)
            log_ratio = np.log(probs / (1 - probs))
            return torch.tensor(log_ratio, dtype=torch.float32)

    def predict(self, data, sigmoid=True):
        _X_y = 2
        out = []
        for batch in data:
            if len(batch) == _X_y:
                x, _ = batch
            else:
                x = batch
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y)
        return torch.cat(out)

    def optimize_and_save_hyperparams(self, X_train, y_train, X_val, y_val, n_iter=10):
        start_time = time.time()
        max_poly_features = int(os.getenv('LOGREGPOL_MAX_POLY_FEATURES', '5000'))
        max_degree = int(os.getenv('LOGREGPOL_MAX_DEGREE', '3'))
        time_budget_sec = int(os.getenv('LOGREGPOL_SEARCH_BUDGET_SEC', '0'))
        candidate_degrees = [d for d in range(1, max_degree + 1)]
        allowed_degrees = []
        for degree in candidate_degrees:
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                transformed_dim = poly.fit_transform(X_train[:1]).shape[1]
            except Exception:
                continue
            if transformed_dim <= max_poly_features:
                allowed_degrees.append(degree)
        if not allowed_degrees:
            allowed_degrees = [1]
        skipped_degrees = [d for d in candidate_degrees if d not in allowed_degrees]
        if skipped_degrees:
            print('LogRegPol: skipping polynomial degrees ' f'{skipped_degrees} due to transformed_dim > {max_poly_features} ' '(set LOGREGPOL_MAX_POLY_FEATURES/LOGREGPOL_MAX_DEGREE to override).')
        if self.experiment == 'use_case_13':
            class_weight_options = [None]
        else:
            class_weight_options = [None, 'balanced']
        param_space = {'poly__degree': Categorical(allowed_degrees), 'poly__interaction_only': Categorical([False, True]), 'logreg__C': Real(1e-4, 1e2, prior='log-uniform'), 'logreg__penalty': Categorical(['l1', 'l2']), 'logreg__class_weight': Categorical(class_weight_options)}
        pipe = Pipeline([('poly', PolynomialFeatures(include_bias=False)), ('logreg', LogisticRegression(random_state=self.random_state, max_iter=2000, solver='saga'))])
        print(f"BayesSearchCV (LogRegPol): testing {n_iter} parameter combinations (Bayesian optimization)...")
        bayes_search = BayesSearchCV(pipe, param_space, n_iter=n_iter, cv=5, scoring='neg_log_loss', n_jobs=1, verbose=0, random_state=self.random_state, return_train_score=True)
        callbacks = [DeadlineStopper(total_time=time_budget_sec)] if time_budget_sec > 0 else None
        bayes_search.fit(X_train, y_train, callback=callbacks)
        self.best_params = {k: (int(v) if hasattr(v, 'item') else v) for k, v in bayes_search.best_params_.items()}
        if self.hyperparams_path:
            os.makedirs(os.path.dirname(self.hyperparams_path), exist_ok=True)
            with open(self.hyperparams_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
        elapsed_time = time.time() - start_time
        return self.best_params, elapsed_time

    def train_loop(self, train_dl, val_dl, epochs, lr=1e-3):
        X_train_list, y_train_list = [], []
        for X, y in train_dl:
            X_train_list.append(X.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        self.N_p = np.sum(y_train == 1)
        self.N_q = np.sum(y_train == 0)
        X_val_list, y_val_list = [], []
        for X, y in val_dl:
            X_val_list.append(X.detach().cpu().numpy())
            y_val_list.append(y.detach().cpu().numpy())
        X_val = np.vstack(X_val_list)
        y_val = np.concatenate(y_val_list)
        n_iter_val = self.n_iter if self.n_iter is not None else 10
        self.best_params, optimization_time = self.optimize_and_save_hyperparams(X_train, y_train, X_val, y_val, n_iter=n_iter_val)
        bp = self.best_params or {}
        class_weight_value = bp.get('logreg__class_weight', None)
        base_pipeline = Pipeline([('poly', PolynomialFeatures(degree=int(bp.get('poly__degree', 1)), interaction_only=bool(bp.get('poly__interaction_only', False)), include_bias=False)), ('logreg', LogisticRegression(C=float(bp.get('logreg__C', 1.0)), penalty=bp.get('logreg__penalty', 'l2'), class_weight=class_weight_value, random_state=self.random_state, max_iter=2000, solver='saga'))])
        base_pipeline.fit(X_train, y_train)
        method = "isotonic" if len(y_val) > 1000 else "sigmoid"
        self.model = CalibratedClassifierCV(base_pipeline, method=method, cv="prefit")
        self.model.fit(X_val, y_val)
        self.is_fitted = True
        y_train_pred = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_pred)
        y_val_pred = self.model.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_pred)
        return {'tr_loss': [train_loss], 'eval_loss': [val_loss], 'total_time': optimization_time}
