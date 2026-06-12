import torch
import numpy as np

from sklearn.utils import shuffle
from .discriminatorMLP import DiscriminatorMLP
from .discriminatorRF import DiscriminatorRF
from .discriminatorLogReg import DiscriminatorLogReg
from .discriminatorLogRegPol import DiscriminatorLogRegPol
from .discriminatorTabPFN import DiscriminatorTabPFN
from sklearn.metrics import accuracy_score, f1_score
from .est_utils import to_dataloader, plot_loss, tensor_to_dataloader


def detection_validation(p_train, q_train, p_eval, q_eval):
    x_train = torch.cat([p_train, q_train]).detach().cpu().numpy()
    y_train = np.array([1.0] * len(p_train) + [0.0] * len(q_train))
    x_eval = torch.cat([p_eval, q_eval]).detach().cpu().numpy()
    y_eval_np = np.array([1.0] * len(p_eval) + [0.0] * len(q_eval))
    y_eval = torch.tensor(y_eval_np, dtype=torch.float32)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    results = {}
    for model in ['RF', 'Log_Reg', 'Log_Reg_Pol', 'MLP']:
        results[model] = {}
        if model == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3, max_depth=2, criterion='entropy')
        elif model == 'Log_Reg':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0)
        elif model == 'Log_Reg_Pol':
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures
            clf = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('logreg', LogisticRegression(random_state=0, max_iter=2000, solver='saga'))])
        elif model == 'MLP':
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=0, max_iter=500, early_stopping=True)
        else:
            raise NotImplementedError(f'Model {model} not available')
        clf.fit(x_train, y_train)
        y_pred_train = clf.predict(x_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        results[model]['train'] = {'accuracy': acc_train, 'f1': f1_train}
        y_pred_eval = clf.predict(x_eval)
        acc_eval = accuracy_score(y_eval_np, y_pred_eval)
        f1_eval = f1_score(y_eval_np, y_pred_eval)
        validation_predictions = clf.predict_proba(x_eval)
        validation_predictions = torch.tensor(validation_predictions[:, 1], dtype=torch.float32)
        loss_i = torch.nn.BCELoss()
        loss_j = loss_i(validation_predictions.clone().detach().requires_grad_(True), y_eval)
        results[model]['eval'] = {'accuracy': acc_eval, 'f1': f1_eval, 'loss': loss_j.item()}
    return results


class Divergence:
    def __init__(self, p, q, x_p, x_q, div, train_m=1.0, n=None, m=None, l=None,
                 discriminator_type='MLP', experiment=None,
                 ratio_correction_mode='auto', ratio_correction_threshold=0.1):
        self.p = p
        self.q = q
        self.x_p = x_p
        self.x_q = x_q
        self.train_m = train_m
        self.n = n
        self.m = m
        self.l = l
        self.div = div
        self.discriminator_type = discriminator_type
        self.experiment = experiment
        self.ratio_correction_mode = ratio_correction_mode
        self.ratio_correction_threshold = ratio_correction_threshold
        valid_modes = {'auto', 'on', 'off'}
        if self.ratio_correction_mode not in valid_modes:
            raise ValueError(f'Invalid ratio_correction_mode={self.ratio_correction_mode}. Use one of {valid_modes}.')
        if self.ratio_correction_threshold < 0:
            raise ValueError('ratio_correction_threshold must be >= 0.')
        if train_m == 1.0:
            self.p_train, self.p_eval = self.x_p, self.x_p
            self.q_train, self.q_eval = self.x_q, self.x_q
            self.p_val = self.p_eval
            self.q_val = self.q_eval

    def _train_ratio_q_over_p(self):
        if len(self.p_train) == 0:
            return 1.0
        return len(self.q_train) / len(self.p_train)

    def _use_ratio_correction(self):
        if self.ratio_correction_mode == 'on':
            return True
        if self.ratio_correction_mode == 'off':
            return False
        if len(self.p_train) == 0 or len(self.q_train) == 0:
            return False
        ratio_q_over_p = self._train_ratio_q_over_p()
        return abs(ratio_q_over_p - 1.0) > self.ratio_correction_threshold

    def split_data(self, train_m, n, m, l):
        assert 0.0 < train_m <= 1.0, 'Fraction must be in (0, 1]'
        self.n = n
        self.m = m
        self.l = l
        split_idx_p = int(len(self.x_p) * train_m)
        self.p_train, self.p_eval = self.x_p[:split_idx_p], self.x_p[split_idx_p:]
        self.p_test, self.p_val = self.p_eval[:len(self.p_eval) // 2], self.p_eval[len(self.p_eval) // 2:]
        split_idx_q = int(len(self.x_q) * train_m)
        self.q_train, self.q_eval = self.x_q[:split_idx_q], self.x_q[split_idx_q:]
        self.q_test, self.q_val = self.q_eval[:len(self.q_eval) // 2], self.q_eval[len(self.q_eval) // 2:]
        real_m = len(self.p_train)
        real_l = len(self.p_val)
        if real_m != self.m or real_l != self.l:
            if self.experiment in ('use_case_4', 'use_case_14'):
                self.effective_m = real_m
                self.effective_l = real_l
            else:
                raise ValueError(f'Expected m={self.m} and l={self.l} but got m={real_m} and l={real_l}')

    def estimate(self, disc_model):
        raise NotImplementedError("Subclasses must implement estimate().")

    def fit(self, disc_model, epochs):
        train_dl = to_dataloader(self.p_train, self.q_train)
        val_dl = to_dataloader(self.p_val, self.q_val)
        train_loop_result = disc_model.train_loop(train_dl, val_dl, epochs)
        tr_loss = train_loop_result['tr_loss']
        eval_loss = train_loop_result['eval_loss']
        total_time = train_loop_result['total_time']
        test_dl = to_dataloader(self.p_test, self.q_test, shuffle=False)
        y_pred = disc_model.predict(test_dl, sigmoid=True).clone().detach().requires_grad_(True)
        y_pred_label = torch.where(y_pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        y_eval = test_dl.dataset.y
        acc = accuracy_score(y_eval.cpu().numpy(), y_pred_label.cpu().numpy())
        f1 = f1_score(y_eval.cpu().numpy(), y_pred_label.cpu().numpy())
        loss = torch.nn.BCELoss()(y_pred, y_eval)
        det_res = detection_validation(self.p_train, self.q_train, self.p_val, self.q_val)
        results = {'tr_loss': tr_loss, 'eval_loss': eval_loss, 'accuracy': acc, 'f1': f1, 'loss': loss.item(), 'detection_validation': det_res}
        if total_time is not None:
            results['total_time'] = total_time
        return results

    def forward(self, epochs, path, layers=(256, 64, 32), 
                n_estimators=100, max_depth=10, random_state=0, optimize_rf_hyperparams=False,
                experiment_results_path=None, n_iter=None, save_plots=False):
        if self.discriminator_type == 'MLP':
            disc_model = DiscriminatorMLP(layers)
        elif self.discriminator_type == 'RF':
            hyperparams_path = path + 'rf_best_hyperparams.json'
            disc_model = DiscriminatorRF(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
                hyperparams_path=hyperparams_path, n_iter=n_iter, experiment=self.experiment)
        elif self.discriminator_type == 'XGBoost':
            from .discriminatorXGBoost import DiscriminatorXGBoost

            hyperparams_path = path + 'xgboost_best_hyperparams.json'
            disc_model = DiscriminatorXGBoost(random_state=random_state, hyperparams_path=hyperparams_path, n_iter=n_iter, experiment=self.experiment)
        elif self.discriminator_type == 'LogReg':
            hyperparams_path = path + 'logreg_best_hyperparams.json'
            disc_model = DiscriminatorLogReg(random_state=random_state, hyperparams_path=hyperparams_path, n_iter=n_iter, experiment=self.experiment)
        elif self.discriminator_type == 'LogRegPol':
            hyperparams_path = path + 'logregpol_best_hyperparams.json'
            disc_model = DiscriminatorLogRegPol(random_state=random_state, hyperparams_path=hyperparams_path, n_iter=n_iter, experiment=self.experiment)
        elif self.discriminator_type == 'TabPFN':
            disc_model = DiscriminatorTabPFN(random_state=random_state)
        else:
            raise ValueError(f"Unknown discriminator type: {self.discriminator_type}")
        training_results = self.fit(disc_model, epochs)
        estimates = self.estimate(disc_model)
        results = {'training_results': training_results, 'estimates': estimates}
        training_results_path = path + self.div + '_' + self.discriminator_type + '_training_results.pkl'
        with open(training_results_path, 'wb') as handle:
            import pickle
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if save_plots and self.discriminator_type == 'MLP':
            plot_loss(training_results['tr_loss'], training_results['eval_loss'], estimates, path, self.div, self.n, self.m, self.l)
        return training_results, estimates


class LogRatioDivergence(Divergence):
    def mc(self, p_test=None):
        if p_test is None:
            p_test = self.p_test
        log_p = self.p.log_prob(p_test)
        log_q = self.q.log_prob(p_test)
        return (log_p - log_q).mean()

    def estimate(self, disc_model):
        test_dl = tensor_to_dataloader(self.p_test)
        log_ratio = disc_model.predict(test_dl, sigmoid=False)
        if self._use_ratio_correction() and len(self.p_train) > 0 and len(self.q_train) > 0:
            prior_correction = torch.log(torch.tensor(self._train_ratio_q_over_p(), dtype=log_ratio.dtype))
            log_ratio = log_ratio + prior_correction
        estimate = torch.mean(log_ratio)
        return estimate


class JS(LogRatioDivergence):
    def disc_model_prob_imbalance(self, prob):
        if not self._use_ratio_correction():
            return prob
        if len(self.p_train) == 0 or len(self.q_train) == 0:
            return prob

        eps = 1e-6
        prob = torch.clamp(prob, min=eps, max=1 - eps)
        prior_ratio = torch.tensor(self._train_ratio_q_over_p(), dtype=prob.dtype, device=prob.device)
        odds = prob / (1 - prob)
        corrected_odds = odds * prior_ratio
        corrected_prob = corrected_odds / (1 + corrected_odds)
        return corrected_prob

    def mc(self, p_test=None, q_test=None):
        if p_test is None or q_test is None:
            p_test = self.p_test
            q_test = self.q_test

        p_p = torch.exp(self.p.log_prob(p_test))
        q_p = torch.exp(self.q.log_prob(p_test))

        p_q = torch.exp(self.p.log_prob(q_test))
        q_q = torch.exp(self.q.log_prob(q_test))

        t1 = torch.log2(2 * p_p) - torch.log2(q_p + p_p)
        t2 = (torch.log2(2 * q_q) - torch.log2(p_q + q_q))

        return (t1.mean() + t2.mean()) / 2

    def bound_divergence(self, disc_model, p, q):
        p_dl = tensor_to_dataloader(p)
        q_dl = tensor_to_dataloader(q)
        prob_p = self.disc_model_prob_imbalance(disc_model.predict(p_dl, sigmoid=True))
        prob_q = 1 - self.disc_model_prob_imbalance(disc_model.predict(q_dl, sigmoid=True))
        prob_p = torch.clamp(prob_p, min=1e-6, max=1)
        prob_q = torch.clamp(prob_q, min=1e-6, max=1)

        log2_prob_p = torch.log2(prob_p)
        log2_prob_q = torch.log2(prob_q)

        estimate = 0.5 * (1 + log2_prob_p.mean() + 1 + log2_prob_q.mean())
        estimate = torch.clamp(estimate, min=0.0)

        estimate_ln = 0.5 * ((torch.log(prob_p)).mean()) + 0.5 * ((torch.log(prob_q)).mean()) + torch.log(
            torch.tensor(2))
        bound = (-2 * estimate_ln) + torch.log(torch.tensor(4))

        return estimate, bound

    def estimate(self, disc_model):
        tr_estimate, tr_bound = self.bound_divergence(disc_model, self.p_train, self.q_train)
        val_estimate, val_bound = self.bound_divergence(disc_model, self.p_test, self.q_test)

        return tr_estimate, tr_bound, val_estimate, val_bound
