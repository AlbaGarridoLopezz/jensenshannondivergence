# Author: UPM
# Email: alba.garrido.lopez@upm.es
# Date: 25/05/2026

import torch
import numpy as np
from torch import distributions as D
from sklearn.mixture import GaussianMixture

def load_mvn(n_dims=5, dist=1):
    cov = np.identity(n_dims).astype("float32")
    choices = [0.3, 0.5, 0.7]
    for row in range(n_dims):
        for col in range(n_dims):
            if row == col:
                cov[row, col] = 10.
            else:
                cov[row, col] = np.random.choice(choices)
    cov /= 10.
    cov = cov @ cov.T
    locs = torch.rand((n_dims,)) * dist
    mvn = torch.distributions.MultivariateNormal(loc=locs, covariance_matrix=torch.tensor(cov, dtype=torch.float32))
    return mvn


def create_independent_gm(component_probs=torch.tensor([0.7, 0.3]), loc=torch.tensor([[1., 1.], [-1., -1.]]),
                          scale=torch.tensor([.5, .5])):
    categorical = D.Categorical(probs=component_probs)
    components = D.Independent(D.Normal(loc=loc, scale=scale), 1)
    return D.MixtureSameFamily(categorical, components)


def create_corr_bimodal_gm():
    categorical = D.Categorical(probs=torch.tensor([0.7, 0.3]))
    components = D.MultivariateNormal(
        loc=torch.Tensor([[1., 1.], [-1., -1.]]),
        covariance_matrix=torch.Tensor([
            [[1., .2], [.2, 1.]],
            [[1., .2], [.2, 1.]],
        ])
    )
    return D.MixtureSameFamily(categorical, components)


def create_high_corr_bimodal_gm(rho=0.2):
    probs = torch.tensor([0.7, 0.3])
    cov = torch.Tensor([
        [[1., rho], [rho, 1.]],
        [[1., rho], [rho, 1.]],
    ])
    categorical = D.Categorical(probs=probs)
    components = D.MultivariateNormal(loc=torch.Tensor([[1., 1.], [-1., -1.]]), covariance_matrix=cov)
    return D.MixtureSameFamily(categorical, components)


def create_corr_mvn(n_dims=2, rho=0.0):
    cov = torch.full((n_dims, n_dims), rho, dtype=torch.float32)
    cov.fill_diagonal_(1.0)
    return D.MultivariateNormal(loc=torch.zeros(n_dims), covariance_matrix=cov)


class GMM:
    def __init__(self, n_components=2, covariance_type='full', random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                   random_state=self.random_state)

    def fit(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        self.gmm.fit(x)
        return self.gmm

    def sample(self, n_samples):
        return self.gmm.sample(n_samples)

    def log_prob(self, x):
        device = x.device if torch.is_tensor(x) else 'cpu'
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        return torch.tensor(self.gmm.score_samples(x_np), device=device)
