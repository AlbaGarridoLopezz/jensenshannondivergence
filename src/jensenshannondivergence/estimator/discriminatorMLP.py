import copy
import torch
import time
from torch import nn, optim
from torch.nn import functional as F

import warnings

warnings.filterwarnings('ignore')

_activations = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.2), 'tanh': nn.Tanh()}


class DenseModule(nn.Module):
    def __init__(self, n_neurons: int, activation: str, *args, batch_norm: bool, dropout: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = nn.LazyLinear(out_features=n_neurons)
        if activation not in _activations.keys():
            msg = f'Expected one of {_activations.keys()}'
            raise ValueError(msg)
        self.activation = _activations[activation]
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.LazyBatchNorm1d()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class DiscriminatorMLP(nn.Module):
    def __init__(self, layers, *args, device='auto', **kwargs):
        super().__init__(*args, **kwargs)
        layers_ = []
        for elem in layers:
            layers_.append(DenseModule(elem, activation='leaky_relu', batch_norm=True, dropout=True))
        layers_ += [nn.LazyLinear(1)]
        self.l = nn.ModuleList(layers_)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, data, sigmoid=False):
        x = data
        for layer in self.l:
            x = layer(x)
        if sigmoid:
            x = F.sigmoid(x)
        return x.reshape(-1)

    @torch.no_grad()
    def predict(self, data, sigmoid=True):
        self.train(False)
        _X_y = 2
        out = []
        for batch in data:
            if len(batch) == _X_y:
                x, _ = batch
            else:
                x = batch
            x = x.to(self.device)
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y.detach().cpu())
        return torch.cat(out)

    def train_loop(self, train_dl, val_dl, epochs, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr)
        tr_loss = []
        eval_loss = []
        patience_0 = 1000
        patience = patience_0
        best_metric = float('inf')
        best_model = None
        t0 = time.time()
        for epoch in range(epochs):
            self.train(True)
            cum_loss = 0.0
            cum_eval_loss = 0.0
            for X, y in train_dl:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logit_x = self(X)
                loss = F.binary_cross_entropy_with_logits(logit_x, y.reshape(-1))
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            avg_loss = cum_loss / len(train_dl)
            tr_loss.append(avg_loss * 2)
            self.eval()
            with torch.no_grad():
                for X_eval, y_eval in val_dl:
                    X_eval = X_eval.to(self.device)
                    y_eval = y_eval.to(self.device)
                    logit_x_eval = self(X_eval)
                    loss_eval = F.binary_cross_entropy_with_logits(logit_x_eval, y_eval.reshape(-1))
                    cum_eval_loss += loss_eval.item()
                avg_loss_eval = cum_eval_loss / (len(val_dl))
                eval_loss.append(avg_loss_eval * 2)
            if avg_loss_eval < best_metric:
                best_metric = avg_loss_eval
                patience = patience_0
                best_model = copy.deepcopy(self.state_dict())
            else:
                patience -= 1
            if patience == 0:
                self.load_state_dict(best_model)
                break
        total_time = time.time() - t0
        return {'tr_loss': tr_loss, 'eval_loss': eval_loss, 'total_time': total_time}
