import time
import warnings
import numpy as np
import torch
from packaging.version import Version, InvalidVersion
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

if not hasattr(torch, 'OutOfMemoryError'):
    torch.OutOfMemoryError = RuntimeError


class DiscriminatorTabPFN:
    def __init__(self, random_state=0, device='auto'):
        self.random_state = random_state
        self.device = device
        self.model = None
        self.is_fitted = False

    def _resolve_device(self):
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device

    def _check_torch_compat(self):
        try:
            torch_v = Version(torch.__version__.split('+')[0])
        except InvalidVersion:
            return
        if torch_v < Version('2.4'):
            raise RuntimeError(f"TabPFN requires torch>=2.4, but found torch=={torch.__version__}. Upgrade PyTorch or use another discriminator.")

    def _build_model(self):
        self._check_torch_compat()
        resolved_device = self._resolve_device()
        from tabpfn import TabPFNClassifier
        try:
            from tabpfn.constants import ModelVersion
            if hasattr(TabPFNClassifier, 'create_default_for_version'):
                return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=resolved_device, random_state=self.random_state, ignore_pretraining_limits=True)
        except Exception:
            pass
        try:
            return TabPFNClassifier(random_state=self.random_state, device=resolved_device, ignore_pretraining_limits=True)
        except TypeError:
            return TabPFNClassifier(device=resolved_device, ignore_pretraining_limits=True)

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

    def train_loop(self, train_dl, val_dl, epochs, lr=1e-3):
        del epochs, lr
        x_train_list = []
        y_train_list = []
        for x, y in train_dl:
            x_train_list.append(x.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())
        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list).astype(int)
        x_val_list = []
        y_val_list = []
        for x, y in val_dl:
            x_val_list.append(x.detach().cpu().numpy())
            y_val_list.append(y.detach().cpu().numpy())
        x_val = np.vstack(x_val_list)
        y_val = np.concatenate(y_val_list).astype(int)
        t0 = time.time()
        self.model = self._build_model()
        self.model.fit(x_train, y_train)
        total_time = time.time() - t0
        y_train_prob = np.clip(self.model.predict_proba(x_train)[:, 1], 1e-6, 1 - 1e-6)
        y_val_prob = np.clip(self.model.predict_proba(x_val)[:, 1], 1e-6, 1 - 1e-6)
        train_loss = log_loss(y_train, y_train_prob)
        val_loss = log_loss(y_val, y_val_prob)
        self.is_fitted = True
        return {'tr_loss': [train_loss], 'eval_loss': [val_loss], 'total_time': total_time}
