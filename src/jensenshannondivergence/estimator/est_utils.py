import torch

import numpy as np

from typing import Union
from torch.utils.data import Dataset, DataLoader


class _NumpyDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = None
        if y is not None:
            self.y = y

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]

    def __len__(self):
        return len(self.X)


def tensor_to_dataloader(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray, None] = None,
                         batch_size=1000000, shuffle=False):
    ds = _NumpyDataset(x, y)
    dl = DataLoader(ds, batch_size, shuffle=shuffle)
    return dl


def to_dataloader(p, q, shuffle=True):
    x = torch.cat([p, q])
    y = torch.tensor([1.0] * len(p) + [0.0] * len(q))
    return tensor_to_dataloader(x, y, shuffle=shuffle)


def plot_loss(tr_loss, eval_loss, estimates, path, div, n, m, l):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.plot(tr_loss)
    plt.plot(eval_loss)
    if str(div).upper() != 'JS':
        plt.legend(['Training', 'Validation'], loc='center right')
    elif 'JS' in div:
        ax.axhline(estimates[1].cpu().item(), color='g', label='Training bound', linestyle='dashed')
        ax.axhline(estimates[3].cpu().item(), color='r', label='Validation bound', linestyle='dashed')
        plt.legend(['Training', 'Validation', 'Training bound', 'Validation bound'], loc='center right')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.5)
    if n is not None:
        plt.title('$D_{' + str(div) + '}$ ' + 'Discriminator Loss (M=' + str(m) + ', L=' + str(l) + ', N=' + str(n) + ')')
        fig_name = f'{div}_discriminator_loss_M={m}_L={l}_N={n}.png'
    else:
        plt.title('$D_{' + str(div) + '}$ ' + 'Discriminator Loss (M=' + str(m) + ', L=' + str(l) + ')')
        fig_name = f'{div}_discriminator_loss_M={m}_L={l}.png'

    fig.savefig(path + fig_name)
    plt.close(fig)
