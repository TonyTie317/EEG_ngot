"""
Deep-learning classification (EEGNet, ShallowConvNet) with LOSO CV.

Operates on raw 10 s epoch tensors (n_channels × n_times). Guarded by torch
availability. Tasks mirror :mod:`swt.ml` (substance, intensity). Adapted from
``pipeline/dl.py`` but kept self-contained.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    pass


if TORCH_AVAILABLE:

    class EEGNet(nn.Module):
        """EEGNet (Lawhern 2018) with lazy final FC for arbitrary n_times."""

        def __init__(self, n_channels=16, n_classes=2, F1=8, D=2, F2=16,
                     dropout=0.5, kernel_length=64):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.sep = nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False),
                nn.Conv2d(F2, F2, (1, 1), bias=False))
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop = nn.Dropout(dropout)
            self.elu = nn.ELU()
            self.fc, self._nc = None, n_classes

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.bn1(self.conv1(x))
            x = self.drop(self.pool1(self.elu(self.bn2(self.depthwise(x)))))
            x = self.drop(self.pool2(self.elu(self.bn3(self.sep(x)))))
            x = x.flatten(1)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], self._nc).to(x.device)
            return self.fc(x)

    class ShallowConvNet(nn.Module):
        """ShallowConvNet (Schirrmeister 2017)."""

        def __init__(self, n_channels=16, n_classes=2, dropout=0.5):
            super().__init__()
            self.tconv = nn.Conv2d(1, 40, (1, 25))
            self.sconv = nn.Conv2d(40, 40, (n_channels, 1))
            self.bn = nn.BatchNorm2d(40)
            self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
            self.drop = nn.Dropout(dropout)
            self.fc, self._nc = None, n_classes

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.bn(self.sconv(self.tconv(x)))
            x = torch.log(torch.clamp(self.pool(x ** 2), min=1e-7))
            x = self.drop(x).flatten(1)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], self._nc).to(x.device)
            return self.fc(x)

    def _make_model(name: str, n_channels: int, n_classes: int):
        if name == 'eegnet':
            return EEGNet(n_channels, n_classes)
        if name == 'shallowconvnet':
            return ShallowConvNet(n_channels, n_classes)
        raise ValueError(name)


def prepare_dl_data(all_epochs: list, task: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stack raw epochs into (X[n,ch,t], y, groups, class_names) for a task."""
    from sklearn.preprocessing import LabelEncoder
    X = np.concatenate([ep.get_data(copy=False) for ep in all_epochs], axis=0)
    meta = pd.concat([ep.metadata for ep in all_epochs], ignore_index=True)
    if task in ('substance', 'intensity'):
        keep = meta['substance'].isin(['Sucrose', 'Sucralose']).values
        X, meta = X[keep], meta[keep].reset_index(drop=True)
        ycol = 'substance' if task == 'substance' else 'intensity'
    else:
        raise ValueError(task)
    le = LabelEncoder()
    y = le.fit_transform(meta[ycol].astype(str).values)
    return X, y, meta['subject'].values, list(le.classes_)


def _train_fold(model, Xtr, ytr, cfg, device='cpu'):
    from sklearn.model_selection import train_test_split
    d = cfg['dl']
    Xt, Xv, yt, yv = train_test_split(
        Xtr, ytr, test_size=d.get('val_ratio', 0.2),
        stratify=ytr if len(np.unique(ytr)) > 1 else None,
        random_state=d.get('random_state', 42))
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt)),
                    batch_size=d.get('batch_size', 16), shuffle=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv)),
                    batch_size=d.get('batch_size', 16))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=d.get('learning_rate', 1e-3),
                           weight_decay=d.get('weight_decay', 1e-4))
    crit = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best, best_state, noimp = float('inf'), None, 0
    for _ in range(d.get('epochs', 80)):
        model.train(); tloss = 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step(); tloss += loss.item() * len(xb)
        model.eval(); vloss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb); vloss += crit(out, yb).item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item(); total += len(yb)
        hist['train_loss'].append(tloss / len(Xt))
        hist['val_loss'].append(vloss / max(len(Xv), 1))
        hist['val_acc'].append(correct / max(total, 1))
        if hist['val_loss'][-1] < best:
            best = hist['val_loss'][-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
            if noimp >= d.get('early_stopping_patience', 15):
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


def run_dl_loso(all_epochs: list, task: str, model_name: str, cfg: Dict[str, Any],
                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """LOSO CV for one DL model & task. Returns a results dict (or {} if no torch)."""
    if not TORCH_AVAILABLE:
        if logger:
            logger.warning("PyTorch unavailable — skipping DL")
        return {}
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    X, y, groups, class_names = prepare_dl_data(all_epochs, task)
    n_ch = X.shape[1]
    torch.manual_seed(cfg['dl'].get('random_state', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logo = LeaveOneGroupOut()
    y_true, y_pred, fold_acc, fold_sids, hists = [], [], [], [], []
    for tr, te in logo.split(X, y, groups):
        mu = X[tr].mean((0, 2), keepdims=True)
        sd = X[tr].std((0, 2), keepdims=True) + 1e-7
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        model = _make_model(model_name, n_ch, len(class_names))
        model, hist = _train_fold(model, Xtr, y[tr], cfg, device)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(Xte).to(device)).argmax(1).cpu().numpy()
        y_true.extend(y[te]); y_pred.extend(p)
        fold_acc.append(accuracy_score(y[te], p)); fold_sids.append(groups[te][0])
        hists.append(hist)
        if logger:
            logger.info(f"  [{task}/{model_name}] fold {groups[te][0]}: "
                        f"acc={fold_acc[-1]:.3f}")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        'task': task, 'model': model_name, 'class_names': class_names,
        'chance': 1.0 / len(class_names),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'per_fold_accuracy': fold_acc, 'fold_subjects': fold_sids,
        'mean_fold_accuracy': float(np.mean(fold_acc)),
        'std_fold_accuracy': float(np.std(fold_acc)), 'fold_histories': hists,
    }
