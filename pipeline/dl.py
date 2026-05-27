"""
Deep Learning Classification — EEGNet, ShallowConvNet, DeepConvNet with LOSO CV.

All code guarded by torch availability. Uses raw epoch data (n_channels × n_times)
as input, not hand-crafted features.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ensure_dir

# Guard torch import
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class EEGNet(nn.Module):
        """EEGNet (Lawhern et al. 2018) adapted for 16-channel input.

        Input: (batch, n_channels, n_times)
        """

        def __init__(self, n_channels=16, n_times=171, n_classes=6,
                     F1=8, D=2, F2=16, dropout_rate=0.5, kernel_length=64):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same')
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.elu = nn.ELU()
            self.avgpool1 = nn.AvgPool2d((1, 4))
            self.dropout = nn.Dropout(dropout_rate)
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 16), padding='same'),
                nn.Conv2d(F2, F2, (1, 1)),
            )
            self.bn3 = nn.BatchNorm2d(F2)
            self.avgpool2 = nn.AvgPool2d((1, 8))
            self.fc = None  # lazy init
            self._n_classes = n_classes
            self._F2 = F2
            self._D = D

        def forward(self, x):
            # x: (batch, n_channels, n_times)
            x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)  # (batch, F1*D, 1, n_times)
            x = self.bn2(x)
            x = self.elu(x)
            x = self.avgpool1(x)
            x = self.dropout(x)
            x = self.separable(x)
            x = self.bn3(x)
            x = self.elu(x)
            x = self.avgpool2(x)
            x = self.dropout(x)
            x = x.flatten(1)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], self._n_classes).to(x.device)
            x = self.fc(x)
            return x

    class ShallowConvNet(nn.Module):
        """ShallowConvNet (Schirrmeister et al. 2017)."""

        def __init__(self, n_channels=16, n_times=171, n_classes=6,
                     dropout_rate=0.5):
            super().__init__()
            self.conv_temporal = nn.Conv2d(1, 40, (1, 25))
            self.conv_spatial = nn.Conv2d(40, 40, (n_channels, 1))
            self.bn = nn.BatchNorm2d(40)
            self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = None
            self._n_classes = n_classes

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv_temporal(x)
            x = self.conv_spatial(x)
            x = self.bn(x)
            x = x ** 2  # square
            x = self.avgpool(x)
            x = torch.log(x + 1e-7)  # log
            x = self.dropout(x)
            x = x.flatten(1)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], self._n_classes).to(x.device)
            x = self.fc(x)
            return x

    class DeepConvNet(nn.Module):
        """DeepConvNet (Schirrmeister et al. 2017)."""

        def __init__(self, n_channels=16, n_times=171, n_classes=6,
                     dropout_rate=0.5):
            super().__init__()
            kernel_size = 5 if n_times < 500 else 10

            self.block1 = nn.Sequential(
                nn.Conv2d(1, 25, (1, kernel_size)),
                nn.Conv2d(25, 25, (n_channels, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 3)),
                nn.Dropout(dropout_rate),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(25, 50, (1, kernel_size)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 3)),
                nn.Dropout(dropout_rate),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(50, 100, (1, kernel_size)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 3)),
                nn.Dropout(dropout_rate),
            )
            self.fc = None
            self._n_classes = n_classes

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = x.flatten(1)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], self._n_classes).to(x.device)
            x = self.fc(x)
            return x

    def create_dl_model(model_name: str, n_channels: int, n_times: int,
                        n_classes: int) -> nn.Module:
        """Factory function for DL models."""
        if model_name == 'eegnet':
            return EEGNet(n_channels, n_times, n_classes)
        elif model_name == 'shallowconvnet':
            return ShallowConvNet(n_channels, n_times, n_classes)
        elif model_name == 'deepconvnet':
            return DeepConvNet(n_channels, n_times, n_classes)
        else:
            raise ValueError(f"Unknown DL model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _train_one_fold(
    model: 'nn.Module',
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    device: str = 'cpu',
) -> Tuple['nn.Module', Dict[str, list]]:
    """Train model for one fold with early stopping.

    Val set được tách từ train (20%) để tránh data leakage với test fold.

    Returns
    -------
    model : trained model
    history : dict with 'train_loss', 'val_loss', 'val_acc'
    """
    from sklearn.model_selection import train_test_split

    dl_cfg = config.get('dl', {})
    batch_size = dl_cfg.get('batch_size', 16)
    lr = dl_cfg.get('learning_rate', 0.001)
    weight_decay = dl_cfg.get('weight_decay', 0.0001)
    max_epochs = dl_cfg.get('epochs', 100)
    patience = dl_cfg.get('early_stopping_patience', 15)
    random_state = dl_cfg.get('random_state', 42)
    val_ratio = dl_cfg.get('val_ratio', 0.2)

    # Tách val từ train (stratified), KHÔNG dùng test fold
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
        random_state=random_state,
    )

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr), torch.LongTensor(y_tr)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)  # len(train_ds) = len(X_tr)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * len(X_batch)
                preds = output.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_loss /= len(val_ds)
        val_acc = correct / total if total > 0 else 0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# LOSO CV
# ──────────────────────────────────────────────────────────────────────────────

def run_dl_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_classes: int,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run LOSO CV for one DL model.

    Returns
    -------
    results : dict with accuracy, f1_macro, confusion_matrix, per_fold.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Skipping DL.")
        return {}

    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    dl_cfg = config.get('dl', {})
    random_state = dl_cfg.get('random_state', 42)
    device = dl_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(random_state)

    n_channels = X.shape[1]
    n_times = X.shape[2]

    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)

    y_true_all = []
    y_pred_all = []
    fold_accs = []
    fold_histories = []   # learning curves per fold
    fold_subjects = []    # subject id per fold

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        # Per-fold normalization: fit scaler trên train, apply lên train + test
        # Reshape (n_epochs, n_ch, n_t) → (n_epochs, n_ch*n_t) để scale
        n_ch, n_t = X.shape[1], X.shape[2]
        X_tr_flat = X[train_idx].reshape(len(train_idx), -1)
        X_te_flat = X[test_idx].reshape(len(test_idx), -1)
        scaler = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
        X_tr_flat = scaler.fit_transform(X_tr_flat)
        X_te_flat = scaler.transform(X_te_flat)
        X_train_fold = X_tr_flat.reshape(len(train_idx), n_ch, n_t)
        X_test_fold  = X_te_flat.reshape(len(test_idx), n_ch, n_t)

        # Reset FC layer by creating new model each fold
        model = create_dl_model(model_name, n_channels, n_times, n_classes)

        # Val set tách từ train bên trong _train_one_fold (không dùng test fold)
        trained_model, history = _train_one_fold(
            model,
            X_train_fold, y[train_idx],
            config, device,
        )

        # Predict trên test fold đã normalize
        trained_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_fold).to(device)
            output = trained_model(X_test_t)
            preds = output.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y[test_idx], preds)
        fold_accs.append(acc)
        fold_histories.append(history)
        fold_subjects.append(unique_subjects[fold])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)

        if logger:
            logger.info(f"    Fold {fold + 1}/{len(unique_subjects)} [{unique_subjects[fold]}]: acc={acc:.3f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    results = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'f1_macro': f1_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'per_fold_accuracy': fold_accs,
        'mean_fold_accuracy': np.mean(fold_accs),
        'std_fold_accuracy': np.std(fold_accs),
        'fold_histories': fold_histories,
        'fold_subjects': fold_subjects,
    }

    if logger:
        logger.info(
            f"  {model_name}: acc={results['accuracy']:.3f}, "
            f"f1={results['f1_macro']:.3f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_dl_results(
    result: Dict[str, Any],
    model_name: str,
    class_names: List[str],
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """Vẽ và lưu toàn bộ hình DL vào results_dir:
    - Confusion matrix
    - Learning curves (train/val loss) trung bình qua các fold
    - Val accuracy learning curve
    - Per-fold accuracy bar chart (overfitting indicator)
    - Train vs Val loss gap chart (overfitting)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as sk_cm

    os.makedirs(results_dir, exist_ok=True)
    prefix = os.path.join(results_dir, model_name)

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────
    cm = sk_cm(result['y_true'], result['y_pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(
        f'{model_name} — Confusion Matrix (jar_group)\n'
        f'Pooled acc={result["accuracy"]:.3f}  '
        f'mean fold={result["mean_fold_accuracy"]:.3f}±{result["std_fold_accuracy"]:.3f}'
    )
    fig.tight_layout()
    fig.savefig(f'{prefix}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {prefix}_confusion_matrix.png")

    # ── 2. Learning Curves (averaged across folds) ───────────────────────────
    histories = result.get('fold_histories', [])
    if histories:
        max_ep = max(len(h['train_loss']) for h in histories)

        def pad(seq, length, val=np.nan):
            return list(seq) + [val] * (length - len(seq))

        train_losses = np.array([pad(h['train_loss'], max_ep) for h in histories])
        val_losses   = np.array([pad(h['val_loss'],   max_ep) for h in histories])
        val_accs     = np.array([pad(h['val_acc'],    max_ep) for h in histories])

        mean_train = np.nanmean(train_losses, axis=0)
        mean_val   = np.nanmean(val_losses,   axis=0)
        std_val    = np.nanstd(val_losses,    axis=0)
        mean_vacc  = np.nanmean(val_accs,     axis=0)
        epochs_x   = np.arange(1, max_ep + 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Loss curves
        ax = axes[0]
        ax.plot(epochs_x, mean_train, label='Train loss (mean)', color='steelblue')
        ax.plot(epochs_x, mean_val,   label='Val loss (mean)',   color='tomato')
        ax.fill_between(epochs_x,
                        mean_val - std_val, mean_val + std_val,
                        alpha=0.2, color='tomato', label='Val ±1 std')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curves — Loss\n(averaged across 28 LOSO folds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gap = val_loss - train_loss (overfitting indicator)
        gap = mean_val - mean_train
        ax2 = ax.twinx()
        ax2.plot(epochs_x, gap, color='orange', linestyle='--',
                 alpha=0.7, label='Val−Train gap')
        ax2.set_ylabel('Val−Train gap', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.axhline(0, color='orange', linestyle=':', alpha=0.4)
        ax2.legend(loc='upper right')

        # Val accuracy curve
        ax = axes[1]
        ax.plot(epochs_x, mean_vacc, color='seagreen', label='Val acc (mean)')
        ax.fill_between(epochs_x,
                        np.nanmean(val_accs, axis=0) - np.nanstd(val_accs, axis=0),
                        np.nanmean(val_accs, axis=0) + np.nanstd(val_accs, axis=0),
                        alpha=0.2, color='seagreen', label='±1 std')
        ax.axhline(1/len(class_names), color='gray', linestyle='--',
                   alpha=0.6, label=f'Chance ({1/len(class_names):.2f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Val Accuracy per Epoch\n(averaged across 28 LOSO folds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'{model_name} — Learning Curves', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{prefix}_learning_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved: {prefix}_learning_curves.png")

    # ── 3. Per-fold Accuracy Bar Chart ───────────────────────────────────────
    fold_accs    = result['per_fold_accuracy']
    fold_subjects = result.get('fold_subjects', [f'F{i+1}' for i in range(len(fold_accs))])
    chance = 1 / len(class_names)

    colors = ['tomato' if a < chance else 'steelblue' for a in fold_accs]
    fig, ax = plt.subplots(figsize=(max(10, len(fold_accs) * 0.55), 4))
    bars = ax.bar(range(len(fold_accs)), fold_accs, color=colors, edgecolor='white', width=0.7)
    ax.axhline(chance, color='gray',   linestyle='--', linewidth=1.2,
               label=f'Chance ({chance:.2f})')
    ax.axhline(result['mean_fold_accuracy'], color='orange', linestyle='-', linewidth=1.5,
               label=f'Mean ({result["mean_fold_accuracy"]:.3f}±{result["std_fold_accuracy"]:.3f})')
    ax.set_xticks(range(len(fold_accs)))
    ax.set_xticklabels(fold_subjects, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f'{model_name} — Per-Fold Accuracy (LOSO)\n'
        f'Đỏ = dưới chance, Xanh = trên chance'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Annotate early stopping epoch count
    if histories:
        for i, (bar, h) in enumerate(zip(bars, histories)):
            n_ep = len(h['train_loss'])
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(n_ep), ha='center', va='bottom', fontsize=6, color='dimgray')
    fig.tight_layout()
    fig.savefig(f'{prefix}_per_fold_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {prefix}_per_fold_accuracy.png")

    # ── 4. Overfitting heatmap: final train_loss vs val_loss per fold ─────────
    if histories:
        final_train = [h['train_loss'][-1] for h in histories]
        final_val   = [h['val_loss'][-1]   for h in histories]
        n_epochs_used = [len(h['train_loss']) for h in histories]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        x = np.arange(len(fold_subjects))
        ax.bar(x - 0.2, final_train, 0.4, label='Final train loss', color='steelblue')
        ax.bar(x + 0.2, final_val,   0.4, label='Final val loss',   color='tomato')
        ax.set_xticks(x)
        ax.set_xticklabels(fold_subjects, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Loss')
        ax.set_title('Final Train vs Val Loss per Fold\n(gap lớn = overfitting)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        ax.bar(x, n_epochs_used, color='mediumpurple', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(fold_subjects, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Epochs (early stopping)')
        ax.set_title('Số epoch chạy trước early stopping\n(ít = converge nhanh hoặc overfit sớm)')
        ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'{model_name} — Overfitting Analysis', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{prefix}_overfitting_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved: {prefix}_overfitting_analysis.png")


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation for DL (raw epoch data)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_dl_data(
    all_epochs,
    all_trial_info: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare raw epoch data for DL classification (jar_group only).

    Returns
    -------
    X : ndarray (n_epochs, n_channels, n_times)
    y : ndarray (n_epochs,)
    groups : ndarray (n_epochs,)
    class_names : list of str
    """
    from sklearn.preprocessing import LabelEncoder

    X = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    ti = all_trial_info.copy().reset_index(drop=True)

    ti = ti.dropna(subset=['jar_group'])
    # Bỏ condition 605 (100% Khong_du, không có giá trị phân loại)
    ti = ti[ti['condition'] != 605]
    X = X[ti.index.values]
    ti = ti.reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(ti['jar_group'].values)
    class_names = list(le.classes_)

    groups = ti['subject_id'].values
    return X, y, groups, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_dl_tasks(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run all DL classification tasks.

    Returns
    -------
    all_results : dict
    """
    logger.info("=" * 60)
    logger.info("STAGE: DL Classification")
    logger.info("=" * 60)

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not installed. Skipping DL stage.")
        return {}

    from .epoching import load_all_epochs
    from .erp_analysis import apply_woody_realign

    all_epochs, all_trial_info = load_all_epochs(config, logger)
    if not all_epochs:
        logger.error("No epochs loaded.")
        return {}

    # Áp dụng Woody realignment (onset mới từ realign_offsets.csv)
    all_epochs, all_trial_info = apply_woody_realign(all_epochs, all_trial_info, logger)

    dl_cfg = config.get('dl', {})
    models = dl_cfg.get('models', ['eegnet'])

    results_dir = os.path.join(config['paths']['results_base'], 'dl')
    ensure_dir(results_dir)

    logger.info("\nTask: jar_group")
    try:
        X, y, groups, class_names = prepare_dl_data(all_epochs, all_trial_info)
    except Exception as e:
        logger.error(f"  Failed to prepare data: {e}")
        return {}

    n_classes = len(class_names)
    logger.info(f"  {X.shape[0]} samples, {X.shape[1]} channels × "
                 f"{X.shape[2]} timepoints, {n_classes} classes")

    all_results = {}
    task_results = {}
    for model_name in models:
        logger.info(f"  Model: {model_name}")
        result = run_dl_loso_cv(
            X, y, groups, model_name, n_classes, config, logger
        )
        if result:
            task_results[model_name] = result
            plot_dl_results(result, model_name, class_names, results_dir, logger)

    all_results['jar_group'] = task_results

    # Save summary
    summary_rows = []
    for mn, res in task_results.items():
        summary_rows.append({
            'task': 'jar_group',
            'model': mn,
            'accuracy': res['accuracy'],
            'f1_macro': res['f1_macro'],
            'mean_fold_acc': res['mean_fold_accuracy'],
            'std_fold_acc': res['std_fold_accuracy'],
        })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(results_dir, 'jar_group_summary.csv'), index=False
        )

    logger.info(f"DL results saved to {results_dir}")
    return all_results
