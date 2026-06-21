"""
Stage 09 — Deep-learning classification (EEGNet / ShallowConvNet, LOSO).

Raw 10 s epoch tensors → sucrose-vs-sucralose and intensity. Guarded by torch
availability (skips cleanly if PyTorch is missing).
"""

import _bootstrap  # noqa: F401
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, dl, viz, report


def learning_curve_fig(res, path):
    hists = res.get('fold_histories', [])
    if not hists:
        return None
    max_ep = max(len(h['train_loss']) for h in hists)
    pad = lambda s: list(s) + [np.nan] * (max_ep - len(s))
    tr = np.array([pad(h['train_loss']) for h in hists])
    va = np.array([pad(h['val_loss']) for h in hists])
    ep = np.arange(1, max_ep + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, np.nanmean(tr, 0), label='train loss', color='steelblue')
    ax.plot(ep, np.nanmean(va, 0), label='val loss', color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f"{res['model']} / {res['task']} — learning curves")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'dl')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 09 — DL classification"); log.info("=" * 60)

    S = ["# Stage 09 — Deep-Learning Classification (LOSO)\n"]
    if not dl.TORCH_AVAILABLE:
        S.append("PyTorch is not installed — deep-learning stage skipped.\n")
        report.write(report_path(cfg, '09_dl.md'), S)
        log.warning("torch unavailable; skipped")
        return

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    S.append("Input: raw 10 s epochs (16 channels × 1000 samples), per-fold "
             "channel standardisation, early stopping on a held-out validation split. "
             "Models: EEGNet, ShallowConvNet.\n")

    summary_rows = []
    for task in cfg['ml'].get('tasks', ['substance', 'intensity']):
        for model_name in cfg['dl'].get('models', ['eegnet']):
            res = dl.run_dl_loso(all_epochs, task, model_name, cfg, log)
            if not res:
                continue
            summary_rows.append({'task': task, 'model': model_name,
                                 'accuracy': res['accuracy'], 'f1_macro': res['f1_macro'],
                                 'mean_fold_acc': res['mean_fold_accuracy'],
                                 'std_fold_acc': res['std_fold_accuracy'],
                                 'chance': res['chance']})
            cm = viz.plot_confusion(
                res['confusion_matrix'], res['class_names'],
                os.path.join(fdir, f'cm_{task}_{model_name}.png'),
                title=f'{model_name} — {task}',
                subtitle=f"acc={res['accuracy']:.3f}, f1={res['f1_macro']:.3f}")
            lc = learning_curve_fig(res, os.path.join(fdir, f'lc_{task}_{model_name}.png'))
            S.append(f"## {task} — {model_name} "
                     f"(acc={res['accuracy']:.3f}, chance={res['chance']:.2f})\n")
            S.append(report.img(cm, rdir, 'Confusion matrix'))
            if lc:
                S.append(report.img(lc, rdir, 'Learning curves'))

    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        summ.to_csv(result_path(cfg, 'dl', 'dl_summary.csv'), index=False)
        S.insert(2, "## Summary\n\n" + report.df_to_md(summ))

    report.write(report_path(cfg, '09_dl.md'), S)
    log.info(f"Stage 09 done → {report_path(cfg, '09_dl.md')}")


if __name__ == '__main__':
    main()
