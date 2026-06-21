"""
Stage 08 — Machine-learning classification (LOSO CV).

Per-trial band-power features → sucrose-vs-sucralose (binary) and sweetness
intensity (3-class). Models: LogReg / SVM / RandomForest / XGBoost.
"""

import _bootstrap  # noqa: F401
import os

import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, ml, viz, report


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'ml')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 08 — ML classification"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    feats = ml.build_trial_features(all_epochs, cfg, log)
    feats.to_csv(result_path(cfg, 'ml', 'trial_features.csv'), index=False)

    summary_rows, S = [], []
    S.append("# Stage 08 — Machine-Learning Classification (LOSO)\n")
    S.append("Features: per-trial relative band power (per channel) + ROI absolute/"
             "relative band power. Leave-One-Subject-Out cross-validation; class-balanced "
             "models. Chance = 1/n_classes.\n")

    for task in cfg['ml'].get('tasks', ['substance', 'intensity']):
        res = ml.run_loso(feats, task, cfg, logger=log)
        meta = res.pop('_meta')
        chance = meta['chance']
        S.append(f"## Task: {task}  (classes = {meta['class_names']}, "
                 f"n_trials = {meta['n_trials']}, chance = {chance:.2f})\n")
        labels, accs = [], []
        for name, r in res.items():
            labels.append(name); accs.append(r['accuracy'])
            summary_rows.append({'task': task, 'model': name, 'accuracy': r['accuracy'],
                                 'f1_macro': r['f1_macro'],
                                 'mean_fold_acc': r['mean_fold_accuracy'],
                                 'std_fold_acc': r['std_fold_accuracy'], 'chance': chance})
            cm_fig = viz.plot_confusion(
                r['confusion_matrix'], meta['class_names'],
                os.path.join(fdir, f'cm_{task}_{name}.png'),
                title=f'{task} — {name}',
                subtitle=f"acc={r['accuracy']:.3f}, f1={r['f1_macro']:.3f}")
        acc_fig = viz.plot_bar_simple(
            labels, accs, os.path.join(fdir, f'acc_{task}.png'),
            title=f'{task}: model accuracy (LOSO)', ylabel='Accuracy', chance=chance)
        S.append(report.img(acc_fig, rdir, f'{task} model accuracy'))
        # best model confusion
        best = max(res, key=lambda k: res[k]['accuracy'])
        S.append(report.img(os.path.join(fdir, f'cm_{task}_{best}.png'), rdir,
                            f'Best model ({best}) confusion matrix'))

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(result_path(cfg, 'ml', 'ml_summary.csv'), index=False)
    S.insert(2, "## Summary\n\n" + report.df_to_md(summ))

    report.write(report_path(cfg, '08_ml.md'), S)
    log.info(f"Stage 08 done → {report_path(cfg, '08_ml.md')}")


if __name__ == '__main__':
    main()
