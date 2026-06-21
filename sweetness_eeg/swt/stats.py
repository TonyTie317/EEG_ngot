"""
Statistics: two-way repeated-measures ANOVA (substance × intensity), paired
contrasts with FDR correction, and cluster-based permutation tests on PSD.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    PINGOUIN_AVAILABLE = False

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    STATSMODELS_AVAILABLE = False


def fdr(pvals, method: str = 'fdr_bh') -> np.ndarray:
    """Benjamini-Hochberg FDR correction (falls back to raw p if unavailable)."""
    p = np.asarray(pvals, float)
    if STATSMODELS_AVAILABLE and len(p):
        return multipletests(p, method=method)[1]
    return p


def two_way_rm_anova(df: pd.DataFrame, dv: str, subject: str = 'subject',
                     within=('substance', 'intensity')
                     ) -> Optional[pd.DataFrame]:
    """Two-way within-subject ANOVA over sweet conditions only (Sucrose/Sucralose).

    Keeps subjects with complete cells. Returns the pingouin ANOVA table or None.
    """
    d = df[df['substance'].isin(['Sucrose', 'Sucralose'])].copy()
    # collapse to one value per subject × cell
    d = (d.groupby([subject, within[0], within[1]], dropna=False)[dv]
         .mean().reset_index())
    # keep only complete subjects (all 6 cells present and finite)
    cells = d.dropna(subset=[dv]).groupby(subject).size()
    complete = cells[cells == len(within[0:1]) * 0 + 6].index   # 2 substances × 3 inten
    d = d[d[subject].isin(complete)]
    if d[subject].nunique() < 3 or not PINGOUIN_AVAILABLE:
        return None
    try:
        aov = pg.rm_anova(data=d, dv=dv, within=list(within), subject=subject,
                          detailed=True)
        return aov
    except Exception:                                   # noqa: BLE001
        return None


def paired_substance_contrasts(subj_cond: pd.DataFrame, dv: str
                               ) -> pd.DataFrame:
    """Sucrose vs Sucralose paired t-test at each intensity (across subjects).

    ``subj_cond`` must have one row per subject × condition with column ``dv``.
    Returns a table with t, p, p_fdr, mean_diff, n per intensity.
    """
    rows = []
    for inten in [5, 7, 12]:
        suc = subj_cond[(subj_cond['substance'] == 'Sucrose') &
                        (subj_cond['intensity'] == inten)].set_index('subject')[dv]
        scl = subj_cond[(subj_cond['substance'] == 'Sucralose') &
                        (subj_cond['intensity'] == inten)].set_index('subject')[dv]
        common = suc.index.intersection(scl.index)
        a, b = suc.loc[common].dropna(), scl.loc[common].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(common) < 3:
            rows.append({'intensity': inten, 'n': len(common), 't': np.nan,
                         'p': np.nan, 'mean_sucrose': a.mean(),
                         'mean_sucralose': b.mean(), 'mean_diff': np.nan})
            continue
        t, p = sps.ttest_rel(a, b)
        rows.append({'intensity': inten, 'n': len(common), 't': t, 'p': p,
                     'mean_sucrose': a.mean(), 'mean_sucralose': b.mean(),
                     'mean_diff': (a - b).mean()})
    out = pd.DataFrame(rows)
    out['p_fdr'] = fdr(out['p'].fillna(1.0).values)
    return out


def condition_vs_water(subj_cond: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Each sweet condition vs water, paired across subjects, FDR corrected."""
    water = subj_cond[subj_cond['ma_mau'] == 'H2O'].set_index('subject')[dv]
    rows = []
    for cond in ['S1_5', 'S1_7', 'S1_12', 'S2_5', 'S2_7', 'S2_12']:
        c = subj_cond[subj_cond['ma_mau'] == cond].set_index('subject')[dv]
        common = water.index.intersection(c.index)
        a, b = c.loc[common].dropna(), water.loc[common].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(common) < 3:
            rows.append({'condition': cond, 'n': len(common), 't': np.nan, 'p': np.nan,
                         'mean_diff': np.nan})
            continue
        t, p = sps.ttest_rel(a, b)
        rows.append({'condition': cond, 'n': len(common), 't': t, 'p': p,
                     'mean_diff': (a - b).mean()})
    out = pd.DataFrame(rows)
    out['p_fdr'] = fdr(out['p'].fillna(1.0).values)
    return out


def anova_per_band_roi(roi_bp: pd.DataFrame, value: str = 'rel_power'
                       ) -> pd.DataFrame:
    """Run the two-way rmANOVA for every band × ROI; collect main/interaction p."""
    from .constants import BAND_ORDER, ROIS
    rows = []
    for roi in ROIS:
        for band in BAND_ORDER:
            sub = roi_bp[(roi_bp['roi'] == roi) & (roi_bp['band'] == band)]
            aov = two_way_rm_anova(sub, value)
            row = {'roi': roi, 'band': band}
            if aov is not None:
                for _, a in aov.iterrows():
                    src = str(a['Source']).replace(' ', '')
                    row[f'p_{src}'] = a.get('p-unc', np.nan)
                    if 'np2' in a:
                        row[f'np2_{src}'] = a.get('np2', np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_permutation_psd(all_epochs: list, cfg: Dict[str, Any], roi: str,
                            cond_a: str = 'Sucrose', cond_b: str = 'Sucralose',
                            logger: Optional[logging.Logger] = None
                            ) -> Optional[Dict[str, Any]]:
    """Cluster-based permutation test on the ROI PSD spectrum: cond_a vs cond_b.

    Builds per-subject mean PSD (ROI-averaged) for each substance and runs an
    independent cluster test across the frequency axis. Returns dict or None.
    """
    from mne.stats import permutation_cluster_test
    from .constants import EEG_CHANNELS, ROIS, SFREQ
    from .spectral import compute_psd

    roi_ch = [EEG_CHANNELS.index(c) for c in ROIS[roi] if c in EEG_CHANNELS]
    A, B, freqs = {}, {}, None
    for ep in all_epochs:
        data = ep.get_data(copy=False)[:, roi_ch, :]
        meta = ep.metadata.reset_index(drop=True)
        f, psd = compute_psd(data, SFREQ, cfg)
        freqs = f
        psd = psd.mean(axis=1)                           # ROI-avg → (n_ep, n_freqs)
        for grp, store in ((cond_a, A), (cond_b, B)):
            idx = np.where(meta['substance'].values == grp)[0]
            if len(idx):
                store[meta['subject'].iloc[0]] = psd[idx].mean(axis=0)
    if not A or not B:
        return None
    Xa = np.log10(np.stack(list(A.values())) + 1e-30)
    Xb = np.log10(np.stack(list(B.values())) + 1e-30)
    try:
        T_obs, clusters, cluster_pv, _ = permutation_cluster_test(
            [Xa, Xb], n_permutations=cfg['stats'].get('n_permutations', 1000),
            seed=cfg['stats'].get('random_state', 42), out_type='mask', verbose=False)
    except Exception as e:                              # noqa: BLE001
        if logger:
            logger.warning(f"cluster test failed for {roi}: {e}")
        return None
    sig = [(freqs[m][0], freqs[m][-1], p)
           for m, p in zip(clusters, cluster_pv) if p < cfg['stats'].get('alpha', 0.05)]
    return {'roi': roi, 'freqs': freqs, 'T_obs': T_obs, 'clusters': clusters,
            'cluster_pv': cluster_pv, 'significant': sig, 'n_a': len(A), 'n_b': len(B)}
