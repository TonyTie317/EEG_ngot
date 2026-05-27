"""
Statistical analysis — rmANOVA, pairwise tests, cluster permutation, JAR analysis.

Tests ERP component differences across concentration levels and JAR groups.
"""

import os
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .constants import CONCENTRATIONS, CONCENTRATION_LABELS, ERP_WINDOWS
from .config import ensure_dir


# ──────────────────────────────────────────────────────────────────────────────
# Repeated-measures ANOVA
# ──────────────────────────────────────────────────────────────────────────────

def repeated_measures_anova(
    measures: pd.DataFrame,
    dv: str,
    within_factor: str = 'condition',
    subject_col: str = 'subject_id',
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run 1-way repeated-measures ANOVA.

    Uses pingouin if available, otherwise falls back to scipy-based
    approximation.

    Parameters
    ----------
    measures : pd.DataFrame
        Must have columns: subject_col, within_factor, dv.
    dv : str
        Dependent variable column name.
    within_factor : str
        Within-subject factor ('condition' or 'jar_group').
    subject_col : str
        Subject ID column.
    logger : logging.Logger, optional

    Returns
    -------
    result : dict
        Keys: F, p_value, df1, df2, np2 (partial eta-squared).
    """
    try:
        import pingouin as pg
        result = pg.rm_anova(
            data=measures,
            dv=dv,
            within=within_factor,
            subject=subject_col,
            detailed=True,
        )
        row = result.iloc[0]
        # Column names vary by pingouin version: p_unc vs p-unc, DF vs ddof1
        p_val = row.get('p_unc', row.get('p-unc', np.nan))
        df1 = row.get('DF', row.get('ddof1', np.nan))
        # Error row for df2
        err_row = result.iloc[1] if len(result) > 1 else pd.Series()
        df2 = err_row.get('DF', row.get('ddof2', np.nan))
        np2 = row.get('np2', row.get('ng2', np.nan))
        return {
            'F': row['F'],
            'p_value': p_val,
            'df1': df1,
            'df2': df2,
            'np2': np2,
            'method': 'pingouin rm_anova',
        }
    except ImportError:
        if logger:
            logger.warning("pingouin not installed. Using scipy F-test approximation.")

    # Fallback: one-way F-test (between-subjects approximation)
    groups = [g[dv].dropna().values for _, g in measures.groupby(within_factor)]
    if len(groups) < 2:
        return {'F': np.nan, 'p_value': np.nan, 'df1': np.nan,
                'df2': np.nan, 'np2': np.nan, 'method': 'insufficient groups'}

    F, p = sp_stats.f_oneway(*groups)
    k = len(groups)
    N = sum(len(g) for g in groups)
    np2 = (F * (k - 1)) / (F * (k - 1) + (N - k)) if F > 0 else 0

    return {
        'F': F,
        'p_value': p,
        'df1': k - 1,
        'df2': N - k,
        'np2': np2,
        'method': 'scipy f_oneway (approximation)',
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise comparisons with FDR
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_comparisons(
    measures: pd.DataFrame,
    dv: str,
    within_factor: str = 'condition',
    subject_col: str = 'subject_id',
    alpha: float = 0.05,
    fdr_method: str = 'fdr_bh',
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Post-hoc pairwise comparisons with FDR correction.

    For concentration: all 15 pairwise (6 choose 2).
    For JAR groups: all 3 pairwise.

    Uses paired t-tests when subjects match across conditions,
    independent t-tests otherwise.
    """
    levels = sorted(measures[within_factor].unique())
    n_levels = len(levels)
    rows = []

    for i in range(n_levels):
        for j in range(i + 1, n_levels):
            lev_a, lev_b = levels[i], levels[j]
            data_a = measures[measures[within_factor] == lev_a].set_index(subject_col)[dv].dropna()
            data_b = measures[measures[within_factor] == lev_b].set_index(subject_col)[dv].dropna()

            # Find common subjects for paired test
            common = data_a.index.intersection(data_b.index)
            if len(common) >= 3:
                a_vals = data_a.loc[common].values
                b_vals = data_b.loc[common].values
                t_stat, p_val = sp_stats.ttest_rel(a_vals, b_vals)
                mean_diff = (b_vals - a_vals).mean()
                # Cohen's d for paired
                diff = b_vals - a_vals
                d = diff.mean() / diff.std() if diff.std() > 0 else 0
                test_type = 'paired t-test'
            else:
                t_stat, p_val = sp_stats.ttest_ind(data_a, data_b)
                mean_diff = data_b.mean() - data_a.mean()
                pooled_std = np.sqrt(
                    (data_a.std()**2 + data_b.std()**2) / 2
                ) if len(data_a) > 0 and len(data_b) > 0 else 1
                d = mean_diff / pooled_std if pooled_std > 0 else 0
                test_type = 'independent t-test'

            rows.append({
                'level_a': lev_a,
                'level_b': lev_b,
                'mean_diff': mean_diff,
                't_stat': t_stat,
                'p_raw': p_val,
                'cohens_d': d,
                'n_common': len(common),
                'test_type': test_type,
            })

    results = pd.DataFrame(rows)

    # FDR correction
    if len(results) > 0 and results['p_raw'].notna().any():
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_fdr, _, _ = multipletests(
                results['p_raw'].fillna(1).values,
                alpha=alpha,
                method=fdr_method,
            )
            results['p_fdr'] = p_fdr
            results['significant_fdr'] = p_fdr < alpha
        except ImportError:
            if logger:
                logger.warning("statsmodels not installed. No FDR correction applied.")
            results['p_fdr'] = results['p_raw']
            results['significant_fdr'] = results['p_raw'] < alpha

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Trend analysis (linear/quadratic)
# ──────────────────────────────────────────────────────────────────────────────

def concentration_trend_analysis(
    measures: pd.DataFrame,
    dv: str,
    subject_col: str = 'subject_id',
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Test for linear and quadratic trends across concentration levels.

    Uses orthogonal polynomial contrasts.
    """
    # Concentration ordering
    cond_order = CONCENTRATIONS
    # Map concentration to ordinal (0, 1, 2, ...)
    measures = measures.copy()
    measures['cond_ordinal'] = measures['condition'].map(
        {c: i for i, c in enumerate(cond_order)}
    )

    # Per-subject correlation of dv with ordinal concentration
    subjects = measures[subject_col].unique()
    corrs_linear = []
    for subj in subjects:
        subj_data = measures[measures[subject_col] == subj].dropna(subset=[dv])
        if len(subj_data) >= 3:
            r, p = sp_stats.pearsonr(subj_data['cond_ordinal'], subj_data[dv])
            corrs_linear.append(r)

    if not corrs_linear:
        return {'linear_trend_r': np.nan, 'linear_trend_p': np.nan}

    # One-sample t-test on correlation values
    t_lin, p_lin = sp_stats.ttest_1samp(corrs_linear, 0)

    return {
        'linear_trend_r': np.mean(corrs_linear),
        'linear_trend_p': p_lin,
        'n_subjects': len(corrs_linear),
    }


# ──────────────────────────────────────────────────────────────────────────────
# JAR group analysis (non-parametric, handles unequal groups)
# ──────────────────────────────────────────────────────────────────────────────

def jar_group_analysis(
    measures: pd.DataFrame,
    dv: str,
    subject_col: str = 'subject_id',
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Full statistical analysis of JAR group differences.

    Uses Kruskal-Wallis (non-parametric, handles unequal N per group).
    If significant, follows up with pairwise Mann-Whitney U.
    """
    groups = []
    group_names = []
    for name in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        g = measures[measures['jar_group'] == name][dv].dropna()
        if len(g) > 0:
            groups.append(g.values)
            group_names.append(name)

    if len(groups) < 2:
        return {'test': 'insufficient groups', 'p_value': np.nan}

    # Kruskal-Wallis
    H, p_kw = sp_stats.kruskal(*groups)

    result = {
        'kruskal_wallis_H': H,
        'kruskal_wallis_p': p_kw,
        'group_sizes': {n: len(g) for n, g in zip(group_names, groups)},
        'group_means': {n: g.mean() for n, g in zip(group_names, groups)},
    }

    # Pairwise Mann-Whitney U if Kruskal-Wallis significant
    if p_kw < alpha and len(groups) >= 2:
        pairwise = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                U, p = sp_stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                pairwise.append({
                    'group_a': group_names[i],
                    'group_b': group_names[j],
                    'U': U,
                    'p_raw': p,
                })

        # FDR correction
        if pairwise:
            pw_df = pd.DataFrame(pairwise)
            try:
                from statsmodels.stats.multitest import multipletests
                _, p_fdr, _, _ = multipletests(
                    pw_df['p_raw'].values, alpha=alpha, method='fdr_bh'
                )
                pw_df['p_fdr'] = p_fdr
            except ImportError:
                pw_df['p_fdr'] = pw_df['p_raw']
            result['pairwise'] = pw_df

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Temporal cluster permutation test
# ──────────────────────────────────────────────────────────────────────────────

def temporal_cluster_test(
    all_epochs_data: np.ndarray,
    all_trial_info: pd.DataFrame,
    contrast_conditions: tuple,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Run MNE spatio-temporal cluster test for condition comparison.

    Parameters
    ----------
    all_epochs_data : ndarray
        Shape (n_epochs, n_channels, n_times).
    all_trial_info : pd.DataFrame
    contrast_conditions : tuple of (int, int)
        (condition_a, condition_b), e.g. (893, 189).
    config : dict
    logger : logging.Logger

    Returns
    -------
    result : dict or None
        Cluster statistics and p-values.
    """
    try:
        from mne.stats import spatio_temporal_cluster_test
    except ImportError:
        logger.warning("MNE cluster test not available.")
        return None

    cond_a, cond_b = contrast_conditions
    mask_a = all_trial_info['condition'] == cond_a
    mask_b = all_trial_info['condition'] == cond_b

    data_a = all_epochs_data[mask_a.values]  # (n_a, n_ch, n_times)
    data_b = all_epochs_data[mask_b.values]

    if len(data_a) < 5 or len(data_b) < 5:
        logger.warning(
            f"Not enough trials for cluster test: "
            f"{cond_a}={len(data_a)}, {cond_b}={len(data_b)}"
        )
        return None

    n_perm = config.get('stats', {}).get('cluster_n_permutations', 1024)
    threshold = config.get('stats', {}).get('cluster_threshold', 0.05)

    # Compute threshold from t-distribution
    import scipy.stats as sp_t
    threshold_df = len(data_a) + len(data_b) - 2
    threshold_val = -sp_t.t.ppf(threshold / 2, threshold_df)

    logger.info(
        f"Running cluster test: {cond_a} ({len(data_a)}) vs "
        f"{cond_b} ({len(data_b)}), {n_perm} permutations..."
    )

    T_obs, clusters, p_values, H0 = spatio_temporal_cluster_test(
        [data_a, data_b],
        n_permutations=n_perm,
        threshold=threshold_val,
        tail=0,
        verbose=False,
    )

    return {
        'contrast': f'{cond_a}_vs_{cond_b}',
        'T_obs': T_obs,
        'n_clusters': len(clusters),
        'p_values': p_values,
        'significant_clusters': sum(p < 0.05 for p in p_values),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_stats(
    erp_results: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run all statistical analyses on ERP results.

    Parameters
    ----------
    erp_results : dict
        Output from erp_analysis.run_erp_analysis().
    config : dict
    logger : logging.Logger

    Returns
    -------
    stats_results : dict
    """
    logger.info("=" * 60)
    logger.info("STAGE: Statistical Analysis")
    logger.info("=" * 60)

    measures = erp_results.get('measures')
    if measures is None or measures.empty:
        logger.error("No ERP measures available.")
        return {}

    stats_cfg = config.get('stats', {})
    alpha = stats_cfg.get('alpha', 0.05)
    fdr_method = stats_cfg.get('fdr_method', 'fdr_bh')

    components = ['P1', 'N1', 'P2', 'N400']
    measure_types = ['mean_amp', 'peak_amp', 'peak_lat']
    all_results = {}

    results_dir = os.path.join(config['paths']['results_base'], 'stats')
    ensure_dir(results_dir)

    # ── 1. rmANOVA by concentration ────────────────────────────────────────
    logger.info("Running rmANOVA by concentration...")
    anova_rows = []
    for comp in components:
        for mtype in measure_types:
            col = f'{comp}_{mtype}'
            if col not in measures.columns:
                continue
            result = repeated_measures_anova(
                measures, dv=col, within_factor='condition',
                logger=logger,
            )
            result['component'] = comp
            result['measure_type'] = mtype
            anova_rows.append(result)

    anova_df = pd.DataFrame(anova_rows)
    all_results['anova_concentration'] = anova_df
    anova_df.to_csv(os.path.join(results_dir, 'anova_concentration.csv'), index=False)

    # ── 2. Pairwise comparisons by concentration ───────────────────────────
    logger.info("Running pairwise comparisons...")
    all_pairwise = []
    for comp in components:
        for mtype in ['mean_amp', 'peak_amp']:
            col = f'{comp}_{mtype}'
            if col not in measures.columns:
                continue
            pw = pairwise_comparisons(
                measures, dv=col, within_factor='condition',
                alpha=alpha, fdr_method=fdr_method, logger=logger,
            )
            pw['component'] = comp
            pw['measure_type'] = mtype
            all_pairwise.append(pw)

    if all_pairwise:
        pairwise_df = pd.concat(all_pairwise, ignore_index=True)
        all_results['pairwise_concentration'] = pairwise_df
        pairwise_df.to_csv(os.path.join(results_dir, 'pairwise_concentration.csv'), index=False)

    # ── 3. Trend analysis ─────────────────────────────────────────────────
    logger.info("Running trend analysis...")
    trend_rows = []
    for comp in components:
        for mtype in ['mean_amp', 'peak_amp']:
            col = f'{comp}_{mtype}'
            if col not in measures.columns:
                continue
            trend = concentration_trend_analysis(measures, dv=col, logger=logger)
            trend['component'] = comp
            trend['measure_type'] = mtype
            trend_rows.append(trend)

    trend_df = pd.DataFrame(trend_rows)
    all_results['trend_analysis'] = trend_df
    trend_df.to_csv(os.path.join(results_dir, 'trend_analysis.csv'), index=False)

    # ── 4. JAR group analysis ─────────────────────────────────────────────
    logger.info("Running JAR group analysis...")
    jar_measures = measures.dropna(subset=['jar_group'])
    jar_rows = []
    if len(jar_measures) > 0:
        for comp in components:
            for mtype in ['mean_amp', 'peak_amp']:
                col = f'{comp}_{mtype}'
                if col not in jar_measures.columns:
                    continue
                jar_result = jar_group_analysis(
                    jar_measures, dv=col, alpha=alpha, logger=logger,
                )
                jar_result['component'] = comp
                jar_result['measure_type'] = mtype
                jar_rows.append(jar_result)

                if 'pairwise' in jar_result:
                    jar_result['pairwise'].to_csv(
                        os.path.join(results_dir, f'jar_pairwise_{comp}_{mtype}.csv'),
                        index=False,
                    )

    jar_df = pd.DataFrame(jar_rows)
    all_results['jar_analysis'] = jar_df

    # ── 5. Cluster permutation test ────────────────────────────────────────
    logger.info("Running cluster permutation tests...")
    # Load epoch data for cluster tests
    from .epoching import load_all_epochs
    all_epochs, all_trial_info = load_all_epochs(config, logger)
    if all_epochs:
        X = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
        for contrast in [(893, 605), (762, 605), (893, 189)]:
            name = f'cluster_{contrast[0]}_vs_{contrast[1]}'
            cl_result = temporal_cluster_test(X, all_trial_info, contrast, config, logger)
            if cl_result:
                all_results[name] = cl_result

    logger.info(f"Statistical results saved to {results_dir}")
    return all_results
