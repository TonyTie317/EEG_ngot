#!/usr/bin/env python3
"""
So sánh ANOVA: unfiltered vs filtered (weak/strict) cho P2 mean_amp.

Usage:
    .venv/bin/python run_anova_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from pipeline.config import load_config, setup_logging
from pipeline.constants import CONCENTRATIONS, CONCENTRATION_LABELS
from pipeline.stats import repeated_measures_anova

UNFILTERED = 'output/results/erp/component_measures.csv'
FILTERED_W = 'output/results/erp_filtered/component_measures_filtered.csv'
FILTERED_S = 'output/results/erp_filtered/component_measures_filtered_strict.csv'


def print_means(df, label):
    """Print P2 mean_amp per condition."""
    print(f'\n  {label}:')
    for c in CONCENTRATIONS:
        cdf = df[df['condition'] == c]['P2_mean_amp']
        if len(cdf):
            print(f'    {CONCENTRATION_LABELS[c]:15s}: {cdf.mean()*1e6:+.2f} ± {cdf.std()*1e6:.2f}µV '
                  f'(n={len(cdf)})')


def anova_balanced(measures, dv, label):
    """Run rmANOVA chỉ trên subjects có đủ 6 conditions."""
    data = measures.dropna(subset=[dv]).copy()
    subj_conds = data.groupby('subject_id')['condition'].nunique()
    complete = subj_conds[subj_conds == 6].index
    balanced = data[data['subject_id'].isin(complete)]

    print(f'  {label}: {len(complete)}/{len(data["subject_id"].unique())} subjects complete, '
          f'{len(balanced)} data points')

    if len(complete) < 5:
        print(f'  {label}: Không đủ subjects cho rmANOVA')
        return None

    result = repeated_measures_anova(
        measures=balanced, dv=dv, within_factor='condition',
        subject_col='subject_id'
    )
    if result:
        p = result['p_value']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f'  {label}: F({result["df1"]:.0f},{result["df2"]:.0f}) = {result["F"]:.3f}, '
              f'p = {p:.4f} {stars}, η²p = {result["np2"]:.3f}')

    return result


def mixedlm_anova(measures, dv, label):
    """Mixed Linear Model (xử lý missing data tốt hơn rmANOVA)."""
    data = measures.dropna(subset=[dv]).copy()
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        formula = f'{dv} ~ C(condition)'
        md = smf.mixedlm(formula, data, groups=data['subject_id'])
        mdf = md.fit(reml=True)
        # Joint test via Likelihood Ratio Test
        md_null = smf.mixedlm(f'{dv} ~ 1', data, groups=data['subject_id'])
        mdf_null = md_null.fit(reml=True)
        lr_stat = 2 * abs(mdf.llf - mdf_null.llf)
        df = len(CONCENTRATIONS) - 1
        from scipy.stats import chi2
        p_val = 1 - chi2.cdf(lr_stat, df)
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'  {label} [MixedLM LRT]: χ²({df}) = {lr_stat:.2f}, p = {p_val:.4f} {stars}')
        # In các hệ số riêng lẻ
        print(f'    Fixed effects:')
        for effect_name, p_eff in mdf.pvalues.items():
            if 'C(condition)' in effect_name:
                print(f'      {effect_name}: coef={mdf.fe_params[effect_name]*1e6:+.2f}µV, p={p_eff:.4f}')
        return {'lr_stat': lr_stat, 'p_value': p_val, 'method': 'mixedlm_lr'}
    except ImportError:
        print(f'  {label} [MixedLM]: statsmodels not available')
        return None
    except Exception as e:
        print(f'  {label} [MixedLM]: {e}')
        return None


def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    print('=' * 65)
    print('  SO SÁNH ANOVA: Unfiltered vs Filtered')
    print('  DV: P2_mean_amp')
    print('=' * 65)

    uf = pd.read_csv(UNFILTERED) if os.path.exists(UNFILTERED) else None
    fw = pd.read_csv(FILTERED_W) if os.path.exists(FILTERED_W) else None
    fs = pd.read_csv(FILTERED_S) if os.path.exists(FILTERED_S) else None

    if uf is not None:
        print(f'\nUnfiltered:        {len(uf)} rows, {len(uf["subject_id"].unique())} subjects')
    if fw is not None:
        print(f'Filtered (weak):   {len(fw)} rows, {len(fw["subject_id"].unique())} subjects')
    if fs is not None:
        print(f'Filtered (strict): {len(fs)} rows, {len(fs["subject_id"].unique())} subjects')

    print('\n' + '=' * 65)
    print('  MEANS')
    print('=' * 65)
    if uf is not None: print_means(uf, 'Unfiltered')
    if fw is not None: print_means(fw, 'Filtered (weak)')
    if fs is not None: print_means(fs, 'Filtered (strict)')

    print('\n' + '=' * 65)
    print('  rmANOVA (balanced subjects có đủ 6 conditions)')
    print('=' * 65)
    if uf is not None: anova_balanced(uf, 'P2_mean_amp', 'Unfiltered')
    if fw is not None: anova_balanced(fw, 'P2_mean_amp', 'Filtered (weak)')
    if fs is not None: anova_balanced(fs, 'P2_mean_amp', 'Filtered (strict)')

    print('\n' + '=' * 65)
    print('  Mixed Model ANOVA (chấp nhận missing data)')
    print('=' * 65)
    if uf is not None: mixedlm_anova(uf, 'P2_mean_amp', 'Unfiltered')
    if fw is not None: mixedlm_anova(fw, 'P2_mean_amp', 'Filtered (weak)')
    if fs is not None: mixedlm_anova(fs, 'P2_mean_amp', 'Filtered (strict)')

    # Thêm JAR analysis
    print('\n' + '=' * 65)
    print('  JAR GROUP ANALYSIS (P2_mean_amp)')
    print('=' * 65)
    for df, label in [(uf, 'Unfiltered'), (fw, 'Filtered(weak)'), (fs, 'Filtered(strict)')]:
        if df is not None and 'jar_group' in df.columns and df['jar_group'].notna().any():
            jar_data = df.dropna(subset=['P2_mean_amp', 'jar_group'])
            print(f'\n  {label}:')
            for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
                jd = jar_data[jar_data['jar_group'] == jg]['P2_mean_amp']
                if len(jd):
                    print(f'    {jg:15s}: {jd.mean()*1e6:+.2f} ± {jd.std()*1e6:.2f}µV (n={len(jd)})')
            # JAR ANOVA (one-way, between subjects since JAR varies)
            try:
                from scipy.stats import f_oneway
                groups = [jar_data[jar_data['jar_group'] == jg]['P2_mean_amp'].values
                          for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']
                          if len(jar_data[jar_data['jar_group'] == jg]) > 1]
                if len(groups) >= 2:
                    f, p = f_oneway(*groups)
                    print(f'    JAR effect: F = {f:.3f}, p = {p:.4f} {"*" if p<0.05 else "ns"}'  )
            except Exception:
                pass

    # Per-condition pairwise effect sizes (Cohen's d)
    print('\n' + '=' * 65)
    print('  PAIRWISE EFFECT SIZES (Cohen\'s d)')
    print('=' * 65)
    for df, label in [(uf, 'Unfiltered'), (fw, 'Filtered(weak)'), (fs, 'Filtered(strict)')]:
        if df is None:
            continue
        print(f'\n  {label}:')
        d = df.dropna(subset=['P2_mean_amp'])
        means = d.groupby('condition')['P2_mean_amp'].mean()
        stds = d.groupby('condition')['P2_mean_amp'].std()
        # High (893) vs Water (605)
        n1 = len(d[d['condition'] == 893])
        n2 = len(d[d['condition'] == 605])
        pooled = np.sqrt(((n1-1)*stds[893]**2 + (n2-1)*stds[605]**2) / (n1+n2-2))
        d_val = (means[893] - means[605]) / pooled if pooled > 0 else 0
        print(f'    High/893 vs Water/605: d = {d_val:.3f} (n1={n1}, n2={n2})')
        # High (893) vs MedLow (453)
        n2 = len(d[d['condition'] == 453])
        pooled = np.sqrt(((n1-1)*stds[893]**2 + (n2-1)*stds[453]**2) / (n1+n2-2))
        d_val = (means[893] - means[453]) / pooled if pooled > 0 else 0
        print(f'    High/893 vs MedLow/453: d = {d_val:.3f} (n1={n1}, n2={n2})')


if __name__ == '__main__':
    main()
