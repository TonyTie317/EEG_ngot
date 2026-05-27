#!/usr/bin/env python3
"""
JAR Analysis — chi tiết cho từng ERP component (P1, N1, P2, N400).

Chạy 1-way ANOVA JAR group → mỗi dependent variable (mean_amp, peak_amp)
cho unfiltered và filtered data.

Usage:
    .venv/bin/python run_jar_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from pipeline.config import load_config, setup_logging
from pipeline.constants import CONCENTRATIONS, CONCENTRATION_LABELS

UNFILTERED = 'output/results/erp/component_measures.csv'
FILTERED_W = 'output/results/erp_filtered/component_measures_filtered.csv'
FILTERED_S = 'output/results/erp_filtered/component_measures_filtered_strict.csv'

JAR_GROUPS = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
COMPONENTS = ['P1', 'N1', 'P2', 'N400']
MEASURES = ['mean_amp', 'peak_amp']


def load(label):
    path = {'Unfiltered': UNFILTERED, 'Filtered(weak)': FILTERED_W,
            'Filtered(strict)': FILTERED_S}[label]
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def anova_jar(measures, dv):
    """1-way between-subjects ANOVA: jar_group → DV."""
    groups = []
    for jg in JAR_GROUPS:
        vals = measures[measures['jar_group'] == jg][dv].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
    if len(groups) >= 2:
        f, p = f_oneway(*groups)
        return f, p, [len(g) for g in groups]
    return None, None, None


def effect_size(measures, dv):
    """Eta-squared for JAR effect: SS_between / SS_total."""
    vals = measures.dropna(subset=[dv, 'jar_group']).copy()
    grand_mean = vals[dv].mean()
    ss_total = ((vals[dv] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0
    ss_between = 0
    for jg in JAR_GROUPS:
        g = vals[vals['jar_group'] == jg][dv]
        if len(g):
            ss_between += len(g) * (g.mean() - grand_mean) ** 2
    return ss_between / ss_total


def jar_means_table(measures, label):
    """Tạo bảng JAR means cho tất cả components."""
    lines = [f'\n{"="*72}']
    lines.append(f'  JAR MEANS — {label}')
    lines.append(f'{"="*72}')

    # Per component
    for comp in COMPONENTS:
        lines.append(f'\n  --- {comp} ---')
        lines.append(f'  {"JAR Group":15s} {"n":5s} {"mean_amp":14s} {"peak_amp":14s} {"peak_lat":12s}')
        lines.append(f'  {"-"*60}')
        for jg in JAR_GROUPS:
            jd = measures[measures['jar_group'] == jg]
            n = len(jd)
            if n == 0:
                continue
            ma = jd[f'{comp}_mean_amp'].mean() * 1e6 if f'{comp}_mean_amp' in jd else 0
            pa = jd[f'{comp}_peak_amp'].mean() * 1e6 if f'{comp}_peak_amp' in jd else 0
            pl = jd[f'{comp}_peak_lat'].mean() * 1e3 if f'{comp}_peak_lat' in jd else 0
            lines.append(f'  {jg:15s} {n:5d} {ma:+.2f}±{jd[f"{comp}_mean_amp"].std()*1e6:.2f}µV  '
                         f'{pa:+.2f}±{jd[f"{comp}_peak_amp"].std()*1e6:.2f}µV  '
                         f'{pl:.0f}ms')

    return '\n'.join(lines)


def jar_anova_table(measures, label):
    """Bảng ANOVA JAR effect cho tất cả components + measures."""
    lines = [f'\n{"="*72}']
    lines.append(f'  JAR ANOVA (1-way between-subjects) — {label}')
    lines.append(f'{"="*72}')
    lines.append(f'  {"Component":8s} {"Measure":10s} {"F":8s} {"p":8s} {"sig":5s} '
                 f'{"eta2":6s} {"n per group":20s}')
    lines.append(f'  {"-"*72}')

    for comp in COMPONENTS:
        for m in MEASURES:
            dv = f'{comp}_{m}'
            if dv not in measures.columns:
                continue
            data = measures.dropna(subset=[dv, 'jar_group'])
            if len(data) < 5:
                continue
            f, p, ns = anova_jar(data, dv)
            eta2 = effect_size(data, dv)
            if f is not None:
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                n_str = ', '.join(str(n) for n in ns)
                lines.append(f'  {comp:8s} {m:10s} {f:8.3f} {p:8.4f} {sig:5s} {eta2:6.3f} [{n_str}]')

    return '\n'.join(lines)


def concentration_response_by_jar(measures, label):
    """P2 mean_amp theo concentration, split by JAR group."""
    lines = [f'\n{"="*72}']
    lines.append(f'  P2 MEAN_AMP × CONCENTRATION × JAR — {label}')
    lines.append(f'{"="*72}')

    for jg in JAR_GROUPS:
        lines.append(f'\n  JAR = {jg}:')
        jd = measures[measures['jar_group'] == jg]
        for c in CONCENTRATIONS:
            cd = jd[jd['condition'] == c]['P2_mean_amp']
            if len(cd):
                lines.append(f'    {CONCENTRATION_LABELS[c]:15s}: {cd.mean()*1e6:+.2f} ± {cd.std()*1e6:.2f}µV '
                             f'(n={len(cd)})')

    return '\n'.join(lines)


def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    labels = ['Unfiltered', 'Filtered(weak)', 'Filtered(strict)']

    for label in labels:
        df = load(label)
        if df is None:
            continue
        print(jar_means_table(df, label))
        print(jar_anova_table(df, label))
        print(concentration_response_by_jar(df, label))

    # Mixed model cho JAR (xử lý missing data)
    print(f'\n{"="*72}')
    print('  MIXED MODEL JAR ANALYSIS (P2_mean_amp ~ JAR, random=subject)')
    print(f'{"="*72}')

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        for label in labels:
            df = load(label)
            if df is None or 'jar_group' not in df.columns:
                continue
            data = df.dropna(subset=['P2_mean_amp', 'jar_group']).copy()
            if len(data) < 10:
                continue
            try:
                md = smf.mixedlm('P2_mean_amp ~ C(jar_group)', data, groups=data['subject_id'])
                mdf = md.fit(reml=True)
                print(f'\n  {label} (n={len(data)}):')
                for eff_name, p_eff in mdf.pvalues.items():
                    if 'C(jar_group)' in eff_name:
                        print(f'    {eff_name}: coef={mdf.fe_params[eff_name]*1e6:+.2f}µV, p={p_eff:.4f}')
            except Exception as e:
                print(f'  {label}: mixed model failed — {e}')
    except ImportError:
        print('  statsmodels not available')


if __name__ == '__main__':
    main()
