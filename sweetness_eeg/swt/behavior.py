"""
Behavioral (sensory) data: liking, sweetness-JAR and sweet-aftertaste.

Reads the curated ``Sheet1`` of ``doc/Đánh giá cảm quan - EEG 31.5.xlsx`` directly
(openpyxl). Sheet1 holds 25 clean participants (IDs 001–025, duplicates resolved,
020 present) with 14 sample-blocks per row: (Code, Liking, JAR_ngọt, JAR_hậu_ngọt).

Each blind code maps to a (substance, intensity) via :data:`swt.constants.CODE_TO_LABEL`,
so behavior can be aggregated per condition and joined to EEG by subject id
(``PXXX`` ↔ Excel id ``XXX`` — same number, the Excel id merely lacks the "P").
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    CODE_TO_LABEL, JAR_CENTER, label_to_intensity, label_to_substance,
    map_jar_to_group,
)

_INT_RE = re.compile(r'-?\d+')


def _to_int(val) -> Optional[int]:
    """Extract the leading integer from a cell like '6 - Hơi thích' or '6'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    m = _INT_RE.search(str(val))
    return int(m.group()) if m else None


def _eeg_id(raw_id) -> Optional[str]:
    """Normalise an Excel participant id to the EEG 'PXXX' form."""
    m = _INT_RE.search(str(raw_id))
    return f"P{int(m.group()):03d}" if m else None


def load_behavior_long(xlsx_path: str, sheet: str = 'Sheet1',
                       logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Return a tidy long-format behavioral table.

    One row per (participant × sample) with columns:
    subject, name, sex, age, code, ma_mau, substance, intensity,
    liking, sweetness_jar, aftertaste, jar_dev, after_dev, jar_group.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
    except ValueError:
        df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)
        if logger:
            logger.warning(f"Sheet {sheet!r} not found; used first sheet instead")

    meta_n = 6  # Timestamp, id, name, sex, age, hometown
    records: List[Dict[str, Any]] = []
    n_blocks = (len(df.columns) - meta_n) // 4
    for _, row in df.iterrows():
        subject = _eeg_id(row.iloc[1])
        if subject is None:
            continue
        for b in range(n_blocks):
            base = meta_n + b * 4
            code = _to_int(row.iloc[base])
            if code is None or code not in CODE_TO_LABEL:
                continue
            label = CODE_TO_LABEL[code]
            liking = _to_int(row.iloc[base + 1])
            jar = _to_int(row.iloc[base + 2])
            after = _to_int(row.iloc[base + 3])
            records.append({
                'subject': subject,
                'name': str(row.iloc[2]).strip(),
                'sex': str(row.iloc[3]).strip(),
                'age': _to_int(row.iloc[4]),
                'code': code,
                'ma_mau': label,
                'substance': label_to_substance(label),
                'intensity': label_to_intensity(label),
                'liking': liking,
                'sweetness_jar': jar,
                'aftertaste': after,
                'jar_dev': None if jar is None else jar - JAR_CENTER,
                'after_dev': None if after is None else after - JAR_CENTER,
                'jar_group': map_jar_to_group(jar),
            })

    out = pd.DataFrame(records)
    if logger:
        logger.info(f"Behavior: {df.shape[0]} participants → {len(out)} sample ratings "
                    f"({out['subject'].nunique()} unique subjects)")
    return out


def subject_condition_means(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average the two repeats per (subject × condition).

    Returns one row per subject × ma_mau with mean liking / JAR / aftertaste.
    """
    agg = (long_df
           .groupby(['subject', 'ma_mau', 'substance', 'intensity'], dropna=False)
           [['liking', 'sweetness_jar', 'aftertaste', 'jar_dev', 'after_dev']]
           .mean()
           .reset_index())
    return agg


def condition_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Group-level mean ± SEM per condition for the three sensory measures."""
    def sem(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    rows = []
    for (label, sub, inten), g in long_df.groupby(
            ['ma_mau', 'substance', 'intensity'], dropna=False):
        row = {'ma_mau': label, 'substance': sub, 'intensity': inten, 'n': len(g)}
        for m in ['liking', 'sweetness_jar', 'aftertaste']:
            row[f'{m}_mean'] = g[m].mean()
            row[f'{m}_sem'] = sem(g[m])
        rows.append(row)
    return pd.DataFrame(rows)
