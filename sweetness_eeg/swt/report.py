"""Markdown report helpers."""

import os
from typing import List, Optional

import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = '.3f', max_rows: Optional[int] = None
             ) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Float columns whose largest magnitude is very small (e.g. PSD power in V²,
    ~1e-12) are shown in scientific notation so they don't collapse to ``0``.
    """
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            vals = d[c].abs()
            mx = vals[vals > 0].max() if (vals > 0).any() else 0.0
            fmt = '.2e' if (mx and mx < 1e-3) else floatfmt
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else format(v, fmt))
    try:
        return d.to_markdown(index=False)
    except Exception:                                   # noqa: BLE001
        return '```\n' + d.to_string(index=False) + '\n```'


def rel(path: str, report_dir: str) -> str:
    """Path relative to the report directory (for figure links in markdown)."""
    return os.path.relpath(path, report_dir)


def img(path: str, report_dir: str, caption: str = '') -> str:
    """Markdown image embed with a relative path."""
    r = rel(path, report_dir)
    return f"![{caption}]({r})\n\n*{caption}*\n" if caption else f"![]({r})\n"


def write(path: str, sections: List[str]) -> str:
    """Write a list of markdown strings (joined by blank lines) to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n\n'.join(sections) + '\n')
    return path
