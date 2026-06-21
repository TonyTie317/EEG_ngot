"""swt — Sweetness-perception EEG analysis package (Sucrose vs Sucralose).

Self-contained pipeline for the ``data/datamoi/`` recordings. Does not modify
or depend on the legacy ``pipeline/`` or ``src/`` code.
"""

__all__ = [
    'constants', 'config', 'loader', 'behavior', 'preprocess', 'epoching',
    'spectral', 'erp', 'connectivity', 'stats', 'ml', 'dl', 'viz',
]
