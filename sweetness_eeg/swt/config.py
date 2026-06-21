"""Config loading, logging, and path helpers for the sweetness EEG study."""

import logging
import os
from typing import Any, Dict

import yaml

# Repo root = two levels up from this file (sweetness_eeg/swt/config.py).
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(PKG_DIR)                 # sweetness_eeg/
REPO_ROOT = os.path.dirname(STUDY_DIR)               # EEG_ngot/
DEFAULT_CONFIG = os.path.join(STUDY_DIR, 'config.yaml')


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load YAML config and resolve all paths to absolutes under the study dir."""
    with open(path, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    paths = cfg.setdefault('paths', {})
    # raw_data is relative to the repo root; everything else to the study dir.
    paths['raw_data'] = os.path.join(REPO_ROOT, paths.get('raw_data', 'data/datamoi'))
    paths['behavior_xlsx'] = os.path.join(
        REPO_ROOT, paths.get('behavior_xlsx', 'doc/Đánh giá cảm quan - EEG 31.5.xlsx'))
    for key in ('figures', 'results', 'reports'):
        paths[key] = os.path.join(STUDY_DIR, paths.get(key, key))
        ensure_dir(paths[key])
    return cfg


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not exist; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def setup_logging(name: str = 'swt', level: int = logging.INFO) -> logging.Logger:
    """Return a console logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                                datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def fig_dir(cfg: Dict[str, Any], sub: str) -> str:
    """Return (and create) a sub-directory under figures/."""
    return ensure_dir(os.path.join(cfg['paths']['figures'], sub))


def result_path(cfg: Dict[str, Any], *parts: str) -> str:
    """Build a path under results/, creating parent dirs."""
    p = os.path.join(cfg['paths']['results'], *parts)
    ensure_dir(os.path.dirname(p))
    return p


def report_path(cfg: Dict[str, Any], name: str) -> str:
    """Build a path under reports/."""
    return os.path.join(cfg['paths']['reports'], name)
