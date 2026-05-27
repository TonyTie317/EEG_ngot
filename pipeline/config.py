"""
Configuration loading, validation, and logging setup.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config file.

    Returns
    -------
    config : dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that required config sections and values are present.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Raises
    ------
    ValueError
        If required sections are missing or values are invalid.
    """
    required_sections = ['paths', 'preprocessing', 'epoching']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Paths
    if 'raw_data' not in config['paths']:
        raise ValueError("config.paths must contain 'raw_data'")

    # Preprocessing sanity
    prep = config['preprocessing']
    if prep.get('l_freq', 0) >= prep.get('h_freq', 999):
        raise ValueError("preprocessing.l_freq must be < h_freq")

    # Epoching sanity
    ep = config['epoching']
    if ep.get('tmin', 0) >= 0:
        raise ValueError("epoching.tmin must be negative (pre-stimulus)")
    if ep.get('tmax', 0) <= 0:
        raise ValueError("epoching.tmax must be positive (post-stimulus)")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Create and return a configured logger.

    Parameters
    ----------
    config : dict
        Configuration dictionary with optional 'logging' section.

    Returns
    -------
    logger : logging.Logger
        Configured logger named 'gerp_pipeline'.
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    logger = logging.getLogger('gerp_pipeline')
    logger.setLevel(level)
    logger.handlers = []

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_config.get('log_to_file', False):
        log_file = log_config.get('log_file', 'output/pipeline.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
