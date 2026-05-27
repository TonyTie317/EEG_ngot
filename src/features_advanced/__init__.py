"""
Advanced feature extraction package for EEG data.

Provides additional feature extraction methods beyond the basic pipeline:
- STFT (Short-Time Fourier Transform)
- Time-Frequency (Morlet wavelets, PLV)
- Wavelet Transform (DWT, CWT)
- Alpha band analysis
- Connectivity (coherence, PLV, correlation)
- Advanced spectral (SEF, centroid, band ratios, 1/f)
"""

import os
import logging
from collections import Counter
from typing import Dict, Any

import numpy as np
import pandas as pd
import mne

__version__ = '0.1.0'


def extract_advanced_features_from_epochs(
    epochs: mne.Epochs,
    config: Dict[str, Any],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Extract all advanced features from epochs.

    Mirrors the pattern of src.features.extract_features_from_epochs().

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    config : dict
        Configuration dictionary with 'features_advanced' section
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    adv_config = config.get('features_advanced', {})
    freq_bands = adv_config.get('freq_bands') or config['stats']['freq_bands']
    rois = config['stats'].get('rois', {})

    all_features = []

    # STFT features
    stft_cfg = adv_config.get('stft', {})
    if stft_cfg.get('enabled', True):
        logger.info("Computing STFT features")
        from .stft_features import compute_stft_features
        features = compute_stft_features(
            epochs, freq_bands,
            window_size=stft_cfg.get('window_size', 64),
            hop_length=stft_cfg.get('hop_length', 32),
            n_fft=stft_cfg.get('n_fft', 128),
        )
        if not features.empty:
            all_features.append(features)

    # Time-frequency features
    tf_cfg = adv_config.get('timefreq', {})
    if tf_cfg.get('enabled', True):
        logger.info("Computing Time-Frequency features")
        from .timefreq_features import compute_tfr_features, compute_plv_features
        features = compute_tfr_features(
            epochs, freq_bands,
            n_cycles=tf_cfg.get('n_cycles', 7),
            use_multitaper=tf_cfg.get('use_multitaper', False),
        )
        if not features.empty:
            all_features.append(features)

        if tf_cfg.get('compute_plv', True):
            logger.info("Computing TFR PLV features")
            features = compute_plv_features(
                epochs, freq_bands,
                pair_strategy=tf_cfg.get('plv_pair_strategy', 'roi'),
            )
            if not features.empty:
                all_features.append(features)

    # Wavelet features
    wavelet_cfg = adv_config.get('wavelet', {})
    if wavelet_cfg.get('enabled', True):
        from .wavelet_features import compute_dwt_features, compute_cwt_features

        logger.info("Computing DWT features")
        features = compute_dwt_features(
            epochs,
            wavelet=wavelet_cfg.get('dwt_wavelet', 'db4'),
            max_level=wavelet_cfg.get('dwt_max_level'),
        )
        if not features.empty:
            all_features.append(features)

        logger.info("Computing CWT features")
        features = compute_cwt_features(
            epochs, freq_bands,
            wavelet=wavelet_cfg.get('cwt_wavelet', 'cmor1.5-1.0'),
        )
        if not features.empty:
            all_features.append(features)

    # Alpha analysis
    alpha_cfg = adv_config.get('alpha', {})
    if alpha_cfg.get('enabled', True):
        logger.info("Computing Alpha analysis features")
        from .alpha_analysis import compute_alpha_features
        features = compute_alpha_features(
            epochs,
            alpha_band=alpha_cfg.get('alpha_band', [8, 12]),
            frontal_pairs=[tuple(p) for p in alpha_cfg.get('frontal_pairs', [['F3', 'F4'], ['F7', 'F8']])],
            compute_coherence=alpha_cfg.get('compute_coherence', True),
        )
        if not features.empty:
            all_features.append(features)

    # Connectivity features
    conn_cfg = adv_config.get('connectivity', {})
    if conn_cfg.get('enabled', True):
        from .connectivity import (
            compute_coherence_features,
            compute_plv_connectivity,
            compute_correlation_features,
        )

        if conn_cfg.get('compute_coherence', True):
            logger.info("Computing Coherence features")
            features = compute_coherence_features(
                epochs, freq_bands,
                pair_strategy=conn_cfg.get('pair_strategy', 'roi'),
            )
            if not features.empty:
                all_features.append(features)

        if conn_cfg.get('compute_plv', True):
            logger.info("Computing PLV connectivity features")
            features = compute_plv_connectivity(
                epochs, freq_bands,
                pair_strategy=conn_cfg.get('pair_strategy', 'roi'),
            )
            if not features.empty:
                all_features.append(features)

        if conn_cfg.get('compute_correlation', True):
            logger.info("Computing Correlation features")
            features = compute_correlation_features(epochs, rois=rois)
            if not features.empty:
                all_features.append(features)

    # Advanced spectral features
    spectral_cfg = adv_config.get('spectral_adv', {})
    if spectral_cfg.get('enabled', True):
        logger.info("Computing Advanced spectral features")
        from .spectral_advanced import compute_advanced_spectral_features
        features = compute_advanced_spectral_features(
            epochs, freq_bands,
            sef_percentiles=spectral_cfg.get('sef_percentiles', [50, 90, 95]),
            band_ratios=spectral_cfg.get('band_ratios'),
            compute_aperiodic=spectral_cfg.get('compute_aperiodic', True),
        )
        if not features.empty:
            all_features.append(features)

    if len(all_features) == 0:
        logger.warning("No advanced features extracted")
        return pd.DataFrame()

    combined = pd.concat(all_features, ignore_index=True)
    logger.info(f"Extracted {len(combined)} advanced feature values")
    return combined


def extract_advanced_subject_features(
    subject_id: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Extract advanced features for a single subject.
    Excludes the second repeat, keeping 4 unique samples.

    Mirrors the pattern of src.features.extract_subject_features().
    """
    output_base = config['paths']['output_base']
    epochs_dir = os.path.join(output_base, 'epochs', subject_id)

    fif_path = os.path.join(epochs_dir, 'all_epochs-epo.fif')
    meta_path = os.path.join(epochs_dir, 'trial_metadata.csv')

    if not os.path.exists(fif_path):
        logger.error(f"Epochs not found: {fif_path}")
        return pd.DataFrame()

    logger.info(f"Loading epochs for {subject_id}")
    epochs = mne.read_epochs(fif_path, preload=True)
    metadata = pd.read_csv(meta_path)

    # Keep only 4 unique samples (exclude second repeat)
    sample_types = metadata['sample_type'].tolist()
    type_counts = Counter(sample_types)
    repeated_types = [t for t, cnt in type_counts.items() if cnt == 2]

    if len(repeated_types) == 1:
        repeat_type = repeated_types[0]
        repeat_mask = metadata['sample_type'] == repeat_type
        repeat_indices = metadata[repeat_mask].index.tolist()

        if len(repeat_indices) == 2:
            second_repeat_idx = repeat_indices[1]
            metadata = metadata[metadata.index != second_repeat_idx].reset_index(drop=True)
            keep_indices = [i for i in range(len(epochs)) if i != second_repeat_idx]
            epochs = epochs[keep_indices]
            metadata['epoch_idx'] = range(len(metadata))
            logger.info(f"Excluding second repeat (index {second_repeat_idx}), keeping 4 unique samples")
    else:
        logger.warning(f"Could not identify unique repeat for {subject_id}, using all samples")

    # Extract advanced features
    features_df = extract_advanced_features_from_epochs(epochs, config, logger)

    if features_df.empty:
        return features_df

    features_df['subject_id'] = subject_id
    features_df = features_df.merge(
        metadata[['epoch_idx', 'sample_type', 'event_code']],
        on='epoch_idx',
        how='left'
    )

    return features_df


def extract_all_advanced_features(
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Extract advanced features for all subjects and save to CSV.

    Mirrors the pattern of src.features.extract_all_features().
    """
    from ..utils import get_subject_list

    subjects = get_subject_list(config)
    logger.info(f"Extracting advanced features for {len(subjects)} subjects")

    all_features = []

    for subject_id in subjects:
        try:
            features_df = extract_advanced_subject_features(subject_id, config, logger)
            if not features_df.empty:
                all_features.append(features_df)
        except Exception as e:
            logger.error(f"Error extracting advanced features for {subject_id}: {e}", exc_info=True)

    if len(all_features) == 0:
        logger.error("No advanced features extracted")
        return

    combined = pd.concat(all_features, ignore_index=True)

    # Pivot to wide format
    features_wide = combined.pivot_table(
        index=['subject_id', 'epoch_idx', 'sample_type', 'event_code'],
        columns='feature_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Save
    output_base = config['paths']['output_base']
    features_dir = os.path.join(output_base, 'features')
    os.makedirs(features_dir, exist_ok=True)

    wide_path = os.path.join(features_dir, 'features_advanced.csv')
    features_wide.to_csv(wide_path, index=False)
    logger.info(f"Saved advanced features (wide) to {wide_path}")

    long_path = os.path.join(features_dir, 'features_advanced_long.csv')
    combined.to_csv(long_path, index=False)
    logger.info(f"Saved advanced features (long) to {long_path}")

    n_features = len(features_wide.columns) - 4
    logger.info(f"Advanced feature extraction completed: {len(features_wide)} epochs, {n_features} features")
