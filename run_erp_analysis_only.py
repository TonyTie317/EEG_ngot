#!/usr/bin/env python3
"""
Chạy chỉ Stage 4: ERP Analysis từ epochs đã lưu trên disk.
Tự động áp dụng Woody realignment (nếu có realign_offsets.csv).

Usage:
    .venv/bin/python run_erp_analysis_only.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mne

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.erp_analysis import run_erp_analysis


EPOCHS_BASE = 'output/epochs'
TRIAL_INFO_CSV = 'output/epochs/all_trial_info.csv'


def load_all_epochs(config, logger):
    """Load epochs từ disk (đã lưu bởi run_epoching_only.py)."""
    from pipeline.constants import ALL_SUBJECTS

    all_epochs = []
    subjects_found = []

    for sid in ALL_SUBJECTS:
        ep_dir = os.path.join(EPOCHS_BASE, sid)
        fif_path = os.path.join(ep_dir, 'epochs_epo.fif')
        if not os.path.exists(fif_path):
            logger.warning(f'[{sid}] Không tìm thấy {fif_path}, bỏ qua')
            continue
        try:
            epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
            all_epochs.append(epochs)
            subjects_found.append(sid)
            logger.info(f'[{sid}] Loaded {len(epochs)} epochs, shape={epochs.get_data().shape}')
        except Exception as e:
            logger.error(f'[{sid}] Lỗi load epochs: {e}')

    logger.info(f'Loaded {len(all_epochs)} subjects, '
                f'{sum(len(e) for e in all_epochs)} epochs tổng')
    return all_epochs, subjects_found


def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    ensure_dir(config['paths']['output_base'])
    ensure_dir(config['paths']['results_base'])
    ensure_dir(config['paths']['figures_base'])

    logger.info('=' * 60)
    logger.info('ERP Analysis — từ epochs đã lưu (có Woody realignment)')
    logger.info('=' * 60)

    # Load epochs từ disk
    logger.info('\n[Load] Đọc epochs từ disk...')
    all_epochs, subjects = load_all_epochs(config, logger)
    if not all_epochs:
        logger.error('Không load được epoch nào. Chạy run_epoching_only.py trước.')
        sys.exit(1)

    # Load trial info
    if not os.path.exists(TRIAL_INFO_CSV):
        logger.error(f'Không tìm thấy {TRIAL_INFO_CSV}. Chạy run_epoching_only.py trước.')
        sys.exit(1)
    all_trial_info = pd.read_csv(TRIAL_INFO_CSV)
    logger.info(f'Trial info: {len(all_trial_info)} rows')

    # Chạy ERP Analysis (bên trong tự load epochs và apply_woody_realign)
    logger.info('\n[Stage 4] ERP Analysis...')
    erp_results = run_erp_analysis(config, logger)

    logger.info('\n' + '=' * 60)
    logger.info('ERP Analysis XONG!')
    logger.info(f'  Kết quả tại: {config["paths"]["results_base"]}')
    logger.info(f'  Figures tại: {config["paths"]["figures_base"]}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
