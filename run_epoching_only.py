#!/usr/bin/env python3
"""
Chạy lại chỉ Stage 1-3: Load → Preprocess → Epoch
Dùng khi thay đổi tmin/tmax/baseline/reject trong config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.loader import load_all_subjects
from pipeline.preprocess import preprocess_all_subjects
from pipeline.epoching import create_epochs_all_subjects, save_all_epochs


def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    ep_cfg = config['epoching']
    logger.info("=" * 60)
    logger.info("Re-Epoching với config mới:")
    logger.info(f"  tmin={ep_cfg['tmin']}s  tmax={ep_cfg['tmax']}s")
    logger.info(f"  baseline={ep_cfg.get('baseline')}  reject={ep_cfg.get('reject')}")
    logger.info("=" * 60)

    ensure_dir(config['paths']['output_base'])

    logger.info("\n[Stage 1] Loading subjects...")
    subjects_data = load_all_subjects(config, logger)
    if not subjects_data:
        logger.error("Không load được subject nào. Dừng.")
        return

    logger.info(f"  Loaded {len(subjects_data)} subjects.")

    logger.info("\n[Stage 2] Preprocessing...")
    subjects_data = preprocess_all_subjects(subjects_data, config, logger)
    logger.info("  Preprocessing xong.")

    logger.info("\n[Stage 3] Epoching...")
    all_epochs, all_trial_info = create_epochs_all_subjects(subjects_data, config, logger)
    subjects_list = [s['subject_id'] for s in subjects_data]
    save_all_epochs(all_epochs, all_trial_info, subjects_list, config, logger)

    # In tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT EPOCH SAU RE-RUN:")
    logger.info(f"  Số subjects: {len(all_epochs)}")
    total = len(all_trial_info)
    logger.info(f"  Tổng epochs: {total}")
    if len(all_epochs) > 0:
        shape = all_epochs[0].get_data().shape
        logger.info(f"  Shape mỗi epoch: {shape[1]} kênh × {shape[2]} mẫu "
                    f"({shape[2]/100:.2f}s)")
    # Reject rate per subject
    for i, (sd, ep) in enumerate(zip(subjects_data, all_epochs)):
        sid = sd['subject_id']
        n_kept = len(ep)
        n_total = len(sd['trials'])
        rejected = n_total - n_kept
        pct = rejected / n_total * 100 if n_total > 0 else 0
        logger.info(f"  {sid}: {n_kept}/{n_total} kept ({rejected} rejected, {pct:.0f}%)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
