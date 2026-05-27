#!/usr/bin/env python3
"""
Main entry point for the gERP Analysis Pipeline.

Runs all stages in sequence:
  1. Load (loader.py)
  2. Preprocess (preprocess.py)
  3. Epoch (epoching.py)
  4. ERP Analysis (erp_analysis.py)
  5. Stats (stats.py)
  6. ML (ml.py)
  7. DL (dl.py)
  8. Visualization (viz.py)
"""

import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.loader import load_all_subjects
from pipeline.preprocess import preprocess_all_subjects
from pipeline.epoching import create_epochs_all_subjects, save_all_epochs
from pipeline.erp_analysis import run_erp_analysis
from pipeline.stats import run_all_stats
from pipeline.ml import run_all_ml_tasks
from pipeline.dl import run_all_dl_tasks
from pipeline.viz import generate_all_figures


def main():
    # ── Config & logging ─────────────────────────────────────────────────
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("gERP Analysis Pipeline — START")
    logger.info("=" * 60)

    # Ensure output directories exist
    ensure_dir(config['paths']['output_base'])
    ensure_dir(config['paths']['results_base'])
    ensure_dir(config['paths']['figures_base'])

    # ── Stage 1: Load ────────────────────────────────────────────────────
    logger.info("\n[Stage 1] Loading subjects...")
    subjects_data = load_all_subjects(config, logger)
    logger.info(f"  Loaded {len(subjects_data)} subjects.")

    if not subjects_data:
        logger.error("No subjects loaded. Aborting.")
        return

    # ── Stage 2: Preprocess ──────────────────────────────────────────────
    logger.info("\n[Stage 2] Preprocessing...")
    subjects_data = preprocess_all_subjects(subjects_data, config, logger)
    logger.info("  Preprocessing complete.")

    # ── Stage 3: Epoch ───────────────────────────────────────────────────
    logger.info("\n[Stage 3] Epoching...")
    all_epochs, all_trial_info = create_epochs_all_subjects(subjects_data, config, logger)
    subjects_list = [s['subject_id'] for s in subjects_data]
    save_all_epochs(all_epochs, all_trial_info, subjects_list, config, logger)
    logger.info(f"  Epochs created for {len(all_epochs)} subjects.")

    # ── Stage 4: ERP Analysis ────────────────────────────────────────────
    logger.info("\n[Stage 4] ERP Analysis...")
    erp_results = run_erp_analysis(config, logger)
    logger.info("  ERP analysis complete.")

    # ── Stage 5: Statistics ──────────────────────────────────────────────
    logger.info("\n[Stage 5] Statistics...")
    stats_results = run_all_stats(erp_results, config, logger)
    logger.info("  Statistics complete.")

    # ── Stage 6: ML Classification ───────────────────────────────────────
    logger.info("\n[Stage 6] ML Classification...")
    ml_results = run_all_ml_tasks(erp_results, config, logger)
    logger.info("  ML complete.")

    # ── Stage 7: Deep Learning ───────────────────────────────────────────
    logger.info("\n[Stage 7] Deep Learning...")
    dl_results = run_all_dl_tasks(config, logger)
    logger.info("  DL complete.")

    # ── Stage 8: Visualization ───────────────────────────────────────────
    logger.info("\n[Stage 8] Visualization...")
    generate_all_figures(erp_results, ml_results, dl_results,
                         config, logger)
    logger.info("  Visualization complete.")

    logger.info("\n" + "=" * 60)
    logger.info("gERP Analysis Pipeline — DONE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
