"""
Fixed domain constants for the gERP analysis pipeline.

All values here are derived from the experimental design and data structure.
No config dependency — these are ground-truth values.
"""

# ── Channels ──────────────────────────────────────────────────────────────────
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
]
ECG_CHANNELS = ['ECG1', 'ECG2']
# Legacy aliases that appear in raw CSV files (old 10-10 naming)
CHANNEL_ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
NON_SIGNAL_COLUMNS = [
    'times', 'timestamp', 'frame_idx',
    'ma_mau', 'NT', 'repeat', 'JAR',
    'ECG1', 'ECG2',
]

# ── Subjects ──────────────────────────────────────────────────────────────────
ALL_SUBJECTS = [f'P{i:03d}' for i in range(1, 31) if i not in (12, 22)]
N_SUBJECTS = len(ALL_SUBJECTS)  # 28

# ── Experimental conditions (ma_mau codes) ────────────────────────────────────
# Ordered by sweetness perception (ascending JAR mean across all subjects):
#   605=1.04 (water/baseline), 258=1.18, 453=2.04, 189=3.04, 762=3.43, 893=3.93
CONCENTRATIONS = [605, 258, 453, 189, 762, 893]
CONCENTRATION_LABELS = {
    605: 'Water/605',
    258: 'Low/258',
    453: 'MedLow/453',
    189: 'Medium/189',
    762: 'MedHigh/762',
    893: 'High/893',
}
WATER_CODE = 605       # baseline water sample
HIGHEST_CODE = 893     # highest sucrose concentration
N_CONDITIONS = len(CONCENTRATIONS)  # 6
N_REPEATS = 5
TRIALS_PER_SUBJECT = N_CONDITIONS * N_REPEATS  # 30

# ── JAR mapping (per-subject per-condition rating) ───────────────────────────
# Original scale: 1-5 → 3 groups
JAR_GROUPS = {
    'Khong_du': [1, 2],   # "Not enough" sweetness
    'Vua_phai': [3],      # "Just right"
    'Qua_nhieu': [4, 5],  # "Too much"
}

JAR_LABELS_VN = {
    'Khong_du': 'Không đủ',
    'Vua_phai': 'Vừa phải',
    'Qua_nhieu': 'Quá nhiều',
}

JAR_NUMERIC = {
    'Khong_du': 0,
    'Vua_phai': 1,
    'Qua_nhieu': 2,
}


def map_jar_to_group(jar_value):
    """Map numeric JAR (1-5) to group name string.

    Returns None if jar_value is NaN or outside expected range.
    """
    import math
    if jar_value is None or (isinstance(jar_value, float) and math.isnan(jar_value)):
        return None
    jar_int = int(jar_value)
    for group_name, values in JAR_GROUPS.items():
        if jar_int in values:
            return group_name
    return None


# ── ERP component time windows (seconds) ─────────────────────────────────────
ERP_WINDOWS = {
    'P1':   (0.080, 0.120),
    'N1':   (0.120, 0.200),
    'P2':   (0.350, 0.450),
    'N400': (0.350, 0.500),
}

# Peak detection mode: 'pos' = most positive, 'neg' = most negative
ERP_PEAK_MODE = {
    'P1':   'pos',
    'N1':   'neg',
    'P2':   'pos',
    'N400': 'neg',
}

# Default ROI per component (from Wilton 2018, Mouillot 2020)
ERP_ROI = {
    'P1':   ['C3', 'C4', 'P3', 'P4'],       # centro-parietal
    'N1':   ['Fp1', 'Fp2', 'F3', 'F4'],      # frontal
    'P2':   ['C3', 'C4', 'P3', 'P4'],         # centro-parietal
    'N400': ['F3', 'F4', 'C3', 'C4'],         # fronto-central
}

# ── ROI channel groups ────────────────────────────────────────────────────────
ROI = {
    'Frontal':   ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
    'Central':   ['C3', 'C4'],
    'Temporal':  ['T3', 'T4', 'T5', 'T6'],
    'Parietal':  ['P3', 'P4'],
    'Occipital': ['O1', 'O2'],
}

# ── Frequency bands (Hz) ─────────────────────────────────────────────────────
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# ── Trial parameters ─────────────────────────────────────────────────────────
SFREQ = 100               # Hz
TRIAL_DURATION_SAMPLES = 1100  # 11 seconds at 100 Hz
