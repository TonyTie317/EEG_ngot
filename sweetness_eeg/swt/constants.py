"""
Ground-truth constants for the Sucrose-vs-Sucralose sweetness EEG study
(data in ``data/datamoi/``).

These values come from the experimental protocol
(``doc/Protocol chất tạo ngọt V2.docx``) and the verified structure of the
``*_KT88_with_times_10s.csv`` recordings. No config dependency.

Sample-code mapping (verified identical across every subject)
------------------------------------------------------------
``ma_mau`` (readable label)  ── substance ── perceived intensity ── blind codes
    S1_5   Sucrose    5  (≈ 5   g/L sucrose)       119, 452
    S1_7   Sucrose    7  (≈ 7.5 g/L sucrose)       781, 299
    S1_12  Sucrose    12 (≈ 12  g/L sucrose)       873, 777
    S2_5   Sucralose  5  (iso-sweet to 5%)         563, 453
    S2_7   Sucralose  7  (iso-sweet to 7.5%)       336, 122
    S2_12  Sucralose  12 (iso-sweet to 12%)        257, 644
    H2O    Water      0  (baseline)                681, 575
"""

# ── Channels ──────────────────────────────────────────────────────────────────
# Names as they appear in the raw datamoi CSV header.
RAW_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
]
ECG_CHANNELS = ['ECG1', 'ECG2']
META_COLUMNS = ['times', 'timestamp', 'code', 'ma_mau', 'lan_lap']

# Old 10-10 → standard 10-20 names (needed for an MNE montage).
CHANNEL_ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}

# Standard names after renaming (used everywhere downstream).
EEG_CHANNELS = [CHANNEL_ALIASES.get(ch, ch) for ch in RAW_EEG_CHANNELS]
N_CHANNELS = len(EEG_CHANNELS)  # 16

# ── Subjects ──────────────────────────────────────────────────────────────────
# 23 subjects present in data/datamoi/ (P013, P018 absent; P012 present here).
ALL_SUBJECTS = [
    'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009',
    'P010', 'P011', 'P012', 'P014', 'P015', 'P016', 'P017', 'P019', 'P020',
    'P021', 'P022', 'P023', 'P024', 'P025',
]
N_SUBJECTS = len(ALL_SUBJECTS)  # 23

# ── Sample / condition mapping ────────────────────────────────────────────────
SUBSTANCES = {'S1': 'Sucrose', 'S2': 'Sucralose', 'H2O': 'Water'}

# label → (substance_code, intensity_int, blind_codes)
LABEL_INFO = {
    'S1_5':  ('S1', 5,  [119, 452]),
    'S1_7':  ('S1', 7,  [781, 299]),
    'S1_12': ('S1', 12, [873, 777]),
    'S2_5':  ('S2', 5,  [563, 453]),
    'S2_7':  ('S2', 7,  [336, 122]),
    'S2_12': ('S2', 12, [257, 644]),
    'H2O':   ('H2O', 0, [681, 575]),
}

# blind code (int) → label
CODE_TO_LABEL = {
    code: label
    for label, (_sub, _inten, codes) in LABEL_INFO.items()
    for code in codes
}

# Ordered condition labels (sweet samples + water).
CONDITIONS = ['H2O', 'S1_5', 'S1_7', 'S1_12', 'S2_5', 'S2_7', 'S2_12']
SWEET_CONDITIONS = ['S1_5', 'S1_7', 'S1_12', 'S2_5', 'S2_7', 'S2_12']
INTENSITIES = [5, 7, 12]
INTENSITY_LABELS = {5: 'Low (~5%)', 7: 'Mid (~7.5%)', 12: 'High (~12%)'}

# Pretty display labels for figures/reports.
COND_DISPLAY = {
    'H2O':   'Water',
    'S1_5':  'Sucrose-5',  'S1_7':  'Sucrose-7.5',  'S1_12':  'Sucrose-12',
    'S2_5':  'Sucralose-5', 'S2_7': 'Sucralose-7.5', 'S2_12': 'Sucralose-12',
}

# Color per condition (for consistent plots).
COND_COLORS = {
    'H2O':   '#7f7f7f',
    'S1_5':  '#9ecae1', 'S1_7':  '#4292c6', 'S1_12':  '#08519c',  # blues = sucrose
    'S2_5':  '#fcae91', 'S2_7':  '#fb6a4a', 'S2_12':  '#a50f15',  # reds  = sucralose
}
SUBSTANCE_COLORS = {'Sucrose': '#08519c', 'Sucralose': '#a50f15', 'Water': '#7f7f7f'}


def label_to_substance(label):
    """Return full substance name ('Sucrose'/'Sucralose'/'Water') for a label."""
    sub = LABEL_INFO.get(label, (None,))[0]
    return SUBSTANCES.get(sub)


def label_to_intensity(label):
    """Return integer intensity (0/5/7/12) for a label, or None."""
    info = LABEL_INFO.get(label)
    return info[1] if info else None


# ── ROIs (channel groups) ─────────────────────────────────────────────────────
# Frontal = reward/hedonic cortex; Gustatory = scalp proxy over insula/operculum
# (true insula is not directly recordable with scalp EEG — documented as a proxy).
ROIS = {
    'Frontal':   ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
    'Gustatory': ['T7', 'T8', 'C3', 'C4'],          # insular/opercular scalp proxy
    'Central':   ['C3', 'C4'],
    'Temporal':  ['T7', 'T8', 'P7', 'P8'],
    'Parietal':  ['P3', 'P4'],
    'Occipital': ['O1', 'O2'],
}
# ROIs that map directly onto the reference paper's regions of interest.
PAPER_ROIS = ['Frontal', 'Gustatory']

# ── Frequency bands (Hz) ──────────────────────────────────────────────────────
FREQ_BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_COLORS = {
    'delta': '#5e3c99', 'theta': '#2c7fb4', 'alpha': '#41ab5d',
    'beta':  '#fb9a29', 'gamma': '#e31a1c',
}

# ── Acquisition parameters ────────────────────────────────────────────────────
SFREQ = 100                       # Hz
TRIAL_SAMPLES = 1000              # samples per trial (10 s @ 100 Hz)
TRIAL_DURATION = TRIAL_SAMPLES / SFREQ  # 10.0 s
N_TRIALS_PER_SUBJECT = 14         # 7 samples × 2 repeats
UV_TO_V = 1e-6                    # microvolts → volts

# ── Behavioral (sensory) scales ───────────────────────────────────────────────
# Liking: 9-point hedonic. JAR & aftertaste: 5-point (3 = "just about right").
LIKING_SCALE = (1, 9)
JAR_SCALE = (1, 5)
AFTERTASTE_SCALE = (1, 5)
JAR_CENTER = 3                    # "Vừa phải" = just-about-right
JAR_LABELS = {
    1: 'Far too little', 2: 'Too little', 3: 'Just right',
    4: 'Too sweet', 5: 'Far too sweet',
}
# 3-group collapse of the 5-point sweetness/JAR scale.
JAR_GROUPS = {'Not_enough': [1, 2], 'Just_right': [3], 'Too_much': [4, 5]}


def map_jar_to_group(value):
    """Map a 1–5 JAR rating to one of {Not_enough, Just_right, Too_much}."""
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    v = int(round(value))
    for grp, vals in JAR_GROUPS.items():
        if v in vals:
            return grp
    return None
