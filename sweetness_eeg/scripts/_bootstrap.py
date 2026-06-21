"""Shared bootstrap: put the study package on sys.path and silence noise."""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)            # sweetness_eeg/
if _STUDY not in sys.path:
    sys.path.insert(0, _STUDY)
