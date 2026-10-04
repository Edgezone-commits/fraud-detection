"""Project settings in one place. Change numbers here, not in the middle of code."""

from pathlib import Path

# Project root = the folder that contains README.md (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_COL = "Class"  # 1 = fraud, 0 = legitimate

# Fixed seed: the same code always produces the same split and the same model.
RANDOM_STATE = 42

# Split sizes as a share of ALL rows. Train = 1 - 0.2 - 0.2 = 60%.
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2

# Business cost ASSUMPTION (not measured): a missed fraud costs 10x a false alarm.
COST_FN = 10
COST_FP = 1

N_BOOTSTRAP = 1000  # resamples for the confidence intervals
CV_FOLDS = 5        # folds for calibration and for the CV comparison
