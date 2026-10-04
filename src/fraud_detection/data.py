"""Load the credit card dataset and split it into train / validation / test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_FRACTION,
    VAL_FRACTION,
)


def load_data(path=DATA_PATH):
    """Return (X, y). X has the 30 features (Time, V1-V28, Amount) in CSV order."""
    df = pd.read_csv(path)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def split_data(X, y):
    """Stratified 60/20/20 split.

    Stratify keeps the fraud rate (~0.17%) the same in every part, which matters
    because each part contains only a few hundred frauds.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Step 1: hold out the test set (20% of rows). It is not used again
    # until the final reporting step.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_STATE
    )

    # Step 2: split the remaining 80% into train and validation.
    # 0.2 / 0.8 = 0.25 of the rest, which is 20% of all rows.
    val_share_of_rest = VAL_FRACTION / (1 - TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share_of_rest, stratify=y_rest, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
