"""Request and response models for the API.

Pydantic checks types and ranges BEFORE the model runs, so bad input gets a clear
422 error with the field name and the rule it broke.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The 28 PCA features (V1-V28) were produced by a PCA transform, so they are roughly
# centred near 0. The largest value in the training data is about 121 in absolute
# value. The bound below is generous on purpose: it rejects garbage, not rare values.
PcaFeature = Annotated[float, Field(ge=-1000, le=1000, allow_inf_nan=False)]


class TransactionIn(BaseModel):
    """One credit card transaction with the same 30 features as the training CSV."""

    # Reject unknown keys. A typo such as "v1" should fail loudly, not be ignored.
    model_config = ConfigDict(extra="forbid")

    Time: Annotated[float, Field(ge=0, le=1e7, allow_inf_nan=False,
                                 description="Seconds since the first transaction in the dataset")]
    V1: PcaFeature
    V2: PcaFeature
    V3: PcaFeature
    V4: PcaFeature
    V5: PcaFeature
    V6: PcaFeature
    V7: PcaFeature
    V8: PcaFeature
    V9: PcaFeature
    V10: PcaFeature
    V11: PcaFeature
    V12: PcaFeature
    V13: PcaFeature
    V14: PcaFeature
    V15: PcaFeature
    V16: PcaFeature
    V17: PcaFeature
    V18: PcaFeature
    V19: PcaFeature
    V20: PcaFeature
    V21: PcaFeature
    V22: PcaFeature
    V23: PcaFeature
    V24: PcaFeature
    V25: PcaFeature
    V26: PcaFeature
    V27: PcaFeature
    V28: PcaFeature
    Amount: Annotated[float, Field(ge=0, le=1e6, allow_inf_nan=False,
                                   description="Transaction amount in euros")]


class PredictOut(BaseModel):
    fraud_probability: float = Field(description="Calibrated probability that this transaction is fraud")
    decision: Literal["fraud", "legitimate"]
    threshold: float = Field(description="Validation-chosen threshold used for the decision")


class Reason(BaseModel):
    feature: str
    value: float = Field(description="The transaction's value for this feature")
    contribution: float = Field(description="SHAP value in log-odds; positive pushes toward fraud")
    direction: Literal["toward_fraud", "toward_legitimate"]


class ExplainOut(PredictOut):
    top_reasons: list[Reason] = Field(description="The 5 features with the largest effect")
    explanation_note: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    threshold: float | None
    calibration: str | None
