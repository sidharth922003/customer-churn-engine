from typing import List, Optional, Any, Dict

try:
    from pydantic import BaseModel as _PydanticBaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class _PydanticBaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            result = {}
            for k, v in self.__dict__.items():
                if isinstance(v, list):
                    result[k] = [item.dict() if hasattr(item, 'dict') else item for item in v]
                elif hasattr(v, 'dict'):
                    result[k] = v.dict()
                else:
                    result[k] = v
            return result

        def model_dump(self) -> Dict[str, Any]:
            return self.dict()

    def Field(default=None, **kwargs):
        return default


class CustomerInput(_PydanticBaseModel):
    customer_id: Optional[str] = Field(default="CUST-PREDICT-01")
    tenure_months: int = Field(default=4)
    contract_type: str = Field(default="Month-to-month")
    monthly_charges: float = Field(default=85.50)
    total_charges: Optional[float] = Field(default=342.0)
    support_tickets_90d: int = Field(default=4)
    failed_transactions_90d: int = Field(default=2)
    avg_daily_usage_mins: float = Field(default=22.0)
    days_since_last_login: int = Field(default=18)
    has_auto_renew: int = Field(default=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure fallbacks have attributes set if not using Pydantic's metaclass
        if not HAS_PYDANTIC:
            self.customer_id = kwargs.get("customer_id", "CUST-PREDICT-01")
            self.tenure_months = int(kwargs.get("tenure_months", 4))
            self.contract_type = str(kwargs.get("contract_type", "Month-to-month"))
            self.monthly_charges = float(kwargs.get("monthly_charges", 85.50))
            self.total_charges = float(kwargs.get("total_charges", 342.0))
            self.support_tickets_90d = int(kwargs.get("support_tickets_90d", 4))
            self.failed_transactions_90d = int(kwargs.get("failed_transactions_90d", 2))
            self.avg_daily_usage_mins = float(kwargs.get("avg_daily_usage_mins", 22.0))
            self.days_since_last_login = int(kwargs.get("days_since_last_login", 18))
            self.has_auto_renew = int(kwargs.get("has_auto_renew", 0))


class FeatureContribution(_PydanticBaseModel):
    feature: str = Field(default="")
    impact_score: float = Field(default=0.0)
    direction: str = Field(default="")  # "INCREASES CHURN RISK" or "REDUCES CHURN RISK"
    explanation: str = Field(default="")

    def __init__(self, feature: str = "", impact_score: float = 0.0, direction: str = "", explanation: str = "", **kwargs):
        super().__init__(feature=feature, impact_score=impact_score, direction=direction, explanation=explanation, **kwargs)
        self.feature = feature or kwargs.get("feature", "")
        self.impact_score = impact_score or kwargs.get("impact_score", 0.0)
        self.direction = direction or kwargs.get("direction", "")
        self.explanation = explanation or kwargs.get("explanation", "")


class PredictionResponse(_PydanticBaseModel):
    customer_id: str = Field(default="")
    churn_probability: float = Field(default=0.0)
    risk_tier: str = Field(default="LOW")  # "LOW", "MEDIUM", "HIGH"
    will_churn: bool = Field(default=False)
    confidence_score: float = Field(default=0.0)
    base_value: float = Field(default=0.24)
    top_risk_factors: List[FeatureContribution] = Field(default_factory=list)
    retention_action: str = Field(default="")

    def __init__(self, customer_id: str = "", churn_probability: float = 0.0, risk_tier: str = "LOW",
                 will_churn: bool = False, confidence_score: float = 0.0, base_value: float = 0.24,
                 top_risk_factors: Optional[List[FeatureContribution]] = None, retention_action: str = "", **kwargs):
        top_risk = top_risk_factors if top_risk_factors is not None else kwargs.get("top_risk_factors", [])
        super().__init__(
            customer_id=customer_id or kwargs.get("customer_id", ""),
            churn_probability=churn_probability or kwargs.get("churn_probability", 0.0),
            risk_tier=risk_tier or kwargs.get("risk_tier", "LOW"),
            will_churn=will_churn if will_churn is not None else kwargs.get("will_churn", False),
            confidence_score=confidence_score or kwargs.get("confidence_score", 0.0),
            base_value=base_value or kwargs.get("base_value", 0.24),
            top_risk_factors=top_risk,
            retention_action=retention_action or kwargs.get("retention_action", ""),
            **kwargs
        )
        self.customer_id = customer_id or kwargs.get("customer_id", "")
        self.churn_probability = churn_probability or kwargs.get("churn_probability", 0.0)
        self.risk_tier = risk_tier or kwargs.get("risk_tier", "LOW")
        self.will_churn = will_churn if will_churn is not None else kwargs.get("will_churn", False)
        self.confidence_score = confidence_score or kwargs.get("confidence_score", 0.0)
        self.base_value = base_value or kwargs.get("base_value", 0.24)
        self.top_risk_factors = top_risk
        self.retention_action = retention_action or kwargs.get("retention_action", "")


class ModelMetricsResponse(_PydanticBaseModel):
    model_name: str = Field(default="")
    dataset_records: int = Field(default=10000)
    accuracy: float = Field(default=0.0)
    precision: float = Field(default=0.0)
    recall: float = Field(default=0.0)
    f1_score: float = Field(default=0.0)
    roc_auc: float = Field(default=0.0)
    global_top_features: List[dict] = Field(default_factory=list)
