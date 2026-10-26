from typing import List, Optional, Any, Dict

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful fallback when pydantic is not installed
    class BaseModel:
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

    def Field(default=None, ge=None, le=None, description=None, **kwargs):
        return default


class CustomerInput(BaseModel):
    customer_id: Optional[str] = "CUST-PREDICT-01"
    tenure_months: int = 4
    contract_type: str = "Month-to-month"
    monthly_charges: float = 85.50
    total_charges: Optional[float] = 342.0
    support_tickets_90d: int = 4
    failed_transactions_90d: int = 2
    avg_daily_usage_mins: float = 22.0
    days_since_last_login: int = 18
    has_auto_renew: int = 0

    def __init__(self, **kwargs):
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


class FeatureContribution(BaseModel):
    feature: str
    impact_score: float
    direction: str  # "INCREASES CHURN RISK" or "REDUCES CHURN RISK"
    explanation: str

    def __init__(self, feature: str, impact_score: float, direction: str, explanation: str):
        self.feature = feature
        self.impact_score = impact_score
        self.direction = direction
        self.explanation = explanation


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str  # "LOW", "MEDIUM", "HIGH"
    will_churn: bool
    confidence_score: float
    base_value: float
    top_risk_factors: List[FeatureContribution]
    retention_action: str

    def __init__(self, customer_id: str, churn_probability: float, risk_tier: str,
                 will_churn: bool, confidence_score: float, base_value: float,
                 top_risk_factors: List[FeatureContribution], retention_action: str):
        self.customer_id = customer_id
        self.churn_probability = churn_probability
        self.risk_tier = risk_tier
        self.will_churn = will_churn
        self.confidence_score = confidence_score
        self.base_value = base_value
        self.top_risk_factors = top_risk_factors
        self.retention_action = retention_action


class ModelMetricsResponse(BaseModel):
    model_name: str
    dataset_records: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    global_top_features: List[dict]

    def __init__(self, model_name: str, dataset_records: int, accuracy: float,
                 precision: float, recall: float, f1_score: float,
                 roc_auc: float, global_top_features: List[dict]):
        self.model_name = model_name
        self.dataset_records = dataset_records
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc
        self.global_top_features = global_top_features
