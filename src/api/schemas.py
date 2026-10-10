"""Response models. They feed the automatic OpenAPI page at /docs."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Disease(str, Enum):
    dengue = "dengue"
    malaria = "malaria"
    chikungunya = "chikungunya"


class Health(BaseModel):
    status: str
    data_version: str
    latest_observed_year: int
    data_granularity: str
    model: Optional[str] = None
    data_note: str


class State(BaseModel):
    state_id: str
    state_name: str


class YearCases(BaseModel):
    year: int
    cases: float


class Historical(BaseModel):
    state_id: str
    state_name: str
    data_granularity: str
    latest_observed_year: int
    history: dict[str, list[YearCases]]


class ForecastItem(BaseModel):
    year: int
    horizon: int
    forecast: int
    persistence: Optional[float] = None          # always present; null only when last year is missing
    interval_80: Optional[list[int]] = None      # horizon 1 only
    confidence: str                              # normal | low
    notes: list[str]
    risk_level: str                              # Low | Moderate | High | Very high | Insufficient evidence
    percentile: Optional[float] = None
    growth_vs_persistence_pct: Optional[float] = None
    risk_reason: list[str]
    experimental: bool                           # true for horizons 2 and 3


class ModelInfo(BaseModel):
    name: str
    version: Optional[str] = None
    summary: str


class ForecastResponse(BaseModel):
    state_id: str
    disease: str
    history: list[YearCases]
    forecasts: list[ForecastItem]
    model: ModelInfo
    disclaimer: str


class RiskState(BaseModel):
    state_id: str
    state_name: str
    risk_level: str
    percentile: Optional[float] = None
    growth_vs_persistence_pct: Optional[float] = None
    forecast: int
    persistence: Optional[float] = None
    interval_80: Optional[list[int]] = None
    confidence: str
    notes: list[str]
    risk_reason: list[str]


class RiskMap(BaseModel):
    disease: str
    year: int
    horizon: int
    experimental: bool
    method: str
    cutoffs: dict[str, float]
    disclaimer: str
    states: list[RiskState]


class Explanation(BaseModel):
    state_id: str
    disease: str
    year: int
    horizon: int
    base_forecast: float
    persistence: Optional[float] = None
    label: str
    method: str
    groups: list[dict]
    summary: str
    notes: list[str]
