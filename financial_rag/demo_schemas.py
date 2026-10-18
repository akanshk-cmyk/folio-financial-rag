"""Context: validated contracts for the Colab demo's four application roles.

These roles are software components, not autonomous subprocesses. Secrets are
excluded from result schemas; every answer retains its exact generation context.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentMetadata(StrictModel):
    document_id: str
    company: str = "AAPL"
    fiscal_year: int = Field(ge=1900, le=2200)
    title: str
    source_url: str
    file_sha256: str


class ChunkMetadata(StrictModel):
    chunk_id: str
    document_id: str
    pdf_page: int = Field(ge=1)
    strategy: Literal["fixed", "semantic"]
    dataset_id: str


class SearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=10)


class ArticleInput(StrictModel):
    url: HttpUrl


class QuestionInput(StrictModel):
    question: str = Field(min_length=1, max_length=6000)
    strategy: Literal["fixed", "semantic"] = "semantic"
    rerank: bool = True
    fiscal_years: list[int] | None = None
    company: str = "AAPL"
    candidates: int = Field(default=10, ge=3, le=50)
    top_n: int = Field(default=5, ge=1, le=10)
    context_chars: int = Field(default=24000, ge=1000, le=50000)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.top_n > self.candidates:
            raise ValueError("top_n must not exceed candidate count")
        if self.fiscal_years is not None and not self.fiscal_years:
            raise ValueError("Use None or a nonempty list of fiscal years")
        if self.fiscal_years and len(set(self.fiscal_years)) > self.top_n:
            raise ValueError("top_n must allow at least one source for each requested filing year")
        if self.fiscal_years and len(set(self.fiscal_years)) != len(self.fiscal_years):
            raise ValueError("Fiscal-year filters must be unique")
        return self


class Citation(StrictModel):
    source_id: str
    document_id: str
    document_title: str
    pdf_page: int | None = Field(default=None, ge=1)
    web_url: str | None = None


class Calculation(StrictModel):
    formula: str
    source_values: list[dict]
    units: str
    result: str
    supporting_source_ids: list[str]


class AnswerResult(StrictModel):
    question: str
    answer: str
    citations: list[Citation]
    calculations: list[Calculation]
    insufficient_evidence: bool
    status: str
    retrieved_passages: dict[str, str]
    run_settings: dict
    latency_seconds: float = Field(ge=0)
    audit: dict = Field(default_factory=dict)


class EvaluationResult(StrictModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    citation_support: float = Field(ge=0, le=1)
    explanations: dict[str, str]
    detected_issues: list[str]


class Observation(StrictModel):
    value: str
    currency: str
    scale: str
    basis: str
    scope: str
    period: str
    period_type: str
    metric: str
    evidence_id: str
    quote: str
    component: str | None = None
    period_quote: str | None = None


class CalculationInput(StrictModel):
    operation: Literal["growth", "margin", "difference", "sum"]
    inputs: list[Observation] = Field(min_length=2, max_length=2)
