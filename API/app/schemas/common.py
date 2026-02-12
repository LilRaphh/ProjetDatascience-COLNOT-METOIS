from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ResponseMeta(BaseModel):
    dataset_id: str
    status: str = Field(default="success")


class ResponseReport(BaseModel):
    message: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BaseResponse(BaseModel):
    meta: ResponseMeta
    result: Any
    report: Optional[ResponseReport] = None
