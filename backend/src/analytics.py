# backend/src/analytics.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
from src.llm_analytics import generate_analytics_insights


class InsightsRequest(BaseModel):
    chartId: str
    metric: Optional[str] = None
    metricLabel: Optional[str] = None
    totals: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    sample: Optional[Dict[str, Any]] = None
    chartImageB64: Optional[str] = None

@router.post("/insights")
async def analytics_insights(payload: InsightsRequest) -> Dict[str, Any]:
    try:
        text = generate_analytics_insights(payload.model_dump(), model="llama3.2-vision")
        if not text:
            # graceful fallback
            return {"text": "AI insights unavailable (LLM call failed). Check Ollama is running and model name is correct."}
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))