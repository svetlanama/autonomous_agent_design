"""Локальні Pydantic-tools (з ДЗ1/ДЗ2): RAG-пошук у ChromaDB, арифметика SLA та SLA-компенсацій."""

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from knowledge import retrieve


def _ok(data: dict) -> str:
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def _err(message: str, tool_name: str = "") -> str:
    return json.dumps({"status": "error", "tool": tool_name, "error": message}, ensure_ascii=False)


def parse_tool_json(content: Any) -> dict:
    """Розбирає результат tool: JSON-рядок, dict або список content-блоків MCP."""
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    try:
        return json.loads(content) if isinstance(content, str) else dict(content)
    except (ValueError, TypeError):
        return {"status": "error", "error": str(content)[:200]}


class SearchKnowledgeArgs(BaseModel):
    """Параметри RAG-пошуку."""

    query: str = Field(description="Пошуковий запит до бази runbook-ів і політик SRE")
    top_k: int = Field(default=2, ge=1, le=5, description="Скільки документів повернути (1..5)")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not 3 <= len(query) <= 300:
            raise ValueError("Запит має містити від 3 до 300 символів")
        return query


class CalculateSlaArgs(BaseModel):
    """Параметри розрахунку SLA (period_days валідується раніше за downtime_minutes)."""

    service: str = Field(min_length=2, max_length=40, description="Назва сервісу")
    period_days: int = Field(default=30, ge=1, le=365, description="Період у днях (1..365)")
    downtime_minutes: float = Field(ge=0, description="Сумарний простій за період, хвилин")
    target_percent: float = Field(default=99.9, ge=90.0, le=100.0, description="Цільова доступність, % (90..100)")

    @field_validator("downtime_minutes")
    @classmethod
    def validate_downtime(cls, value: float, info: ValidationInfo) -> float:
        period_minutes = info.data.get("period_days", 30) * 1440
        if value > period_minutes:
            raise ValueError(f"Простій {value} хв перевищує тривалість періоду ({period_minutes} хв)")
        return round(value, 2)


class SlaCreditArgs(BaseModel):
    """Параметри розрахунку SLA-компенсації."""

    availability_percent: float = Field(ge=0.0, le=100.0, description="Фактична доступність за місяць, %")
    target_percent: float = Field(default=99.9, ge=90.0, le=100.0, description="Ціль SLA з тарифу клієнта, %")
    monthly_fee_usd: float = Field(gt=0, le=1e7, description="Місячна плата клієнта, USD")


CREDIT_TIERS = [(99.0, 10), (95.0, 25), (0.0, 50)]  # (нижня межа доступності, % компенсації) — див. faq://sla-policy


@tool("search_knowledge", args_schema=SearchKnowledgeArgs)
def search_knowledge(query: str, top_k: int = 2) -> str:
    """RAG: шукає у базі знань runbook-и та політики SRE (SLA, ескалація, перезапуск, post-mortem, чергування).
    Повертає документи зі similarity — якщо similarity низька, переформулюй запит і спробуй ще раз."""
    return _ok({"query": query, "documents": retrieve(query, top_k)})


@tool("calculate_sla", args_schema=CalculateSlaArgs)
def calculate_sla(service: str, downtime_minutes: float, period_days: int = 30, target_percent: float = 99.9) -> str:
    """Рахує фактичну доступність сервісу за період, бюджет помилок і чи порушено SLA."""
    period_minutes = period_days * 1440
    availability = (1 - downtime_minutes / period_minutes) * 100
    budget = (1 - target_percent / 100) * period_minutes
    return _ok({"service": service, "period_days": period_days, "downtime_minutes": downtime_minutes,
                "availability_percent": round(availability, 4), "target_percent": target_percent,
                "error_budget_minutes": round(budget, 1), "budget_remaining_minutes": round(budget - downtime_minutes, 1),
                "sla_breached": availability < target_percent})


@tool("estimate_sla_credit", args_schema=SlaCreditArgs)
def estimate_sla_credit(availability_percent: float, monthly_fee_usd: float, target_percent: float = 99.9) -> str:
    """Рахує SLA-компенсацію клієнту (% від місячної плати та сума в USD) за політикою компенсацій."""
    percent = 0 if availability_percent >= target_percent else next(p for low, p in CREDIT_TIERS if availability_percent >= low)
    return _ok({"availability_percent": availability_percent, "target_percent": target_percent,
                "credit_percent": percent, "credit_usd": round(monthly_fee_usd * percent / 100, 2),
                "monthly_fee_usd": monthly_fee_usd})


LOCAL_TOOLS: list[BaseTool] = [search_knowledge, calculate_sla, estimate_sla_credit]


def try_tool(tool_obj: BaseTool, args: dict) -> str:
    """Викликає tool; будь-яку помилку валідації/виконання перетворює на JSON {status: error}."""
    try:
        result = tool_obj.invoke(args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except ValidationError as error:  # помилка валідації стає Observation для агента
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return _err(f"[ValidationError] {details}", tool_obj.name)
    except Exception as error:  # noqa: BLE001
        return _err(f"[{type(error).__name__}] {str(error).strip().splitlines()[0]}", tool_obj.name)


if __name__ == "__main__":
    for t in LOCAL_TOOLS:
        print(f"{t.name:20} ({', '.join(t.args_schema.model_fields)})\n{'':22}{t.description[:90]}")
    print(try_tool(calculate_sla, {"service": "payments", "downtime_minutes": 90}))
    print(try_tool(estimate_sla_credit, {"availability_percent": 99.79, "monthly_fee_usd": 5000}))
    print(try_tool(calculate_sla, {"service": "payments", "downtime_minutes": -5}))
