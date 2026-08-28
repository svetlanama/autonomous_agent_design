"""Тести SRE-асистента: Pydantic-схеми, інструменти, ReAct-цикл."""
import json

import pytest
from pydantic import ValidationError

try:
    import incident_agent as A          # локальний запуск: pytest -v
except ImportError:                     # запуск із ноутбука (Colab): об'єкти вже в __main__
    import __main__ as A


# ---------- Pydantic-схеми: коректні входи ----------
def test_service_status_args_normalizes_service():
    assert A.ServiceStatusArgs(service="  Payments ").service == "payments"


def test_query_logs_args_defaults_and_level_normalization():
    args = A.QueryLogsArgs(service="api_gateway", level="error")
    assert (args.service, args.level, args.last_minutes) == ("api-gateway", "ERROR", 30)


def test_calculate_sla_args_valid():
    args = A.CalculateSlaArgs(service="payments", downtime_minutes=45, period_days=30)
    assert args.downtime_minutes == 45.0 and args.target_percent == 99.9


def test_restart_args_valid():
    args = A.RestartServiceArgs(service="payments", reason="Деградація сервісу після деплою v2.14.0")
    assert args.service == "payments"


# ---------- Pydantic-схеми: некоректні входи ----------
def test_service_status_args_unknown_service():
    with pytest.raises(ValidationError, match="Невідомий сервіс"):
        A.ServiceStatusArgs(service="billing")


def test_query_logs_args_invalid_level_and_window():
    with pytest.raises(ValidationError, match="Невідомий рівень"):
        A.QueryLogsArgs(service="payments", level="DEBUG")
    with pytest.raises(ValidationError):
        A.QueryLogsArgs(service="payments", last_minutes=0)


def test_calculate_sla_args_downtime_exceeds_period():
    with pytest.raises(ValidationError, match="перевищує тривалість періоду"):
        A.CalculateSlaArgs(service="payments", downtime_minutes=5000, period_days=1)


def test_restart_args_protected_service_and_short_reason():
    with pytest.raises(ValidationError, match="заборонено політикою"):
        A.RestartServiceArgs(service="postgres-db", reason="повільні запити у базі")
    with pytest.raises(ValidationError, match="щонайменше 10 символів"):
        A.RestartServiceArgs(service="payments", reason="коротко")


def test_search_runbooks_args_bounds():
    with pytest.raises(ValidationError):
        A.SearchRunbooksArgs(query="ab")
    with pytest.raises(ValidationError):
        A.SearchRunbooksArgs(query="політика SLA", top_k=9)


# ---------- Інструменти ----------
def test_get_service_status_returns_standard_json():
    payload = json.loads(A.get_service_status.invoke({"service": "payments"}))
    assert payload["status"] == "ok"
    assert payload["data"]["service"] == "payments" and payload["data"]["status"] == "degraded"


def test_try_tool_returns_error_json_for_invalid_input():
    payload = json.loads(A.try_tool("get_service_status", {"service": "billing"}))
    assert payload["status"] == "error" and "Невідомий сервіс" in payload["error"]


def test_query_logs_filters_by_level_and_window():
    payload = json.loads(A.query_logs.invoke({"service": "payments", "level": "ERROR", "last_minutes": 10}))
    assert payload["data"]["count"] == 2
    assert all(e["level"] == "ERROR" for e in payload["data"]["entries"])


def test_calculate_sla_numbers():
    data = json.loads(A.calculate_sla.invoke({"service": "payments", "downtime_minutes": 45, "period_days": 30}))["data"]
    assert data["error_budget_minutes"] == pytest.approx(43.2)
    assert data["sla_breached"] is True
    assert data["availability_percent"] == pytest.approx(99.8958, abs=1e-3)


def test_search_runbooks_returns_relevant_document():
    data = json.loads(A.search_runbooks.invoke({"query": "політика перезапуску сервісів підтвердження людини", "top_k": 2}))["data"]
    assert len(data["documents"]) == 2
    assert data["documents"][0]["doc_id"] == "restart_policy"


def test_query_logs_fallback_to_status():
    result, used = A.try_tool_with_fallback("query_logs", {"service": "notifications"})
    payload = json.loads(result)
    assert used == "get_service_status"
    assert payload["status"] == "error" and payload["fallback"]["result"]["data"]["status"] == "down"


def test_restart_service_is_marked_risky():
    assert "restart_service" in A.RISKY_TOOLS
    data = json.loads(A.restart_service.invoke({"service": "auth-service", "reason": "Тестовий перезапуск stateless-сервісу"}))["data"]
    assert data["status"] == "restarting" and len(data["restart_id"]) == 8


# ---------- ReAct-цикл ----------
def test_react_loop_completes_with_tool_call():
    record = A.run_agent("Перевір статус сервісу payments.", llm=A.ScriptedSreLLM(), tag="test_react")
    assert not record["halted"]
    assert [c["tool"] for c in record["tool_calls"]] == ["get_service_status"]
    assert "degraded" in record["final"]
    assert [e["type"] for e in record["trajectory"]] == ["thought", "observation", "thought", "final_answer"]


def test_react_loop_detection_guardrail():
    record = A.run_agent("Перевір зациклення: питай статус api-gateway знову і знову.",
                         llm=A.ScriptedSreLLM(), config=A.AgentConfig(max_steps=10), tag="test_loop")
    assert record["halted"] and record["halt_reason"].startswith("loop")
    assert record["steps"] == 2


def test_react_max_steps_guardrail():
    record = A.run_agent("Порахуй довгий ланцюг SLA крок за кроком.",
                         llm=A.ScriptedSreLLM(), config=A.AgentConfig(max_steps=3), tag="test_max_steps")
    assert record["halted"] and record["halt_reason"].startswith("max_steps")
    assert record["steps"] == 3
