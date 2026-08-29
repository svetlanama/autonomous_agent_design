"""Async-тести MCP-сервера: tools, resources, prompts (pytest-asyncio)."""

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import mcp_server
from mcp_server import mcp

pytestmark = pytest.mark.asyncio


async def call(name: str, args: dict) -> dict:
    """Викликає tool через mcp.call_tool і розбирає JSON-текст відповіді."""
    content = await mcp.call_tool(name, args)
    return json.loads(content[0].text)


async def test_list_tools_exposes_five_documented_tools():
    tools = await mcp.list_tools()
    assert sorted(t.name for t in tools) == ["get_billing_account", "get_service_status", "open_incident",
                                              "query_logs", "restart_service"]
    assert all(t.description and len(t.description) > 40 for t in tools)


async def test_get_service_status_known_and_unknown():
    ok = await call("get_service_status", {"service": "payments"})
    assert ok["status"] == "ok" and ok["data"]["status"] == "degraded"
    bad = await call("get_service_status", {"service": "billing"})
    assert bad["status"] == "error" and "невідомий" in bad["error"]


async def test_query_logs_filters_by_level_and_window():
    result = await call("query_logs", {"service": "payments", "level": "ERROR", "last_minutes": 10})
    entries = result["data"]["entries"]
    assert entries and all(e["level"] == "ERROR" and e["minutes_ago"] <= 10 for e in entries)
    assert (await call("query_logs", {"service": "notifications"}))["status"] == "error"


async def test_schema_validation_rejects_bad_arguments():
    with pytest.raises(ToolError):
        await mcp.call_tool("query_logs", {"service": "payments", "last_minutes": 99999})
    with pytest.raises(ToolError):
        await mcp.call_tool("restart_service", {"service": "payments", "reason": "коротко"})
    with pytest.raises(ToolError):
        await mcp.call_tool("get_service_status", {"service": "BAD NAME!"})


async def test_restart_service_protects_database_and_heals_stateless():
    denied = await call("restart_service", {"service": "postgres-db", "reason": "спроба перезапустити базу даних"})
    assert denied["status"] == "error" and "postgres-db" in denied["error"]
    before = dict(mcp_server.SERVICES_DB["payments"])
    healed = await call("restart_service", {"service": "payments", "reason": "тест: вичерпано пул з'єднань"})
    assert healed["data"]["new_status"]["status"] == "healthy"
    mcp_server.SERVICES_DB["payments"].update(before)


async def test_open_incident_has_side_effect():
    count = len(mcp_server.INCIDENTS)
    ticket = await call("open_incident", {"service": "payments", "severity": "P1", "summary": "пул з'єднань вичерпано"})
    assert ticket["data"]["id"].startswith("INC-") and ticket["data"]["response_minutes"] == 5
    assert len(mcp_server.INCIDENTS) == count + 1


async def test_get_billing_account():
    acc = await call("get_billing_account", {"customer_id": "acme"})
    assert acc["data"]["plan"] == "enterprise" and acc["data"]["monthly_fee_usd"] == 5000.0
    assert (await call("get_billing_account", {"customer_id": "nobody"}))["status"] == "error"


async def test_resources_static_and_template():
    uris = [str(r.uri) for r in await mcp.list_resources()]
    assert "faq://sla-policy" in uris
    assert "service://{name}" in [t.uriTemplate for t in await mcp.list_resource_templates()]
    policy = await mcp.read_resource("faq://sla-policy")
    assert "99.9%" in policy[0].content
    card = json.loads((await mcp.read_resource("service://postgres-db"))[0].content)
    assert card["restart_allowed"] is False


async def test_prompts_render_with_arguments():
    names = sorted(p.name for p in await mcp.list_prompts())
    assert names == ["incident_report", "sla_credit_reply"]
    prompt = await mcp.get_prompt("incident_report", {"service": "payments", "severity": "P1", "tone": "friendly"})
    text = prompt.messages[0].content.text
    assert "payments" in text and "P1" in text and "дружнім" in text
