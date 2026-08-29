"""HITL для ризикового MCP-tool restart_service у MAS: три сценарії — approve / reject / edit (interrupt + Command(resume))."""

import json
from pathlib import Path

from mas_langgraph import build_mas_graph, open_sqlite_saver, run_mas
from mcp_client import McpSession
from tools import parse_tool_json

OUTPUT_DIR = Path(__file__).parent
HITL_RESULTS_PATH = OUTPUT_DIR / "hitl_results.json"

SCENARIOS = [
    {"name": "reject", "query": "Перевір статус payments і перезапусти сервіс", "approve": False,
     "comment": "on-call: спершу збільшити пул з'єднань", "edit_args": None},
    {"name": "edit", "query": "Перевір статус payments і перезапусти сервіс", "approve": True,
     "comment": "on-call: уточнив причину", "edit_args": {"reason": "INC-1001: connection pool exhausted, схвалено лідом Payments"}},
    {"name": "approve", "query": "Перевір статус notifications і перезапусти сервіс", "approve": True,
     "comment": "on-call схвалив", "edit_args": None},
]


def service_status(tools: list, service: str) -> str:
    tool = next(t for t in tools if t.name == "get_service_status")
    return parse_tool_json(tool.invoke({"service": service}))["data"]["status"]


def run_scenarios(mcp_tools: list, graph=None) -> list[dict]:
    """Проганяє три сценарії HITL на одному графі; повертає журнал рішень і стан сервісу до/після."""
    graph = graph or build_mas_graph(mcp_tools)
    rows = []
    for sc in SCENARIOS:
        service = "notifications" if "notifications" in sc["query"] else "payments"
        before = service_status(mcp_tools, service)
        print(f"\n▶ [{sc['name']}] {sc['query']}  (стан до: {before})")
        result = run_mas(graph, sc["query"], f"hitl-{sc['name']}", approve=sc["approve"], comment=sc["comment"], edit_args=sc["edit_args"])
        after = service_status(mcp_tools, service)
        decision = result["hitl_decisions"][-1]
        print(f"  рішення: approved={decision['approved']}, edited={decision['edited']}, args={json.dumps(decision['args'], ensure_ascii=False)}")
        print(f"  стан після: {after} · відповідь: {result['final_answer'][:160]}")
        rows.append({"scenario": sc["name"], "query": sc["query"], "service": service, "status_before": before, "status_after": after,
                     "decision": decision, "final_answer": result["final_answer"][:300]})
    return rows


def main() -> None:
    session = McpSession().open()
    saver, conn = open_sqlite_saver()
    graph = build_mas_graph(session.tools, checkpointer=saver)
    rows = run_scenarios(session.tools, graph)
    conn.close()
    session.close()
    HITL_RESULTS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗбережено {HITL_RESULTS_PATH.name}: " + ", ".join(f"{r['scenario']} → {r['status_before']}→{r['status_after']}" for r in rows))


if __name__ == "__main__":
    main()
