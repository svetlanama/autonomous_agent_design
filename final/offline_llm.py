"""LLM-провайдер: OpenAI за наявності ключа, інакше детермінований офлайн-фолбек (supervisor, агенти, planner, replanner)."""

import os
import re
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from tools import parse_tool_json

ROLE_RX = re.compile(r"\[\[role=(\w+)\]\]")
KNOWN_SERVICES = ("api-gateway", "auth-service", "payments", "postgres-db", "notifications")
SERVICE_STEMS = [
    ("api-gateway", "api-gateway"), ("gateway", "api-gateway"), ("auth-service", "auth-service"), ("auth", "auth-service"),
    ("автентиф", "auth-service"), ("payments", "payments"), ("платеж", "payments"), ("платіж", "payments"),
    ("postgres", "postgres-db"), ("база даних", "postgres-db"), ("бази даних", "postgres-db"),
    ("notifications", "notifications"), ("сповіщен", "notifications"),
    ("checkout", "checkout"),  # навмисно невідомий сервіс — демонстрація помилки та replanning
]
ROUTE_RULES = [  # порядок важливий: вужчі категорії перед ширшою tech
    ("billing", ("компенсац", "credit", "рахунок", "тариф", "оплат", "білінг", "billing", "клієнт", "бюджет помилок", "доступніст")),
    ("researcher", ("runbook", "ранбук", "що робити", "як діяти", "інструкц", "політик", "ескалац", "post-mortem", "постмортем",
                    "шаблон", "чергув", "поясни", "що таке", "рекоменд")),
    ("tech", ("статус", "стан", "лог", "помилк", "перезапуст", "рестарт", "restart", "деград", "не працює", "інцидент",
              "розслід", "перевір", "тікет", "латент", "впав", "down")),
]


def services_in(text: str) -> list[str]:
    low = text.lower()
    found = sorted({(low.find(stem), service) for stem, service in SERVICE_STEMS if stem in low})
    return list(dict.fromkeys(service for _, service in found))


def _number_before(text: str, pattern: str, default: float) -> float:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*" + pattern, text.lower())
    return float(match.group(1).replace(",", ".")) if match else default


def _section(prompt: str, header: str) -> list[str]:
    if header not in prompt:
        return []
    block = prompt.split(header, 1)[1].split("\n\n", 1)[0]
    return [line[2:].strip() for line in block.splitlines() if line.strip().startswith("- ")]


def describe_observation(name: str, payload: dict) -> str:
    """Перетворює JSON-результат tool на речення для відповіді агента."""
    if payload.get("status") != "ok":
        return f"{name}: помилка — {payload.get('error', '?')}"
    d = payload["data"]
    if name == "get_service_status":
        return f"{d['service']}: статус {d['status']}, p95 {d['latency_p95_ms']} мс, помилки {d['error_rate_percent']}%, реплік {d['replicas']}"
    if name == "query_logs":
        return f"логи {d['service']} ({d['level']}, {d['last_minutes']} хв): " + " | ".join(e["message"] for e in d["entries"][:3])
    if name == "get_billing_account":
        return f"клієнт {d['customer']} (план {d['plan']}, ${d['monthly_fee_usd']:.0f}/міс, ціль SLA {d['sla_target_percent']}%)"
    if name == "open_incident":
        return f"відкрито тікет {d['id']} ({d['severity']}, власник {d['owner']}, реакція {d['response_minutes']} хв)"
    if name == "restart_service":
        return f"{d['service']} перезапущено (причина: {d['reason']}), новий стан {d['new_status']['status']}"
    if name == "calculate_sla":
        verdict = "SLA ПОРУШЕНО" if d["sla_breached"] else "SLA у нормі"
        return (f"доступність {d['service']} {d['availability_percent']:.2f}% при цілі {d['target_percent']}%, "
                f"залишок бюджету помилок {d['budget_remaining_minutes']} хв — {verdict}")
    if name == "estimate_sla_credit":
        return f"компенсація {d['credit_percent']}% від ${d['monthly_fee_usd']:.0f} = ${d['credit_usd']:.2f}"
    if name == "search_knowledge":
        return "; ".join(f"«{doc['doc_id']}» (similarity {doc['similarity']}): {doc['text'][:150]}…" for doc in d["documents"])
    return str(payload)[:160]


class ScriptedLLM(BaseChatModel):
    """Детермінований офлайн-фолбек: емулює structured output supervisor-а/planner-а та tool calling агентів."""

    bound_tools: list[dict] = Field(default_factory=list)
    latency_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def bind_tools(self, tools: list, **kwargs: Any) -> "ScriptedLLM":
        return self.model_copy(update={"bound_tools": [convert_to_openai_tool(t) for t in tools]})

    def _generate(self, messages: list[AnyMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.latency_s:
            time.sleep(self.latency_s)
        bound = [t["function"]["name"] for t in self.bound_tools]
        last_human = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        if bound == ["RouteDecision"]:
            message = self._route(last_human)
        elif bound == ["Plan"]:
            message = self._plan(last_human)
        elif bound == ["ReplanDecision"]:
            message = self._replan(last_human)
        else:
            system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
            role = ROLE_RX.search(system)
            message = self._agent_step(role.group(1) if role else "general", messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _structured(name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name.lower()}"}])

    # ---------- supervisor ----------
    def _route(self, query: str) -> AIMessage:
        low = query.lower()
        for action, keywords in ROUTE_RULES:
            hit = next((k for k in keywords if k in low), None)
            if hit:
                return self._structured("RouteDecision", {"action": action, "reasoning": f"ключова ознака «{hit}» → {action}"})
        return self._structured("RouteDecision", {"action": "general", "reasoning": "запит не стосується білінгу, інцидентів чи бази знань"})

    # ---------- agents ----------
    def _agent_step(self, role: str, messages: list[AnyMessage]) -> AIMessage:
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        query = messages[humans[-1]].content
        task = query.split("Поточний крок:", 1)[-1].strip()
        after = messages[humans[-1] + 1:]
        observations = {m.name: parse_tool_json(m.content) for m in after if isinstance(m, ToolMessage)}
        turn = sum(1 for m in after if isinstance(m, AIMessage) and m.tool_calls)
        low = task.lower()

        def obs(name: str) -> dict | None:
            return observations.get(name)

        def data(name: str) -> dict | None:
            payload = obs(name)
            return payload["data"] if payload and payload.get("status") == "ok" else None

        def act(name: str, args: dict, thought: str) -> AIMessage:
            return AIMessage(content=f"Thought: {thought}", name=role,
                             tool_calls=[{"name": name, "args": args, "id": f"call_{role}_{turn}"}])

        def summarize(prefix: str = "") -> AIMessage:
            parts = [describe_observation(n, p) for n, p in observations.items()]
            return AIMessage(content=(f"{prefix}: " if prefix else "") + "; ".join(parts) + ".", name=role)

        services = services_in(query) or ["payments"]
        service = services[0]
        if "зациклення" in low:  # демонстрація guardrail «loop» з ДЗ1
            return act("get_service_status", {"service": "api-gateway"}, "перевіряю статус ще раз")

        if role == "billing":
            customer = re.search(r"клієнт\w*\s+([a-z][a-z0-9-]+)", low)
            downtime = _number_before(task, r"хв", 45)
            period = int(_number_before(task, r"дн", 30))
            account = data("get_billing_account")
            target = account["sla_target_percent"] if account else 99.9
            if customer and obs("get_billing_account") is None:
                return act("get_billing_account", {"customer_id": customer.group(1)}, "потрібен тариф і ціль SLA клієнта")
            if obs("calculate_sla") is None:
                return act("calculate_sla", {"service": service, "downtime_minutes": downtime, "period_days": period,
                                             "target_percent": target}, "рахую доступність і бюджет помилок")
            sla = data("calculate_sla")
            if account and sla and obs("estimate_sla_credit") is None:
                return act("estimate_sla_credit", {"availability_percent": sla["availability_percent"], "target_percent": target,
                                                   "monthly_fee_usd": account["monthly_fee_usd"]}, "рахую компенсацію за політикою")
            return summarize("Розрахунок SLA")

        if role == "tech":
            if "статус" in low and obs("get_service_status") is None:
                return act("get_service_status", {"service": service}, f"перевіряю стан {service}")
            if "лог" in low and obs("query_logs") is None:
                minutes = int(_number_before(task, r"хвилин", 30))
                return act("query_logs", {"service": service, "level": "ERROR", "last_minutes": minutes}, "збираю логи помилок")
            if ("тікет" in low or "відкрий інцидент" in low) and obs("open_incident") is None:
                severity = re.search(r"\bP[1-4]\b", task)
                return act("open_incident", {"service": service, "severity": severity.group(0) if severity else "P2",
                                             "summary": f"Деградація {service}: помилки за даними моніторингу та логів"},
                           "реєструю інцидент")
            if ("перезапуст" in low or "restart" in low) and obs("restart_service") is None:
                return act("restart_service", {"service": service, "reason": f"Деградація {service}: вичерпано пул з'єднань, за runbook"},
                           "потрібен перезапуск (ризикова дія)")
            if observations:
                return summarize()
            context = _section(query, "Контекст виконаних кроків:")
            if context:
                return AIMessage(content="Підсумок інциденту: " + " | ".join(c.split(" → ", 1)[-1][:160] for c in context), name=role)
            return AIMessage(content=f"Крок «{task[:60]}» не потребує інструментів.", name=role)

        if role == "researcher":
            first = obs("search_knowledge")
            if first is None:
                return act("search_knowledge", {"query": task[:200], "top_k": 2}, "шукаю в базі знань")
            docs = (first.get("data") or {}).get("documents", [])
            if docs and docs[0]["similarity"] < 0.25 and turn < 2:  # agentic RAG: слабкий збіг → переформулювати запит
                refined = " ".join(dict.fromkeys(["runbook", *services_in(task), *re.findall(r"[а-яіїєґ]{5,}", low)[:4]]))
                return act("search_knowledge", {"query": refined, "top_k": 3}, f"similarity {docs[0]['similarity']} низька — уточнюю запит")
            best = max((d for p in observations.values() if p.get("status") == "ok" for d in p["data"]["documents"]),
                       key=lambda d: d["similarity"], default=None)
            if not best:
                return AIMessage(content="У базі знань немає релевантних документів.", name=role)
            return AIMessage(content=f"За базою знань («{best['doc_id']}», similarity {best['similarity']}): {best['text']}", name=role)

        return AIMessage(content="Я SRE-асистент центру реагування на інциденти. Можу перевірити стан сервісів і логи, "
                                 "розрахувати SLA-компенсацію клієнту або знайти runbook і політику у базі знань.", name=role)

    # ---------- planner ----------
    def _plan(self, query: str) -> AIMessage:
        low, services = query.lower(), services_in(query) or ["payments"]
        steps: list[str] = []
        for s in services:
            if any(k in low for k in ("статус", "стан", "перевір", "розслід", "деград")):
                steps.append(f"Перевір статус сервісу {s}")
            if "лог" in low and s in KNOWN_SERVICES:
                steps.append(f"Переглянь логи помилок сервісу {s} за {int(_number_before(query, r'хвилин', 30))} хвилин")
        if "тікет" in low or "відкрий інцидент" in low:
            severity = re.search(r"\bP[1-4]\b", query)
            steps.append(f"Відкрий тікет інциденту {severity.group(0) if severity else 'P2'} для {services[0]}")
        if any(k in low for k in ("перезапуст", "рестарт", "restart")):
            steps.append(f"Перезапусти сервіс {services[0]}")
        steps.append("Сформуй підсумок інциденту для користувача")
        return self._structured("Plan", {"steps": steps})

    # ---------- replanner ----------
    def _replan(self, prompt: str) -> AIMessage:
        remaining, done = _section(prompt, "Залишок плану:"), _section(prompt, "Виконані кроки:")
        last = done[-1].split(" → ", 1)[-1] if done else ""
        low = last.lower()
        fix = "Перевір статус сервісу payments"
        if "помилка" in low and "невідомий" in low and fix not in remaining and not any(d.startswith(fix) for d in done):
            return self._structured("ReplanDecision", {"action": "revise", "steps": [fix] + remaining,
                                                       "reason": "сервіс невідомий моніторингу; за каталогом ці функції обслуговує payments"})
        if "відхилено" in low:
            return self._structured("ReplanDecision", {"action": "finish", "reason": "людина відхилила ризикову дію",
                                                       "response": "Перезапуск скасовано за рішенням людини. Зібрана діагностика: "
                                                                   + " | ".join(d.split(" → ", 1)[-1][:120] for d in done[:-1])})
        if remaining:
            return self._structured("ReplanDecision", {"action": "continue", "reason": "план актуальний"})
        response = last if low.startswith("підсумок") else "Готово. " + " | ".join(d.split(" → ", 1)[-1][:150] for d in done)
        return self._structured("ReplanDecision", {"action": "finish", "response": response, "reason": "усі кроки плану виконано"})


def build_llm(model_name: str = "gpt-4.1", temperature: float = 0.0) -> BaseChatModel:
    """OpenAI за наявності OPENAI_API_KEY, інакше офлайн-фолбек."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
    return ScriptedLLM()
