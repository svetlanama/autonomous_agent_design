"""ДЗ1-ядро: ReAct-цикл із guardrails (max_steps, timeout, детекція зациклення), TrajectoryLogger з agent_name, HITL."""

import json
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from guardrails import log_security_event, tool_guardrail
from tools import _err, parse_tool_json, try_tool

RISKY_TOOLS = {"restart_service"}


class AgentConfig(BaseModel):
    """Налаштування агента та його захисних механізмів (з ДЗ1/ДЗ2)."""

    max_steps: int = Field(default=6, ge=1, le=50, description="Максимум ітерацій циклу LLM→tools")
    timeout_s: float = Field(default=60.0, gt=0, description="Загальний тайм-аут виконання агента, секунд")
    detect_loops: bool = Field(default=True, description="Зупиняти при повторному ідентичному tool call")
    max_plan_steps: int = Field(default=8, ge=1, le=20, description="Максимальна довжина плану")
    max_total_steps: int = Field(default=12, ge=1, le=50, description="Ліміт виконаних кроків плану")
    model_name: str = Field(default="gpt-4.1")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


DEFAULT_CONFIG = AgentConfig()


class TrajectoryLogger:
    """Журнал траєкторії MAS: кожен запис має agent_name, step, type та час від старту."""

    def __init__(self):
        self.events: list[dict] = []
        self.started_at = time.perf_counter()

    def log(self, agent_name: str, step: int, event_type: str, **payload: Any) -> dict:
        event = {"agent_name": agent_name, "step": step, "type": event_type,
                 "elapsed_s": round(time.perf_counter() - self.started_at, 3), **payload}
        self.events.append(event)
        return event

    def clear(self) -> None:
        self.events.clear()
        self.started_at = time.perf_counter()

    def summary(self) -> dict:
        by_agent: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for e in self.events:
            by_agent[e["agent_name"]] = by_agent.get(e["agent_name"], 0) + 1
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        return {"events": len(self.events), "by_agent": by_agent, "by_type": by_type}

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps({"summary": self.summary(), "events": self.events}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


TRAJECTORY = TrajectoryLogger()


def call_signature(name: str, args: dict) -> str:
    return f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"


def _schema_of(tool_obj: BaseTool | None):
    schema = getattr(tool_obj, "args_schema", None)
    return schema if isinstance(schema, type) and issubclass(schema, BaseModel) else None


def execute_tool_call(agent_name: str, call: dict, tools_by_name: dict[str, BaseTool]) -> tuple[str, dict | None]:
    """Виконує один tool call: tool guardrail → HITL (interrupt) для ризикових → виклик. Повертає (observation, hitl-рішення)."""
    name, args = call["name"], call["args"]
    tool_obj = tools_by_name.get(name)
    verdict = tool_guardrail(agent_name, name, args, _schema_of(tool_obj))
    if not verdict.allowed:
        return _err(f"GUARDRAIL: {verdict.reason}", name), None
    args, decision = verdict.normalized_args, None
    if name in RISKY_TOOLS:  # --- Human-in-the-Loop: стан зберігається у checkpointer, граф чекає Command(resume=...)
        answer = interrupt({"agent": agent_name, "tool": name, "args": args,
                            "question": f"Дозволити {name}({json.dumps(args, ensure_ascii=False)})?"})
        approved = bool(answer.get("approved"))
        if approved and answer.get("args"):  # сценарій edit: людина відредагувала аргументи
            edited = tool_guardrail(agent_name, name, {**args, **answer["args"]}, _schema_of(tool_obj))
            if not edited.allowed:
                return _err(f"GUARDRAIL після редагування: {edited.reason}", name), None
            args = edited.normalized_args
        decision = {"tool": name, "args": args, "approved": approved, "comment": answer.get("comment", ""),
                    "edited": bool(answer.get("args"))}
        log_security_event("hitl", "approved" if approved else "rejected", call_signature(name, args), agent=agent_name)
        if not approved:
            return _err(f"ВІДХИЛЕНО людиною: {decision['comment'] or 'без коментаря'}", name), decision
    return try_tool(tool_obj, args), decision


def react_loop(agent_name: str, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str,
               messages: list[AnyMessage], config: AgentConfig = DEFAULT_CONFIG,
               logger: TrajectoryLogger = TRAJECTORY) -> dict:
    """ReAct-цикл ДЗ1: LLM ⇄ tools з max_steps, timeout, детекцією зациклення та журналом траєкторії."""
    mark = len(logger.events)
    try:
        return _react_loop(agent_name, llm, tools, system_prompt, messages, config, logger)
    except GraphInterrupt:  # вузол буде виконано повторно після resume — не дублюємо події у журналі
        del logger.events[mark:]
        raise


def _react_loop(agent_name, llm, tools, system_prompt, messages, config, logger) -> dict:
    tools_by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    history = list(messages)
    started, seen, tools_used, hitl = time.perf_counter(), set(), [], []
    step, halted, halt_reason = 0, False, ""
    while True:
        elapsed = time.perf_counter() - started
        if step >= config.max_steps:
            halted, halt_reason = True, f"max_steps: досягнуто ліміт {config.max_steps} ітерацій"
        elif elapsed > config.timeout_s:
            halted, halt_reason = True, f"timeout: перевищено {config.timeout_s:.2f} с (пройшло {elapsed:.2f} с)"
        if halted:
            logger.log(agent_name, step, "guardrail", reason=halt_reason)
            break
        t0 = time.perf_counter()
        response = bound.invoke([SystemMessage(system_prompt), *history])
        response.name = agent_name
        step += 1
        history.append(response)
        logger.log(agent_name, step, "thought", content=response.content,
                   tool_calls=[{"name": c["name"], "args": c["args"]} for c in response.tool_calls],
                   llm_latency_s=round(time.perf_counter() - t0, 3))
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            signature, decision = call_signature(call["name"], call["args"]), None
            if config.detect_loops and signature in seen:
                halted, halt_reason = True, f"loop: повторний виклик {signature}"
                observation = _err("Цей виклик уже виконувався — змініть аргументи або завершіть відповідь", call["name"])
            else:
                seen.add(signature)
                observation, decision = execute_tool_call(agent_name, call, tools_by_name)
            if decision:
                hitl.append(decision)
            tools_used.append(call["name"])
            history.append(ToolMessage(content=observation, name=call["name"], tool_call_id=call["id"]))
            logger.log(agent_name, step, "observation", tool=call["name"], args=call["args"], observation=observation[:400],
                       is_error=parse_tool_json(observation).get("status") != "ok", risky=call["name"] in RISKY_TOOLS,
                       hitl=decision)
        if halted:
            logger.log(agent_name, step, "guardrail", reason=halt_reason)
            break
    last = history[-1]
    final = last.content if isinstance(last, AIMessage) and not halted else ""
    if halted:
        final = (f"Виконання зупинено захисним механізмом → {halt_reason}. "
                 f"Часткові дані отримано інструментами: {', '.join(dict.fromkeys(tools_used)) or 'немає'}.")
    logger.log(agent_name, step, "final_answer", answer=final)
    return {"final": final, "messages": history[len(messages):], "tools_used": tools_used, "steps": step,
            "halted": halted, "halt_reason": halt_reason, "hitl": hitl}
