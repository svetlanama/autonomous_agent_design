"""ДЗ2: Plan-and-Execute (planner → executor → replanner) поверх ReAct-кроку з ДЗ1; executor підтримує HITL через interrupt."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from react import DEFAULT_CONFIG, TRAJECTORY, AgentConfig, TrajectoryLogger, react_loop


class Plan(BaseModel):
    """Structured output планувальника."""

    steps: list[str] = Field(min_length=1, max_length=10, description="Кроки плану у порядку виконання, кожен — одна дія")


class ReplanDecision(BaseModel):
    """Structured output replanner-а після виконання чергового кроку."""

    action: Literal["continue", "revise", "finish"] = Field(
        description="continue — виконувати далі; revise — замінити залишок плану; finish — дати фінальну відповідь")
    steps: list[str] = Field(default_factory=list, description="Новий залишок плану (лише для revise)")
    response: str = Field(default="", description="Фінальна відповідь користувачу (лише для finish)")
    reason: str = Field(description="Коротке пояснення рішення")


PLANNER_PROMPT = """Ти — планувальник tech-агента SRE-центру. Склади мінімальний покроковий план розслідування інциденту.
Доступні дії: перевірити статус сервісу, переглянути логи помилок, відкрити тікет інциденту, перезапустити сервіс.
Правила: один крок — одна дія; перезапуск — лише якщо користувач явно просить; останній крок — підсумок для користувача.
Відповідай structured output-ом Plan."""

REPLANNER_PROMPT = """Ти — replanner tech-агента. Проаналізуй виконані кроки та залишок плану і виріши:
- continue — план актуальний і кроки ще залишились;
- revise — останній результат містить помилку або план застарів (передай новий залишок плану у steps);
- finish — даних достатньо або дію відхилила людина (передай фінальну відповідь у response).
Відповідай structured output-ом ReplanDecision."""


class PlanExecState(TypedDict, total=False):
    input: str
    plan: list[str]
    initial_plan: list[str]
    past_steps: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    hitl: Annotated[list[dict], operator.add]
    response: str


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {s}" for s in items) or "(порожньо)"


def _past(past: list[dict]) -> str:
    return "\n".join(f"- {p['step']} → {p['result']}" for p in past) or "(порожньо)"


def build_plan_execute_graph(agent_name: str, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str,
                             config: AgentConfig = DEFAULT_CONFIG, logger: TrajectoryLogger = TRAJECTORY, checkpointer=None):
    """Збирає граф planner → executor → replanner; executor виконує один крок плану ReAct-циклом з ДЗ1."""
    planner_llm = llm.with_structured_output(Plan)
    replanner_llm = llm.with_structured_output(ReplanDecision)
    step_config = config.model_copy(update={"max_steps": 4})

    def planner(state: PlanExecState) -> dict:
        plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(state["input"])])
        steps = plan.steps[:config.max_plan_steps]
        logger.log(agent_name, 0, "plan", steps=steps)
        return {"plan": steps, "initial_plan": steps, "decisions": [{"node": "planner", "action": "plan", "steps": steps}]}

    def executor(state: PlanExecState) -> dict:
        step = state["plan"][0]
        prompt = (f"Запит користувача: {state['input']}\nКонтекст виконаних кроків:\n{_past(state['past_steps'])}\n\n"
                  f"Поточний крок: {step}")
        result = react_loop(agent_name, llm, tools, system_prompt, [HumanMessage(prompt)], step_config, logger)
        return {"plan": state["plan"][1:], "hitl": result["hitl"],
                "past_steps": [{"step": step, "tools": result["tools_used"], "react_steps": result["steps"], "result": result["final"]}]}

    def replanner(state: PlanExecState) -> dict:
        if len(state["past_steps"]) >= config.max_total_steps:
            reason = f"guardrail: досягнуто ліміт {config.max_total_steps} кроків"
            logger.log(agent_name, len(state["past_steps"]), "guardrail", reason=reason)
            return {"response": f"Зупинено захисним механізмом → {reason}", "decisions": [{"node": "replanner", "action": "finish", "reason": reason}]}
        prompt = (f"Запит користувача: {state['input']}\nЗалишок плану:\n{_bullets(state['plan'])}\n\n"
                  f"Виконані кроки:\n{_past(state['past_steps'])}\n\nВиріши: continue / revise / finish.")
        decision = replanner_llm.invoke([SystemMessage(REPLANNER_PROMPT), HumanMessage(prompt)])
        event = {"node": "replanner", "action": decision.action, "reason": decision.reason}
        logger.log(agent_name, len(state["past_steps"]), "replan", action=decision.action, reason=decision.reason)
        if decision.action == "finish":
            return {"response": decision.response, "decisions": [event]}
        if decision.action == "revise":
            return {"plan": decision.steps[:config.max_plan_steps], "decisions": [event | {"steps": decision.steps}]}
        return {"decisions": [event]}

    def after_replanner(state: PlanExecState) -> Literal["executor", "__end__"]:
        return "__end__" if state.get("response") else "executor"

    graph = StateGraph(PlanExecState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("replanner", replanner)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", after_replanner, {"executor": "executor", "__end__": END})
    return graph.compile(checkpointer=checkpointer)


def initial_plan_state(query: str) -> dict:
    return {"input": query, "plan": [], "initial_plan": [], "past_steps": [], "decisions": [], "hitl": [], "response": ""}
