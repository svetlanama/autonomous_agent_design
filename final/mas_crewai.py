"""Бонус: той самий кейс у CrewAI (billing / tech / researcher) з тими ж tools, guardrails і HITL-gate + порівняння з LangGraph."""

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM as CrewBaseLLM
from crewai.tools import BaseTool as CrewBaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from guardrails import AGENT_TOOL_ALLOWLIST, input_guardrail, log_security_event, output_guardrail, tool_guardrail
from mcp_client import McpSession
from observability import LocalTracer, setup_tracing
from offline_llm import ScriptedLLM, build_llm
from react import RISKY_TOOLS
from tools import LOCAL_TOOLS, try_tool

OUTPUT_DIR = Path(__file__).parent
COMPARISON_PATH = OUTPUT_DIR / "comparison.json"
CREW_TRACER = LocalTracer("crewai-mas")

CASE = [  # той самий кейс, що у LangGraph-демо (mas_langgraph.DEMO_QUERIES)
    ("tech", "Перевір статус payments, подивись логи помилок і перезапусти сервіс"),
    ("researcher", "Що робити при деградації payments за runbook і яка політика перезапуску?"),
    ("billing", "Клієнт acme: сервіс payments простоював 90 хв за 30 днів — яка компенсація по SLA?"),
]


class ApprovalGate:
    """Черга рішень людини для ризикових tools (у CrewAI немає interrupt/resume — gate у tool-обгортці)."""

    def __init__(self):
        self.decisions: list[dict] = []
        self.log: list[dict] = []

    def preset(self, approved: bool, comment: str = "") -> None:
        self.decisions.append({"approved": approved, "comment": comment})

    def ask(self, tool: str, args: dict) -> dict:
        decision = self.decisions.pop(0) if self.decisions else {"approved": False, "comment": "рішення людини відсутнє"}
        self.log.append({"tool": tool, "args": args, **decision})
        log_security_event("hitl", "approved" if decision["approved"] else "rejected", f"{tool}({args})", framework="crewai")
        print(f"  ⏸  HITL(CrewAI): {tool}({json.dumps(args, ensure_ascii=False)}) → {'СХВАЛЕНО' if decision['approved'] else 'ВІДХИЛЕНО'} {decision['comment']}")
        return decision


APPROVAL_GATE = ApprovalGate()


class GuardedTool(CrewBaseTool):
    """CrewAI-tool поверх LangChain/MCP-tool з tool guardrail та HITL-gate для ризикових дій."""

    name: str
    description: str
    agent_name: str
    args_schema: type[BaseModel]
    inner: BaseTool

    def _run(self, **kwargs) -> str:
        verdict = tool_guardrail(self.agent_name, self.name, kwargs, self.args_schema)
        with CREW_TRACER.span(self.name, "tool", input_chars=len(json.dumps(kwargs, ensure_ascii=False)), agent=self.agent_name) as sp:
            if not verdict.allowed:
                text = json.dumps({"status": "error", "error": f"GUARDRAIL: {verdict.reason}"}, ensure_ascii=False)
            elif self.name in RISKY_TOOLS and not APPROVAL_GATE.ask(self.name, verdict.normalized_args)["approved"]:
                text = json.dumps({"status": "error", "error": "ВІДХИЛЕНО людиною"}, ensure_ascii=False)
            else:
                text = try_tool(self.inner, verdict.normalized_args)
            sp.output_chars = len(text)
        return text


def make_crew_tools(agent_name: str, all_tools: dict[str, BaseTool]) -> list[GuardedTool]:
    """Tools агента лише з його allowlist (той самий принцип найменших привілеїв)."""
    return [GuardedTool(name=n, description=all_tools[n].description, agent_name=agent_name, args_schema=all_tools[n].args_schema, inner=all_tools[n])
            for n in sorted(AGENT_TOOL_ALLOWLIST[agent_name]) if n in all_tools]


class ScriptedCrewLLM(CrewBaseLLM):
    """Офлайн-фолбек для CrewAI: перетворює ReAct-транскрипт CrewAI на повідомлення й делегує логіку ScriptedLLM."""

    def __init__(self):
        super().__init__(model="scripted-crew-offline")
        self._brain = ScriptedLLM()

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None):
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        role = (re.search(r"You are (\w+)", system) or re.search(r"(billing|tech|researcher)", system))
        role = role.group(1).lower() if role else "general"
        task_text = next((m["content"] for m in msgs if m["role"] == "user" and "Current Task" in m["content"]), "")
        query = re.search(r"Current Task:\s*(.+?)(?:\n\n|$)", task_text, re.S)
        history = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
        actions = re.findall(r"Action: (\w+)\nAction Input: (\{.*?\})", history, re.S)
        observations = re.findall(r"Observation: (\{.*?\})(?=\n|$)", history, re.S)
        fake = [SystemMessage(f"[[role={role}]]"), HumanMessage(query.group(1) if query else task_text)]
        for i, ((name, args), obs) in enumerate(zip(actions, observations)):
            fake.append(AIMessage("", tool_calls=[{"name": name, "args": json.loads(args), "id": f"c{i}"}], name=role))
            fake.append(ToolMessage(obs, name=name, tool_call_id=f"c{i}"))
        step = self._brain._agent_step(role, fake)
        if step.tool_calls:
            call = step.tool_calls[0]
            answer = f"{step.content}\nAction: {call['name']}\nAction Input: {json.dumps(call['args'], ensure_ascii=False)}"
        else:
            answer = f"Thought: I now know the final answer\nFinal Answer: {step.content}"
        CREW_TRACER.record_llm(sum(len(str(m.get("content", ""))) for m in msgs), len(answer))
        return answer

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 128_000


def build_crew_llm():
    llm = build_llm()
    if isinstance(llm, ScriptedLLM):
        return ScriptedCrewLLM()
    from crewai import LLM

    return LLM(model=f"openai/{llm.model_name}", temperature=0)


def build_crew(mcp_tools: list[BaseTool], llm=None) -> Crew:
    """Той самий кейс у CrewAI: 3 агенти, 3 послідовні задачі з контекстом (allow_delegation=False — без циклів)."""
    llm = llm or build_crew_llm()
    all_tools = {t.name: t for t in [*LOCAL_TOOLS, *mcp_tools]}
    specs = {
        "tech": ("Діагностувати інцидент і безпечно усунути його (перезапуск лише після підтвердження)", "On-call інженер; єдиний з доступом до restart_service."),
        "researcher": ("Знайти runbook і політики, релевантні інциденту", "Інженер бази знань; має доступ лише до search_knowledge."),
        "billing": ("Розрахувати доступність і SLA-компенсацію клієнту", "Фінансовий аналітик SRE; має доступ лише до білінгу та SLA-калькуляторів."),
    }
    agents = {name: Agent(role=name, goal=goal, backstory=story, tools=make_crew_tools(name, all_tools), llm=llm,
                          verbose=False, max_iter=6, allow_delegation=False) for name, (goal, story) in specs.items()}
    tasks, previous = [], []
    for agent_name, query in CASE:
        task = Task(description=query, expected_output="Стисла відповідь українською з фактами з інструментів", agent=agents[agent_name], context=list(previous))
        tasks.append(task)
        previous.append(task)
    return Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=False)


def kickoff_blocking(crew: Crew, inputs: dict | None = None):
    """CrewAI забороняє синхронний kickoff усередині запущеного event loop (Jupyter) — виконуємо в окремому потоці."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return crew.kickoff(inputs=inputs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(crew.kickoff, inputs=inputs).result()


def run_crew(mcp_tools: list[BaseTool], approve: bool = True) -> dict:
    """Запуск CrewAI з тими ж guardrails: input guardrail на кожну задачу → crew → output guardrail."""
    for _, query in CASE:
        verdict = input_guardrail(query)
        if not verdict.allowed:
            return {"blocked": True, "final_answer": f"Запит відхилено input guardrail-ом ({', '.join(verdict.reasons)})", "tasks": []}
    APPROVAL_GATE.preset(approve, "on-call схвалив" if approve else "on-call відхилив")
    crew = build_crew(mcp_tools)
    started = time.perf_counter()
    with CREW_TRACER.span("crew.kickoff", "chain"):
        result = kickoff_blocking(crew)
    outputs = [{"agent": o.agent, "raw": output_guardrail(o.raw).text} for o in result.tasks_output]
    return {"blocked": False, "tasks": outputs, "duration_s": round(time.perf_counter() - started, 3),
            "llm_calls": CREW_TRACER.llm_calls, "tool_calls": CREW_TRACER.tool_calls, "hitl": list(APPROVAL_GATE.log)}


def run_langgraph_case(mcp_tools: list[BaseTool]) -> dict:
    """Той самий кейс через LangGraph MAS з трейсером — для чесного порівняння чисел."""
    from mas_langgraph import build_mas_graph, run_mas

    callbacks, tracer = setup_tracing("langgraph-case")
    graph = build_mas_graph(mcp_tools)
    started = time.perf_counter()
    answers = [run_mas(graph, q, f"compare-{a}", approve=True, comment="on-call схвалив", callbacks=callbacks, quiet=True)["final_answer"] for a, q in CASE]
    return {"duration_s": round(time.perf_counter() - started, 3), "llm_calls": tracer.llm_calls, "tool_calls": tracer.tool_calls, "answers": answers}


def count_loc(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#"))


def compare(mcp_tools: list[BaseTool]) -> dict:
    crew = run_crew(mcp_tools)
    lg = run_langgraph_case(mcp_tools)
    table = {
        "LOC оркестрації": {"langgraph": count_loc(OUTPUT_DIR / "mas_langgraph.py") + count_loc(OUTPUT_DIR / "plan_execute.py"),
                            "crewai": count_loc(OUTPUT_DIR / "mas_crewai.py")},
        "Викликів LLM за кейс (3 запити)": {"langgraph": lg["llm_calls"], "crewai": crew["llm_calls"]},
        "Викликів tools за кейс": {"langgraph": lg["tool_calls"], "crewai": crew["tool_calls"]},
        "Час виконання, с (офлайн-LLM)": {"langgraph": lg["duration_s"], "crewai": crew["duration_s"]},
        "Маршрутизація": {"langgraph": "supervisor + with_structured_output(RouteDecision)", "crewai": "Process.sequential з context між задачами"},
        "Стан між агентами": {"langgraph": "TypedDict + reducers + SqliteSaver", "crewai": "текст TaskOutput у context"},
        "HITL": {"langgraph": "interrupt() / Command(resume) з паузою стану", "crewai": "ApprovalGate у tool-обгортці (без паузи)"},
        "Plan-and-Execute": {"langgraph": "підграф planner→executor→replanner", "crewai": "немає (ReAct у агента)"},
    }
    return {"table": table, "langgraph": lg, "crewai": crew}


def main() -> None:
    session = McpSession().open()
    report = compare(session.tools)
    session.close()
    print("\nCrewAI — результати задач:")
    for t in report["crewai"]["tasks"]:
        print(f"  [{t['agent']}] {t['raw'][:150]}")
    print("\nПорівняння LangGraph vs CrewAI:")
    for k, v in report["table"].items():
        print(f"  {k:34} | {str(v['langgraph']):48} | {v['crewai']}")
    COMPARISON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено {COMPARISON_PATH.name}")


if __name__ == "__main__":
    main()
