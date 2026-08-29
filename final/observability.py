"""Observability: LangSmith / Langfuse за наявності ключів + локальний трейсер (дерево spans → JSON/PNG) завжди."""

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

OUTPUT_DIR = Path(__file__).parent
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
HAS_LANGFUSE = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
HAS_LANGSMITH = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


@dataclass
class Span:
    span_id: str
    name: str
    kind: str  # chain | llm | tool
    parent_id: str | None
    start: float
    end: float | None = None
    input_chars: int = 0
    output_chars: int = 0
    meta: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000


class LocalTracer(BaseCallbackHandler):
    """Збирає spans LangGraph (chain → llm → tool) через callbacks; дає дерево, зведення та JSON."""

    HIDDEN = ("RunnableSequence", "RunnableLambda", "ChannelWrite", "_write", "__start__", "Branch", "RunnableCallable")

    def __init__(self, name: str):
        self.name = name
        self.spans: dict[str, Span] = {}
        self.roots: list[Span] = []
        self._stack: list[Span] = []
        self.llm_calls = self.tool_calls = self.prompt_chars = self.completion_chars = 0

    def _open(self, span_id: str, name: str, kind: str, parent_id: str | None, input_chars: int = 0, **meta) -> Span:
        span = Span(span_id, name, kind, parent_id, time.time(), input_chars=input_chars, meta=meta)
        self.spans[span_id] = span
        parent = self.spans.get(parent_id) if parent_id else None
        (parent.children if parent else self.roots).append(span)
        return span

    def _close(self, span_id: str, output_chars: int = 0, **meta) -> None:
        if span := self.spans.get(span_id):
            span.end, span.output_chars = time.time(), output_chars
            span.meta.update(meta)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kw):
        self._open(str(run_id), name or (serialized or {}).get("name") or "chain", "chain", str(parent_run_id) if parent_run_id else None)

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._close(str(run_id))

    def on_chain_error(self, error, *, run_id, **kw):
        self._close(str(run_id), error=type(error).__name__)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        chars = sum(len(str(m.content)) for batch in messages for m in batch)
        self.llm_calls += 1
        self.prompt_chars += chars
        self._open(str(run_id), (serialized or {}).get("name", "llm"), "llm", str(parent_run_id) if parent_run_id else None, chars)

    def on_llm_end(self, response, *, run_id, **kw):
        text = " ".join(g.text or json.dumps(getattr(g.message, "tool_calls", ""), ensure_ascii=False)
                        for gens in response.generations for g in gens)
        self.completion_chars += len(text)
        self._close(str(run_id), len(text))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        self.tool_calls += 1
        self._open(str(run_id), (serialized or {}).get("name", "tool"), "tool", str(parent_run_id) if parent_run_id else None, len(input_str))

    def on_tool_end(self, output, *, run_id, **kw):
        self._close(str(run_id), len(str(output)))

    def on_tool_error(self, error, *, run_id, **kw):
        self._close(str(run_id), error=type(error).__name__)

    @contextmanager
    def span(self, name: str, kind: str = "chain", input_chars: int = 0, **meta):
        """Ручний span (для CrewAI та власних вузлів)."""
        if kind == "tool":
            self.tool_calls += 1
        span = self._open(uuid.uuid4().hex, name, kind, self._stack[-1].span_id if self._stack else None, input_chars, **meta)
        self._stack.append(span)
        try:
            yield span
        finally:
            self._stack.pop()
            self._close(span.span_id, span.output_chars, **span.meta)

    def record_llm(self, prompt_chars: int, completion_chars: int) -> None:
        self.llm_calls += 1
        self.prompt_chars += prompt_chars
        self.completion_chars += completion_chars

    def render(self, max_depth: int = 7) -> str:
        lines: list[str] = []

        def walk(span: Span, depth: int) -> None:
            hidden = any(span.name.startswith(h) for h in self.HIDDEN)
            if not hidden:
                mark = {"llm": "🧠", "tool": "🔧"}.get(span.kind, "▸")
                extra = f" ({span.input_chars}→{span.output_chars} симв.)" if span.kind != "chain" else ""
                lines.append(f"{'  ' * depth}{mark} {span.name} — {span.duration_ms:.1f} мс{extra}")
            for child in span.children:
                if depth < max_depth:
                    walk(child, depth + (0 if hidden else 1))

        for root in self.roots:
            walk(root, 0)
        return "\n".join(lines)

    def summary(self) -> dict:
        return {"tracer": self.name, "spans": len(self.spans), "llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
                "prompt_chars": self.prompt_chars, "completion_chars": self.completion_chars,
                "total_ms": round(sum(r.duration_ms for r in self.roots), 1)}

    def to_json(self) -> dict:
        def dump(span: Span) -> dict:
            return {"name": span.name, "kind": span.kind, "duration_ms": round(span.duration_ms, 2), "input_chars": span.input_chars,
                    "output_chars": span.output_chars, "meta": span.meta, "children": [dump(c) for c in span.children]}
        return {"summary": self.summary(), "spans": [dump(r) for r in self.roots]}


def setup_tracing(run_name: str, project: str = "mas-sre-final") -> tuple[list, LocalTracer]:
    """Вмикає Langfuse / LangSmith (якщо є ключі) і завжди додає LocalTracer. Повертає (callbacks, local)."""
    local = LocalTracer(run_name)
    callbacks: list = [local]
    if HAS_LANGFUSE:
        from langfuse.langchain import CallbackHandler

        callbacks.append(CallbackHandler())
        print("Tracing: Langfuse увімкнено → https://cloud.langfuse.com")
    if HAS_LANGSMITH:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        print(f"Tracing: LangSmith увімкнено → https://smith.langchain.com (проєкт {os.environ['LANGSMITH_PROJECT']})")
    if not (HAS_LANGFUSE or HAS_LANGSMITH):
        print("Tracing: ключів Langfuse/LangSmith немає — LocalTracer (дерево spans + JSON + PNG)")
    return callbacks, local


def text_to_png(text: str, path: Path, title: str = "") -> Path:
    """«Скриншот» консольного виводу: рендерить текст у PNG (matplotlib, без GUI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import textwrap

    lines = [wrapped for line in text.splitlines() if line.strip()
             for wrapped in textwrap.wrap(line, width=150, subsequent_indent="      ", drop_whitespace=False)][:120]
    width = min(max(len(line) for line in lines) * 0.085 + 1, 14)
    fig, ax = plt.subplots(figsize=(width, 0.19 * len(lines) + (0.6 if title else 0.3)))
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color="#333")
    ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.5, va="top", ha="left", color="#0b0b0b", transform=ax.transAxes)
    fig.patch.set_facecolor("#fcfcfb")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    from mas_langgraph import DEMO_QUERIES, build_mas_graph, run_mas
    from mcp_client import McpSession

    session = McpSession().open()
    callbacks, tracer = setup_tracing("mas-langgraph")
    graph = build_mas_graph(session.tools)
    query = DEMO_QUERIES[1][1]
    print(f"▶ {query}")
    result = run_mas(graph, query, "trace-demo", approve=True, comment="on-call схвалив", callbacks=callbacks)
    tree = tracer.render()
    print(tree)
    print("Зведення:", tracer.summary())
    (OUTPUT_DIR / "trace_mas.json").write_text(json.dumps(tracer.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    text_to_png(f"▶ {query}\n{tree}\n\n{json.dumps(tracer.summary(), ensure_ascii=False)}",
                SCREENSHOTS_DIR / "observability_trace.png", "LocalTracer: дерево spans MAS (chain → llm → tool)")
    print(f"Збережено trace_mas.json і {SCREENSHOTS_DIR.name}/observability_trace.png; відповідь: {result['final_answer'][:80]}…")
    session.close()


if __name__ == "__main__":
    main()
