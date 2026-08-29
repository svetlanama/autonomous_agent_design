"""Підключення MCP-сервера до LangGraph: MultiServerMCPClient (stdio) + синхронні обгортки tools / resources / prompts."""

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field, create_model

MCP_SERVER_PATH = Path(__file__).with_name("mcp_server.py")


class BackgroundLoop:
    """Окремий event loop у фоновому потоці: у ньому живе stdio-сесія MCP (Jupyter має власний loop)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="mcp-loop", daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


BACKGROUND_LOOP = BackgroundLoop()


def run_sync(coro):
    """Виконує корутину синхронно з будь-якого контексту (ноутбук, pytest, графи)."""
    return BACKGROUND_LOOP.run(coro)


_JSON_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def schema_from_json(name: str, schema: dict) -> type[BaseModel]:
    """Будує Pydantic-модель з JSON-схеми MCP-tool, щоб tool guardrail валідував аргументи на боці клієнта."""
    fields: dict[str, Any] = {}
    required = set(schema.get("required", []))
    for key, spec in schema.get("properties", {}).items():
        spec = dict(spec)
        py_type = Literal[tuple(spec["enum"])] if "enum" in spec else _JSON_TYPES.get(spec.get("type"), str)
        constraints = {k: spec[k] for k in ("min_length", "max_length", "pattern", "ge", "le") if k in spec}
        constraints.update({"min_length": spec.get("minLength", constraints.get("min_length")),
                            "max_length": spec.get("maxLength", constraints.get("max_length")),
                            "ge": spec.get("minimum", constraints.get("ge")), "le": spec.get("maximum", constraints.get("le"))})
        constraints = {k: v for k, v in constraints.items() if v is not None}
        default = ... if key in required else spec.get("default")
        fields[key] = (py_type, Field(default, description=spec.get("description", ""), **constraints))
    return create_model(f"{name}_args", **fields)


class McpSession:
    """Постійна stdio-сесія до окремого процесу mcp_server.py: один процес на весь запуск, стан сервера зберігається."""

    def __init__(self, server_path: Path = MCP_SERVER_PATH):
        self.client = MultiServerMCPClient({"sre": {"command": sys.executable, "args": [str(server_path)], "transport": "stdio"}})
        self.session = None
        self.async_tools: list[BaseTool] = []
        self.tools: list[BaseTool] = []
        self._ready, self._stop, self._task = asyncio.Event(), asyncio.Event(), None

    async def _lifetime(self) -> None:
        # anyio вимагає входити і виходити з контексту сесії в одній задачі — тримаємо її відкритою у власній task.
        async with self.client.session("sre") as session:
            self.session = session
            self.async_tools = await load_mcp_tools(session)
            self._ready.set()
            await self._stop.wait()

    async def _open(self) -> None:
        self._task = asyncio.create_task(self._lifetime())
        await self._ready.wait()

    async def _close(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def open(self) -> "McpSession":
        run_sync(self._open())
        self.tools = [self._sync_tool(t) for t in self.async_tools]
        return self

    def close(self) -> None:
        run_sync(self._close())

    def _sync_tool(self, tool_obj: BaseTool) -> StructuredTool:
        """Синхронна обгортка MCP-tool: граф LangGraph лишається синхронним (SqliteSaver, interrupt), виклик іде у фоновий loop."""
        schema = tool_obj.args_schema if isinstance(tool_obj.args_schema, type) else schema_from_json(tool_obj.name, tool_obj.args_schema)

        def call(**kwargs) -> str:
            result = run_sync(tool_obj.ainvoke(kwargs))
            if isinstance(result, list):
                result = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
            try:  # компактний однорядковий JSON — зручно для журналу та текстових ReAct-транскриптів (CrewAI)
                return json.dumps(json.loads(result) if isinstance(result, str) else result, ensure_ascii=False)
            except (ValueError, TypeError):
                return str(result)

        return StructuredTool(name=tool_obj.name, description=tool_obj.description, args_schema=schema, func=call)

    # --- resources & prompts ---
    def list_resources(self) -> list[dict]:
        static = run_sync(self.session.list_resources()).resources
        templates = run_sync(self.session.list_resource_templates()).resourceTemplates
        return ([{"uri": str(r.uri), "name": r.name, "description": r.description} for r in static]
                + [{"uri": t.uriTemplate, "name": t.name, "description": t.description} for t in templates])

    def read_resource(self, uri: str) -> str:
        return run_sync(self.session.read_resource(uri)).contents[0].text

    def list_prompts(self) -> list[dict]:
        return [{"name": p.name, "description": p.description, "arguments": [a.name for a in (p.arguments or [])]}
                for p in run_sync(self.session.list_prompts()).prompts]

    def get_prompt(self, name: str, arguments: dict) -> str:
        return run_sync(self.session.get_prompt(name, arguments)).messages[0].content.text


if __name__ == "__main__":
    session = McpSession().open()
    print("MCP tools:", [t.name for t in session.tools])
    print("MCP resources:", [r["uri"] for r in session.list_resources()])
    print("MCP prompts:", [p["name"] for p in session.list_prompts()])
    print(session.tools[0].invoke({"service": "payments"}))
    print(session.read_resource("service://postgres-db"))
    print(session.get_prompt("incident_report", {"service": "payments", "severity": "P1"})[:120])
    session.close()
