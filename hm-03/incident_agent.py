"""SRE-асистент з управління інцидентами: інструменти, ReAct, Plan-and-Execute (бібліотечна частина ноутбука)."""

import hashlib
import json
import math
import operator
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationInfo, field_validator

OUTPUT_DIR = Path.cwd()
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory.json"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints.sqlite"


class AgentConfig(BaseModel):
    """Налаштування агента та його захисних механізмів."""

    max_steps: int = Field(default=10, ge=1, le=50, description="Максимум ітерацій циклу LLM→tools")
    timeout_s: float = Field(default=120.0, gt=0, description="Загальний тайм-аут виконання, секунд")
    detect_loops: bool = Field(default=True, description="Зупиняти при повторному ідентичному tool call")
    max_plan_steps: int = Field(default=8, ge=1, le=20, description="Максимальна довжина плану")
    max_total_steps: int = Field(default=12, ge=1, le=50, description="Ліміт виконаних кроків плану")
    model_name: str = Field(default="gpt-4.1", description="Назва моделі провайдера")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


DEFAULT_CONFIG = AgentConfig()


def save_json(payload: Any, path: Path) -> None:
    """Зберігає структуру у JSON-файл з підтримкою кирилиці."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено: {path.name} ({path.stat().st_size / 1024:.1f} КБ)")


def fmt_duration(seconds: float) -> str:
    """Форматує тривалість: мілісекунди для швидких запусків, секунди — для повільних."""
    return f"{seconds * 1000:.1f} мс" if seconds < 1 else f"{seconds:.2f} с"


print(f"Python {sys.version.split()[0]}")
print(f"Guardrails: max_steps={DEFAULT_CONFIG.max_steps}, timeout={DEFAULT_CONFIG.timeout_s:.0f} с, "
      f"детекція повторів={DEFAULT_CONFIG.detect_loops}")


# Стан сервісів: імітація системи моніторингу.
SERVICES_DB = {
    "api-gateway": {"status": "healthy", "latency_p95_ms": 120, "error_rate_percent": 0.2, "replicas": 3},
    "auth-service": {"status": "healthy", "latency_p95_ms": 95, "error_rate_percent": 0.1, "replicas": 2},
    "payments": {"status": "degraded", "latency_p95_ms": 1850, "error_rate_percent": 7.4, "replicas": 2},
    "postgres-db": {"status": "healthy", "latency_p95_ms": 14, "error_rate_percent": 0.0, "replicas": 1},
    "notifications": {"status": "down", "latency_p95_ms": 0, "error_rate_percent": 100.0, "replicas": 0},
}

# Логи: імітація лог-колектора. Для notifications логи навмисно відсутні (демонстрація fallback-стратегії).
LOGS_DB = {
    "api-gateway": [
        {"minutes_ago": 3, "level": "WARN", "message": "upstream payments timeout after 2000ms"},
        {"minutes_ago": 12, "level": "INFO", "message": "config reloaded"},
    ],
    "auth-service": [
        {"minutes_ago": 40, "level": "INFO", "message": "token cache warmed"},
    ],
    "payments": [
        {"minutes_ago": 2, "level": "ERROR", "message": "connection pool exhausted (max=20)"},
        {"minutes_ago": 5, "level": "ERROR", "message": "PSP callback timeout: stripe"},
        {"minutes_ago": 9, "level": "WARN", "message": "retry budget 80% consumed"},
        {"minutes_ago": 25, "level": "ERROR", "message": "deadlock detected on table transactions"},
        {"minutes_ago": 70, "level": "INFO", "message": "deploy v2.14.0 finished"},
    ],
    "postgres-db": [
        {"minutes_ago": 26, "level": "WARN", "message": "long running query 4.2s from payments"},
    ],
}

# База знань «Runbook-и та політики SRE»: 10 документів для ChromaDB.
KNOWLEDGE_DOCS = {
    "runbook_api_gateway": "Runbook api-gateway. Симптоми: зростання p95 latency понад 500 мс, помилки 502/504. "
                           "Дії: перевірити upstream-сервіси (payments, auth-service), збільшити таймаут до 3 с, "
                           "за потреби масштабувати до 5 реплік. Перезапуск gateway допускається лише по одній репліці.",
    "runbook_auth_service": "Runbook auth-service. Симптоми: помилки 401 у здорових клієнтів, прострочені токени. "
                            "Дії: перевірити синхронізацію часу (NTP), прогріти кеш токенів, перевірити з'єднання з postgres-db. "
                            "Перезапуск безпечний у будь-який час — сервіс stateless.",
    "runbook_payments": "Runbook payments. Симптоми: connection pool exhausted, таймаути PSP, deadlock у таблиці transactions. "
                        "Дії: 1) перевірити статус та ERROR-логи; 2) збільшити пул з'єднань до 40; 3) якщо помилки тривають "
                        "понад 15 хвилин — перезапустити сервіс зі згодою чергового ліда; 4) відкрити інцидент рівня P1.",
    "runbook_postgres": "Runbook postgres-db. Симптоми: повільні запити понад 3 с, блокування, зростання реплікаційного лагу. "
                        "Дії: знайти довгі запити через pg_stat_activity, завершити їх pg_terminate_backend. "
                        "ПЕРЕЗАПУСК БАЗИ ДАНИХ ЗАБОРОНЕНО без затвердження DBA та вікна обслуговування.",
    "runbook_notifications": "Runbook notifications. Симптоми: черга повідомлень зростає, воркери не відповідають, статус down. "
                             "Дії: перевірити брокер повідомлень, перезапустити воркери, після відновлення повторно надіслати чергу. "
                             "Логи сервісу експортуються лише у S3 і недоступні у лог-колекторі — використовуйте статус сервісу.",
    "sla_policy": "Політика SLA. Цільова доступність для критичних сервісів (payments, api-gateway, auth-service) — 99.9% "
                  "за 30-денний період, тобто бюджет помилок 43.2 хвилини на місяць. Для некритичних сервісів "
                  "(notifications) — 99.5%. Якщо бюджет помилок вичерпано, релізи заморожуються до кінця періоду.",
    "escalation_policy": "Політика ескалації. P1 — повна недоступність платежів або автентифікації: повідомити чергового ліда "
                         "протягом 5 хвилин, зібрати war room. P2 — деградація одного сервісу: чергового інженера протягом 15 хвилин. "
                         "P3/P4 — тікет у беклог без негайної ескалації.",
    "restart_policy": "Політика перезапуску сервісів. Перезапуск — ризикова дія: він скидає з'єднання користувачів і може "
                      "втратити дані в пам'яті. Перед перезапуском обов'язково перевірити статус і логи сервісу, "
                      "отримати явне підтвердження людини (чергового інженера або ліда) та зафіксувати причину.",
    "postmortem_template": "Шаблон post-mortem. Розділи: 1) хронологія інциденту з часовими мітками; 2) вплив на користувачів "
                           "та SLA; 3) корінна причина (5 whys); 4) що спрацювало добре; 5) план дій з відповідальними. "
                           "Post-mortem пишеться протягом 48 годин після закриття інциденту, без пошуку винних.",
    "oncall_rotation": "Графік чергувань. Тиждень A — команда Platform (лід: Оксана), тиждень B — команда Payments (лід: Андрій). "
                       "Чергування триває 7 днів з понеділка 10:00. Контакт чергового — канал #oncall та PagerDuty.",
}

import chromadb
from chromadb.api.types import EmbeddingFunction


class HashingEmbedding(EmbeddingFunction):
    """Детермінований локальний ембеддинг без мережі: хешування слів і символьних 3-грам у вектор.

    3-грами покривають морфологію української («перезапуск» ↔ «перезапустити» мають спільні грами),
    md5-хеш дає стабільні індекси, тож результати пошуку відтворювані на будь-якій машині.
    """

    def __init__(self, dim: int = 512, word_weight: float = 3.0):
        self.dim = dim
        self.word_weight = word_weight

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = [t for t in re.findall(r"[\w-]+", text.lower()) if len(t) > 2]
            for token in tokens:
                vec[self._bucket(token)] += self.word_weight
                for i in range(max(len(token) - 2, 1)):
                    vec[self._bucket(token[i:i + 3])] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "hashing-embedding"

    def get_config(self) -> dict:
        return {"dim": self.dim, "word_weight": self.word_weight}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding(**config)


chroma_client = chromadb.EphemeralClient()
knowledge = chroma_client.get_or_create_collection(
    "sre_runbooks",
    embedding_function=HashingEmbedding(),
    metadata={"hnsw:space": "cosine"},
)
knowledge.upsert(ids=list(KNOWLEDGE_DOCS), documents=list(KNOWLEDGE_DOCS.values()))
print(f"ChromaDB-колекція '{knowledge.name}': {knowledge.count()} документів")

probe = knowledge.query(query_texts=["перезапуск сервісу payments"], n_results=3)
for doc_id, doc, dist in zip(probe["ids"][0], probe["documents"][0], probe["distances"][0]):
    print(f"  {doc_id:22} similarity={1 - dist:.3f}  {doc[:70]}…")


LOG_LEVELS = ("ERROR", "WARN", "INFO")
PROTECTED_SERVICES = {"postgres-db"}  # перезапуск заборонено політикою (див. runbook_postgres)


def _ok(data: dict) -> str:
    """Стандартний успішний результат інструмента."""
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def _err(message: str, tool_name: str = "") -> str:
    """Стандартний результат-помилка інструмента."""
    return json.dumps({"status": "error", "tool": tool_name, "error": message}, ensure_ascii=False)


def _normalize_service(value: str) -> str:
    service = value.strip().lower().replace("_", "-").replace(" ", "-")
    if service not in SERVICES_DB:
        raise ValueError(f"Невідомий сервіс '{value}'. Доступні: {', '.join(SERVICES_DB)}")
    return service


class SearchRunbooksArgs(BaseModel):
    """Параметри інструмента search_runbooks."""

    query: str = Field(description="Пошуковий запит до бази runbook-ів та SRE-політик")
    top_k: int = Field(default=2, ge=1, le=5, description="Скільки документів повернути (1..5)")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if len(query) < 3:
            raise ValueError("Запит закороткий (мінімум 3 символи)")
        if len(query) > 300:
            raise ValueError("Запит задовгий (максимум 300 символів)")
        return query


class ServiceStatusArgs(BaseModel):
    """Параметри інструмента get_service_status."""

    service: str = Field(description=f"Назва сервісу: {', '.join(SERVICES_DB)}")

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        return _normalize_service(value)


class QueryLogsArgs(BaseModel):
    """Параметри інструмента query_logs."""

    service: str = Field(description=f"Назва сервісу: {', '.join(SERVICES_DB)}")
    level: str = Field(default="ERROR", description=f"Мінімальний рівень: {', '.join(LOG_LEVELS)}")
    last_minutes: int = Field(default=30, ge=1, le=1440, description="Часове вікно у хвилинах (1..1440)")

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        return _normalize_service(value)

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"Невідомий рівень '{value}'. Доступні: {', '.join(LOG_LEVELS)}")
        return level


class CalculateSlaArgs(BaseModel):
    """Параметри інструмента calculate_sla. Порядок полів важливий: period_days валідується раніше за downtime_minutes."""

    service: str = Field(description=f"Назва сервісу: {', '.join(SERVICES_DB)}")
    period_days: int = Field(default=30, ge=1, le=365, description="Період розрахунку у днях (1..365)")
    downtime_minutes: float = Field(ge=0, description="Сумарний простій за період, хвилин")
    target_percent: float = Field(default=99.9, ge=90.0, le=100.0, description="Цільова доступність, % (90..100)")

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        return _normalize_service(value)

    @field_validator("downtime_minutes")
    @classmethod
    def validate_downtime(cls, value: float, info: ValidationInfo) -> float:
        period_minutes = info.data.get("period_days", 30) * 1440
        if value > period_minutes:
            raise ValueError(f"Простій {value} хв перевищує тривалість періоду ({period_minutes} хв)")
        return round(value, 2)


class RestartServiceArgs(BaseModel):
    """Параметри ризикового інструмента restart_service."""

    service: str = Field(description=f"Назва сервісу для перезапуску: {', '.join(SERVICES_DB)}")
    reason: str = Field(description="Причина перезапуску для журналу інцидентів (мінімум 10 символів)")

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        service = _normalize_service(value)
        if service in PROTECTED_SERVICES:
            raise ValueError(f"Перезапуск '{service}' заборонено політикою: потрібне вікно обслуговування та DBA")
        return service

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        reason = value.strip()
        if len(reason) < 10:
            raise ValueError("Причина перезапуску має містити щонайменше 10 символів")
        return reason


@tool("search_runbooks", args_schema=SearchRunbooksArgs)
def search_runbooks(query: str, top_k: int = 2) -> str:
    """Шукає у базі знань runbook-и та SRE-політики (RAG). Використовуй для інструкцій «що робити», політик SLA, ескалації, перезапуску."""
    found = knowledge.query(query_texts=[query], n_results=top_k)
    documents = [
        {"doc_id": doc_id, "similarity": round(1 - dist, 3), "text": doc}
        for doc_id, doc, dist in zip(found["ids"][0], found["documents"][0], found["distances"][0])
    ]
    return _ok({"query": query, "documents": documents})


@tool("get_service_status", args_schema=ServiceStatusArgs)
def get_service_status(service: str) -> str:
    """Повертає поточний стан сервісу з моніторингу: статус, p95 latency, частку помилок, кількість реплік."""
    return _ok({"service": service, **SERVICES_DB[service]})


@tool("query_logs", args_schema=QueryLogsArgs)
def query_logs(service: str, level: str = "ERROR", last_minutes: int = 30) -> str:
    """Повертає записи логів сервісу не нижче заданого рівня за останні N хвилин."""
    if service not in LOGS_DB:
        raise RuntimeError(f"Логи сервісу '{service}' недоступні у лог-колекторі")
    allowed = LOG_LEVELS[:LOG_LEVELS.index(level) + 1]
    entries = [e for e in LOGS_DB[service] if e["level"] in allowed and e["minutes_ago"] <= last_minutes]
    return _ok({"service": service, "level": level, "last_minutes": last_minutes,
                "count": len(entries), "entries": entries})


@tool("calculate_sla", args_schema=CalculateSlaArgs)
def calculate_sla(service: str, downtime_minutes: float, period_days: int = 30, target_percent: float = 99.9) -> str:
    """Рахує фактичну доступність сервісу за період, бюджет помилок та чи порушено SLA."""
    period_minutes = period_days * 1440
    availability = (1 - downtime_minutes / period_minutes) * 100
    budget = (1 - target_percent / 100) * period_minutes
    return _ok({
        "service": service, "period_days": period_days, "downtime_minutes": downtime_minutes,
        "availability_percent": round(availability, 4), "target_percent": target_percent,
        "error_budget_minutes": round(budget, 1), "budget_remaining_minutes": round(budget - downtime_minutes, 1),
        "sla_breached": availability < target_percent,
    })


@tool("restart_service", args_schema=RestartServiceArgs)
def restart_service(service: str, reason: str) -> str:
    """РИЗИКОВА ДІЯ: перезапускає сервіс (скидає з'єднання користувачів). Потребує підтвердження людини."""
    restart_id = hashlib.md5(f"{service}|{reason}".encode()).hexdigest()[:8].upper()
    SERVICES_DB[service] = {**SERVICES_DB[service], "status": "restarting", "error_rate_percent": 0.0}
    return _ok({"restart_id": restart_id, "service": service, "reason": reason,
                "status": "restarting", "expected_downtime_s": 20})


TOOLS: list[BaseTool] = [search_runbooks, get_service_status, query_logs, calculate_sla, restart_service]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
RISKY_TOOLS = {"restart_service"}
# Fallback-стратегія: інструмент → (альтернативний інструмент, перетворення аргументів).
FALLBACK_TOOLS = {"query_logs": ("get_service_status", lambda args: {"service": args.get("service", "")})}


def try_tool(name: str, args: dict) -> str:
    """Викликає інструмент; будь-яку помилку валідації або виконання перетворює на JSON {status: error}."""
    if name not in TOOLS_BY_NAME:
        return _err(f"Невідомий інструмент '{name}'. Доступні: {', '.join(TOOLS_BY_NAME)}", name)
    try:
        return TOOLS_BY_NAME[name].invoke(args)
    except Exception as error:
        reasons = list(dict.fromkeys(re.findall(r"Value error, (.+?)(?: \[type=|$)", str(error))))
        details = "; ".join(reasons) or str(error).strip().splitlines()[0]
        return _err(f"[{type(error).__name__}] {details}", name)


def try_tool_with_fallback(name: str, args: dict) -> tuple[str, str | None]:
    """Викликає інструмент; при помилці пробує альтернативу з FALLBACK_TOOLS. Повертає (результат, назва fallback-а)."""
    result = try_tool(name, args)
    if json.loads(result)["status"] == "ok" or name not in FALLBACK_TOOLS:
        return result, None
    alt_name, transform = FALLBACK_TOOLS[name]
    alt_result = try_tool(alt_name, transform(args))
    merged = json.loads(result) | {"fallback": {"tool": alt_name, "result": json.loads(alt_result)}}
    return json.dumps(merged, ensure_ascii=False), alt_name


for tool_obj in TOOLS:
    marker = " [РИЗИКОВИЙ → HITL]" if tool_obj.name in RISKY_TOOLS else ""
    params = ", ".join(tool_obj.args_schema.model_fields)
    print(f"{tool_obj.name:20}{marker}\n{'':22}({params})\n{'':22}{tool_obj.description}")


class Plan(BaseModel):
    """Structured output планувальника: впорядкований список кроків."""

    steps: list[str] = Field(min_length=1, max_length=10,
                             description="Кроки плану у порядку виконання, кожен — одна конкретна дія")


class ReplanDecision(BaseModel):
    """Structured output replanner-а: рішення після виконання чергового кроку."""

    action: Literal["continue", "revise", "finish"] = Field(
        description="continue — виконувати план далі; revise — замінити залишок плану; finish — дати фінальну відповідь")
    steps: list[str] = Field(default_factory=list, description="Новий залишок плану (лише для revise)")
    response: str = Field(default="", description="Фінальна відповідь користувачу (лише для finish)")
    reason: str = Field(description="Коротке пояснення рішення")


REACT_SYSTEM_PROMPT = f"""Ти — SRE-асистент чергового інженера. Працюй за циклом ReAct:
Thought: коротко поміркуй, яких даних бракує для діагностики інциденту.
Action: виклич потрібний інструмент (кілька одразу, якщо вони незалежні).
Observation: проаналізуй JSON-результат інструмента (status: ok / error).
Повторюй, доки не зможеш дати повну відповідь.

Інструменти: {', '.join(TOOLS_BY_NAME)}.
Правила:
- Факти про стан сервісів, логи та SLA бери лише з інструментів, не вигадуй.
- Інструкції «що робити», політики SLA/ескалації/перезапуску шукай у базі знань (search_runbooks) —
  але не звертайся до неї для простих запитів про статус чи арифметику SLA.
- Не викликай той самий інструмент з тими самими аргументами двічі.
- Якщо інструмент повернув status=error — прочитай текст помилки та виправ аргументи або обери інший шлях.
- restart_service — ризикова дія: викликай лише коли користувач явно просить перезапуск.
- Коли даних достатньо, дай фінальну відповідь українською мовою."""

PLANNER_PROMPT = f"""Ти — планувальник SRE-асистента. Склади мінімальний покроковий план розслідування інциденту.
Доступні інструменти: {', '.join(TOOLS_BY_NAME)}.
Правила:
- Кожен крок — одна конкретна дія, яку можна виконати одним інструментом, або підсумок.
- Інструкції та політики бери З БАЗИ ЗНАНЬ (search_runbooks), а не з пам'яті.
- Не додавай зайвих кроків: для простого запиту про статус база знань не потрібна.
- Останній крок — підсумок для користувача.
Відповідай structured output-ом Plan."""

REPLANNER_PROMPT = """Ти — replanner SRE-асистента. Проаналізуй виконані кроки та залишок плану і виріши:
- continue — якщо план актуальний і кроки ще залишились;
- revise — якщо останній результат містить помилку або план застарів (передай новий залишок плану у steps);
- finish — якщо даних достатньо або дію відхилила людина (передай фінальну відповідь у response).
Відповідай structured output-ом ReplanDecision."""

# Морфологічні стеми для розпізнавання сервісів у тексті (офлайн-провайдер).
SERVICE_STEMS: list[tuple[str, str]] = [
    ("api-gateway", "api-gateway"), ("gateway", "api-gateway"), ("гейтвей", "api-gateway"),
    ("auth-service", "auth-service"), ("auth", "auth-service"), ("автентиф", "auth-service"),
    ("payments", "payments"), ("платеж", "payments"), ("платіж", "payments"),
    ("postgres", "postgres-db"), ("база даних", "postgres-db"), ("бази даних", "postgres-db"),
    ("notifications", "notifications"), ("сповіщен", "notifications"),
    ("billing", "billing"),  # навмисно неіснуючий сервіс для демонстрації обробки помилок і replanning
]


def _services_in(text: str) -> list[str]:
    """Повертає сервіси, згадані у тексті, у порядку появи."""
    low = text.lower()
    found = sorted({(low.find(stem), service) for stem, service in SERVICE_STEMS if stem in low})
    return list(dict.fromkeys(service for _, service in found))


def _number_before(text: str, pattern: str, default: float) -> float:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*" + pattern, text.lower())
    return float(match.group(1).replace(",", ".")) if match else default


def _intent_calls(text: str, context_services: list[str] | None = None) -> list[list[tuple[str, dict]]]:
    """Перетворює текст задачі на етапи tool calls: [[status...], [logs...], [sla...], [runbook], [restart]]."""
    low = text.lower()
    services = _services_in(text) or (context_services or [])
    stages: list[list[tuple[str, dict]]] = []
    if any(k in low for k in ("статус", "стан ", "стан.", "перевір", "розслідуй", "деград", "не працює")):
        stages.append([("get_service_status", {"service": s}) for s in services])
    if any(k in low for k in ("лог", "помилк")):
        level = "WARN" if "warn" in low or "попереджен" in low else "ERROR"
        minutes = int(_number_before(text, r"хвилин", 30))
        stages.append([("query_logs", {"service": s, "level": level, "last_minutes": minutes})
                       for s in services if s in SERVICES_DB])
    if any(k in low for k in ("sla", "доступн", "бюджет помилок")):
        downtime = _number_before(text, r"хв(?:илин)?\.? простою", 45)
        period = int(_number_before(text, r"дн", 30))
        stages.append([("calculate_sla", {"service": s, "downtime_minutes": downtime, "period_days": period})
                       for s in services if s in SERVICES_DB][:1])
    if any(k in low for k in ("runbook", "ранбук", "інструкц", "що робити", "політик", "ескалац", "post-mortem", "шаблон")):
        query = f"runbook {' '.join(services)}: {text.strip()}" if services else text.strip()
        stages.append([("search_runbooks", {"query": query[:200], "top_k": 2})])
    if any(k in low for k in ("перезапуст", "рестарт", "restart")):
        reason = "Деградація сервісу за даними моніторингу та логів, запит чергового інженера"
        stages.append([("restart_service", {"service": s, "reason": reason}) for s in services if s in SERVICES_DB][:1])
    return [stage for stage in stages if stage]


def _describe_observation(name: str, payload: dict) -> str:
    """Перетворює JSON-результат інструмента на речення для фінальної відповіді."""
    if payload.get("status") != "ok":
        text = f"{name}: помилка — {payload.get('error', '?')}"
        if payload.get("fallback"):
            fb = payload["fallback"]
            text += f"; fallback {fb['tool']} → {_describe_observation(fb['tool'], fb['result'])}"
        return text
    d = payload["data"]
    if name == "get_service_status":
        return (f"{d['service']}: статус {d['status']}, p95 latency {d['latency_p95_ms']} мс, "
                f"помилки {d['error_rate_percent']}%, реплік {d['replicas']}")
    if name == "query_logs":
        last = d["entries"][0]["message"] if d["entries"] else "немає записів"
        return f"{d['service']}: {d['count']} записів рівня {d['level']}+ за {d['last_minutes']} хв, останній: «{last}»"
    if name == "calculate_sla":
        verdict = "SLA ПОРУШЕНО" if d["sla_breached"] else "SLA у нормі"
        return (f"{d['service']}: доступність {d['availability_percent']}% при цілі {d['target_percent']}%, "
                f"залишок бюджету помилок {d['budget_remaining_minutes']} хв — {verdict}")
    if name == "search_runbooks":
        top = d["documents"][0]
        return f"runbook «{top['doc_id']}» (similarity {top['similarity']}): {top['text'][:160]}…"
    if name == "restart_service":
        return f"{d['service']} перезапущено (id {d['restart_id']}), очікуваний простій {d['expected_downtime_s']} с"
    return json.dumps(payload, ensure_ascii=False)[:160]


def _section(prompt: str, header: str) -> list[str]:
    """Повертає рядки '- …' з відповідного блоку промпту."""
    if header not in prompt:
        return []
    block = prompt.split(header, 1)[1].split("\n\n", 1)[0]
    return [line[2:].strip() for line in block.splitlines() if line.strip().startswith("- ")]


class ScriptedSreLLM(BaseChatModel):
    """Детермінований офлайн-фолбек: емулює ReAct tool calling, планувальник і replanner.

    Підтримує `bind_tools` (звідси працює й `with_structured_output`) та штучну затримку
    `latency_s` для демонстрації тайм-ауту.
    """

    bound_tools: list[dict] = Field(default_factory=list)
    latency_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def bind_tools(self, tools: list, **kwargs: Any) -> "ScriptedSreLLM":
        return self.model_copy(update={"bound_tools": [convert_to_openai_tool(t) for t in tools]})

    def _generate(self, messages: list[AnyMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.latency_s:
            time.sleep(self.latency_s)
        bound = [t["function"]["name"] for t in self.bound_tools]
        if bound == ["Plan"]:
            message = self._plan(messages[-1].content)
        elif bound == ["ReplanDecision"]:
            message = self._replan(messages[-1].content)
        else:
            message = self._react_step(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _structured(name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name.lower()}"}])

    # ---------- ReAct ----------
    def _react_step(self, messages: list[AnyMessage]) -> AIMessage:
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        query = messages[humans[-1]].content
        task = query.split("Поточний крок:", 1)[-1].strip()
        after = messages[humans[-1] + 1:]
        turn = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        low = task.lower()

        # Спеціальні сценарії для демонстрації guardrails.
        if "зациклення" in low:
            calls = [("get_service_status", {"service": "api-gateway"})]
        elif "довгий ланцюг" in low:
            calls = [("calculate_sla", {"service": "payments", "downtime_minutes": float(turn + 1), "period_days": 30})]
        else:
            done = {(c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
                    for m in after if isinstance(m, AIMessage) for c in m.tool_calls}
            earlier = " ".join(messages[i].content for i in humans[:-1])
            stages = _intent_calls(task, context_services=_services_in(earlier))
            calls = next((stage for stage in (
                [c for c in stage if (c[0], json.dumps(c[1], sort_keys=True, ensure_ascii=False)) not in done]
                for stage in stages) if stage), [])

        if calls:
            tool_calls = [{"name": n, "args": a, "id": f"call_{turn}_{i}"} for i, (n, a) in enumerate(calls)]
            names = ", ".join(c["name"] for c in tool_calls)
            return AIMessage(content=f"Thought: потрібні дані з інструментів ({names}).", tool_calls=tool_calls)

        observations = [m for m in after if isinstance(m, ToolMessage)]
        if observations:
            parts = [_describe_observation(m.name, json.loads(m.content)) for m in observations]
            return AIMessage(content="Підсумок: " + "; ".join(parts) + ".")
        context = _section(query, "Контекст виконаних кроків:")
        if context:
            results = [c.split(" → ", 1)[-1].removeprefix("Підсумок: ")[:200] for c in context]
            return AIMessage(content="Підсумок інциденту: " + " | ".join(results))
        return AIMessage(content=f"Для запиту «{task[:80]}» інструменти не потрібні — уточніть, який сервіс перевірити.")

    # ---------- Planner ----------
    def _plan(self, query: str) -> AIMessage:
        steps: list[str] = []
        for stage in _intent_calls(query):
            for name, args in stage:
                if name == "get_service_status":
                    steps.append(f"Перевір статус сервісу {args['service']}")
                elif name == "query_logs":
                    steps.append(f"Переглянь логи рівня {args['level']} сервісу {args['service']} за {args['last_minutes']} хвилин")
                elif name == "calculate_sla":
                    steps.append(f"Порахуй SLA сервісу {args['service']} при {args['downtime_minutes']:g} хв простою за {args['period_days']} днів")
                elif name == "search_runbooks":
                    services = _services_in(query)
                    steps.append(f"Знайди runbook у базі знань про інцидент з {services[0]}" if services
                                 else f"Знайди в базі знань: {query[:80]}")
                elif name == "restart_service":
                    steps.append(f"Перезапусти сервіс {args['service']}")
        if not steps:
            return self._structured("Plan", {"steps": [f"Дай пряму відповідь на запит: {query[:80]}"]})
        steps.append("Сформуй підсумок інциденту для користувача")
        return self._structured("Plan", {"steps": steps})

    # ---------- Replanner ----------
    def _replan(self, prompt: str) -> AIMessage:
        remaining = _section(prompt, "Залишок плану:")
        done = _section(prompt, "Виконані кроки:")
        last_result = done[-1].split(" → ", 1)[-1] if done else ""
        low = last_result.lower()
        fix = "Перевір статус сервісу payments"
        already_fixed = any(d.startswith(fix) for d in done) or fix in remaining
        if "помилка" in low and "billing" in low and not low.startswith("підсумок інциденту") and not already_fixed:
            return self._structured("ReplanDecision", {
                "action": "revise", "steps": [fix] + remaining,
                "reason": "сервіс billing невідомий моніторингу; за runbook-ами платежі обслуговує payments"})
        if "відхилено" in low:
            return self._structured("ReplanDecision", {
                "action": "finish", "reason": "людина відхилила ризикову дію",
                "response": "Перезапуск скасовано за рішенням людини. Зібрана діагностика збережена у кроках плану."})
        if remaining:
            return self._structured("ReplanDecision", {"action": "continue", "reason": "план актуальний"})
        response = last_result if low.startswith("підсумок") else "Готово. " + " | ".join(d[:150] for d in done)
        return self._structured("ReplanDecision", {"action": "finish", "response": response,
                                                   "reason": "усі кроки плану виконано"})


def build_llm(config: AgentConfig = DEFAULT_CONFIG) -> BaseChatModel:
    """Створює LLM: OpenAI за наявності ключа, інакше детермінований офлайн-фолбек."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model_name, temperature=config.temperature)
    return ScriptedSreLLM()


ACTIVE_LLM = build_llm()
OFFLINE_LLM = ScriptedSreLLM()
print(f"Активний провайдер: {ACTIVE_LLM._llm_type} | модель: {getattr(ACTIVE_LLM, 'model_name', DEFAULT_CONFIG.model_name)}")
for model in (Plan, ReplanDecision):
    print(f"Structured output {model.__name__}: {', '.join(model.model_fields)}")


class ReActState(TypedDict):
    """Стан графа ReAct-агента."""

    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    started_at: float
    seen_calls: list[str]
    halted: bool
    halt_reason: str
    trajectory: Annotated[list[dict], operator.add]
    final: str


def _event(state: ReActState, event_type: str, **payload: Any) -> dict:
    """Створює запис траєкторії з часом від початку виконання."""
    return {"step": state["step"], "type": event_type,
            "elapsed_s": round(time.perf_counter() - state["started_at"], 3), **payload}


def _call_signature(name: str, args: dict) -> str:
    """Підпис tool call для детекції повторів."""
    return f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"


def build_react_graph(llm: BaseChatModel, config: AgentConfig = DEFAULT_CONFIG,
                      checkpointer: SqliteSaver | None = None, interrupt_before: list[str] | None = None):
    """Збирає та компілює LangGraph-граф ReAct-агента із guardrails, HITL-вузлом і checkpointer-ом."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: ReActState) -> dict:
        elapsed = time.perf_counter() - state["started_at"]
        if state["step"] >= config.max_steps:
            reason = f"max_steps: досягнуто ліміт {config.max_steps} ітерацій"
            return {"halted": True, "halt_reason": reason, "trajectory": [_event(state, "guardrail", reason=reason)]}
        if elapsed > config.timeout_s:
            reason = f"timeout: перевищено {config.timeout_s:.2f} с (пройшло {elapsed:.2f} с)"
            return {"halted": True, "halt_reason": reason, "trajectory": [_event(state, "guardrail", reason=reason)]}
        started = time.perf_counter()
        response = llm_with_tools.invoke([SystemMessage(REACT_SYSTEM_PROMPT), *state["messages"]])
        event = _event(state, "thought", content=response.content,
                       tool_calls=[{"name": c["name"], "args": c["args"]} for c in response.tool_calls],
                       llm_latency_s=round(time.perf_counter() - started, 3))
        return {"messages": [response], "step": state["step"] + 1, "trajectory": [event]}

    def tools_node(state: ReActState) -> dict:
        calls = state["messages"][-1].tool_calls
        messages, events, signatures = [], [], []
        halted, halt_reason = False, ""
        for call in calls:
            signature = _call_signature(call["name"], call["args"])
            fallback = None
            if config.detect_loops and signature in state["seen_calls"]:
                halted, halt_reason = True, f"loop: повторний виклик {signature}"
                observation = _err("Цей виклик уже виконувався — змініть аргументи або завершіть відповідь", call["name"])
            else:
                signatures.append(signature)
                observation, fallback = try_tool_with_fallback(call["name"], call["args"])
            messages.append(ToolMessage(content=observation, name=call["name"], tool_call_id=call["id"]))
            events.append(_event(state, "observation", tool=call["name"], args=call["args"],
                                 observation=observation, is_error=json.loads(observation)["status"] != "ok",
                                 fallback_tool=fallback, risky=call["name"] in RISKY_TOOLS))
        if halted:
            events.append(_event(state, "guardrail", reason=halt_reason))
        return {"messages": messages, "seen_calls": state["seen_calls"] + signatures,
                "trajectory": events, "halted": halted, "halt_reason": halt_reason}

    def finalize_node(state: ReActState) -> dict:
        last = state["messages"][-1]
        if state["halted"]:
            used = sorted({m.name for m in state["messages"] if isinstance(m, ToolMessage)})
            final = (f"Виконання зупинено захисним механізмом → {state['halt_reason']}. "
                     f"Часткові дані отримано інструментами: {', '.join(used) or 'немає'}.")
        else:
            final = last.content if isinstance(last, AIMessage) else ""
        return {"final": final, "trajectory": [_event(state, "final_answer", answer=final)]}

    def route_after_agent(state: ReActState) -> Literal["tools", "risky_tools", "finalize"]:
        if state["halted"]:
            return "finalize"
        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        if not calls:
            return "finalize"
        return "risky_tools" if any(c["name"] in RISKY_TOOLS for c in calls) else "tools"

    def route_after_tools(state: ReActState) -> Literal["agent", "finalize"]:
        return "finalize" if state["halted"] else "agent"

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("risky_tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent,
                                {"tools": "tools", "risky_tools": "risky_tools", "finalize": "finalize"})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "finalize": "finalize"})
    graph.add_conditional_edges("risky_tools", route_after_tools, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before or [])


def initial_state(query: str) -> dict:
    """Початковий стан для нового запиту (лічильники скидаються, історія повідомлень накопичується)."""
    return {"messages": [HumanMessage(query)], "step": 0, "started_at": time.perf_counter(),
            "seen_calls": [], "halted": False, "halt_reason": "", "trajectory": [], "final": ""}


def thread_config(thread_id: str, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": config.max_steps * 2 + 10}


RUN_LOG: list[dict] = []


def run_agent(query: str, llm: BaseChatModel | None = None, config: AgentConfig = DEFAULT_CONFIG,
              tag: str = "run", graph=None, thread_id: str | None = None) -> dict:
    """Запускає ReAct-агента на одному запиті та повертає повний запис виконання (додається до RUN_LOG)."""
    llm = llm or ACTIVE_LLM
    graph = graph or build_react_graph(llm, config)
    run_config = thread_config(thread_id, config) if thread_id else {"recursion_limit": config.max_steps * 2 + 10}
    already_logged = len(graph.get_state(run_config).values.get("trajectory", [])) if thread_id else 0
    started = time.perf_counter()
    state = graph.invoke(initial_state(query), run_config)
    trajectory = state["trajectory"][already_logged:]
    record = {
        "tag": tag, "query": query, "provider": llm._llm_type, "thread_id": thread_id,
        "config": {"max_steps": config.max_steps, "timeout_s": config.timeout_s, "detect_loops": config.detect_loops},
        "steps": state["step"],
        "tool_calls": [{"tool": e["tool"], "args": e["args"], "is_error": e["is_error"], "fallback_tool": e["fallback_tool"]}
                       for e in trajectory if e["type"] == "observation"],
        "duration_s": round(time.perf_counter() - started, 4),
        "halted": state["halted"], "halt_reason": state["halt_reason"],
        "interrupted": bool(state.get("__interrupt__")),
        "final": state["final"], "trajectory": trajectory,
    }
    RUN_LOG.append(record)
    return record


def print_run(record: dict, show_trajectory: bool = False) -> None:
    """Друкує компактне зведення запуску агента."""
    status = f"ЗУПИНЕНО ({record['halt_reason']})" if record["halted"] else "ЗАВЕРШЕНО"
    tools = ", ".join(c["tool"] + ("→" + c["fallback_tool"] if c["fallback_tool"] else "") for c in record["tool_calls"]) or "—"
    print(f"Запит: {record['query'][:100]}")
    print(f"Статус: {status}")
    print(f"Кроків LLM: {record['steps']} | tool calls: {len(record['tool_calls'])} ({tools}) | час: {fmt_duration(record['duration_s'])}")
    print(f"Відповідь: {record['final'][:500]}")
    if show_trajectory:
        print("\nТраєкторія (JSON-лог):")
        for e in record["trajectory"]:
            detail = {"thought": lambda e: f"{e['content'][:60]} → {[c['name'] for c in e['tool_calls']]}",
                      "observation": lambda e: f"{e['tool']}{' [fallback→' + e['fallback_tool'] + ']' if e['fallback_tool'] else ''} → {e['observation'][:90]}",
                      "guardrail": lambda e: e["reason"],
                      "final_answer": lambda e: e["answer"][:90]}[e["type"]](e)
            print(f"  [{e['elapsed_s']:>6.3f} с] step {e['step']} {e['type']:<13} {detail}")


DEFAULT_GRAPH = build_react_graph(ACTIVE_LLM, DEFAULT_CONFIG)
print("ReAct-граф скомпільовано. Вузли:",
      ", ".join(n for n in DEFAULT_GRAPH.get_graph().nodes if not n.startswith("__")))


class PlanExecState(TypedDict):
    """Стан Plan-and-Execute графа."""

    input: str
    plan: list[str]
    past_steps: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    response: str


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {s}" for s in items) or "(порожньо)"


def _format_past(past: list[dict]) -> str:
    return "\n".join(f"- {p['step']} → {p['result']}" for p in past) or "(порожньо)"


def build_plan_execute_graph(llm: BaseChatModel, config: AgentConfig = DEFAULT_CONFIG,
                             checkpointer: SqliteSaver | None = None):
    """Збирає граф planner → executor → replanner; executor запускає вкладений ReAct-агент."""
    planner_llm = llm.with_structured_output(Plan)
    replanner_llm = llm.with_structured_output(ReplanDecision)
    step_config = config.model_copy(update={"max_steps": 4})
    step_agent = build_react_graph(llm, step_config)

    def planner_node(state: PlanExecState) -> dict:
        plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(state["input"])])
        steps = plan.steps[:config.max_plan_steps]
        return {"plan": steps, "decisions": [{"node": "planner", "action": "plan", "steps": steps}]}

    def executor_node(state: PlanExecState) -> dict:
        step = state["plan"][0]
        prompt = (f"Запит користувача: {state['input']}\n"
                  f"Контекст виконаних кроків:\n{_format_past(state['past_steps'])}\n\n"
                  f"Поточний крок: {step}")
        result = step_agent.invoke(initial_state(prompt), {"recursion_limit": step_config.max_steps * 2 + 10})
        tools = [e["tool"] for e in result["trajectory"] if e["type"] == "observation"]
        return {"plan": state["plan"][1:],
                "past_steps": [{"step": step, "tools": tools, "react_steps": result["step"], "result": result["final"]}]}

    def replanner_node(state: PlanExecState) -> dict:
        if len(state["past_steps"]) >= config.max_total_steps:
            reason = f"guardrail: досягнуто ліміт {config.max_total_steps} кроків"
            return {"response": f"Зупинено захисним механізмом → {reason}",
                    "decisions": [{"node": "replanner", "action": "finish", "reason": reason}]}
        prompt = (f"Запит користувача: {state['input']}\n"
                  f"Залишок плану:\n{_format_list(state['plan'])}\n\n"
                  f"Виконані кроки:\n{_format_past(state['past_steps'])}\n\nВиріши: continue / revise / finish.")
        decision = replanner_llm.invoke([SystemMessage(REPLANNER_PROMPT), HumanMessage(prompt)])
        event = {"node": "replanner", "action": decision.action, "reason": decision.reason}
        if decision.action == "finish":
            return {"response": decision.response, "decisions": [event]}
        if decision.action == "revise":
            return {"plan": decision.steps[:config.max_plan_steps], "decisions": [event | {"steps": decision.steps}]}
        return {"decisions": [event]}

    def route_after_replanner(state: PlanExecState) -> Literal["executor", "__end__"]:
        return "__end__" if state["response"] else "executor"

    graph = StateGraph(PlanExecState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", route_after_replanner, {"executor": "executor", "__end__": END})
    return graph.compile(checkpointer=checkpointer)


PLAN_LOG: list[dict] = []


def run_plan_execute(query: str, llm: BaseChatModel | None = None, config: AgentConfig = DEFAULT_CONFIG,
                     tag: str = "plan", graph=None, thread_id: str | None = None) -> dict:
    """Запускає Plan-and-Execute агента та повертає зведення (додається до PLAN_LOG)."""
    llm = llm or ACTIVE_LLM
    graph = graph or build_plan_execute_graph(llm, config)
    run_config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    started = time.perf_counter()
    state = graph.invoke({"input": query, "plan": [], "past_steps": [], "decisions": [], "response": ""}, run_config)
    record = {
        "tag": tag, "query": query, "provider": llm._llm_type,
        "plan": next(d["steps"] for d in state["decisions"] if d["node"] == "planner"),
        "steps_done": len(state["past_steps"]),
        "llm_calls": 2 + sum(p["react_steps"] for p in state["past_steps"]) + len(state["past_steps"]) - 1,
        "tool_calls": [t for p in state["past_steps"] for t in p["tools"]],
        "replans": sum(1 for d in state["decisions"] if d.get("action") == "revise"),
        "duration_s": round(time.perf_counter() - started, 4),
        "past_steps": state["past_steps"], "decisions": state["decisions"], "response": state["response"],
    }
    PLAN_LOG.append(record)
    return record


def print_plan_run(record: dict) -> None:
    """Друкує план, виконані кроки та рішення replanner-а."""
    print(f"Запит: {record['query'][:100]}")
    print("План:")
    for i, step in enumerate(record["plan"], 1):
        print(f"  {i}. {step}")
    print("Виконання:")
    for i, past in enumerate(record["past_steps"], 1):
        print(f"  крок {i} [{', '.join(past['tools']) or 'без інструмента'}] {past['step'][:70]}\n      → {past['result'][:140]}")
    for d in record["decisions"]:
        if d["node"] == "replanner":
            print(f"  replanner: {d['action']} — {d['reason']}")
    print(f"Разом: кроків плану {record['steps_done']}, викликів LLM {record['llm_calls']}, "
          f"tool calls {len(record['tool_calls'])}, replans {record['replans']}, час {fmt_duration(record['duration_s'])}")
    print(f"Відповідь: {record['response'][:500]}")


PLAN_GRAPH = build_plan_execute_graph(ACTIVE_LLM, DEFAULT_CONFIG)
print("Plan-and-Execute граф скомпільовано. Вузли:",
      ", ".join(n for n in PLAN_GRAPH.get_graph().nodes if not n.startswith("__")))
print(PLAN_GRAPH.get_graph().draw_mermaid())
