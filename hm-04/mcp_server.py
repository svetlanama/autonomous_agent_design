"""MCP-сервер SRE-центру (FastMCP): моніторинг, логи, runbook-и та ризиковий перезапуск."""

import logging
import re
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

# Через stdio лог сервера йде у stderr клієнта — лишаємо лише помилки, щоб не засмічувати вивід ноутбука.
logging.getLogger("FastMCP").setLevel(logging.ERROR)
logging.getLogger("fastmcp").setLevel(logging.ERROR)

mcp = FastMCP("sre-incident-tools")

# --- Імітація систем моніторингу, лог-колектора та бази знань ---------------------------------
SERVICES_DB = {
    "api-gateway": {"status": "healthy", "latency_p95_ms": 120, "error_rate_percent": 0.2, "replicas": 3},
    "auth-service": {"status": "healthy", "latency_p95_ms": 95, "error_rate_percent": 0.1, "replicas": 2},
    "payments": {"status": "degraded", "latency_p95_ms": 1850, "error_rate_percent": 7.4, "replicas": 2},
    "postgres-db": {"status": "healthy", "latency_p95_ms": 14, "error_rate_percent": 0.0, "replicas": 1},
    "notifications": {"status": "down", "latency_p95_ms": 0, "error_rate_percent": 100.0, "replicas": 0},
}

# Логи містять персональні дані (PII) — їх має вирізати output guardrail агента.
LOGS_DB = {
    "api-gateway": [
        {"minutes_ago": 3, "level": "WARN", "message": "upstream payments timeout after 2000ms for client ip 203.0.113.42"},
        {"minutes_ago": 12, "level": "INFO", "message": "config reloaded"},
    ],
    "auth-service": [
        {"minutes_ago": 40, "level": "INFO", "message": "token cache warmed"},
    ],
    "payments": [
        {"minutes_ago": 2, "level": "ERROR", "message": "connection pool exhausted (max=20)"},
        {"minutes_ago": 4, "level": "ERROR", "message": "charge failed for user ivan.petrenko@example.com card 4111 1111 1111 1111"},
        {"minutes_ago": 6, "level": "ERROR", "message": "retry storm from client +380501234567 (ip 198.51.100.7)"},
        {"minutes_ago": 15, "level": "WARN", "message": "slow query 1900ms on payments_ledger"},
        {"minutes_ago": 55, "level": "INFO", "message": "deploy v2.14.0 finished"},
    ],
    "postgres-db": [
        {"minutes_ago": 8, "level": "WARN", "message": "checkpoint took 3200ms"},
    ],
    "notifications": [],
}

RUNBOOKS = {
    "rb-payments-pool": {
        "title": "Runbook: вичерпання пулу з'єднань у payments",
        "text": "Симптоми: connection pool exhausted, зростання p95. Кроки: 1) перевірити стан postgres-db; "
                "2) перевірити активні з'єднання; 3) якщо пул вичерпано через витік — перезапустити payments "
                "з причиною у тікеті інциденту; 4) після перезапуску спостерігати 15 хвилин.",
    },
    "rb-restart-policy": {
        "title": "Політика перезапуску сервісів",
        "text": "Перезапуск stateful-сервісів (postgres-db) заборонено без DBA. Перезапуск stateless-сервісів "
                "дозволено on-call інженеру після підтвердження людиною; причина обов'язкова.",
    },
    "rb-notifications-down": {
        "title": "Runbook: сервіс notifications недоступний",
        "text": "Якщо notifications має 0 реплік — перевірити deployment, черги та квоти; масштабувати до 2 реплік.",
    },
    "rb-sla": {
        "title": "SLA та бюджет помилок",
        "text": "Ціль доступності 99.9% на 30 днів = 43.2 хв бюджету простою. При витраті >50% бюджету — заморозка релізів.",
    },
    "rb-pii": {
        "title": "Політика обробки персональних даних у логах",
        "text": "Email, телефони, IP-адреси та номери карток не можна виводити у звіти інцидентів — лише маскувати.",
    },
}

ServiceName = Annotated[str, Field(min_length=2, max_length=40, pattern=r"^[a-z0-9-]+$",
                                   description="Технічна назва сервісу, напр. payments")]


def _ok(data: dict) -> dict:
    return {"status": "ok", "data": data}


def _err(message: str) -> dict:
    return {"status": "error", "error": message}


@mcp.tool
def get_service_status(service: ServiceName) -> dict:
    """Повертає поточний стан сервісу з моніторингу: статус, p95-латентність, частку помилок, кількість реплік."""
    record = SERVICES_DB.get(service.lower())
    if record is None:
        return _err(f"сервіс «{service}» невідомий моніторингу; відомі: {', '.join(sorted(SERVICES_DB))}")
    return _ok({"service": service.lower(), **record})


@mcp.tool
def query_logs(
    service: ServiceName,
    level: Annotated[Literal["ERROR", "WARN", "INFO"], Field(description="Мінімальний рівень записів")] = "ERROR",
    last_minutes: Annotated[int, Field(ge=1, le=1440, description="Вікно пошуку у хвилинах")] = 30,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> dict:
    """Повертає записи логів сервісу за вказаним рівнем (і суворішими) за останні N хвилин."""
    if service.lower() not in LOGS_DB:
        return _err(f"логи для сервісу «{service}» відсутні у лог-колекторі")
    rank = {"ERROR": 0, "WARN": 1, "INFO": 2}
    entries = [e for e in LOGS_DB[service.lower()]
               if e["minutes_ago"] <= last_minutes and rank[e["level"]] <= rank[level]][:limit]
    if not entries:
        return _err(f"за {last_minutes} хв записів рівня {level} для «{service}» не знайдено")
    return _ok({"service": service.lower(), "level": level, "last_minutes": last_minutes, "entries": entries})


@mcp.tool
def search_runbooks(
    query: Annotated[str, Field(min_length=3, max_length=300, description="Пошуковий запит")],
    top_k: Annotated[int, Field(ge=1, le=5)] = 2,
) -> dict:
    """Шукає runbook-и та SRE-політики за запитом; повертає top_k найрелевантніших документів зі скорингом."""
    tokens = [t for t in re.findall(r"[a-zа-яіїєґ0-9'-]+", query.lower()) if len(t) > 2]
    scored = []
    for doc_id, doc in RUNBOOKS.items():
        haystack = (doc["title"] + " " + doc["text"]).lower()
        score = sum(haystack.count(tok) for tok in tokens)
        if score:
            scored.append({"id": doc_id, "title": doc["title"], "score": score, "text": doc["text"]})
    scored.sort(key=lambda d: -d["score"])
    if not scored:
        return _err("релевантних runbook-ів не знайдено")
    return _ok({"query": query, "results": scored[:top_k]})


@mcp.tool
def restart_service(
    service: ServiceName,
    reason: Annotated[str, Field(min_length=10, max_length=300, description="Причина перезапуску для журналу")],
) -> dict:
    """РИЗИКОВА ДІЯ: перезапускає сервіс. Потребує підтвердження людини; stateful-сервіси перезапускати заборонено."""
    name = service.lower()
    if name not in SERVICES_DB:
        return _err(f"сервіс «{service}» невідомий")
    if name == "postgres-db":
        return _err("перезапуск stateful-сервісу postgres-db заборонено політикою (потрібен DBA)")
    SERVICES_DB[name].update({"status": "healthy", "latency_p95_ms": 140, "error_rate_percent": 0.3,
                              "replicas": max(SERVICES_DB[name]["replicas"], 2)})
    return _ok({"service": name, "action": "restarted", "reason": reason, "new_status": SERVICES_DB[name]})


if __name__ == "__main__":
    mcp.run(show_banner=False)  # транспорт за замовчуванням — stdio
