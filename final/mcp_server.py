"""MCP-сервер SRE-центру (офіційний MCP Python SDK, FastMCP): tools, resources, prompts."""

import json
import logging
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logging.getLogger("mcp").setLevel(logging.ERROR)  # stdio: stderr сервера потрапляє у вивід клієнта

mcp = FastMCP("sre-incident-center")

# --- Імітація моніторингу, лог-колектора, білінгу та каталогу сервісів -------------------------
SERVICES_DB = {
    "api-gateway": {"status": "healthy", "latency_p95_ms": 120, "error_rate_percent": 0.2, "replicas": 3},
    "auth-service": {"status": "healthy", "latency_p95_ms": 95, "error_rate_percent": 0.1, "replicas": 2},
    "payments": {"status": "degraded", "latency_p95_ms": 1850, "error_rate_percent": 7.4, "replicas": 2},
    "postgres-db": {"status": "healthy", "latency_p95_ms": 14, "error_rate_percent": 0.0, "replicas": 1},
    "notifications": {"status": "down", "latency_p95_ms": 0, "error_rate_percent": 100.0, "replicas": 0},
}

# Логи навмисно містять PII — їх має замаскувати output guardrail агента.
LOGS_DB = {
    "api-gateway": [
        {"minutes_ago": 3, "level": "WARN", "message": "upstream payments timeout after 2000ms for client ip 203.0.113.42"},
        {"minutes_ago": 12, "level": "INFO", "message": "config reloaded"},
    ],
    "auth-service": [{"minutes_ago": 40, "level": "INFO", "message": "token cache warmed"}],
    "payments": [
        {"minutes_ago": 2, "level": "ERROR", "message": "connection pool exhausted (max=20)"},
        {"minutes_ago": 4, "level": "ERROR", "message": "charge failed for user ivan.petrenko@example.com card 4111 1111 1111 1111"},
        {"minutes_ago": 6, "level": "ERROR", "message": "retry storm from client +380501234567 (ip 198.51.100.7)"},
        {"minutes_ago": 15, "level": "WARN", "message": "slow query 1900ms on payments_ledger"},
        {"minutes_ago": 55, "level": "INFO", "message": "deploy v2.14.0 finished"},
    ],
    "postgres-db": [{"minutes_ago": 8, "level": "WARN", "message": "checkpoint took 3200ms"}],
    "notifications": [],
}

BILLING_DB = {
    "acme": {"customer": "ACME Corp", "plan": "enterprise", "monthly_fee_usd": 5000.0, "sla_target_percent": 99.9,
             "services": ["payments", "api-gateway"], "contact": "ops@acme.example.com"},
    "startup-io": {"customer": "Startup.io", "plan": "basic", "monthly_fee_usd": 200.0, "sla_target_percent": 99.5,
                   "services": ["notifications"], "contact": "cto@startup.io"},
}

SERVICE_CATALOG = {
    "api-gateway": {"owner": "Platform", "tier": "critical", "stateless": True, "restart_allowed": True},
    "auth-service": {"owner": "Platform", "tier": "critical", "stateless": True, "restart_allowed": True},
    "payments": {"owner": "Payments", "tier": "critical", "stateless": True, "restart_allowed": True},
    "postgres-db": {"owner": "Data", "tier": "critical", "stateless": False, "restart_allowed": False},
    "notifications": {"owner": "Growth", "tier": "standard", "stateless": True, "restart_allowed": True},
}

SLA_POLICY = """Політика SLA та компенсацій (SLA credits).
Цільова доступність: критичні сервіси (payments, api-gateway, auth-service) — 99.9% за 30 днів (бюджет помилок 43.2 хв);
некритичні (notifications) — 99.5%.
Компенсація на наступний рахунок залежить від фактичної доступності за місяць:
- >= цілі SLA: 0%;
- від 99.0% до цілі: 10% місячної плати;
- від 95.0% до 99.0%: 25%;
- нижче 95.0%: 50%.
Компенсація нараховується лише за запитом клієнта протягом 30 днів після інциденту; максимум — 50% місячної плати.
Якщо бюджет помилок вичерпано, релізи сервісу заморожуються до кінця періоду."""

INCIDENTS: list[dict] = []
PROTECTED_SERVICES = {"postgres-db"}

ServiceName = Annotated[str, Field(min_length=2, max_length=40, pattern=r"^[a-z0-9-]+$",
                                   description="Технічна назва сервісу, напр. payments")]


def _ok(data: dict) -> dict:
    return {"status": "ok", "data": data}


def _err(message: str) -> dict:
    return {"status": "error", "error": message}


# ---------------------------------------------------------------- TOOLS ----------------------------------------
@mcp.tool()
def get_service_status(service: ServiceName) -> dict:
    """Повертає поточний стан сервісу з моніторингу: статус (healthy/degraded/down), p95-латентність у мс,
    частку помилок у % та кількість реплік. Використовуй першим кроком діагностики інциденту.
    Відомі сервіси: api-gateway, auth-service, payments, postgres-db, notifications."""
    record = SERVICES_DB.get(service.lower())
    if record is None:
        return _err(f"сервіс «{service}» невідомий моніторингу; відомі: {', '.join(sorted(SERVICES_DB))}")
    return _ok({"service": service.lower(), **record})


@mcp.tool()
def query_logs(
    service: ServiceName,
    level: Annotated[Literal["ERROR", "WARN", "INFO"], Field(description="Мінімальний рівень записів")] = "ERROR",
    last_minutes: Annotated[int, Field(ge=1, le=1440, description="Вікно пошуку у хвилинах")] = 30,
    limit: Annotated[int, Field(ge=1, le=50, description="Максимум записів")] = 10,
) -> dict:
    """Повертає записи логів сервісу рівня level і суворіших за останні last_minutes хвилин (не більше limit).
    Записи можуть містити персональні дані — виводити їх у звіт заборонено без маскування."""
    name = service.lower()
    if name not in LOGS_DB:
        return _err(f"логи для сервісу «{service}» відсутні у лог-колекторі")
    rank = {"ERROR": 0, "WARN": 1, "INFO": 2}
    entries = [e for e in LOGS_DB[name] if e["minutes_ago"] <= last_minutes and rank[e["level"]] <= rank[level]][:limit]
    if not entries:
        return _err(f"за {last_minutes} хв записів рівня {level} для «{service}» не знайдено")
    return _ok({"service": name, "level": level, "last_minutes": last_minutes, "entries": entries})


@mcp.tool()
def get_billing_account(
    customer_id: Annotated[str, Field(min_length=2, max_length=40, pattern=r"^[a-z0-9-]+$",
                                      description="Ідентифікатор клієнта, напр. acme")],
) -> dict:
    """Повертає білінговий акаунт клієнта: назву, тарифний план, місячну плату в USD, ціль SLA у %
    та перелік сервісів, які він використовує. Потрібен для розрахунку SLA-компенсації."""
    record = BILLING_DB.get(customer_id.lower())
    if record is None:
        return _err(f"клієнт «{customer_id}» не знайдений; відомі: {', '.join(sorted(BILLING_DB))}")
    return _ok({"customer_id": customer_id.lower(), **record})


@mcp.tool()
def open_incident(
    service: ServiceName,
    severity: Annotated[Literal["P1", "P2", "P3", "P4"], Field(description="Рівень інциденту: P1 — найвищий")],
    summary: Annotated[str, Field(min_length=10, max_length=300, description="Короткий опис інциденту")],
) -> dict:
    """Створює тікет інциденту у журналі (побічний ефект: нове INC-число) і повертає його номер,
    команду-власника сервісу та час реакції за політикою ескалації."""
    name = service.lower()
    if name not in SERVICE_CATALOG:
        return _err(f"сервіс «{service}» відсутній у каталозі")
    ticket = {"id": f"INC-{1000 + len(INCIDENTS) + 1}", "service": name, "severity": severity, "summary": summary,
              "owner": SERVICE_CATALOG[name]["owner"], "status": "open",
              "response_minutes": {"P1": 5, "P2": 15, "P3": 240, "P4": 1440}[severity]}
    INCIDENTS.append(ticket)
    return _ok(ticket)


@mcp.tool()
def restart_service(
    service: ServiceName,
    reason: Annotated[str, Field(min_length=10, max_length=300, description="Причина перезапуску для журналу")],
) -> dict:
    """РИЗИКОВА ДІЯ: перезапускає сервіс (скидає з'єднання користувачів). Виконуй лише після підтвердження людини
    і з чіткою причиною. Stateful-сервіси (postgres-db) перезапускати заборонено політикою."""
    name = service.lower()
    if name not in SERVICES_DB:
        return _err(f"сервіс «{service}» невідомий")
    if name in PROTECTED_SERVICES or not SERVICE_CATALOG[name]["restart_allowed"]:
        return _err(f"перезапуск stateful-сервісу {name} заборонено політикою (потрібен DBA і вікно обслуговування)")
    SERVICES_DB[name].update({"status": "healthy", "latency_p95_ms": 140, "error_rate_percent": 0.3,
                              "replicas": max(SERVICES_DB[name]["replicas"], 2)})
    return _ok({"service": name, "action": "restarted", "reason": reason, "new_status": SERVICES_DB[name]})


# ---------------------------------------------------------------- RESOURCES ------------------------------------
@mcp.resource("faq://sla-policy", name="sla_policy", description="Політика SLA та SLA-компенсацій (read-only довідник)",
              mime_type="text/plain")
def sla_policy_resource() -> str:
    """Текст політики SLA і таблиця компенсацій."""
    return SLA_POLICY


@mcp.resource("service://{name}", name="service_card", description="Картка сервісу з каталогу: власник, tier, чи дозволено перезапуск",
              mime_type="application/json")
def service_card_resource(name: str) -> str:
    """Картка сервісу з каталогу за назвою."""
    card = SERVICE_CATALOG.get(name.lower())
    if card is None:
        return json.dumps(_err(f"сервіс «{name}» відсутній у каталозі"), ensure_ascii=False)
    return json.dumps({"service": name.lower(), **card}, ensure_ascii=False)


# ---------------------------------------------------------------- PROMPTS --------------------------------------
@mcp.prompt(name="incident_report", description="Шаблон звіту про інцидент для каналу #incidents")
def incident_report_prompt(service: str, severity: str, tone: str = "formal") -> str:
    """Шаблон звіту про інцидент: сервіс, рівень, тон."""
    style = {"formal": "офіційним діловим стилем", "friendly": "дружнім, але точним стилем"}.get(tone, "нейтральним стилем")
    return (f"Склади звіт про інцидент {severity} для сервісу {service} {style}. Структура: 1) вплив на користувачів; "
            f"2) поточний стан за моніторингом; 3) виконані дії; 4) наступні кроки та відповідальний. "
            f"Не включай персональні дані (email, телефони, IP, картки) — лише маскуй їх.")


@mcp.prompt(name="sla_credit_reply", description="Шаблон відповіді клієнту про SLA-компенсацію")
def sla_credit_reply_prompt(customer: str, service: str, availability_percent: str) -> str:
    """Шаблон листа клієнту про результат розрахунку SLA-компенсації."""
    return (f"Напиши коротку відповідь клієнту {customer} щодо доступності сервісу {service} за місяць: "
            f"{availability_percent}%. Вкажи ціль SLA, чи її порушено, розмір компенсації за політикою faq://sla-policy "
            f"і як її отримати. Тон — ввічливий, без технічного жаргону.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
