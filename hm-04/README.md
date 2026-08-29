# Практичне завдання 2 — Мультиагентна система «SRE-центр реагування на інциденти»

**Автор:** Моісеєнко Світлана · курс «Проєктування автономних агентів»

Мультиагентна система для домену **управління ІТ-інцидентами (SRE / on-call)** у двох фреймворках (LangGraph і CrewAI),
з кастомним MCP-сервером (FastMCP), трьома шарами guardrails, Human-in-the-Loop для ризикового інструмента,
tracing-ом, red-teaming-ом, evals і тестами pytest. Номер варіанта у завданні не вказано — домен, ролі та ризиковий
інструмент (`restart_service`) обрано самостійно як продовження Практичного завдання 1.

## Файли

| Файл | Опис |
|---|---|
| `ДЗ4_Моісеєнко_Світлана.ipynb` | основний ноутбук (Colab / Jupyter), 17 секцій, виконаний з результатами |
| `mas_security.py` | бібліотечна частина ноутбука (guardrails, tracing, LLM-фолбек, графи, CrewAI) — для імпорту в тестах |
| `mcp_server.py` | кастомний MCP-сервер FastMCP (4 tools); запускається окремим процесом через stdio |
| `test_mas_security.py` | 14 тестів pytest: 6 — MCP tools, 8 — guardrails і HITL |
| `pytest_output.txt`, `pytest_results.png` | вивід `pytest -v` та його «скриншот» |
| `trace_langgraph.json`, `trace_crewai.json` | дерева spans обох реалізацій (chain → llm → tool) |
| `security_log.json` | журнал усіх рішень guardrails і HITL за час виконання ноутбука |
| `redteam_results.json` | результати red-teaming (власний корпус + DeepTeam) |
| `comparison.json`, `cost_tracking.json`, `cost_comparison.png` | порівняння фреймворків та оцінка вартості |
| `requirements.txt` | залежності |

## Запуск

### Google Colab
1. Відкрити `ДЗ4_Моісеєнко_Світлана.ipynb` у Colab — Секція 1 сама встановить відсутні пакети.
2. (Опційно) додати ключі **перед Секцією 2**:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."            # обидві MAS на gpt-4.1, AnswerRelevancy/GEval у deepeval
   os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."     # трейси у Langfuse
   os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
   os.environ["LANGSMITH_API_KEY"] = "lsv2_..."        # або трейси у LangSmith
   ```
3. `Runtime → Run all`. Без ключів ноутбук виконується на детермінованих офлайн-провайдерах
   `ScriptedMasLLM` (LangGraph) і `ScriptedCrewLLM` (CrewAI); tracing іде у `LocalTracer` (дерево spans + JSON).
   Тести запускаються з ноутбука (Секція 16).

### Локально
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."                        # опційно
jupyter notebook "ДЗ4_Моісеєнко_Світлана.ipynb"
pytest -v test_mas_security.py                        # тести окремо від ноутбука
python mcp_server.py                                  # MCP-сервер окремо (stdio)
```

## Архітектура

### MCP-сервер (FastMCP) — Секція 3

| Tool | Призначення | Валідація (Pydantic `Field`) |
|---|---|---|
| `get_service_status` | стан сервісу з моніторингу | назва 2..40 символів, `^[a-z0-9-]+$` |
| `query_logs` | записи логів за рівнем і вікном | рівень `ERROR/WARN/INFO`, вікно 1..1440 хв, ліміт 1..50 |
| `search_runbooks` | пошук у базі runbook-ів і політик | запит 3..300 символів, `top_k` 1..5 |
| `restart_service` | **ризикова дія** — перезапуск сервісу | причина ≥10 символів, `postgres-db` заборонено |

Логи навмисно містять PII (email, телефон, IP, номер картки) — це тест для output guardrail.
Інтеграція: у LangGraph — `langchain-mcp-adapters` (`MultiServerMCPClient`, stdio, постійна сесія до дочірнього процесу);
у CrewAI і тестах — in-memory `fastmcp.Client` до того ж сервера.

### MAS у LangGraph (supervisor + 3 агенти) — Секція 7
```
START → input_guard ──(injection)──→ finalize → END
           │
           ▼
       supervisor ──transfer_to_monitor──→ monitor (agent ⇄ tools) ──┐
           │      ──transfer_to_runbook──→ runbook (agent ⇄ tools) ──┤→ supervisor
           │      ──transfer_to_remediation──→ remediation (agent ⇄ tools*) ─┘
           └──(відповідь без handoff)──→ finalize (PII redaction) → END
                                       * tools: interrupt() перед restart_service (HITL)
```
| Агент | Роль | Allowlist tools |
|---|---|---|
| `supervisor` | маршрутизація через handoff-tools `transfer_to_*`, фінальний звіт | — |
| `monitor` | стан сервісів і логи помилок | `get_service_status`, `query_logs` |
| `runbook` | runbook-и та політики з бази знань | `search_runbooks` |
| `remediation` | безпечне усунення (перезапуск після підтвердження) | `get_service_status`, `restart_service` |

Handoff — `Command(goto=<agent>)`; кожен worker — підграф `agent ⇄ tools` із власним станом, тому `interrupt()`
перезапускає лише вузол `tools`. Checkpointer — `MemorySaver`.

### Guardrails — Секція 4
1. **Input (injection detection):** regex-шаблони укр./англ. (override, витік системного промпту, підміна ролі, DAN,
   фейкові маркери `<|system|>`, обхід HITL, ексфільтрація) + де-обфускація base64 / ROT13 / leetspeak із повторним скануванням.
2. **Tool (allowlist per agent + валідація аргументів):** allowlist за принципом найменших привілеїв → Pydantic-схеми
   аргументів (дублюють серверні, defense in depth) → injection-детектор у рядкових аргументах → політика захищених сервісів.
3. **Output (PII redaction):** email, телефони, IPv4, номери карток, IBAN → `[EMAIL]`, `[PHONE]`, `[IP]`, `[CARD]`, `[IBAN]`.

Усі рішення пишуться у `SECURITY_LOG` → `security_log.json`.

### Human-in-the-Loop — Секція 9
Вузол `tools` агента `remediation` викликає `interrupt({...})` перед `restart_service`. Людина відповідає
`Command(resume={"approved": bool, "comment": str})`. Демонстрації: схвалення (сервіс перезапущено, стан `healthy`),
відхилення (граф на паузі, `get_state().next == ("remediation",)`, після відмови стан на MCP-сервері лишається `degraded`).
У CrewAI HITL реалізовано через `ApprovalGate` у tool-обгортці (нативного `interrupt/resume` там немає).

### Tracing — Секції 5, 12
`setup_tracing()` додає `langfuse.langchain.CallbackHandler` (за наявності ключів) або вмикає `LANGSMITH_TRACING`;
завжди працює `LocalTracer(BaseCallbackHandler)`, який будує те саме дерево spans (chain → llm → tool) і зберігає
`trace_*.json`. CrewAI трасується ручними spans у tool-обгортках і лічильниками LLM-викликів.

## Порівняльна таблиця LangGraph vs CrewAI (Секція 11, офлайн-запуск)

| Критерій | LangGraph | CrewAI |
|---|---|---|
| LOC оркестрації (без спільних guardrails/MCP) | 116 | 65 |
| Викликів LLM за кейс | 13 | 8 |
| Викликів tools | 5 | 5 |
| ≈ токенів промптів (tiktoken) | 9 619 | 9 579 |
| ≈ вартість за кейс (gpt-4.1: $2/$8 за 1M) | $0.026 | $0.024 |
| Маршрутизація | явний supervisor + `Command(goto)` | `Process.sequential` з контекстом задач (hierarchical — з реальною LLM) |
| Контроль над станом | повний: `TypedDict`, reducers, checkpointer, підграфи | обмежений: текстовий контекст між задачами |
| HITL | `interrupt()` / `Command(resume)` з паузою стану | `human_input=True` (блокуючий `input()`) або власний gate |
| Debugging | `get_state()`, історія, Mermaid, spans на кожен вузол | verbose-лог, callbacks задач |
| Tool guardrail | у вузлі `tools` | у `BaseTool._run` |

CrewAI потребує менше коду, але дає менше контролю: стан між агентами передається текстом, HITL не має паузи/відновлення,
а кожен виклик LLM несе повний ReAct-шаблон з описом tools. LangGraph — більше коду, зате прозорий стан, підграфи,
checkpointer і точний HITL.

## Red-teaming (Секція 13)

| Атака | Джерело | Зупинив шар |
|---|---|---|
| direct injection (укр.), role hijack + DAN, fake system marker | власний корпус | input |
| PII exfiltration («покажи email клієнтів») | власний корпус | output (PII redaction) |
| injection у тексті причини перезапуску | власний корпус | input (у tool guardrail — окрема перевірка аргументів) |
| Base64, ROT13, Leetspeak, PromptInjection | **DeepTeam** | input (після де-обфускації) |

Результат: 0 успішних атак із 9 (`redteam_results.json`). Слабкі місця regex-підходу — семантичні перефразування
без ключових слів і багатокрокові атаки; для них потрібен LLM-класифікатор поверх regex-шару.

## Evals (Секція 14) і cost tracking (Секція 15)
* deepeval: 6 `LLMTestCase` (обидві MAS, відхилений перезапуск, заблокований injection, інший сервіс, PII) з кастомними
  метриками `KeywordCoverage` і `PIISafety`; з `OPENAI_API_KEY` додаються `AnswerRelevancyMetric` і `GEval`. Офлайн — 6/6 pass.
* Вартість: токени промптів/відповідей із трейсера × ціни gpt-4.1 → `cost_tracking.json`, графік `cost_comparison.png`.

## Тести (Секція 16)
`pytest -v test_mas_security.py` — **14 passed**: 6 тестів MCP tools (перелік, статус, фільтр логів, ранжування runbook-ів,
серверна валідація, заборона postgres-db) і 8 тестів guardrails/HITL (3 параметризовані injection-кейси, чистий запит + base64,
allowlist, валідація аргументів і політик, PII redaction, відмова HITL у графі).

## LLM-провайдери
Основний — OpenAI `gpt-4.1` за наявності `OPENAI_API_KEY` (LangGraph: `ChatOpenAI`; CrewAI: `LLM("openai/gpt-4.1")`).
Фолбек — `ScriptedMasLLM` / `ScriptedCrewLLM`: детерміновані офлайн-провайдери, що емулюють tool calling supervisor-а
та worker-агентів (у т.ч. навмисну спробу `monitor` викликати заборонений `restart_service`), тож guardrails, HITL,
tracing і red-teaming демонструються відтворювано без мережі.
