# Практичне завдання 1 — Автономна агентна система «SRE-асистент з управління інцидентами»

**Автор:** Моісеєнко Світлана · курс «Проєктування автономних агентів»

Повний цикл побудови автономного LLM-агента для предметної області **управління ІТ-інцидентами (SRE / on-call)**:
доменні інструменти з Pydantic v2, ReAct-агент у LangGraph із guardrails, Plan-and-Execute зі structured outputs,
checkpointer `SqliteSaver`, agentic RAG на ChromaDB, human-in-the-loop через `interrupt_before` та тести pytest.

## Файли

| Файл | Опис |
|---|---|
| `ДЗ3_Моісеєнко_Світлана.ipynb` | основний ноутбук (Colab / Jupyter), 14 секцій, виконаний з результатами |
| `incident_agent.py` | бібліотечна частина ноутбука (інструменти, LLM, графи) — для імпорту в тестах |
| `test_incident_agent.py` | 19 тестів pytest: 9 схем, 7 інструментів, 3 ReAct-циклу |
| `pytest_output.txt`, `pytest_results.png` | вивід `pytest -v` та його скриншот |
| `trajectory.json` | JSON-лог траєкторій усіх запусків ReAct і Plan-and-Execute + порівняння |
| `checkpoints.sqlite` | БД checkpointer-а після виконання ноутбука (thread-и `incident-42`, `persist-demo`, `restart-approve`, `restart-reject`) |
| `agent_results.png` | графік: кроки на запуск та порівняння ReAct vs Plan-and-Execute |

База знань (10 документів) вбудована у код — словник `KNOWLEDGE_DOCS` у Секції 2.

## Запуск

### Google Colab
1. Відкрити `ДЗ3_Моісеєнко_Світлана.ipynb` у Colab.
2. (Опційно) додати ключ OpenAI **перед Секцією 4**:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."
   ```
3. `Runtime → Run all`. Без ключа ноутбук виконується на детермінованому офлайн-провайдері
   `ScriptedSreLLM` — усі демонстрації відтворювані. Тести запускаються з ноутбука (Секція 13).

### Локально
```bash
pip install langgraph langgraph-checkpoint-sqlite langchain-core langchain-openai chromadb "pydantic>=2" pytest matplotlib
export OPENAI_API_KEY="sk-..."          # опційно
jupyter notebook "ДЗ3_Моісеєнко_Світлана.ipynb"
pytest -v test_incident_agent.py         # тести окремо від ноутбука
```

## Архітектура

### Інструменти (Секція 3)

| Інструмент | Призначення | Валідація (`field_validator`) |
|---|---|---|
| `search_runbooks` | **RAG**: пошук у базі runbook-ів та SRE-політик | запит 3..300 символів, `top_k` 1..5 |
| `get_service_status` | стан сервісу з моніторингу | назва сервісу з довідника, нормалізація регістру |
| `query_logs` | записи логів за рівнем і вікном | рівень `ERROR/WARN/INFO`, вікно 1..1440 хв |
| `calculate_sla` | доступність, бюджет помилок, порушення SLA | простій ≤ періоду, ціль 90..100 % |
| `restart_service` | **ризикова дія** — перезапуск сервісу | причина ≥10 символів, `postgres-db` заборонено |

Усі інструменти повертають JSON `{"status": "ok", "data": {...}}` або `{"status": "error", "error": "..."}`.
**Fallback:** якщо `query_logs` повертає помилку, вузол `tools` автоматично викликає `get_service_status`.

### ReAct-агент (Секція 5)
```
START → agent ──(tool calls)──→ tools ──→ agent … → finalize → END
            └──(restart_service)──→ risky_tools   ← interrupt_before (HITL)
```
Guardrails у `AgentConfig`: `max_steps=10`, `timeout_s=120`, детекція повторних викликів (ідентичний
tool call → зупинка з причиною `loop`). Кожен запуск пише траєкторію (`thought / observation / guardrail /
final_answer`) у `trajectory.json`.

### Plan-and-Execute (Секція 7)
```
START → planner → executor → replanner ──continue / revise──→ executor
                     ↑                └──finish──→ END
              вкладений ReAct-агент (max_steps=4)
```
`planner` — `with_structured_output(Plan)`; `replanner` — `with_structured_output(ReplanDecision)`
з діями `continue / revise / finish`. Демо: невідомий сервіс `billing` → помилка → `revise`.

### Checkpointer (Секція 8)
`SqliteSaver` поверх файлу `checkpoints.sqlite`:
* пам'ять між запитами в одному `thread_id` («перевір payments» → «а тепер його логи»);
* відновлення після «збою»: граф з `interrupt_before=["tools"]` зупиняється, з'єднання закривається,
  новий процес відкриває файл і продовжує через `invoke(None, config)`;
* `get_state()` — знімок стану, `get_state_history()` — історія checkpoint-ів.

### Agentic RAG (Секція 9)
Агент викликає `search_runbooks` лише для питань «що робити / яка політика»; для статусу чи SLA —
не викликає. Рішення ухвалює LLM за описом інструмента та системним промптом.

### Human-in-the-Loop (Секція 10)
Граф з `interrupt_before=["risky_tools"]`: перед `restart_service` виконання зупиняється,
людина бачить аргументи. **Approve** — `invoke(None, config)`; **reject** — `update_state(..., as_node="risky_tools")`
підмінює результат інструмента повідомленням «ВІДХИЛЕНО», сервіс не перезапускається.

## Приклади використання
```python
run_agent("Перевір статус сервісу payments і подивись його логи помилок за 30 хвилин.")
run_plan_execute("Розслідуй інцидент з payments: перевір статус, переглянь логи, знайди runbook і порахуй SLA при 45 хв простою за 30 днів.")
```

## Результати
* Порівняння на одній задачі: ReAct — 5 викликів LLM, 4 tool calls; Plan-and-Execute — 15 викликів LLM,
  4 tool calls, 5 кроків плану (planner + replanner після кожного кроку).
* Тести: `19 passed` (`pytest_output.txt`).

## Бонуси
* Порівняння ReAct vs Plan-and-Execute (Секція 11) — числова таблиця + графік.
* Mermaid-візуалізація обох графів через `get_graph().draw_mermaid()`.
* Fallback-стратегія інструментів (`FALLBACK_TOOLS`).

## LLM-провайдери
Основний — OpenAI `gpt-4.1` (`ChatOpenAI`) за наявності `OPENAI_API_KEY`. Фолбек — `ScriptedSreLLM`:
детермінований офлайн-провайдер, що емулює tool calling, `Plan` і `ReplanDecision` правилами, тож
guardrails, replanning, checkpointing та HITL демонструються відтворювано без мережі.
