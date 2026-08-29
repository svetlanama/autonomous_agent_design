"""База знань «Runbook-и та політики SRE» у ChromaDB (з ДЗ2): детермінований ембеддинг без мережі."""

import hashlib
import math
import re

import chromadb
from chromadb.api.types import EmbeddingFunction

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
    "sla_credits": "Політика SLA-компенсацій (credits). Компенсація на наступний рахунок: доступність не нижче цілі — 0%; "
                   "від 99.0% до цілі — 10% місячної плати; від 95.0% до 99.0% — 25%; нижче 95.0% — 50%. "
                   "Компенсація нараховується за запитом клієнта протягом 30 днів після інциденту, максимум 50%.",
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


class HashingEmbedding(EmbeddingFunction):
    """Детермінований локальний ембеддинг: хешування слів і символьних 3-грам (покриває українську морфологію)."""

    def __init__(self, dim: int = 512, word_weight: float = 3.0):
        self.dim = dim
        self.word_weight = word_weight

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vec = [0.0] * self.dim
            for token in (t for t in re.findall(r"[\w-]+", text.lower()) if len(t) > 2):
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


def build_knowledge():
    """Створює ChromaDB-колекцію з документами бази знань."""
    collection = chromadb.EphemeralClient().get_or_create_collection(
        "sre_runbooks", embedding_function=HashingEmbedding(), metadata={"hnsw:space": "cosine"})
    collection.upsert(ids=list(KNOWLEDGE_DOCS), documents=list(KNOWLEDGE_DOCS.values()))
    return collection


knowledge = build_knowledge()


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Повертає top_k документів із similarity (1 − cosine distance)."""
    found = knowledge.query(query_texts=[query], n_results=top_k)
    return [{"doc_id": doc_id, "similarity": round(1 - dist, 3), "text": doc}
            for doc_id, doc, dist in zip(found["ids"][0], found["documents"][0], found["distances"][0])]


if __name__ == "__main__":
    print(f"ChromaDB-колекція '{knowledge.name}': {knowledge.count()} документів")
    for doc in retrieve("компенсація за порушення SLA", 3):
        print(f"  {doc['doc_id']:22} similarity={doc['similarity']:.3f}  {doc['text'][:70]}…")
