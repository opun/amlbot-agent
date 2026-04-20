# Логика системы и промпты

Документ описывает полный pipeline AML-tracer'а: от входной точки (CLI/API) до финального отчёта и визуализации, а также приводит **полный текст всех LLM-промптов** и указывает, где именно в коде они вызываются.

> Версия — по состоянию `main` на 2026-04-18. Основной модуль — `src/agent/base_tracer.py` (≈3 700 строк).

---

## 1. Общая картина

Агент решает одну задачу: **проследить украденные средства** от кошелька жертвы через цепочку адресов до «терминала» (CEX-депозит, мост, миксер) или тупика. Он сочетает:

- **Детерминированный код** (Python) — основной цикл обхода (hop-by-hop), хронологическая аккумуляция, FIFO-учёт, планировщик путей, приоритетная очередь.
- **LLM-вызовы** (OpenAI) в трёх точках: классификация каждого адреса (роль + терминал?), выбор транзакций для продолжения, финальная валидация JSON.

Главный «агентный» промпт лежит в `docs/prompts/trace_orchestrator.reference.md` — это справочный документ, описывающий правила, на которых построен код. Runtime его не загружает. Живые LLM-вызовы идут через `trace_hop_classifier.md`, `trace_hop_selector.md` и `trace_validator.md`.

---

## 2. Точки входа

### 2.1 CLI — `src/agent/cli.py`
- Интерактивный ввод: описание кейса, адрес жертвы / tx_hash, сеть, актив, дата (опц.).
- Два режима:
  1. **Address mode** — дан только адрес; theft-транзакция детектится по `token_stats` + `all_txs`.
  2. **Tx-hash mode** — дан хэш кражи; адрес жертвы, токен и время извлекаются из него.
- Описание кейса, если передано, парсится LLM-ом (`parse_case_description_with_llm` в `theft_detection.py`).
- Поднимает MCP-сервер (stdio через Docker) и вызывает `tracer.trace(config)`.

### 2.2 HTTP API — `src/agent/api.py`
FastAPI-сервис. Ключевые endpoints:

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/chat` | Мультиходовый чат: собирает параметры, подтверждает, запускает трейс (стейт-машина `initial → collecting → confirming → tracing → …`). |
| POST | `/api/trace` | Прямой запуск трейса с явными параметрами. Возвращает стрим JSON-событий (`trace_started`, `status`, `result`, `error`). |
| GET  | `/api/session/{id}` | Состояние чат-сессии. |
| DELETE | `/api/session/{id}` | Удалить сессию. |
| GET  | `/api/health` | Health-check. |
| GET  | `/openapi.yaml`, `/docs`, `/redoc` | API-схема и доки. |

Важные переменные окружения:
- `MCP_USE_HTTP` — использовать HTTP-MCP вместо stdio.
- `MCP_SERVER_URL` — URL MCP-сервера (по умолч. `http://localhost:8001`).
- `MCP_USER_ID` — для stdio-клиента.
- `SESSION_MAX_COUNT` (1000), `SESSION_TTL_SECONDS` (3600) — ограничения сессий чата.
- `CORS_ALLOWED_ORIGINS` — фронт.
- `AGENT_PARALLEL_TOOL_CALLS`, `AGENT_MAX_CONCURRENT_TOOLS` — включает параллельные tool-calls в оркестраторе.
- `AGENT_DISABLE_OPENAI_TRACING` — отключить OpenAI tracing spans.

---

## 3. Конфигурация моделей — `src/agent/config.py`

```python
ORCHESTRATOR_MODEL = os.getenv("OPENAI_ORCHESTRATOR_MODEL", "gpt-5-mini")
SELECTOR_MODEL     = os.getenv("OPENAI_SELECTOR_MODEL",     "gpt-5-mini")
VALIDATOR_MODEL    = os.getenv("OPENAI_VALIDATOR_MODEL",    "gpt-4o")
JSON_RETRY_MODEL   = os.getenv("OPENAI_JSON_RETRY_MODEL",   "gpt-4o")
```

- `SELECTOR_MODEL` используется и для hop-селектора, и для hop-классификатора.
- `VALIDATOR_MODEL` — финальная проверка/чинка JSON.
- `JSON_RETRY_MODEL` — восстановление malformed-JSON.

---

## 4. Модели данных — `src/agent/models.py`

- **TracerConfig** — вход: `victim_address`, `blockchain_name`, `asset_symbol`, `approx_date`, `tx_hash`, `known_tx_hashes`, `theft_asset`, `stolen_amount`, `cex_single_cluster_threshold` (0.60), `traced_amount_tolerance` (0.03).
- **CaseMeta** — `case_id`, `trace_id`, `victim_address`, `chains`, `asset_symbol`, `token_id`, …
- **Step** — один перевод в пути: `step_index`, `from`, `to`, `tx_hash`, `chain`, `asset`, `amount_estimate`, `time`, `direction`, `step_type` ∈ {`direct_transfer`, `bridge_in/out/transfer/arrival`, `service_deposit`, `internal_transfer`}, `service_label`, `protocol`, `reasoning`, `attributed_amount`.
- **Path** — линейная цепочка `steps`, `path_id`, `stop_reason`.
- **Entity** — `address`, `chain`, `role` ∈ {`victim`, `perpetrator`, `intermediate`, `bridge_service`, `cex_deposit`, `dex_service`, `otc_service`, `unidentified_service`, `cluster`}, `risk_score`, `labels`, `riskscore_signals`, `notes`.
- **TraceResult** — финал: `case_meta`, `paths[]`, `entities[]`, `annotations[]`, `trace_stats`, `visualization_url`.

---

## 5. Ядро — `BaseTracer`

Класс `BaseTracer` (`base_tracer.py:405+`) абстрактный; транспорт (stdio-MCP / HTTP-MCP) реализуют подклассы `MCPTracer`, `HTTPTracer` через `execute_tool()`.

### 5.1 Загрузка промптов
```python
self.validator_prompt_path      = .../prompts/trace_validator.md
self.selector_prompt_path       = .../prompts/trace_hop_selector.md
self.hop_classifier_prompt_path = .../prompts/trace_hop_classifier.md
```
Функции `_load_validator_prompt`, `_load_selector_prompt`, `_load_hop_classifier_prompt` — ленивое чтение файла. Справочный `trace_orchestrator.reference.md` живёт в `docs/prompts/` и в рантайм не грузится.

### 5.2 Планировщик — `HopScheduler` (`base_tracer.py:77-140`)
Приоритетная очередь hop-джобов. Приоритет:
1. `-attributed_amount` (большие ветки вперёд);
2. `hop_index` (меньше — раньше, чтобы быстро добирать терминалы);
3. FIFO-тайбрейк.

Бюджет: `max_completed` завершённых путей. Hard-limit — `max_iterations = max_completed * 64`. Это фикс прежней проблемы, когда большое число мелких dead-end «братьев» съедало бюджет до того, как дойдёт хоп с реальным CEX.

### 5.3 FIFO-учёт — `FIFOLedger` (`base_tracer.py:142-251`)
Каждый адрес — очередь inflow-ов, у каждого своя `theft_share`. При outflow — FIFO-attribution пропорционально доле кражи:

- `record_inflow(addr, amount, theft_share)` — приход.
- `attribute_outflow(addr, outflow) → attributed` — сколько «украденного» уходит с адреса.
- `claim_terminal(attributed)` — списание против **глобального cap** `stolen_amount * (1 + tolerance)` — **только** на терминале (CEX/мост/тупик). Промежуточные хопы cap не тратят.

### 5.4 HopJob (`base_tracer.py:63-74`)
```python
@dataclass
class HopJob:
    path_id, current_address, incoming_tx_hash, incoming_amount,
    incoming_time, chain, asset, token_id, hop_index,
    attributed_amount: float = 0.0
```

### 5.5 Главный цикл — `_run_agentic_trace` (`base_tracer.py:1485-3169`)

Псевдокод:

```
init: entities={}, paths=[], ledger=FIFOLedger(stolen_amount),
      scheduler=HopScheduler(max_completed), enqueue(victim → первая theft-tx)

while scheduler.should_continue(completed_paths):
    batch = pop(N) jobs

    # Phase 1 — параллельно для каждого job:
    gather(
        get_address(chain, address),
        get_extra_address_info(address, asset),
        bridge_analyze(chain, incoming_tx)  # только если hop_index <= 3
    )

    # Phase 2 — последовательно внутри job (но job'ы батча — параллельно):
    for job in batch:
        role, terminal = hop_classifier(address, get_addr, get_extra, owner_hint)
                                    # → LLM: trace_hop_classifier.md
        early_hop_safeguard()         # hop ≤ 3 + нет owner → не терминал
        build Entity

        if role == bridge_service and bridge_analyzer.is_bridge:
            create bridge step, enqueue HopJob на dst_chain

        if role in {cex_deposit, unidentified_service(mixer)}:
            ledger.claim_terminal(job.attributed_amount)
            complete path; continue

        # OTC: continue tracing + annotation "Ownership Change Risk"

        # Fetch outgoing + хронологическая аккумуляция
        txs = all_txs(address, filter={time: >=incoming_time, token_id})
        selected = hop_selector(txs, incoming_amount, incoming_time)
                                    # → LLM: trace_hop_selector.md
        for tx in selected:
            tt = token_transfers(tx.hash)
            recipient, block_time, owner_hint = tt.output.address, tt.block_time, tt.owner_hint
            attributed = ledger.attribute_outflow(address, tx.amount)
            create Step
            if recipient in current_path: stop (cycle)
            scheduler.push(HopJob(recipient, tx.hash, tx.amount, block_time,
                                  chain, asset, token_id, hop_index+1, attributed))

postprocess_trace_result(...)
generate_visualization_payload(...)
```

### 5.6 Методы, вызывающие LLM

| Метод | Промпт | Модель | Когда |
|---|---|---|---|
| `_run_hop_classifier(ctx)` (`1097-1126`) | `trace_hop_classifier.md` | `SELECTOR_MODEL` | После Phase 1 на каждом адресе |
| `_run_selector(ctx)` (`1065-1095`) | `trace_hop_selector.md` | `SELECTOR_MODEL` | Когда `all_txs` вернул список исходящих |
| `_run_validator(payload)` (`1130-1157`) | `trace_validator.md` | `VALIDATOR_MODEL` | Финальная нормализация результата |

Все LLM-ответы обёрнуты в OpenAI `generation_span(...)` для трейсинга; JSON извлекается `_strip_code_fences` и парсится `json.loads`.

---

## 6. Инструменты (MCP tools)

Определены в `base_tracer.py:TOOLS` и продублированы во всех клиентах.

| Инструмент | Что принимает | Что возвращает |
|---|---|---|
| `expert_search(hash, filter)` | address или tx | поисковая выдача |
| `get_address(blockchain, address)` | **адрес** (42 симв. ETH, 34 TRON) | `owner`, `risk_score`, `riskscore_signals` |
| `get_extra_address_info(address, asset)` | адрес + актив | `services.use_platform` (платформы, с которыми адрес взаимодействовал) |
| `token_stats(blockchain, address)` | адрес | балансы по токенам, `total_out` |
| `all_txs(address, blockchain, filter, limit, offset, direction, order, transaction_type)` | адрес + фильтр `{time: {">=": t}, token_id: [id]}` | список **хэшей** (не адресов!) |
| `get_transaction(address, tx_hash, blockchain, token_id, path)` | tx + адрес-якорь | детали транзакции |
| `token_transfers(tx_hash, blockchain)` | tx | `input.address`, `output.address`, `amount`, `block_time` — **отсюда берётся recipient** |
| `bridge_analyze(chain, tx_hash)` | tx | `is_bridge`, `dst_chain`, `destination_address`, `protocol` |
| `get_position(...)` | tx | позиция транзакции |
| `save_and_share_visualization(...)` | payload | shareable URL |

**Транспорт**:
- `src/agent/mcp_client.py` — stdio через Docker, на базе openai-agents Agent/Runner.
- `src/agent/mcp_http_client.py` — async httpx, **с кэшем и дедупликацией in-flight запросов**. TTL по умолчанию 600 с, для `bridge_analyze` — 3600 с, для `all_txs` — 120 с. Это критично: одна и та же «хаб»-адрес может ловить десятки запросов из разных путей — кэш экономит вызовы.
- `src/agent/tool_dispatch.py` — enum `ToolName` + `dispatch_tool(client, name, args)`.

Таймауты: базово 30 с, 60 с для «тяжёлых» (`all_txs`, `bridge_analyze`, `get_extra_address_info`).

---

## 7. Полный flow: от входа до выхода

```
CLI / POST /api/trace
        │
        ▼
parse_case_description_with_llm()   (если есть description)
        │
        ▼
extract_victim_from_tx_hash()        (если дан только tx_hash)
infer_asset_symbol()                  (если актив не указан)
        │
        ▼
BaseTracer.trace(config)
        │
        ▼
_run_agentic_trace(payload)
        │
        ├── init: FIFOLedger, HopScheduler, entities, paths
        │
        └── main loop:
             pop HopJob
             ├─ Phase 1 (parallel): get_address, get_extra_address_info, bridge_analyze
             ├─ LLM: trace_hop_classifier.md → role, terminal?
             ├─ bridge path    → enqueue на dst_chain
             ├─ CEX/mixer      → claim_terminal, complete path
             ├─ OTC            → annotation, продолжить
             ├─ otherwise:
             │   ├─ all_txs (фильтр по time + token_id)
             │   ├─ LLM: trace_hop_selector.md → selected_hashes
             │   └─ для каждого: token_transfers → recipient → enqueue HopJob
             └─ cycle detection, dead-end → stop_reason
        │
        ▼
postprocess_trace_result()
  · линейность путей (split siblings)
  · нормализация step_type / amounts
  · добавление отсутствующих entity
  · cycle detection
  · сбор списка chains
        │
        ▼
generate_visualization_payload()     (visualization.py)
  · node/edge-граф
  · currencies.json → symbol/unit/issuer
  · нативные активы, синтетические хэши
        │
        ▼
reporting.build_summary_text / build_graph / build_mermaid_graph
        │
        ▼
TraceResult → стрим (API) или вывод (CLI)
```

---

## 8. Классификация и терминалы

Базовые правила (из `trace_hop_classifier.md` + эвристики в коде):

| Условие | role | terminal |
|---|---|---|
| `owner` содержит exchange / binance / coinbase / kraken / huobi / okx / kucoin / bitfinex / mxc / gate / poloniex / bybit | `cex_deposit` | **да** |
| `owner` или `services.use_platform` содержит bridge / layerzero / stargate / wormhole / allbridge / synapse / hop / multichain / across / router / bridgers | `bridge_service` | **да** (но продолжаем на `dst_chain`) |
| ключевое слово dex / swap / uniswap / sushi / pancake / curve | `dex_service` | да |
| mixer / tornado / blender / sinbad | `unidentified_service` | **да** всегда |
| otc | `otc_service` | **нет** (трассируем дальше, помечаем "Ownership Change Risk") |
| `risk_score > 0.75` | не меняет role | **нет** (только label "High Risk") |
| `owner == null` и `services == {}` | `intermediate` | нет |
| hop 0, без identity | `perpetrator`, label "Suspected Perpetrator" | нет |

**Слабый vs сильный сигнал**: ключевое слово в `owner`/`owner_hint` — сильный (адрес принадлежит сервису). Ключевое слово **только** в `services.use_platform` — слабый (адрес лишь взаимодействовал). Слабый сигнал — не повод помечать terminal, кроме миксеров.

**Early-hop safeguard** (`2587-2593`): на первых 3 хопах `terminal=true` для адресов без подтверждённого owner'а принудительно сбрасывается — чтобы не застревать на ложных «биржах» рядом с перпетратором.

---

## 9. Мосты и кросс-чейн

1. На первых ≤3 хопах заранее параллельно вызывается `bridge_analyze(chain, incoming_tx_hash)`.
2. Если `role=bridge_service` и `bridge_analyze.is_bridge`:
   - создаётся шаг `bridge_transfer`/`bridge_out`;
   - из ответа берётся `dst_chain`, `destination_address`, `protocol`;
   - в HopScheduler пушится новый `HopJob` со свежим `chain`, обновлённым `token_id`, `attributed_amount` из FIFO.
3. Агрегация на мосту: если output-сумма отличается от input > 20 %, добавляется аннотация, дальше трассируем по output-сумме.

---

## 10. Хронологическая аккумуляция

Реализована в `trace_hop_selector.md` (LLM) **и** зеркалом в коде (`_run_agentic_trace`, вокруг 2762-3033). Алгоритм:

```
txs := результаты all_txs, отсортированные по block_time asc, отфильтрованные по token_id
accumulated := 0
selected := []
for tx in txs:
    selected.append(tx.hash)
    accumulated += tx.amount
    if accumulated >= incoming_amount: break
    gap := (incoming_amount - accumulated) / incoming_amount
    if gap <= 0.015:                          # 1.5 % slippage
        break
```

Ключевое:
- порядок транзакций **не меняется** — никаких сортировок по сумме;
- 1.5 % slippage покрывает комиссии и свопы;
- если `incoming_amount == 0` — берём только **первую** транзакцию;
- затем на каждый выбранный hash — `token_transfers` для получения recipient'а;
- рассылка по веткам: максимум 1–3 ветки на хоп, каждая — **отдельный `path_id`** (линейный).

---

## 11. Детерминированный трейсер — `deterministic_tracer.py`

Альтернатива без LLM (778 строк):

- Классификация — чистый keyword-match (`_classify_address`, `_detect_role`).
- Обход — DFS-стек (не приоритетная очередь).
- Для каждого адреса: bridge-check → terminal-check → fetch outgoing → хронологическая аккумуляция → push.
- Быстрее и предсказуемее, но слабее на неоднозначных адресах.

Используется как fallback / для быстрых проверок.

---

## 12. Theft detection — `theft_detection.py`

- `parse_case_description_with_llm(desc)` — Agent, извлекающий из произвольного описания victim/tx/chain/asset/date.
- `infer_approx_date_from_description(desc)` — regex-парсер дат (YYYY-MM-DD, DD.MM.YYYY, «Month DD YYYY» и варианты).
- `extract_victim_from_tx_hash(tx, chain, client)`:
  1. `token_transfers(tx)` → `input.address` как victim, + `token_id`, `asset`, `block_time`;
  2. fallback для нативных tx: `expert_search(tx)` → `get_transaction(any_involved_addr, tx)`.
- `infer_asset_symbol(config, client)` — если актив не задан, берём `token_stats(victim)` и выбираем токен с максимальным `total_out`.

---

## 13. Post-processing — `trace_postprocess.py`

`postprocess_trace_result(result)`:

1. **Линейность пути** — если `steps[i].from != steps[i-1].to`, путь разбивается на несколько (с суффиксами `path_id` вида `"1.2"`).
2. **Нормализация step'ов** — `step_type` приводится к допустимому множеству (иначе `direct_transfer`), amounts/times — к числам, step_index — к последовательному.
3. **Полнота entities** — любые адреса из steps, не попавшие в `entities`, добавляются минимальной записью (role=intermediate, risk_score=0, note="Auto-added by validator").
4. **Циклы** — если адрес повторяется в пути, добавляется annotation и ставится `stop_reason`.
5. **Chains** — собираются в `case_meta.chains`.
6. **Stats** — `explored_paths` обновляется.

---

## 14. Визуализация и отчёт

### `visualization.py`
- `generate_visualization_payload(trace_result, txs_collected, tx_list_collected, address_info)` — строит node/edge payload для фронта.
- `_lookup_currency(chain, token_id)` — `src/currencies.json` (через `@lru_cache`).
- `_build_currency_info(chain, token_id, asset_hint)` — symbol/name/unit/issuer; fallback на хардкод-мапу.
- Token-id достаётся даже из «синтетических» хэшей формата `{hash}-{chain}-{token_id}-{idx}` для нативных активов.
- Помечает терминальные узлы по `stop_reason` (свежий коммит `36edc0e`).

### `reporting.py`
- `build_summary_text` — человекочитаемое резюме (пути, terminal'ы, риски).
- `build_graph` — JSON node/edge.
- `build_mermaid_graph` — Mermaid-граф со стилизацией по role.
- В `api.py`: `build_report` объединяет текст + Mermaid + ASCII-дерево в один объект ответа.

---

## 15. Параллельность

- `asyncio.gather` внутри Phase 1 каждого job'а.
- Если `AGENT_PARALLEL_TOOL_CALLS=true` — Phase 1 и Phase 2 сразу для батча джобов.
- `AGENT_MAX_CONCURRENT_TOOLS` (3 по умолч.) — семафор на одновременные tool-calls.
- HTTP-MCP: `InFlightDedup` — одинаковые запросы в рантайме склеиваются, затем всем раздаётся один результат.

---

## 16. Промпты — полный текст и точки вызова

### 16.1 `trace_orchestrator.reference.md` — справочный, рантайм его не грузит

Определяет правила, ссылки на инструменты и критерии классификации. Назначение — единый источник правды для команды при изменении classifier/selector-логики. Исторически был system-prompt'ом legacy-оркестратора; сегодня ни один runtime-путь его не читает.

**Структура**:
1. Task и User Inputs (`{victim_address}`, `{tx_hash}`, `{blockchain_name}`, `{asset_symbol}`, `{approx_date}`, `{description}`).
2. Список MCP-инструментов с разделением адрес vs tx hash и явным запретом их путать (66 символов tx vs 42 address).
3. **Tracing Rules**, 11 секций: Input Processing, Theft Transaction Selection, Entity Classification (strong/weak signals, автоконтинуация, critical-do-not-stop-early), Bridge Detection (3 шага), Path Following (хронологическая аккумуляция с 1.5 % slippage и примерами), Pattern Detection, Output Format (JSON schema), Decision Style, Output Format Requirements, Efficiency & Anti-Stuck, Selector Results.
4. Полная JSON-схема `TraceResult` и примеры корректного/некорректного заполнения `paths`.

Полный файл — `docs/prompts/trace_orchestrator.reference.md`.

### 16.2 `trace_hop_classifier.md` — классификатор адреса на хопе

**Вызов**: `BaseTracer._run_hop_classifier(context)` (`base_tracer.py:1097-1126`).
**Модель**: `SELECTOR_MODEL` (по умолч. `gpt-5-mini`). `max_tokens=250`. Ответ — чистый JSON.

**Вход (user message)** — JSON с полями:
- `address`, `chain`, `asset`
- `incoming_tx_hash`, `incoming_amount`
- `get_address` (ответ MCP)
- `get_extra_address_info` (ответ MCP)
- `owner_hint` (опц., из `token_transfers`)

**Выход**:
```json
{
  "role": "intermediate | victim | perpetrator | bridge_service | cex_deposit | dex_service | otc_service | unidentified_service | cluster",
  "terminal": true | false,
  "stop_reason": "string | null",
  "labels": ["..."],
  "notes": "string | null",
  "service_label": "string | null",
  "protocol": "string | null"
}
```

**Полный текст промпта** (`src/agent/prompts/trace_hop_classifier.md`):
```
# Hop Classifier (Recipient Address)

You are a hop-level classifier. Given address metadata, decide the role and whether this hop is terminal.

## Input (JSON)
You will receive:
- `address` (string)
- `chain` (string)
- `asset` (string)
- `incoming_tx_hash` (string | null)
- `incoming_amount` (number | null)
- `get_address` (object) result
- `get_extra_address_info` (object) result
- `owner_hint` (object | null) optional owner info from token_transfers output

## Classification Rules

There are two signal sources with different strengths:
- **Strong signal (owner)**: `get_address.data.owner` (name, slug, subtype) and `owner_hint` — these confirm who **owns** the address.
- **Weak signal (services)**: `get_extra_address_info.data.services.use_platform` — this only means the address has **interacted with** a platform, NOT that it belongs to that platform.

### Keyword matching
- Mixer keywords: mixer, tornado, blender, sinbad → `role=unidentified_service`, `terminal=true`
- OTC keywords: otc → `role=otc_service`, `terminal=false` (OTC-like entities are NOT terminal; tracing continues through them)
- Exchange keywords: exchange, binance, coinbase, kraken, okx, huobi, kucoin, bybit, gate, bitfinex, mxc, gate.io, poloniex → `role=cex_deposit`
- Bridge keywords: bridge, layerzero, stargate, wormhole, allbridge, synapse, hop, multichain, across, router, bridgers → `role=bridge_service`
- DEX keywords: dex, swap, uniswap, sushiswap, pancakeswap, curve → `role=dex_service`
- Otherwise: `role=intermediate`, `terminal=false`

### Terminal decision
- If a keyword match comes from the **owner** field (strong signal) → `terminal=true`
- If a keyword match comes **only** from `services.use_platform` (weak signal) → `terminal=false` — the address used that platform but is not owned by it; tracing must continue.
- Mixers are always terminal regardless of signal source.

**Important**: Only match keywords against owner identity fields (name, slug, subtype), owner_hint, and service platform names. Do NOT match against risk signal names, annotations, or other metadata.

If risk score > 0.75, add label "High Risk" but do not mark terminal unless the above owner-based rules match.

## Output (JSON only)
Return:
{
  "role": "intermediate | victim | perpetrator | bridge_service | cex_deposit | dex_service | otc_service | unidentified_service | cluster",
  "terminal": true | false,
  "stop_reason": "string | null",
  "labels": ["..."],
  "notes": "string | null",
  "service_label": "string | null",
  "protocol": "string | null"
}

No markdown. No extra text.
```

### 16.3 `trace_hop_selector.md` — выбор транзакций для продолжения

**Вызов**: `BaseTracer._run_selector(context)` (`base_tracer.py:1065-1095`).
**Модель**: `SELECTOR_MODEL`. `max_tokens=300`. Ответ — JSON.

**Вход**: JSON с `chain`, `asset`, `incoming_amount`, `incoming_time`, `txs[]` (из `all_txs`).

**Выход**:
```json
{"selected_hashes": ["tx1", "tx2"], "reasoning": "..."}
```

**Полный текст** (`src/agent/prompts/trace_hop_selector.md`):
```
# Hop Selector (Chronological Accumulation)

You are a hop-level selector. Given outgoing transactions for a single address, follow the **chronological accumulation algorithm** from `trace_orchestrator.md`.

## Input (JSON)
You will receive a JSON object with:
- `chain` (string)
- `asset` (string)
- `incoming_amount` (number | null)
- `incoming_time` (number | null)
- `txs` (array) from `all-txs` containing fields like `hash`, `amount`, `amount_coerced`, `block_time`, `token_id`, `type`

## Rules
- Process transactions **in the order provided** (chronological ascending).
- Initialize `accumulated_amount = 0`, `selected_hashes = []`.
- For each tx in order:
  - Add tx hash to `selected_hashes`.
  - Add tx amount to `accumulated_amount`.
  - Stop immediately when:
    1) `accumulated_amount >= incoming_amount`, OR
    2) remaining gap `(incoming_amount - accumulated_amount) / incoming_amount <= 0.015`
- **Do not skip transactions**. Do not reorder by amount.
- If `incoming_amount` is null or 0, pick the **first** transaction hash only.

## Output (JSON only)
Return a single JSON object:
{
  "selected_hashes": ["tx_hash_1", "tx_hash_2"],
  "reasoning": "Short explanation (chronological accumulation)"
}

No markdown. No extra text.
```

### 16.4 `trace_validator.md` — финальный JSON-чинитель

**Вызов**: `BaseTracer._run_validator(payload)` (`base_tracer.py:1130-1157`). Таймаут 60 с.
**Модель**: `VALIDATOR_MODEL` (`gpt-4o`). Нет `max_tokens` — ответ может быть большой (полный TraceResult).

**Вход**: «сырой» `TraceResult` (может быть битым).

**Выход**: валидный `TraceResult`-JSON.

**Полный текст** (`src/agent/prompts/trace_validator.md`):
```
You are a JSON validator and fixer for crypto trace results.

Input: a TraceResult-like JSON object (may be malformed or missing fields).
Output: a valid JSON object that strictly conforms to the TraceResult schema.

Rules:
1. Output ONLY raw JSON. No markdown, no commentary.
2. Ensure all required top-level fields exist: case_meta, paths, entities, annotations, trace_stats.
3. Ensure each path is LINEAR: for steps within a path, step[i].from must equal step[i-1].to.
   - If sibling branches exist, split into separate paths with new path_id suffixes (e.g. "1.2").
4. step_index must be sequential starting from 0 within each path.
5. step_type must be one of:
   ["direct_transfer","bridge_in","bridge_out","bridge_transfer","bridge_arrival","service_deposit","internal_transfer"].
   If not, set to "direct_transfer".
6. amount_estimate and trace_stats.initial_amount_estimate must be numeric (float).
7. If any address is used in steps but missing from entities, add a minimal entity:
   role="intermediate", risk_score=0.0, labels=[], riskscore_signals={}, notes="Auto-added by validator".
8. If a path has no stop_reason, set a sensible default like "Trace completed".
9. Preserve existing data as much as possible; only repair or normalize when required.
```

Важно: при разбитии невалидного JSON из валидатора срабатывает `postprocess_trace_result`, поэтому этот промпт, по сути, — первая линия «быстрого ремонта», а код — вторая.

---

## 17. Сводная таблица промптов

| Промпт | Файл | Функция | Модель (env) | Когда | Ответ |
|---|---|---|---|---|---|
| Hop Classifier | `trace_hop_classifier.md` | `_run_hop_classifier` | `OPENAI_SELECTOR_MODEL` = `gpt-5-mini` | На каждом адресе после Phase 1 | `{role, terminal, stop_reason, labels, notes, service_label, protocol}` |
| Hop Selector | `trace_hop_selector.md` | `_run_selector` | `OPENAI_SELECTOR_MODEL` = `gpt-5-mini` | После `all_txs` | `{selected_hashes, reasoning}` |
| Validator | `trace_validator.md` | `_run_validator` | `OPENAI_VALIDATOR_MODEL` = `gpt-4o` | Финальный проход | валидный TraceResult |
| Orchestrator (reference) | `docs/prompts/trace_orchestrator.reference.md` | — | — | Не вызывается; документирует правила | — |

---

## 18. Где смотреть при изменениях

- Правила классификации / терминалов → `trace_hop_classifier.md` **и** эвристики в `_agentic_hop_after_phase1` (`base_tracer.py:2535-2593`).
- Правила аккумуляции сумм → `trace_hop_selector.md` **и** код аккумуляции в `_run_agentic_trace` (`base_tracer.py:2762+`), плюс «старый» детерминированный вариант в `deterministic_tracer.py`.
- Форма финального JSON → `models.py::TraceResult` + `trace_postprocess.py` + `trace_validator.md`.
- Список инструментов → `base_tracer.py::TOOLS` + `mcp_client.py` / `mcp_http_client.py` / `tool_dispatch.py`.
- Приоритизация путей → `HopScheduler` (`base_tracer.py:77-140`).
- FIFO-учёт кражи → `FIFOLedger` (`base_tracer.py:142-251`).
- Визуализация → `visualization.py` + `currencies.json`.

Любое изменение правил классификации или аккумуляции должно идти **и в промпт, и в код** — иначе детерминированная и LLM-ветки начинают расходиться.
