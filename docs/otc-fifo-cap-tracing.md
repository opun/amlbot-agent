# OTC-like Classification, FIFO Attribution & Global Cap

## Overview

This document describes the behavioral OTC-like address classification, FIFO-based theft-origin attribution, and global traced-amount cap enforcement added to the tracing system.

## 1. OTC-like Behavioral Classification

### Purpose
Automatically detect addresses that behave like unidentified services (OTC desks, payment processors, etc.) based on activity metrics rather than label databases alone.

### Criteria (all must hold)
| Metric | Threshold |
|---|---|
| Total incoming volume | >= $50,000 |
| Transaction count (in + out) | >= 100 |
| Address age | >= 180 days (6 months) |

### Behavior
- OTC-like entities are **non-terminal**: tracing continues through them.
- They are labeled `Potential Service / OTC-like Entity`.
- An `Ownership Change Risk` annotation is emitted for every OTC-like entity.

### CEX Concentration Gate
When an OTC-like address has >= X% of outflows directed to a single known CEX cluster, tracing is explicitly continued to that CEX endpoint.

- **Config field**: `cex_single_cluster_threshold` (default: `0.60`)
- A `CEX Concentration` annotation records the share and destination.

## 2. FIFO Attribution Ledger

### Purpose
Prevent over-attribution of stolen funds when they are mixed with non-theft funds at intermediate addresses.

### Algorithm
1. Each address maintains a FIFO queue of inflows: `[(amount, theft_share), ...]`.
2. When funds leave an address, the theft-attributed share is computed by draining the queue in FIFO order.
3. For each queue entry, the theft ratio `theft_share / amount` determines what fraction of the outflow is theft-attributed.
4. The attributed amount (not the raw tx amount) is propagated downstream.
5. **Cap is NOT consumed on intermediate hops** — it is only consumed at terminal endpoints (CEX, dead-end, mixer, etc.) via `claim_terminal()`.

### Step-level field
Each `Step` now includes an optional `attributed_amount` field representing the FIFO-computed theft-origin share.

## 3. Global Traced Amount Cap

### Purpose
Ensure the total attributed amount across all traced paths does not exceed the original stolen amount (plus a tolerance for fees/slippage).

### Formula
```
cap = stolen_amount * (1 + tolerance)
```

### Config fields
| Field | Default | Description |
|---|---|---|
| `stolen_amount` | auto-detected from initial theft tx | Known theft amount |
| `traced_amount_tolerance` | `0.03` (3%) | Allowed tolerance above stolen amount |

### Behavior
- The cap is consumed **only at terminal endpoints** (CEX deposit, dead-end, bridge with unknown destination, max hops, etc.) — not on intermediate hops.
- Intermediate hops propagate the FIFO-attributed theft share without consuming the cap budget.
- When the cap is reached, further terminal claims are clipped to zero.
- A `Cap Reached` annotation is emitted with the exact figures.
- Paths that hit the cap receive stop reason `Global traced amount cap reached`.
- Exception: when `stolen_amount` is 0 or not set, no cap is enforced.

## 4. Visualization URL

The fallback share URL format changed from `/visualization/{hash}` to `/ai/{hash}`.
The `VISUALIZATION_URL_TEMPLATE` environment variable continues to override this when set.

## 5. Report Output

The summary text now includes:
- `Stolen Amount` and `Total Traced (FIFO-attributed)` in trace stats.
- A dedicated **Risk Warnings** section listing all ownership change, OTC-like, cap, and CEX concentration annotations.

## Files Changed

| File | Changes |
|---|---|
| `src/agent/models.py` | Added `stolen_amount`, `cex_single_cluster_threshold`, `traced_amount_tolerance` to `TracerConfig`; `attributed_amount` to `Step`; `total_traced_amount`, `stolen_amount` to `TraceStats` |
| `src/agent/base_tracer.py` | Added `FIFOLedger` class, `_classify_otc_like` method, OTC behavioral analysis in trace loop, cap guard, FIFO attribution at each step, updated visualization URL |
| `src/agent/deterministic_tracer.py` | OTC keyword no longer marks terminal; ownership change annotation for OTC entities |
| `src/agent/reporting.py` | Risk warnings section, FIFO stats, attributed_amount in edges |
| `src/agent/api.py` | New fields in `TraceRequest`, wired to `TracerConfig` |
| `src/agent/cli.py` | New CLI parameters for stolen_amount, thresholds |
| `docs/openapi.yaml` | New fields in TraceRequest schema |
| `tests/test_fifo_otc_cap.py` | Unit tests for FIFO, cap, OTC classification, config defaults |
