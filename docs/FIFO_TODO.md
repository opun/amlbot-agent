# FIFO Tracing — Future Improvements

## TODO

### 1. Multi-transaction theft support
Currently `stolen_amount` is auto-detected from a single `tx_hash`. If the theft was split across multiple transactions, the user must manually sum and provide `stolen_amount`.

**Proposed change**: Accept a list of `tx_hash` values in `TracerConfig`, aggregate their amounts to compute `stolen_amount`, and seed one path per theft tx (or merge into a single path if recipients converge).

### 2. Proportional attribution mode
The current FIFO model assigns theft-share based on queue ordering. When funds split into multiple branches from the same address, the branch processed first gets all the theft attribution. This is mathematically correct for FIFO but can concentrate attribution on one path even when the thief split funds evenly.

**Proposed change**: Add a `attribution_mode: Literal["fifo", "proportional"]` config option. In proportional mode, outflows from the same address share the theft attribution pro-rata by amount rather than by queue order. This gives investigators a choice based on their jurisdiction's standards.

### 3. Confidence scores on traced paths
Paths built entirely via deterministic chronological accumulation are more reliable than paths where the LLM selector chose transactions. Currently there's no way to distinguish them in the output.

**Proposed change**: Add a `confidence: float` field to each Path, computed from:
- How many hops used accumulation vs LLM selector (higher = more accumulation)
- Whether amounts matched closely at each hop (lower slippage = higher confidence)
- Whether entity classifications came from heuristic keywords vs LLM-only (keyword match = higher confidence)
