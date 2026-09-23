# Batched QSA decode (prepared 2026-09-24, not applied)

Branch `qsa-batched-decode` (worktree `~/projects/mtplx-qsa-batched`), on top of `qsa-cache-batching`.
Flag `MTPLX_QSA_BATCHED_DECODE=1`, default off. Production is untouched.

## Why
With the Flash-Next batching PR the 12 QSA layers run once per row inside the batched lane (every
selection kernel and the indexer's offsets are single-sequence). At B=2 that loop is roughly a third
of the step, at B=4 about half: per-stream 27-29 tok/s at B=2 and 21 at B=4.

## What this does (v1, plain MLX ops)
`_qsa_batched_decode` in `mtplx/models/qwen4_exp.py`, reached from `_qsa_batched_forward` when S=1,
B>1, every row is in the sparse regime (`T // ratio > block_topk`), no vision rope, no fixed-capacity
banks. Per row (cheap, kept per row): indexer query prep, raw-key write, pooled-key extension, rope
with the row's own position, KV append. Batched: the scoring matmul over pooled keys padded to the
longest row, the masked top-k, a fixed-width gather (k selected blocks + one tail block per row,
invalid slots masked, no host syncs) and one SDPA over `[B, H_kv, W, D]`.
Parity: `tests/test_qsa_batched_decode.py` (tiny layer, rows of different lengths, < 1e-4 vs the
per-row loop; caches identical up to fp32 rope rounding on the appended key).

## What it does NOT do yet
- No Metal kernel: the single-row lane uses `qsa_flash_skip` (a fused selected-block attention
  kernel); this path gathers + dense SDPA. Whether B x kernel beats one batched eager pass is the
  open question `tests/qsa_batched_decode_bench.py` answers (quiet window, second instance on :8001).
- Rows in the dense regime (short contexts) fall back to the loop; mixed batches too.
- Prefill (S>1) stays per row.

## Expected (unmeasured)
If the batched eager pass wins: B=2 from ~28 toward 35-40 per stream, B=4 from ~21 toward 26-30.
If it loses, the next step is a batched `qsa_flash_skip` (grid over batch x head, per-row block
lists and offsets), which is the multi-day kernel work.

## Measured 2026-09-24 (prod daemon, quiet box, flag off vs on, 160 tokens, temperature 0)

| context | streams | off (per-row loop) tok/s per stream | on (batched eager) |
|---|---|---|---|
| 7k | 2 | 31.6 / 31.6 | 25.7 / 25.6 |
| 7k | 4 | 18.3 / 26.3 / 11.7 / 11.7 | 24.0 / 22.1 / 22.1 / 23.0 |
| 20k | 2 | 29.5 / 29.5 | 5.6 / 6.2 |
| 20k | 4 | 2.8 / 29.4 / 6.0 / 7.5 | 2.7 / 6.0 / 23.5 / 5.7 |

Solo (B=1) unchanged at 78-80 (never enters this path). Verdict: **not a win as written**. At 7k it
evens out a 4-stream batch (+30% on the average) but costs 19% at 2 streams; at 20k it is 5x slower
than the per-row loop, so something in the eager path scales with context that the single-row lane's
kernels (compiled indexer + `qsa_flash_skip`) do not. Next step before any further tuning: profile
one batched step at 20k (scoring matmul over nb_max pooled blocks, the per-row `mx.take` gathers from
the 20k KV, the f32 pooled mirror rebuild after a bank restore) to find the term that grows with T.
The output text differs from the loop's (different numeric path), as expected. First A/B was a null
result because the path declined on the verify-glue rope kernel gate (now removed; a once-per-process
"[qwen4_exp] batched QSA decode: engaged|<reason>" line confirms engagement).
