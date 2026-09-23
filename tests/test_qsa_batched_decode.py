"""Batched QSA decode (MTPLX_QSA_BATCHED_DECODE=1): one scoring matmul + one SDPA for
the whole batch instead of the per-row loop. Parity against the per-row loop on a tiny
random layer with two rows of different lengths, both past the sparse threshold."""

from __future__ import annotations

import mlx.core as mx
import pytest

import mtplx.models.qwen4_exp as Q
from mtplx.models.qwen4_exp import Attention, BatchQSACache, QSACache, TextArgs


def _layer():
    args = TextArgs(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
        indexer_compress_ratio=4, rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=1e4,
    )
    mx.random.seed(0)
    attn = Attention(args)
    mx.eval(attn.parameters())
    return attn


def _prefilled_rows(attn, lens):
    rows = []
    for n in lens:
        mx.random.seed(100 + n)
        x = mx.random.normal((1, n, 64))
        c = QSACache(4)
        y = attn(x, c)
        mx.eval(y, c.state)
        rows.append(c)
    return rows


@pytest.mark.parametrize("lens", [(22, 30), (25, 25), (41, 22, 33)])
def test_batched_decode_matches_the_per_row_loop(monkeypatch, lens):
    monkeypatch.delenv("MTPLX_QSA_FLASH", raising=False)
    monkeypatch.delenv("MTPLX_QSA_GATHER_DECODE", raising=False)
    attn = _layer()
    mx.random.seed(7)
    x = mx.random.normal((len(lens), 1, 64))
    # per-row loop (flag off)
    monkeypatch.setenv("MTPLX_QSA_BATCHED_DECODE", "0")
    rows_a = _prefilled_rows(attn, lens)
    y_loop = Q._qsa_batched_forward(attn, x, BatchQSACache(rows_a))
    mx.eval(y_loop)
    # batched path (flag on), fresh identical rows
    monkeypatch.setenv("MTPLX_QSA_BATCHED_DECODE", "1")
    rows_b = _prefilled_rows(attn, lens)
    cache_b = BatchQSACache(rows_b)
    y_batched = Q._qsa_batched_decode(attn, x, cache_b)
    assert y_batched is not None, "all rows are in the sparse regime: the batched path must engage"
    mx.eval(y_batched)
    diff = float(mx.abs(y_batched - y_loop).max())
    assert diff < 1e-4, f"batched vs per-row decode differ by {diff}"
    for ra, rb in zip(rows_a, rows_b):
        assert ra.offset == rb.offset and ra.pooled_len == rb.pooled_len
        # the appended key goes through the eager rope tables (vs the loop's lane): fp32 rounding only
        assert float(mx.abs(ra.kv.keys[..., : ra.offset, :] - rb.kv.keys[..., : rb.offset, :]).max()) < 1e-5
        assert float(mx.abs(ra.raw_keys[:, : ra.offset] - rb.raw_keys[:, : rb.offset]).max()) < 1e-5  # batched vs single-row matmul accumulation


def test_batched_decode_declines_dense_rows(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_BATCHED_DECODE", "1")
    attn = _layer()
    rows = _prefilled_rows(attn, (22, 10))  # 10 // 4 = 2 <= block_topk: dense regime
    x = mx.random.normal((2, 1, 64))
    assert Q._qsa_batched_decode(attn, x, BatchQSACache(rows)) is None
    y = Q._qsa_batched_forward(attn, x, BatchQSACache(_prefilled_rows(attn, (22, 10))))  # falls back to the loop
    mx.eval(y); assert y.shape == (2, 1, 64)
