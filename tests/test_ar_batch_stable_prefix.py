"""Batched lane session handling (#420 follow-up, 2026-09-24 opencode receipt).

(1) `_prepare_stable_prefix`: bank the STABLE prefix (prompt minus MTPLX's transient trailing hint)
as an exact-prefix entry before the batch generator runs, advancing the restored or fresh cache to
the stable edge with the runtime's own forward. (2) `_wait_pending_postcommit`: a same-session
follow-up turn waits for the previous turn's pending postcommit like the solo lane does.
CPU-only, fakes for runtime / bank / sessions; no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import mtplx.server.openai as srv
from mtplx.sampling import SamplerConfig
from mtplx.server.openai import _BatchedARGenerationService, _BatchedARJob


class _Entry:
    def __init__(self):
        self.state = mx.zeros((1, 1))

    def merge(self, others):
        return self


class _Runtime:
    def __init__(self):
        self.forwarded = []
        self.model_path = "m"

    def make_cache(self):
        return [_Entry(), _Entry()]

    def forward_ar(self, ids, cache=None, return_hidden=False):
        self.forwarded.append(ids.tolist()[0])
        return mx.zeros((1, ids.shape[1], 4))


class _Bank:
    def __init__(self):
        self.puts = []

    def put_snapshot(self, **kw):
        self.puts.append(kw)
        return object()

    def restore(self, *a, **k):
        return None


class _Session:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def wait_for_pending_postcommit(self):
        self.calls += 1
        return self.outcome


def _service(runtime, session=None):
    svc = _BatchedARGenerationService.__new__(_BatchedARGenerationService)
    svc.state = SimpleNamespace(runtime=runtime, sessions=SimpleNamespace(peek=lambda sid: session))
    return svc


def _job(prompt_ids, *, stable=None, bank=None, session_id="s1"):
    obs = {} if stable is None else {"stable_prefix_len": stable}
    job = _BatchedARJob(
        request_id="req", prompt_ids=prompt_ids, max_tokens=8, sampler=SamplerConfig(), seed=0,
        stop_token_ids=set(), token_callback=None, prefill_callback=None, request_observability=obs,
        mtp_disabled_reason=None, generation_limits={}, seed_is_explicit=False,
    )
    job.session_bank = bank
    job.session_id = session_id
    job.session_template_hash = "t"
    job.session_policy_fingerprint = "p"
    return job


def test_stable_prefix_is_prefilled_in_chunks_banked_and_split_from_the_hint(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    monkeypatch.setattr(srv, "_STABLE_PREFIX_PREFILL_CHUNK", 3)
    monkeypatch.setattr(srv, "snapshot_cache", lambda cache: SimpleNamespace(states=[], meta_states=[]))
    rt = _Runtime(); bank = _Bank(); svc = _service(rt)
    prompt = list(range(100, 112))  # 12 tokens: stable 9, hint 3
    job = _job(prompt, stable=9, bank=bank)
    svc._prepare_stable_prefix(job)
    assert rt.forwarded == [prompt[0:3], prompt[3:6], prompt[6:9]], "stable edge reached in 3-token chunks, hint untouched"
    assert len(bank.puts) == 1 and bank.puts[0]["token_ids"] == prompt[:9] and bank.puts[0]["snapshot_epoch"] == 9
    assert bank.puts[0]["session_id"] == "s1" and bank.puts[0]["template_hash"] == "t"
    assert job.insert_all_tokens == prompt[:9] and job.insert_prompt_ids == prompt[9:], "generator prefills only the hint"
    assert job.insert_cache is not None and job.request_observability["ar_batch_stable_prefix_bank_stored"] is True
    assert job.request_observability["ar_batch_stable_prefix_prefilled"] == 9


def test_stable_prefix_extends_a_shorter_restore_and_skips_a_longer_one(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    monkeypatch.setattr(srv, "snapshot_cache", lambda cache: SimpleNamespace(states=[], meta_states=[]))
    rt = _Runtime(); bank = _Bank(); svc = _service(rt)
    prompt = list(range(20))
    job = _job(prompt, stable=15, bank=bank)
    restored = rt.make_cache(); job.insert_cache = restored; job.insert_all_tokens = prompt[:10]; job.insert_prompt_ids = prompt[10:]
    svc._prepare_stable_prefix(job)
    assert rt.forwarded == [prompt[10:15]], "only the gap between the restore point and the stable edge is forwarded"
    assert job.insert_cache is restored and job.insert_all_tokens == prompt[:15] and job.insert_prompt_ids == prompt[15:]
    # a restore that already covers the stable edge is left alone
    rt2 = _Runtime(); bank2 = _Bank(); svc2 = _service(rt2)
    job2 = _job(prompt, stable=15, bank=bank2)
    job2.insert_cache = rt2.make_cache(); job2.insert_all_tokens = prompt[:17]; job2.insert_prompt_ids = prompt[17:]
    svc2._prepare_stable_prefix(job2)
    assert rt2.forwarded == [] and bank2.puts == [] and job2.insert_all_tokens == prompt[:17]


def test_stable_prefix_gates(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    rt = _Runtime(); svc = _service(rt); prompt = list(range(12))
    for job in (
        _job(prompt, stable=None, bank=_Bank()),   # no stable length known
        _job(prompt, stable=9, bank=None),         # no bank
        _job(prompt, stable=12, bank=_Bank()),     # stable == whole prompt: nothing to split
        _job(prompt, stable=2, bank=_Bank()),      # below the minimum
    ):
        svc._prepare_stable_prefix(job)
        assert rt.forwarded == [] and job.insert_cache is None and job.insert_prompt_ids == prompt


def test_stable_prefix_failure_leaves_the_job_untouched(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    rt = _Runtime()
    rt.forward_ar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    svc = _service(rt); prompt = list(range(12)); job = _job(prompt, stable=9, bank=_Bank())
    svc._prepare_stable_prefix(job)
    assert job.insert_cache is None and job.insert_prompt_ids == prompt
    assert job.request_observability["ar_batch_stable_prefix_error"].startswith("RuntimeError")


def test_restore_waits_for_the_sessions_pending_postcommit():
    session = _Session({"waited": True, "outcome": "completed", "elapsed_s": 0.2})
    rt = _Runtime(); svc = _service(rt, session=session); bank = _Bank()
    job = _job(list(range(600)), bank=bank)
    svc._prepare_session_bank_restore(job)
    assert session.calls == 1 and job.request_observability["ar_batch_postcommit_wait"]["outcome"] == "completed"
    # no session / no bank: no wait, no error
    job2 = _job(list(range(600)), bank=bank, session_id=None); svc._prepare_session_bank_restore(job2)
    assert session.calls == 1
