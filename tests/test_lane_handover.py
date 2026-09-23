"""Solo MTP -> batched AR lane handover (server side; the generator's poll is exercised live).

When another request is waiting on the batched lane behind a solo MTP owner, the solo run returns
finish_reason="handover" with its trunk cache holding prompt + tokens[:-1]; the solo runner submits a
continuation job (insert the cache plus the last token, remaining max_tokens, penalty counts seeded)
and the dispatcher waits for it off the owner thread and finalizes both segments as one response.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
from mtplx.sampling import SamplerConfig


class _Tok:
    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


class _Service:
    def __init__(self, pending=False, unavailable=None):
        self.jobs = []
        self._pending = pending
        self.ar_batch_unavailable_reason = unavailable

    def has_pending(self):
        return self._pending

    def submit(self, job):
        self.jobs.append(job)
        job.future = Future()
        return job.future


def _state(service, mode="ar_batch"):
    st = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=_Tok()),
        ar_batch_service=service,
        args=SimpleNamespace(scheduler_mode=mode),
        model_scheduler=SimpleNamespace(foreground_pending=lambda: 0),
        fg=0,
    )
    st.begin_foreground = lambda: setattr(st, "fg", st.fg + 1)
    st.end_foreground = lambda: setattr(st, "fg", st.fg - 1)
    return st


def _out(tokens, cache=("cache",)):
    return SimpleNamespace(
        tokens=list(tokens),
        finish_reason="handover",
        final_state=SimpleNamespace(final_trunk_cache=list(cache)),
        stats=SimpleNamespace(to_dict=lambda: {"mode": "mtpk", "verify_calls": 3}),
    )


def test_handover_check_gates(monkeypatch):
    monkeypatch.delenv("MTPLX_LANE_HANDOVER", raising=False)
    assert srv._make_handover_check(_state(_Service()), seed_is_explicit=False) is None
    monkeypatch.setenv("MTPLX_LANE_HANDOVER", "1")
    assert srv._make_handover_check(_state(_Service()), seed_is_explicit=True) is None, "seeded streams never hand over"
    assert srv._make_handover_check(_state(_Service(), mode="serial"), seed_is_explicit=False) is None
    assert srv._make_handover_check(_state(_Service(unavailable="no merge")), seed_is_explicit=False) is None
    check = srv._make_handover_check(_state(_Service(pending=False)), seed_is_explicit=False)
    assert check is not None and check() is False
    st = _state(_Service(pending=True))
    assert srv._make_handover_check(st, seed_is_explicit=False)() is True
    st2 = _state(_Service(pending=False)); st2.model_scheduler = SimpleNamespace(foreground_pending=lambda: 1)
    assert srv._make_handover_check(st2, seed_is_explicit=False)() is True


def test_submit_lane_continuation_builds_the_insertable_job(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    service = _Service(); st = _state(service)
    prompt = list(range(10, 20)); solo = [30, 31, 32, 31]
    calls = []
    marker = srv._submit_lane_continuation(
        st, prompt, _out(solo), request_id="req-1", response_max=100, sampler=SamplerConfig(), generation_seed=7,
        generation_limits={"x": 1}, request_observability={"request_id": "req-1"}, token_callback=calls.append,
        prefill_callback=None, cancel_event=None, session_id="s1", session_bank="bank", session_restore_mode="cold",
        session_template_hash="t", session_draft_head_identity=None, session_policy_fingerprint="p",
        token_times=[1.0, 2.0, 3.0, 4.0], started=0.0,
    )
    job = service.jobs[0]
    assert job.continuation is True and job.insert_cache == ["cache"]
    assert job.insert_all_tokens == prompt + solo[:-1] and job.insert_prompt_ids == [solo[-1]], "cache holds prompt + g[:-1]; the last token is inserted"
    assert job.max_tokens == 96 and job.cached_tokens == len(prompt) + 3 and job.session_cache_hit is True
    assert dict(job.completion_token_counts) == {30: 1, 31: 2, 32: 1}, "penalty counts seeded with the solo tokens"
    assert job.token_callback == calls.append and job.session_bank == "bank" and job.session_id == "s1"
    assert job.request_observability["scheduler_lane"] == "solo_mtp->ar_batch"
    assert marker["_handover_job"] is job and marker["_handover_solo_tokens"] == solo and marker["finish_reason"] == "handover"
    assert marker["_handover_solo_token_times"] == [1.0, 2.0, 3.0, 4.0]


def test_submit_lane_continuation_refuses_without_cache_or_tokens(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: set())
    st = _state(_Service())
    common = dict(request_id=None, response_max=10, sampler=SamplerConfig(), generation_seed=0, generation_limits={},
                  request_observability=None, token_callback=None, prefill_callback=None, cancel_event=None, session_id=None,
                  session_bank=None, session_restore_mode="cold", session_template_hash=None, session_draft_head_identity=None,
                  session_policy_fingerprint=None, token_times=[], started=0.0)
    with pytest.raises(RuntimeError):
        srv._submit_lane_continuation(st, [1, 2], _out([]), **common)
    with pytest.raises(RuntimeError):
        srv._submit_lane_continuation(st, [1, 2], SimpleNamespace(tokens=[3], finish_reason="handover", final_state=None, stats=None), **common)


def test_finish_lane_handover_merges_both_segments(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    monkeypatch.setattr(srv, "_strip_terminal_stop", lambda toks, stops: [t for t in toks if t not in stops])
    captured = {}
    def fake_finalize(state, prompt_ids, generated, **kw):
        captured.update(generated=generated, kw=kw); return {"ok": True}
    monkeypatch.setattr(srv, "_finalize_batched_ar_generation", fake_finalize)
    st = _state(_Service())
    job = SimpleNamespace(future=Future(), request_observability={"scheduler_lane": "solo_mtp->ar_batch"})
    job.future.set_result({"tokens": [5, 6, 99], "text": "fg", "stats": {"mode": "ar", "tok_s": 20.0}, "_token_times": [7.0, 8.0], "elapsed_s": 2.0})
    marker = {"_handover_job": job, "_handover_solo_tokens": [1, 2, 3], "_handover_solo_token_times": [1.0, 2.0, 3.0],
              "_handover_solo_stats": {"mode": "mtpk"}, "_handover_started": 0.0}
    out = srv._finish_lane_handover(st, [10, 11], marker, {"session_id": "s1", "session_cache_hit": True, "cache_miss_reason": None})
    assert out == {"ok": True} and st.fg == 0, "foreground accounting balanced"
    g = captured["generated"]
    assert g["tokens"] == [1, 2, 3, 5, 6, 99] and g["text"] == "bcdfg" and g["completion_tokens"] == 6
    assert g["_token_times"] == [1.0, 2.0, 3.0, 7.0, 8.0]
    assert g["stats"]["scheduler_lane"] == "solo_mtp->ar_batch" and g["stats"]["lane_handover"] == {"solo_tokens": 3, "batched_tokens": 3, "solo_stats": {"mode": "mtpk"}}
    assert captured["kw"]["session_restore_mode"] == "solo_mtp->ar_batch" and captured["kw"]["session_id"] == "s1"


def test_prepare_prompt_inputs_skips_continuations():
    from mtplx.server.openai import _BatchedARGenerationService
    svc = _BatchedARGenerationService.__new__(_BatchedARGenerationService)
    seen = []
    svc._prepare_session_bank_restore = lambda job: seen.append(("restore", job.request_id))
    svc._prepare_shared_prefix = lambda jobs: seen.append(("shared", [j.request_id for j in jobs]))
    svc._prepare_stable_prefix = lambda job: seen.append(("stable", job.request_id))
    fresh = SimpleNamespace(request_id="a", continuation=False); cont = SimpleNamespace(request_id="b", continuation=True)
    svc._prepare_prompt_inputs([fresh, cont])
    assert seen == [("restore", "a"), ("shared", ["a"]), ("stable", "a")]
