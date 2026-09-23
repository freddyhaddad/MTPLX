#!/usr/bin/env python3
"""GPU A/B for the batched QSA decode (run in a QUIET window, never beside a busy daemon:
a second Flash-Next instance needs ~80 GB and tripped the leash on a loaded box).

  1. Start a test instance on :8001 from this worktree with the flag OFF, run this script,
     then restart it with MTPLX_QSA_BATCHED_DECODE=1 and run again:
       PYTHONPATH=~/projects/mtplx-qsa-batched MTPLX_SESSION_BANK_MAX_BYTES=8G FREE_FLOOR_GB=40 \\
       ~/projects/kimi-k26-local/scripts/leash.sh 120 -- /opt/homebrew/bin/mtplx serve \\
         --model ~/models/qwen3.8-flash-next-mtplx-opt --model-id flash-next --host 127.0.0.1 --port 8001 \\
         --no-auth --profile turbo --mtp --generation-mode ar --scheduler-mode ar_batch --batching-preset agent \\
         --max-active-requests 4 --ssd-session-cache off --no-stats-footer
  2. python3 tests/qsa_batched_decode_bench.py --port 8001 --label off|on
Reports per-stream tok/s at B=1,2,4 for ~7k and ~20k contexts (temperature 0, 160 tokens) and the
output text hash per stream so the two arms can be compared for divergence."""
import argparse, hashlib, json, random, threading, time, urllib.request

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001); ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=160); a = ap.parse_args()
    U = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    random.seed(5); words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    def ctx(tag, n): return f"Reference notes {tag}:\n" + "\n".join(f"- item{i}: " + " ".join(random.choice(words) for _ in range(9)) for i in range(n))
    def one(prompt, res, key):
        body = {"model": "flash-next", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "Summarize the notes in exactly 10 numbered points."}], "max_tokens": a.tokens, "temperature": 0}
        r = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(r, timeout=1800).read()); st = d["mtplx_stats"]
        txt = (d["choices"][0]["message"].get("reasoning_content") or "") + "|" + (d["choices"][0]["message"].get("content") or "")
        res[key] = (st.get("prompt_tokens"), round(st.get("tok_s") or 0, 1), st.get("scheduler_lane"), hashlib.sha1(txt.encode()).hexdigest()[:8])
    out = {}
    for name, n in (("7k", 400), ("20k", 1150)):
        prompts = [ctx(f"{name}-{i}", n) for i in range(4)]
        for B in (1, 2, 4):
            res = {}; th = [threading.Thread(target=one, args=(prompts[i], res, i)) for i in range(B)]
            t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
            out[f"{name}_B{B}"] = {"per_stream_tok_s": [res[i][1] for i in range(B)], "wall_s": round(wall, 1), "lanes": [res[i][2] for i in range(B)], "hashes": [res[i][3] for i in range(B)]}
            print(name, f"B={B}", out[f"{name}_B{B}"], flush=True)
    json.dump(out, open(f"bench_qsa_batched_{a.label}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
