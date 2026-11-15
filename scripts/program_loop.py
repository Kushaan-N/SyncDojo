"""Opus writes ONE program per task; simulator workers (scripts/worker.py) test and record it.

  develop   Opus writes run(api) from an example scene; it is dry-run (physics only, no rendering) on the
            --dev layouts, and Opus revises from the logs until all of them pass or --rounds run out.
  rollout   the program is recorded on every --layouts layout, then each solved layout gets failure /
            collision variants (the perturbations are injected inside the skills) and noisy replays.

  python scripts/program_loop.py --task stack_blocks --queue QDIR --out OUT --dev 0 1 2 --layouts 0 1 2 3 4 5
  python scripts/program_loop.py ... --program OUT/<task>/programs/v3.py      # skip develop, reuse a program
Runs anywhere that sees the queue directory (no GPU): it only talks to the workers through files."""
import argparse, json, os, random, sys, time, uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--queue", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dev", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--layouts", type=int, nargs="*", default=[])
p.add_argument("--rounds", type=int, default=6)
p.add_argument("--variants", type=int, default=2, help="variants per kind per solved layout")
p.add_argument("--kinds", default="jitter,jitter_large,grasp_offset,yaw_error,high_release,early_open,sweep,noisy_replay")
p.add_argument("--program", default="")
p.add_argument("--effort", default="high")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--timeout-min", type=float, default=12, help="per request; a batch waits this long times its size")
a = p.parse_args()
root = os.path.join(a.out, a.task)
os.makedirs(os.path.join(root, "programs"), exist_ok=True)
rng = random.Random(a.seed)
usage = []


def submit(req):
    rid = uuid.uuid4().hex[:12]
    tmp = os.path.join(a.queue, f".tmp_{rid}")
    json.dump(req, open(tmp, "w"))
    os.rename(tmp, os.path.join(a.queue, f"req_{rid}.json"))
    return rid


def wait(ids):
    out, t0 = {}, time.time()
    while len(out) < len(ids) and time.time() - t0 < a.timeout_min * 60 * max(1, len(ids)):
        for i in ids:
            f = os.path.join(a.queue, f"res_{i}.json")
            if i not in out and os.path.exists(f):
                out[i] = json.load(open(f)); os.remove(f)
        time.sleep(1.0)
    return [out.get(i, {"ok": False, "error": "no worker answered in time", "success": False, "log": []}) for i in ids]


def call(reqs):
    return wait([submit(r) for r in reqs])


def objs_text(o):
    return "\n".join(f"  - {k}: xyz {v['xyz']}, yaw {v['yaw_deg']}" for k, v in o.items())


def feedback(results):
    parts = []
    for L, r in results:
        if r.get("success"):
            parts.append(f"Layout {L}: PASSED."); continue
        if not r.get("ok"):
            parts.append(f"Layout {L}: could not be run ({r.get('error')})."); continue
        log = r.get("log", [])
        parts.append(f"Layout {L}: FAILED (success check ever true: {r.get('success_latched')}; true at the end: "
                     f"{r.get('final_state_ok')})." + (f"\nProgram error: {r['error']}" if r.get("error") else "")
                     + f"\nObjects at the start:\n{objs_text(r.get('start_objects', {}))}\nExecution log"
                     + (" (last 60 lines)" if len(log) > 60 else "") + ":\n" + "\n".join("  " + l for l in log[-60:])
                     + f"\nObjects at the end:\n{objs_text(r.get('final_objects', {}))}")
    return ("The simulator ran your program on these layouts (each from its own fresh start):\n\n" + "\n\n".join(parts) +
            "\n\nWork out what went wrong and write a corrected, complete program that passes every layout.")


# ------------------------------------------------------------------------------------ develop
if a.program:
    code, version = open(a.program).read(), os.path.basename(a.program)
else:
    from syncdojo import opus_planner as op
    from syncdojo.opus_planner import describe_scene
    sc = call([{"op": "scene", "layout": a.dev[0]}])[0]
    if not sc.get("ok"):
        sys.exit(f"scene request failed: {sc.get('error')}")
    scene = describe_scene(sc["instruction"], sc["objects"], sc["arms"], sc["table"], sc["success_text"]).replace(
        "Write the motion script.", f"This is layout {a.dev[0]}, one example; other layouts place the same kinds of objects "
        "elsewhere. Write the program.")
    best, convo, prog = (-1, None, None), None, None
    for rnd in range(a.rounds):
        t0 = time.time()
        prog, convo = (op.plan_program(scene, a.effort) if convo is None else op.revise_program(convo, fb, a.effort))
        usage[:] = convo["usage"]
        version = f"v{rnd}.py"
        open(os.path.join(root, "programs", version), "w").write(prog.code)
        res = [(a.dev[0], call([{"op": "run", "layout": a.dev[0], "code": prog.code}])[0])]
        if res[0][1].get("success") and len(a.dev) > 1:  # it works on the example: now the other layouts, in parallel
            res += list(zip(a.dev[1:], call([{"op": "run", "layout": L, "code": prog.code} for L in a.dev[1:]])))
        n_ok = sum(bool(r.get("success")) for _, r in res)
        print("ROUND", json.dumps({"round": rnd, "version": version, "passed": [L for L, r in res if r.get("success")],
                                   "failed": [L for L, r in res if not r.get("success")], "opus_s": round(time.time() - t0),
                                   "output_tokens": convo["usage"][-1].get("output_tokens")}), flush=True)
        if n_ok >= best[0]:  # ties: the later version (it has seen more feedback)
            best = (n_ok, version, prog.code)
        if n_ok == len(a.dev):
            break
        fb = feedback(res)
    _, version, code = best
    print("DEVELOP_DONE", json.dumps({"program": version, "dev_passed": best[0], "of": len(a.dev), "opus_calls": len(usage),
                                      "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
                                      "input_tokens": sum(u.get("input_tokens", 0) for u in usage)}), flush=True)
    json.dump(usage, open(os.path.join(root, "programs", "usage.json"), "w"))
    open(os.path.join(root, "programs", "final.py"), "w").write(code)

# ------------------------------------------------------------------------------------ rollout
summ = open(os.path.join(root, "program_summary.jsonl"), "a")


def record(reqs):
    for q, r in zip(reqs, call(reqs)):
        row = {"name": os.path.basename(q["out_dir"]), "layout": q["layout"], "kind": q["kind"], "success": r.get("success"),
               "frames": r.get("frames"), "error": r.get("error") if not r.get("ok") else None, "dir": q["out_dir"]}
        summ.write(json.dumps(row) + "\n"); summ.flush()
        print("EPISODE", json.dumps(row), flush=True)
        yield q, r


if a.layouts:
    base = {"op": "run", "code": code, "record": True, "program_name": version, "model": "claude-opus-5-5"}
    todo = [dict(base, layout=L, kind="program", out_dir=os.path.join(root, "prog", f"program_{L:03d}")) for L in a.layouts]
    todo = [q for q in todo if not os.path.exists(os.path.join(q["out_dir"], "metadata.json"))]
    solved = {}
    for q, r in record(todo):
        if r.get("success"):
            solved[q["layout"]] = (q["out_dir"], sum(1 for l in r.get("log", []) if l.startswith("pick ") and l.endswith("OK")))
    var = []
    for L, (ndir, n_pick) in solved.items():
        for kind in a.kinds.split(","):
            for i in range(a.variants):
                out = os.path.join(root, "prog", f"{kind}_{L:03d}_{i}")
                if os.path.exists(os.path.join(out, "metadata.json")):
                    continue
                at, seed = rng.randrange(max(1, n_pick)), rng.randrange(10 ** 6)
                ang, d = rng.uniform(0, 6.283), rng.uniform(0.015, 0.04)
                import math
                pert = {"jitter": {"jitter_m": 0.003}, "jitter_large": {"jitter_m": 0.012, "jitter_yaw_deg": 8.0},
                        "grasp_offset": {"grasp_offset": [d * math.cos(ang), d * math.sin(ang)], "at": at},
                        "yaw_error": {"yaw_error_deg": rng.choice([-1, 1]) * rng.uniform(25, 60), "at": at},
                        "high_release": {"high_release_m": rng.uniform(0.06, 0.12), "at": at},
                        "early_open": {"early_open": True, "early_frac": rng.uniform(0.3, 0.7), "at": at},
                        "sweep": {"sweep": True}}.get(kind)
                if kind == "noisy_replay":
                    var.append({"op": "replay", "layout": L, "base": ndir, "sigma": [0.01, 0.03, 0.06][i % 3], "seed": seed,
                                "record": True, "kind": kind, "out_dir": out, "source": "expansion"})
                elif pert is not None:
                    var.append(dict(base, layout=L, kind=kind, out_dir=out, perturb=dict(pert, seed=seed), source="expansion"))
    rng.shuffle(var)
    n_s = sum(bool(r.get("success")) for _, r in record(var))
    print("ROLLOUT_DONE", json.dumps({"layouts": len(a.layouts), "solved": sorted(solved), "variants": len(var), "variant_successes": n_s}), flush=True)
