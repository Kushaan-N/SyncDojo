"""A simulator that stays up and takes requests, so nothing waits for a queue or a 90 s start-up.

Start one or more per task inside a GPU allocation; they all serve the same queue directory, so N
workers run N rollouts in parallel on one GPU:

  python scripts/worker.py --task stack_blocks --queue QDIR --headless --enable_cameras

A request is QDIR/req_<id>.json; the worker claims it (rename), and writes QDIR/res_<id>.json.
  {"op": "scene", "layout": L}                       -> the scene description inputs
  {"op": "run", "layout": L, "code": "...", "record": false}             dry run: physics only, no rendering
  {"op": "run", ..., "record": true, "out_dir": D, "kind": K, "perturb": {...}}   recorded episode in D
  {"op": "replay", "layout": L, "base": DIR, "sigma": s, "seed": n, "out_dir": D}  noisy replay of DIR's stream
  {"op": "quit"}
The label of a run is RoboDojo's success check AND the check still holding at the end."""
import argparse, glob, json, os, signal, sys, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--queue", required=True)
p.add_argument("--idle-exit-min", type=float, default=0, help="exit after this long without a request (0: never)")
p.add_argument("--request-timeout-min", type=float, default=25, help="give up on one request after this long")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from syncdojo import rd_sim
app = rd_sim.launch(a)
import numpy as np
from syncdojo.executor import SUCCESS_TEXT, scene_facts
from syncdojo.recorder import Recorder
import importlib
from syncdojo import skills

sim = rd_sim.Sim(a.task, app)


def _timeout(signum, frame):
    raise TimeoutError(f"request took longer than {a.request_timeout_min:.0f} min")


signal.signal(signal.SIGALRM, _timeout)
os.makedirs(a.queue, exist_ok=True)
wid = f"{os.uname().nodename}-{os.getpid()}"
print("WORKER_READY", a.task, wid, flush=True)


def reset(L):
    for t in range(3):
        try:
            sim.reset(L); return True
        except Exception as e:
            print("RESET_FAILED", L, t, repr(e)[:160], flush=True)
    return False


def summary():
    return {k: {"xyz": o["xyz"], "quat_wxyz": o["quat_wxyz"], "yaw_deg": o["yaw_deg"]} for k, o in sim.object_info().items()}


def handle(r):
    t0 = time.time()
    L = int(r.get("layout", 0))
    if not reset(L):
        return {"ok": False, "error": "the simulator rejected this layout as unstable (3 tries)"}
    if r["op"] == "scene":
        objs, arms, table = scene_facts(sim)
        return {"ok": True, "instruction": sim.instruction, "objects": objs, "arms": arms, "table": table,
                "success_text": SUCCESS_TEXT.get(a.task, "")}
    rec = bool(r.get("record"))
    start = summary()
    if rec:
        sim.recorder = Recorder()
    err = None
    try:
        if r["op"] == "replay":
            s = np.load(os.path.join(r["base"], "stream.npz"))["targets"].astype(np.float64)
            g = np.random.default_rng(int(r.get("seed", 0)))
            for off in (0, 14):
                for j in range(6):
                    n = np.convolve(g.normal(0, 1, len(s) + 200), np.hanning(200), mode="valid")[:len(s)]
                    s[:, off + j] += float(r["sigma"]) * n / (n.std() + 1e-9)
            sim.replay(s); log = [f"replayed {r['base']} with joint noise sigma {r['sigma']}"]
        else:
            importlib.reload(skills)  # skill fixes take effect without restarting the simulator
            log, err = skills.run_program(sim, r["code"], r.get("perturb"))
        latched = sim.success(); final_ok = sim.final_state_ok()
    except Exception as e:
        traceback.print_exc()
        log, err, latched, final_ok = [f"simulator error: {e!r}"[:300]], f"simulator error: {e!r}"[:300], False, False
    out = {"ok": True, "success": bool(latched and final_ok), "success_latched": bool(latched), "final_state_ok": bool(final_ok),
           "error": err, "log": log, "start_objects": start, "final_objects": summary(), "physics_steps": sim.n_step,
           "wall_s": round(time.time() - t0, 1)}
    if rec:
        meta = {"task": a.task, "layout": L, "instruction": sim.instruction, "source": r.get("source", "opus_program"),
                "kind": r.get("kind", "program"), "success": out["success"], "success_latched": out["success_latched"],
                "final_state_ok": out["final_state_ok"], "info": {"perturb": r.get("perturb"), "base": r.get("base"),
                                                                   "sigma": r.get("sigma"), "program": r.get("program_name")},
                "program": r.get("code"), "program_error": err, "exec_log": log, "physics_steps": sim.n_step, "dt": rd_sim.DT,
                "model": r.get("model"), "wall_s": out["wall_s"]}
        try:
            out["frames"] = sim.recorder.save(r["out_dir"], sim.stream, meta)
        except Exception as e:
            out.update(ok=False, error=f"save failed: {e!r}"[:200])
        sim.recorder = None
    return out


last = time.time()
while True:
    job = None
    for f in sorted(glob.glob(os.path.join(a.queue, "req_*.json"))):
        mine = f.replace("req_", f"run_{wid}_")
        try:
            os.rename(f, mine); job = mine; break
        except OSError:
            continue
    if job is None:
        if a.idle_exit_min and time.time() - last > a.idle_exit_min * 60:
            break
        time.sleep(1.0); continue
    last = time.time()
    r = json.load(open(job)); rid = os.path.basename(job).split("_")[-1][:-5]
    if r.get("op") == "quit":
        os.remove(job); break
    signal.alarm(int(a.request_timeout_min * 60))  # a hung sim or planner call must not hold a worker forever
    try:
        res = handle(r)
    except Exception as e:
        traceback.print_exc(); res = {"ok": False, "error": f"worker error: {e!r}"[:300]}
        sim.recorder = None
    signal.alarm(0)
    res["id"] = rid
    tmp = os.path.join(a.queue, f".tmp_{rid}")
    json.dump(res, open(tmp, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.rename(tmp, os.path.join(a.queue, f"res_{rid}.json")); os.remove(job)
    print("DONE", rid, r.get("op"), r.get("layout"), r.get("kind", ""), res.get("success"), res.get("wall_s"), flush=True)
print("WORKER_EXIT", flush=True)
sys.stdout.flush(); os._exit(0)
