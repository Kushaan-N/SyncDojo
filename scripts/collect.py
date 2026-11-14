"""Opus-in-the-loop collection: for each layout, Opus writes a motion script from the scene
description, the sim runs it (recorded), and on failure Opus gets the outcome and revises.
Every attempt is saved (successes AND failures) in SyncSim's episode layout.

  python scripts/collect.py --task stack_blocks --layouts 0 1 2 --out OUT [--attempts 4]
Run inside the RoboDojo is6 container (see rd6/collect.sbatch)."""
import argparse, json, os, sys, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import anthropic  # noqa: F401  load before Isaac Sim puts its older typing_extensions on sys.path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="stack_blocks")
p.add_argument("--layouts", type=int, nargs="+", default=[0])
p.add_argument("--attempts", type=int, default=4)
p.add_argument("--stop-after", type=int, default=0, help="stop once this many layouts succeeded (0: run all)")
p.add_argument("--out", required=True)
p.add_argument("--effort", default="high")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from syncdojo import rd_sim
app = rd_sim.launch(a)

from syncdojo import opus_planner as op
from syncdojo.executor import SUCCESS_TEXT, feedback_text, run_script, scene_facts
from syncdojo.recorder import Recorder

sim = rd_sim.Sim(a.task, app)
root = os.path.join(a.out, a.task)
os.makedirs(root, exist_ok=True)
summary_path = os.path.join(root, "collect_summary.jsonl")
prev = [json.loads(l) for l in open(summary_path)] if os.path.exists(summary_path) else []
solved = {r["layout"] for r in prev if r["success"]}
tried = {r["layout"] for r in prev}
n_succ = len(solved)


def safe_reset(L, tries=3):
    """RoboDojo's layout generator sometimes rejects a seed as unstable; retry, then give up on it."""
    for t in range(tries):
        try:
            sim.reset(L)
            return True
        except Exception as e:
            print("RESET_FAILED", L, t, repr(e)[:200], flush=True)
    return False


for L in a.layouts:
    if L in tried:  # resumed run: layouts already attempted (solved or exhausted) are skipped
        continue
    if a.stop_after and n_succ >= a.stop_after:
        break
    if not safe_reset(L):
        continue
    objs, arms, table = scene_facts(sim)
    scene = op.describe_scene(sim.instruction, objs, arms, table, SUCCESS_TEXT[a.task])
    convo, script = None, None
    for k in range(a.attempts):
        t0 = time.time()
        try:
            if k == 0:
                script, convo = op.plan(scene, effort=a.effort)
            else:
                script, convo = op.revise(convo, fb, effort=a.effort)
        except Exception as e:
            print("OPUS_ERROR", L, k, repr(e)[:300], flush=True)
            break
        if not safe_reset(L):
            break
        sim.recorder = Recorder()
        try:
            log, failed = run_script(sim, script)
            success = sim.success()
        except Exception as e:
            traceback.print_exc()
            log, failed, success = [f"simulator error: {e!r}"[:300]], -1, False
        fb = feedback_text(sim, log, failed, success)
        ep = os.path.join(root, f"layout_{L:03d}", f"attempt_{k}")
        meta = {"task": a.task, "layout": L, "instruction": sim.instruction, "source": "opus_motion_script",
                "attempt": k, "success": success, "failed_step": failed, "exec_log": log,
                "script": script.model_dump(), "scene_prompt": scene if k == 0 else None,
                "physics_steps": sim.n_step, "dt": rd_sim.DT, "model": op.MODEL,
                "usage": (convo or {}).get("usage"),  # token counts of every Opus call for this layout so far
                "wall_s": round(time.time() - t0, 1)}
        try:
            T = sim.recorder.save(ep, sim.stream, meta)
        except Exception:
            traceback.print_exc(); T = -1
        sim.recorder = None
        rec = {"layout": L, "attempt": k, "success": success, "failed_step": failed, "frames": T,
               "steps": len(script.steps), "dir": ep}
        print("ATTEMPT", json.dumps(rec), flush=True)
        with open(summary_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if success:
            n_succ += 1
            break
        print("FEEDBACK", L, k, fb.replace("\n", " | ")[:1500], flush=True)
    if a.stop_after and n_succ >= a.stop_after:
        break
print("COLLECT_DONE successes", n_succ, flush=True)
sys.stdout.flush(); os._exit(0)  # SimulationApp.close() can hang for minutes after the work is saved
