"""Expand Opus's successful motion scripts into a balanced eval set.

For every successful Opus attempt under OUT/<task>/layout_*/attempt_*/ this runs variants in
the simulator (all recorded in SyncSim's layout under OUT/<task>/expand/<kind>_<layout>_<i>/):

  rerun          the script again unchanged (sim nondeterminism check)
  jitter         every move target jittered by a few mm (mostly successes)
  jitter_large   every move target jittered ~1 cm and yaw ~8 deg (pushes miss, placements drift)
  noisy_replay   the recorded control stream replayed with smooth joint noise (small or large)
  grasp_offset   the grasp pose of one pick shifted 1.5-4 cm sideways (missed / weak grasps)
  yaw_error      one grasp yaw rotated 25-60 deg
  high_release   one release done 3-8 cm higher (dropped blocks, toppled stacks)
  early_open     the gripper opens part-way through one transport
  sweep          after the script, the free arm sweeps through the stack at mid height (collision)

Every rollout is labelled by the task's own success check. `select` keeps an exactly 50/50
success/failure set per task (balanced by outcome, then spread across kinds and layouts).

  python scripts/expand.py --task stack_blocks --out OUT --per-kind 4
"""
import argparse, copy, glob, json, os, random, sys, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import anthropic  # noqa: F401  load before Isaac Sim puts its older typing_extensions on sys.path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="stack_blocks")
p.add_argument("--out", required=True)
p.add_argument("--per-kind", type=int, default=3, help="variants per kind per successful script")
p.add_argument("--kinds", default="rerun,jitter,jitter_large,noisy_replay,grasp_offset,yaw_error,high_release,early_open,sweep")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--extra-bases", nargs="*", default=[],
               help="attempt dirs used as bases even though they failed (near-misses); outcomes are still labelled by the sim")
p.add_argument("--suffix", default="", help="appended to rollout names, so parallel expanders never collide")
p.add_argument("--max-minutes", type=float, default=200)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from syncdojo import rd_sim
app = rd_sim.launch(a)

import numpy as np
from syncdojo.executor import run_script
from syncdojo.opus_planner import MotionScript
from syncdojo.recorder import Recorder

rng = random.Random(a.seed)
SCENE_PROPS = {"camera_stand"}  # fixed scene furniture, not part of the task
nrng = np.random.default_rng(a.seed)
root = os.path.join(a.out, a.task)
bases = []
for m in sorted(glob.glob(os.path.join(root, "layout_*", "attempt_*", "metadata.json"))):
    meta = json.load(open(m))
    if meta.get("success"):
        bases.append((os.path.dirname(m), meta))
for b in a.extra_bases:
    bases.append((b.rstrip("/"), json.load(open(os.path.join(b, "metadata.json")))))
print("BASES", len(bases), [b[0] for b in bases], flush=True)
sim = rd_sim.Sim(a.task, app)
log_path = os.path.join(root, "expand_summary.jsonl")
done = set()
if os.path.exists(log_path):
    done = {json.loads(l)["name"] for l in open(log_path)}


# ----------------------------------------------------------------------------- script edits
def pick_indices(steps):
    """Indices of 'close' steps (grasps) and 'open' steps that follow a close (releases)."""
    closes = [i for i, s in enumerate(steps) if s["op"] == "close"]
    releases = [i for i, s in enumerate(steps) if s["op"] == "open" and any(c < i for c in closes)]
    return closes, releases


def prev_move(steps, i, arm):
    for j in range(i - 1, -1, -1):
        if steps[j]["op"] == "move" and steps[j]["arm"] == arm:
            return j
    return None


def edit_script(script, kind):
    s = copy.deepcopy(script)
    st = s["steps"]
    closes, releases = pick_indices(st)
    info = {}
    if kind in ("jitter", "jitter_large"):
        sig = 0.003 if kind == "jitter" else 0.012
        for x in st:
            if x["op"] == "move":
                x["xyz"] = [v + float(nrng.normal(0, sig)) for v in x["xyz"]]
                if kind == "jitter_large" and x.get("yaw_deg") is not None:
                    x["yaw_deg"] += float(nrng.normal(0, 8))
        info["sigma_m"] = sig
    elif kind == "grasp_offset" and closes:
        c = rng.choice(closes); j = prev_move(st, c, st[c]["arm"])
        ang, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.015, 0.04)
        dx, dy = d * np.cos(ang), d * np.sin(ang)
        k = j  # shift the whole approach column (hover, descend) so the descent stays vertical
        while k is not None and st[k]["op"] == "move" and k >= 0:
            st[k]["xyz"] = [st[k]["xyz"][0] + dx, st[k]["xyz"][1] + dy, st[k]["xyz"][2]]
            k2 = prev_move(st, k, st[c]["arm"])
            if k2 is None or k - k2 > 2:
                break
            k = k2
        info.update(step=c, offset_xy=[dx, dy])
    elif kind == "yaw_error" and closes:
        c = rng.choice(closes); j = prev_move(st, c, st[c]["arm"])
        dyaw = rng.choice([-1, 1]) * rng.uniform(25, 60)
        for k in (j, prev_move(st, j, st[c]["arm"]) if j is not None else None):
            if k is not None:
                st[k]["yaw_deg"] = (st[k].get("yaw_deg") or 0.0) + dyaw
        info.update(step=c, dyaw_deg=dyaw)
    elif kind == "high_release" and releases:
        r = rng.choice(releases); j = prev_move(st, r, st[r]["arm"])
        dz = rng.uniform(0.03, 0.08)
        if j is not None:
            st[j]["xyz"] = [st[j]["xyz"][0], st[j]["xyz"][1], st[j]["xyz"][2] + dz]
        info.update(step=r, dz=dz)
    elif kind == "early_open" and closes:
        c = rng.choice(closes)
        later = [i for i in range(c + 1, len(st)) if st[i]["op"] == "move" and st[i]["arm"] == st[c]["arm"]]
        if later:
            i = rng.choice(later[: max(1, len(later) - 1)])
            st.insert(i + 1, {"arm": st[c]["arm"], "op": "open", "note": "premature release (expansion)"})
            info.update(after_step=i)
    else:
        return None, info
    return MotionScript.model_validate(s), info


def sweep_steps(sim):
    """Free arm sweeps across the object cluster at the top object's height. Scene props (the camera
    stand) are left out, so the sweep aims at the task objects rather than between them and the stand."""
    objs = {k: v for k, v in sim.objects().items() if k not in SCENE_PROPS}
    xyz = np.array([v[:3] for v in objs.values()])
    c = xyz.mean(0)
    side = rng.choice(["left", "right"])
    base = np.asarray(sim.arms[side].entity_origin_pose, dtype=float)[:3]
    d = c[:2] - base[:2]; d /= np.linalg.norm(d) + 1e-9
    perp = np.array([-d[1], d[0]])
    z = float(xyz[:, 2].max()) + rng.uniform(-0.01, 0.02)
    a0 = np.r_[c[:2] + 0.12 * perp, z + 0.10]
    a1 = np.r_[c[:2] + 0.12 * perp, z]
    a2 = np.r_[c[:2] - 0.12 * perp, z]
    yaw = float(np.degrees(np.arctan2(perp[1], perp[0])))
    return side, [a0, a1, a2], yaw


# ----------------------------------------------------------------------------- rollouts
def rollout(name, kind, layout, fn, info):
    ep = os.path.join(root, "expand", name)
    t0 = time.time()
    for t in range(3):
        try:
            sim.reset(layout)
            break
        except Exception as e:
            print("RESET_FAILED", layout, t, repr(e)[:200], flush=True)
    else:
        return
    sim.recorder = Recorder()
    try:
        log = fn()
        success = sim.success()
    except Exception as e:
        traceback.print_exc(); log, success = [f"simulator error: {e!r}"[:300]], False
    meta = {"task": a.task, "layout": layout, "instruction": sim.instruction, "source": "expansion",
            "kind": kind, "success": success, "info": info, "exec_log": log,
            "physics_steps": sim.n_step, "dt": rd_sim.DT, "wall_s": round(time.time() - t0, 1)}
    try:
        T = sim.recorder.save(ep, sim.stream, meta)
    except Exception as e:
        print("SAVE_FAILED", name, repr(e)[:200], flush=True)
        sim.recorder = None
        return
    sim.recorder = None
    rec = {"name": name, "layout": layout, "kind": meta["kind"], "success": success, "frames": T, "dir": ep}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("EXPAND", json.dumps(rec), flush=True)


def noisy_stream(stream, sigma):
    s = np.asarray(stream, dtype=np.float64).copy()
    T = len(s)
    # smooth noise on arm joint targets only (per arm: q[6], dq[6], g[2])
    for off in (0, 14):
        for j in range(6):
            n = nrng.normal(0, 1, T + 200)
            n = np.convolve(n, np.hanning(200), mode="valid")[:T]
            s[:, off + j] += sigma * n / (n.std() + 1e-9)
    return s


t_start = time.time()
kinds = a.kinds.split(",")
# round-robin (variant index, kind, base) so a time budget still covers every kind and layout
order = [(i, kind, b) for i in range(a.per_kind) for kind in kinds for b in bases]
for i, kind, (bdir, meta) in order:
    L = meta["layout"]
    script = meta["script"]
    stream = np.load(os.path.join(bdir, "stream.npz"))["targets"]
    tag = f"{L:03d}_{os.path.basename(bdir)[-1]}"
    if True:
        if True:
            if (time.time() - t_start) / 60 > a.max_minutes:
                print("TIME_BUDGET_REACHED", flush=True); break
            name = f"{kind}_{tag}_{i}{a.suffix}"
            if name in done:
                continue
            if kind == "rerun":
                ms = MotionScript.model_validate(script)
                rollout(name, kind, L, lambda ms=ms: run_script(sim, ms)[0], {})
            elif kind == "noisy_replay":
                sig = [0.01, 0.03, 0.06][i % 3]
                ns = noisy_stream(stream, sig)
                rollout(name, kind, L, lambda ns=ns: (sim.replay(ns), ["replayed noisy stream"])[1], {"sigma_rad": sig, "base": bdir})
            elif kind == "sweep":
                ms = MotionScript.model_validate(script)

                def fn(ms=ms):
                    log = run_script(sim, ms, settle_steps=20)[0]
                    before = {k: np.asarray(v[:3], float) for k, v in sim.objects().items()}
                    side, pts, yaw = sweep_steps(sim)
                    for k, pt in enumerate(pts):
                        ok, msg = sim.move_tcp(side, pt, yaw, 0.0, straight=(k == 2))
                        log.append(f"sweep {side} {k}: {ok} {msg}")
                    sim.move_home(side); sim.hold(60)
                    moved = {k: round(float(np.linalg.norm(np.asarray(v[:3], float) - before[k])) * 100, 1)
                             for k, v in sim.objects().items() if k in before and k not in SCENE_PROPS}
                    hit = max(moved.values(), default=0.0) >= 1.0
                    log.append(f"sweep displaced (cm): {json.dumps(moved)}; collision: {hit}")
                    return log
                rollout(name, kind, L, fn, {"base": bdir})
            else:
                ms, info = edit_script(script, kind)
                if ms is None:
                    continue
                info["base"] = bdir
                rollout(name, kind, L, lambda ms=ms: run_script(sim, ms)[0], info)
print("EXPAND_DONE", flush=True)
sys.stdout.flush(); os._exit(0)  # SimulationApp.close() can hang for minutes after the work is saved
