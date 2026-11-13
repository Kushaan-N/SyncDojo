"""Pick an exactly 50/50 success/failure eval set per task from all recorded rollouts
(Opus attempts + expansions), spreading picks across kinds and layouts, and lay it out as
    EVAL/<task>/episode_<k>/ -> symlink to the rollout dir
plus EVAL/<task>/manifest.jsonl (episode, success, kind, layout, source dir).

  python scripts/select_balanced.py --runs RUNS_DIR --task stack_blocks --per-class 30 --out EVAL
"""
import argparse, glob, json, os, random
from collections import Counter, defaultdict

import numpy as np

MIN_PATH_M = 0.10   # total TCP travel below this: nothing happens in the episode
DUP_TOL_M = 0.001   # arm and object paths closer than this at every frame: the same episode twice

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True)
p.add_argument("--task", required=True)
p.add_argument("--per-class", type=int, default=30, help="successes = failures = this (capped by availability)")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = random.Random(a.seed)

pool, skipped, calib = [], Counter(), {}
for m in sorted(glob.glob(os.path.join(a.runs, a.task, "**", "metadata.json"), recursive=True)):
    d = json.load(open(m))
    ep = os.path.dirname(m)
    if d.get("source") == "calibration":  # calibration clips are context, not episodes
        calib[d.get("layout")] = ep
        continue
    if not os.path.exists(os.path.join(ep, "agentview_rgb", "video.mp4")):
        continue
    kind = d.get("kind") or ("opus_success" if d.get("success") else "opus_failure")
    if kind == "sweep" and "final_state_ok" not in d:
        skipped["sweep labelled before the final-state check existed"] += 1
        continue
    if kind == "sweep" and not any("collision: True" in l for l in d.get("exec_log", [])):
        skipped["sweep that touched nothing"] += 1
        continue
    A = np.load(os.path.join(ep, "actions.npy"))
    xyz = A[:, [0, 1, 2, 8, 9, 10]]
    if np.abs(np.diff(xyz, axis=0)).sum() < MIN_PATH_M:
        skipped["arms never move (script stopped at its first step)"] += 1
        continue
    obj = np.load(os.path.join(ep, "object_poses.npz"))["poses"][..., :3]
    pool.append({"dir": ep, "success": bool(d.get("success")), "kind": kind, "layout": d.get("layout"),
                 "_xyz": xyz, "_obj": obj})

# Reruns of a script can come out identical to the original (the sim is often deterministic). Keep one
# of each group with identical arm AND object trajectories; originals (Opus attempts) are kept in preference.
pool.sort(key=lambda r: (r["kind"] not in ("opus_success", "opus_failure"), r["dir"]))
kept = []
for r in pool:
    dup = any(k["layout"] == r["layout"] and k["_xyz"].shape == r["_xyz"].shape and k["_obj"].shape == r["_obj"].shape
              and np.abs(k["_xyz"] - r["_xyz"]).max() < DUP_TOL_M and np.abs(k["_obj"] - r["_obj"]).max() < DUP_TOL_M
              for k in kept)
    if dup:
        skipped["identical to another rollout"] += 1
    else:
        kept.append(r)
pool = [{k: v for k, v in r.items() if not k.startswith("_")} for r in kept]

by = {True: defaultdict(list), False: defaultdict(list)}
for r in pool:
    by[r["success"]][(r["kind"], r["layout"])].append(r)
n = min(a.per_class, sum(map(len, by[True].values())), sum(map(len, by[False].values())))


def take(groups, n):
    """Round-robin over (kind, layout) groups so no single kind or layout dominates."""
    groups = {k: rng.sample(v, len(v)) for k, v in groups.items()}
    keys = sorted(groups, key=lambda k: (k[0], str(k[1])))
    rng.shuffle(keys)
    out = []
    while len(out) < n and keys:
        for k in list(keys):
            if groups[k]:
                out.append(groups[k].pop())
                if len(out) == n:
                    break
            else:
                keys.remove(k)
    return out


sel = take(by[True], n) + take(by[False], n)
rng.shuffle(sel)
dst = os.path.join(a.out, a.task)
os.makedirs(dst, exist_ok=True)
for old in glob.glob(os.path.join(dst, "episode_*")):  # a rebuilt set may be smaller: no stale links
    if os.path.islink(old):
        os.unlink(old)
# one calibration clip per layout (scripts/calibrate.py), linked next to the episodes
cdir = os.path.join(dst, "calibration")
os.makedirs(cdir, exist_ok=True)
for old in glob.glob(os.path.join(cdir, "layout_*")):
    if os.path.islink(old):
        os.unlink(old)
for L in sorted({r["layout"] for r in sel} & set(calib)):
    os.symlink(os.path.abspath(calib[L]), os.path.join(cdir, f"layout_{L:03d}"))
with open(os.path.join(dst, "manifest.jsonl"), "w") as f:
    for k, r in enumerate(sel):
        link = os.path.join(dst, f"episode_{k}")
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(os.path.abspath(r["dir"]), link)
        cal = f"calibration/layout_{r['layout']:03d}" if r["layout"] in calib else None
        f.write(json.dumps({"episode": f"episode_{k}", **r, "calibration": cal}) + "\n")
kinds = defaultdict(lambda: [0, 0])
for r in sel:
    kinds[r["kind"]][0 if r["success"] else 1] += 1
print(f"{a.task}: calibration clips for {len({r['layout'] for r in sel} & set(calib))} of {len({r['layout'] for r in sel})} layouts")
print(f"{a.task}: pool {len(pool)} ({sum(r['success'] for r in pool)} succ); selected {n} succ + {n} fail")
for why, c in skipped.items():
    print(f"  left out: {c} ({why})")
for k, (s, f_) in sorted(kinds.items()):
    print(f"  {k:16s} succ {s:3d} fail {f_:3d}")
