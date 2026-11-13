"""Pick an exactly 50/50 success/failure eval set per task from all recorded rollouts
(Opus attempts + expansions), spreading picks across kinds and layouts, and lay it out as
    EVAL/<task>/episode_<k>/ -> symlink to the rollout dir
plus EVAL/<task>/manifest.jsonl (episode, success, kind, layout, source dir).

  python scripts/select_balanced.py --runs RUNS_DIR --task stack_blocks --per-class 30 --out EVAL
"""
import argparse, glob, json, os, random
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True)
p.add_argument("--task", required=True)
p.add_argument("--per-class", type=int, default=30, help="successes = failures = this (capped by availability)")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = random.Random(a.seed)

pool = []
for m in glob.glob(os.path.join(a.runs, a.task, "**", "metadata.json"), recursive=True):
    d = json.load(open(m))
    ep = os.path.dirname(m)
    if not os.path.exists(os.path.join(ep, "agentview_rgb", "video.mp4")):
        continue
    kind = d.get("kind") or ("opus_success" if d.get("success") else "opus_failure")
    pool.append({"dir": ep, "success": bool(d.get("success")), "kind": kind, "layout": d.get("layout")})

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
with open(os.path.join(dst, "manifest.jsonl"), "w") as f:
    for k, r in enumerate(sel):
        link = os.path.join(dst, f"episode_{k}")
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(os.path.abspath(r["dir"]), link)
        f.write(json.dumps({"episode": f"episode_{k}", **r}) + "\n")
kinds = defaultdict(lambda: [0, 0])
for r in sel:
    kinds[r["kind"]][0 if r["success"] else 1] += 1
print(f"{a.task}: pool {len(pool)} ({sum(r['success'] for r in pool)} succ); selected {n} succ + {n} fail")
for k, (s, f_) in sorted(kinds.items()):
    print(f"  {k:16s} succ {s:3d} fail {f_:3d}")
