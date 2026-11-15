"""Skill layer for Opus-written task PROGRAMS.

Instead of a fixed list of gripper moves for one scene (opus_planner.MotionScript), Opus writes a
small Python function `run(api)` that reads the live object poses and calls skills (`api.pick`,
`api.place`, ...). The same program then runs on every layout of the task, and the skills absorb the
low-level failures a fixed script dies on (an unreachable grasp yaw, the wrong arm).

`run_program(sim, code, perturb)` executes a program in a restricted namespace, then opens the
grippers and homes both arms. `perturb` injects the expansion errors (jitter, grasp offset, yaw
error, high release, early open, collision sweep) inside the skills, so one program yields the
failure variants too.
"""
import builtins
import json
import math
import random

import numpy as np

from .rd_sim import DT

SCENE_PROPS = {"camera_stand"}
MAX_STEPS = 75000          # 300 s of simulated time per program
HOVER = 0.10               # m above a grasp / release point
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "range", "len", "min", "max", "sorted", "abs", "round", "enumerate", "zip", "list", "dict", "tuple", "set",
    "float", "int", "str", "bool", "sum", "any", "all", "isinstance", "reversed", "map", "filter", "ValueError",
    "Exception")}

API_DOC = """You write ONE Python function `run(api)` that solves the task on ANY layout of it: read the object
poses at run time, never hard-code coordinates from the example scene. No imports (`math` and `np` (numpy) are
provided). World frame, metres, z up. The executor opens both grippers and returns both arms to their start
poses after `run` returns, so do not do that yourself.

State
  api.objects() -> {name: {"xyz": [x,y,z] centre, "yaw_deg": yaw about z, "size": [sx,sy,sz] world-axis bounding
                    box, "top_z": float, "bottom_z": float, "description": [...]}}   (live, re-read after moving things)
  api.table_z   -> table surface height
  api.arms      -> {"left": {"base_xyz": [...]}, "right": {"base_xyz": [...]}}   comfortable reach ~0.55 m from a base
  api.tcp(arm)  -> [x,y,z] of that arm's tool centre point (between the fingertips)
  api.holding(arm) -> name of the object that arm picked and has not released, else None

Skills (each returns True/False and never raises; a False is already written to the log with the reason)
  api.pick(name, arm=None, yaw=None, dz=0.0, dxy=(0.0, 0.0))
      Top-down grasp: hover 10 cm above, descend straight, close, lift 10 cm, and check the object came up.
      arm None: the nearer arm first, then the other. yaw None: tries the object's yaw, yaw+90, 0, 90, 45, -45
      (yaw 0 = jaws close along world y; the jaws open ~10 cm, so close them across a dimension under ~8 cm).
      The grasp point is the object's centre in x/y (shift it with dxy, e.g. to a bowl's rim) and 3 cm below its
      top (shift with dz).
  api.place(arm, xyz, yaw=None, release_dz=0.005)
      Carries the held object so that ITS CENTRE ends at xyz (the grasp offset is accounted for), hover ->
      descend straight -> open -> retreat 10 cm up. To put A on B: xyz = [B.x, B.y, B.top_z + A.size_z / 2].
      yaw None keeps the grasp yaw (the object's orientation is unchanged).
  api.move(arm, xyz, yaw=0.0, tilt=0.0, straight=False)   raw TCP move (for pushing, tools, odd grasps)
  api.open(arm), api.close(arm), api.home(arm), api.wait(seconds)
  api.log(text)   write a line to the execution log (you get the log back if the run fails)

Time: a program may use at most 300 s of simulated time (it is stopped after that); one pick + place takes
about 15-20 s, a hand-over between the arms about 35 s. Prefer the arm that can reach both the object and its
target, and avoid hand-overs and repeated adjustments unless needed.

Rules: never move a TCP below api.table_z + 0.005. Keep the two arms apart: finish and home (api.home) one arm
before the other works in the same area. Check the return values and stop or try the other arm on False.
Aim for a short, readable function with small helpers; put per-task geometry (order, target slots) in code."""


class Api:
    def __init__(self, sim, perturb=None):
        self.sim, self.p = sim, dict(perturb or {})
        self.lines, self.held, self.n_pick, self.n_place = [], {}, 0, 0
        self.rng = random.Random(self.p.get("seed", 0))
        self.nrng = np.random.default_rng(self.p.get("seed", 0))
        self.table_z = float(sim.surface_z)
        self.arms = {s: {"base_xyz": np.asarray(r.entity_origin_pose, dtype=float)[:3].round(3).tolist()}
                     for s, r in sim.arms.items()}

    # ----------------------------------------------------------------- state
    def log(self, text):
        self.lines.append(str(text)[:300])

    def objects(self):
        out = {}
        for lab, o in self.sim.object_info().items():
            if lab in SCENE_PROPS:
                continue
            lo, hi = o.get("bbox_world_min"), o.get("bbox_world_max")
            size = o.get("bbox_size_xyz") or [0.05, 0.05, 0.05]
            out[lab] = {"xyz": list(o["xyz"]), "yaw_deg": o["yaw_deg"], "size": list(size),
                        "top_z": float(hi[2]) if hi else o["xyz"][2] + size[2] / 2,
                        "bottom_z": float(lo[2]) if lo else o["xyz"][2] - size[2] / 2,
                        "description": o.get("description") or []}
        return out

    def tcp(self, arm):
        return self.sim.tcp(arm)[:3].round(4).tolist()

    def holding(self, arm):
        return (self.held.get(arm) or {}).get("name")

    def _budget(self):
        if self.sim.n_step > MAX_STEPS:
            raise TimeoutError(f"program used more than {MAX_STEPS * DT:.0f} s of simulated time")

    def _move(self, arm, xyz, yaw, tilt=0.0, straight=False, what=""):
        self._budget()
        xyz = [float(v) for v in xyz]
        if self.p.get("jitter_m"):
            xyz = [v + float(self.nrng.normal(0, self.p["jitter_m"])) for v in xyz]
            yaw = float(yaw) + float(self.nrng.normal(0, self.p.get("jitter_yaw_deg", 0.0)))
        if xyz[2] < self.table_z + 0.005:
            self.log(f"{arm} move {what}: refused, z {xyz[2]:.3f} is below the table ({self.table_z:.3f})")
            return False
        ok, msg = self.sim.move_tcp(arm, xyz, float(yaw), float(tilt), bool(straight))
        if not ok:
            self.log(f"{arm} move {what} to {np.round(xyz, 3).tolist()} yaw {yaw:.0f}: FAILED - {msg}")
        return bool(ok)

    # ----------------------------------------------------------------- skills
    def move(self, arm, xyz, yaw=0.0, tilt=0.0, straight=False):
        ok = self._move(arm, xyz, yaw, tilt, straight, "(raw)")
        if ok:
            self.log(f"{arm} move to {np.round(xyz, 3).tolist()} yaw {float(yaw):.0f}: OK")
        return ok

    def open(self, arm):
        self._budget(); self.sim.grip(arm, 1.0); self.held.pop(arm, None)
        self.log(f"{arm} open"); return True

    def close(self, arm):
        self._budget(); self.sim.grip(arm, 0.0); self.log(f"{arm} close (gripper {self.sim.gripper(arm):.2f} open)")
        return True

    def home(self, arm):
        self._budget(); ok, msg = self.sim.move_home(arm); self.log(f"{arm} home: {msg}"); return bool(ok)

    def wait(self, seconds):
        self._budget(); self.sim.hold(max(1, int(float(seconds) / DT))); return True

    def _arm_order(self, xyz):
        return sorted(self.sim.arms, key=lambda s: np.linalg.norm(np.asarray(self.arms[s]["base_xyz"][:2]) - np.asarray(xyz[:2])))

    def pick(self, name, arm=None, yaw=None, dz=0.0, dxy=(0.0, 0.0)):
        self._budget()
        objs = self.objects()
        if name not in objs:
            self.log(f"pick {name}: no such object (have {sorted(objs)})"); return False
        o = objs[name]
        k = self.n_pick; self.n_pick += 1
        g = np.array([o["xyz"][0] + dxy[0], o["xyz"][1] + dxy[1],
                      max(o["top_z"] - 0.03, o["bottom_z"] + 0.012, self.table_z + 0.012) + dz])
        yaw_err = 0.0
        # A perturbed grasp is "blind": one attempt only, and the skill reports success whatever happened, the
        # way an open-loop controller would not notice a bad grasp. The program then carries on with the error.
        blind = False
        if self.p.get("grasp_offset") and self.p.get("at", 0) == k:
            g[:2] += np.asarray(self.p["grasp_offset"], dtype=float); blind = True
        if self.p.get("yaw_error_deg") and self.p.get("at", 0) == k:
            yaw_err = float(self.p["yaw_error_deg"]); blind = True
        yaws = [yaw] if yaw is not None else [o["yaw_deg"], o["yaw_deg"] + 90, 0.0, 90.0, 45.0, -45.0]
        yaws = [((y + 90) % 180) - 90 + yaw_err for y in yaws]
        arms = [arm] if arm else self._arm_order(o["xyz"])
        if blind:
            self.log(f"(expansion) perturbed grasp of {name}: offset {self.p.get('grasp_offset')}, yaw error {yaw_err:.0f}")
        for a in arms:
            if self.holding(a):
                continue
            for y in yaws:
                if not self._move(a, g + [0, 0, HOVER], y, what=f"hover over {name}"):
                    continue
                self.sim.grip(a, 1.0)
                if not self._move(a, g, y, straight=True, what=f"descend to {name}"):
                    continue
                self.sim.grip(a, 0.0)
                z0 = self.objects()[name]["xyz"][2]
                self._move(a, g + [0, 0, HOVER], y, straight=True, what=f"lift {name}")
                now = self.objects()[name]
                if blind:  # report success whatever happened; place() then works with what is (not) in the hand
                    self.held[a] = {"name": name, "yaw": y, "off": (np.asarray(self.sim.tcp(a)[:3]) - np.asarray(now["xyz"])).tolist()
                                    if now["xyz"][2] - z0 >= 0.03 else [0.0, 0.0, 0.0], "obj_yaw": now["yaw_deg"]}
                    self.log(f"pick {name} with {a} at yaw {y:.0f}: OK"); return True
                if now["xyz"][2] - z0 < 0.03:
                    self.log(f"pick {name} with {a} at yaw {y:.0f}: closed but the object did not come up "
                             f"(gripper {self.sim.gripper(a):.2f} open)")
                    self.sim.grip(a, 1.0)
                    return False
                self.held[a] = {"name": name, "yaw": y, "off": (np.asarray(self.sim.tcp(a)[:3]) - np.asarray(now["xyz"])).tolist(),
                                "obj_yaw": now["yaw_deg"]}
                self.log(f"pick {name} with {a} at yaw {y:.0f}: OK")
                return True
        self.log(f"pick {name}: no arm/yaw could reach it (tried arms {arms}, yaws {[round(v) for v in yaws]})")
        return False

    def place(self, arm, xyz, yaw=None, release_dz=0.005):
        self._budget()
        h = self.held.get(arm)
        if not h:
            self.log(f"place with {arm}: that arm holds nothing"); return False
        k = self.n_place; self.n_place += 1
        y = h["yaw"] if yaw is None else float(yaw)
        off = np.asarray(h["off"], dtype=float)
        if yaw is not None:  # the grasp offset rotates with the gripper
            d = math.radians(y - h["yaw"]); c, s = math.cos(d), math.sin(d)
            off = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]])
        t = np.asarray(xyz, dtype=float) + off + [0, 0, float(release_dz)]
        if self.p.get("high_release_m") and self.p.get("at", 0) == k:
            t[2] += float(self.p["high_release_m"])
        if self.p.get("early_open") and self.p.get("at", 0) == k:
            # drop it part-way through the carry (not just above the target, where it would still land on it)
            c = np.asarray(self.sim.tcp(arm)[:3]); m = c + (t + [0, 0, HOVER] - c) * float(self.p.get("early_frac", 0.5))
            m[2] = max(m[2], t[2] + HOVER)
            self._move(arm, m, y, what=f"carry {h['name']} (part-way)")
            self.sim.grip(arm, 1.0); self.log(f"(expansion) {arm}: premature release of {h['name']} part-way")
        if not self._move(arm, t + [0, 0, HOVER], y, what=f"carry {h['name']}"):
            return False
        if not self._move(arm, t, y, straight=True, what=f"lower {h['name']}"):
            return False
        self.sim.grip(arm, 1.0)
        self.held.pop(arm, None)
        self._move(arm, t + [0, 0, HOVER], y, straight=True, what="retreat")
        self.sim.hold(30)
        got = self.objects().get(h["name"], {}).get("xyz")
        self.log(f"place {h['name']} with {arm} at {np.round(xyz, 3).tolist()}: OK, it now rests at {np.round(got, 3).tolist()}")
        return True


def sweep(sim, api):
    """Collision: an arm closes its gripper and sweeps through the task objects at the top object's height."""
    objs = {k: np.asarray(v["xyz"], dtype=float) for k, v in api.objects().items()}
    before = dict(objs)
    xyz = np.array(list(objs.values())); c = xyz.mean(0)
    z = max(float(xyz[:, 2].max()) + api.rng.uniform(-0.01, 0.01), sim.surface_z + 0.015)
    done = False
    for side in api._arm_order(c):
        base = np.asarray(api.arms[side]["base_xyz"][:2]); d = c[:2] - base; d /= np.linalg.norm(d) + 1e-9
        for u in (np.array([-d[1], d[0]]), -np.array([-d[1], d[0]]), -d):
            for yaw in (0.0, 90.0):
                a0 = np.r_[c[:2] + 0.12 * u, z + 0.10]
                ok, _ = sim.move_tcp(side, a0, yaw, 0.0, False)
                if not ok:
                    continue
                sim.grip(side, 0.0)
                sim.move_tcp(side, np.r_[c[:2] + 0.12 * u, z], yaw, 0.0, False)
                sim.move_tcp(side, np.r_[c[:2] - 0.12 * u, z], yaw, 0.0, True)
                sim.move_tcp(side, np.r_[c[:2] - 0.12 * u, z + 0.10], yaw, 0.0, True)
                sim.grip(side, 1.0); sim.move_home(side)
                done = True
                break
            if done:
                break
        if done:
            break
    sim.hold(60)
    moved = {k: round(float(np.linalg.norm(np.asarray(v["xyz"]) - before[k])) * 100, 1) for k, v in api.objects().items() if k in before}
    api.log(f"sweep displaced (cm): {json.dumps(moved)}; collision: {max(moved.values(), default=0.0) >= 1.0}")


def run_program(sim, code, perturb=None, settle_steps=60):
    """Run `code` (defines run(api)) on the current scene. Returns (log lines, error string or None)."""
    api = Api(sim, perturb)
    err = None
    env = {"__builtins__": dict(SAFE_BUILTINS, print=lambda *a: api.log(" ".join(map(str, a)))), "math": math, "np": np}
    try:
        exec(compile(code, "<opus program>", "exec"), env)
        if not callable(env.get("run")):
            raise ValueError("the program does not define run(api)")
        env["run"](api)
    except Exception as e:  # a bug in the program is feedback, not a crash
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        where = next((f"line {f.lineno}: {f.line}" for f in reversed(tb) if f.filename == "<opus program>"), "")
        err = f"{type(e).__name__}: {e} ({where})"[:400]
        api.log("PROGRAM ERROR " + err)
    for s in ("left", "right"):
        if sim.gripper(s) < 0.8:
            sim.grip(s, 1.0)
    for s in ("left", "right"):
        ok, msg = sim.move_home(s)
        api.log(f"auto: {s} arm home: {msg}")
    sim.hold(settle_steps)
    if (perturb or {}).get("sweep"):
        sweep(sim, api)
    return api.lines, err
