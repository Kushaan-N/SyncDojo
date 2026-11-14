"""Calibration clip per layout, following SyncSim: each arm moves each of its 6 end-effector DoFs
(x, y, z translations in the world frame; roll, pitch, yaw about world axes through the TCP)
out -> hold -> back -> hold while the other arm and both grippers stay still.

Probe, then film (as in SyncSim): every leg is first tried unrecorded; a leg is accepted only if
the TCP moved the commanded DoF by the commanded amount (within 25 %) with little off-axis motion,
otherwise the opposite sign or a smaller magnitude is tried. Only accepted legs are then replayed
and recorded, at constant joint speed, so the clip has no failed or off-axis motion.

  python scripts/calibrate.py --task stack_blocks --layouts 0 1 2 --out OUT
writes OUT/<task>/calibration/layout_<LLL>/ in the usual episode layout; metadata.json lists the legs
(arm, DoF, signed amount, measured motion, frame range of out / back)."""
import argparse, json, os, sys, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="stack_blocks")
p.add_argument("--layouts", type=int, nargs="+", default=[0])
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from syncdojo import rd_sim
app = rd_sim.launch(a)
import numpy as np
from syncdojo.rd_sim import DT, qmat, qmul, tcp_to_link6
from syncdojo.recorder import Recorder

TRANS_M, ROT_DEG = (0.05, 0.08), (12.0, 20.0)       # SyncSim's ranges
TRANS_SPEED, ROT_SPEED = 0.08, 20.0                 # m/s, deg/s
PAUSE, TRACK_TOL, MAX_SPUR_ROT, MAX_SPUR_TRANS = 60, 0.25, 5.0, 0.03
MAX_JOINT_DELTA = 0.8  # rad; a single-DoF leg of this size never needs more
AXES = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}
sim = rd_sim.Sim(a.task, app)


def joints(side):
    return np.asarray(sim.rm.get_joint(sim.arms[side])[0], dtype=float).copy()


def interp(side, q0, q1, n):
    r = sim.arms[side]
    name = sim.rm.process_name(r.arm_name)
    sim.run_controls([{name: {"position": q0 + (q1 - q0) * (i + 1) / n, "velocity": np.zeros_like(q0)}} for i in range(n)])


def go_to(side, q, speed=0.6):
    """Joint-interpolate to q at <= `speed` rad/s, then wait until the arm has really arrived."""
    q0 = joints(side)
    n = max(24, int(np.abs(q - q0).max() / speed / DT))
    interp(side, q0, q, n)
    for _ in range(40):
        if np.abs(joints(side) - q).max() < 2e-3:
            return True
        sim.hold(10)
    return False


def measured(a7, b7):
    dp = b7[:3] - a7[:3]
    R = qmat(b7[3:7]) @ qmat(a7[3:7]).T
    return dp, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def tracks(dof, amount, dp, rot):
    if dof in ("x", "y", "z"):
        ax = AXES[dof]; off = float(np.linalg.norm(np.delete(dp, ax)))
        if abs(dp[ax] - amount) > TRACK_TOL * abs(amount):
            return False, f"moved {dp[ax]:+.3f} m along {dof}, commanded {amount:+.3f}"
        if off > 0.01 + TRACK_TOL * abs(amount):
            return False, f"off-axis translation {off:.3f} m"
        if rot > MAX_SPUR_ROT:
            return False, f"spurious rotation {rot:.1f} deg"
    else:
        if abs(rot - abs(amount)) > TRACK_TOL * abs(amount):
            return False, f"rotated {rot:.1f} deg, commanded {abs(amount):.1f}"
        if np.linalg.norm(dp) > MAX_SPUR_TRANS:
            return False, f"spurious translation {np.linalg.norm(dp):.3f} m"
    return True, ""


def probe(layout, rng):
    """Unrecorded pass: which legs can this arm really do? Returns the leg list (accepted ones carry q0/q1/n)."""
    legs = []
    sides = list(rng.permutation(["left", "right"]))
    for side in sides:
        dofs = list(rng.permutation(["x", "y", "z"])) + list(rng.permutation(["roll", "pitch", "yaw"]))
        for dof in dofs:
            is_rot = dof in ("roll", "pitch", "yaw")
            mag = float(rng.uniform(*(ROT_DEG if is_rot else TRANS_M)))
            sign = float(rng.choice([-1.0, 1.0]))
            leg = {"arm": side, "dof": dof, "attempts": []}
            for shrink in (1.0, 0.6, 0.35):
                for sg in (sign, -sign):
                    amount = sg * mag * shrink
                    q0 = sim.home_q[side].copy()  # every leg starts from, and returns to, the reset pose
                    if not go_to(side, q0):
                        leg["attempts"].append({"amount": amount, "ok": False, "why": "arm did not return to its start pose"})
                        continue
                    s7 = sim.tcp(side)
                    xyz, q = s7[:3].copy(), s7[3:7].copy()
                    if is_rot:
                        h = np.deg2rad(amount) / 2; ax = np.zeros(3); ax[AXES[dof]] = 1.0
                        q = qmul(np.r_[np.cos(h), np.sin(h) * ax], q)
                    else:
                        xyz[AXES[dof]] += amount
                    plan = sim.plan_link6(side, np.r_[tcp_to_link6(xyz, q), q])
                    if plan.get("status") != "Success":
                        leg["attempts"].append({"amount": amount, "ok": False, "why": "no plan"})
                        continue
                    sim.run_controls(sim.rm.plan_ee(0, sim.arms[side].arm_name, result=plan))
                    sim.hold(20)
                    q1 = joints(side); dp, rot = measured(s7, sim.tcp(side))
                    n = max(24, int(round((abs(amount) / (ROT_SPEED if is_rot else TRANS_SPEED)) / DT)))
                    go_to(side, q0)
                    ok, why = tracks(dof, amount, dp, rot)
                    if ok and np.abs(q1 - q0).max() > MAX_JOINT_DELTA:
                        ok, why = False, f"joint change {np.abs(q1 - q0).max():.2f} rad (the planner switched arm configuration)"
                    leg["attempts"].append({"amount": amount, "ok": ok, "why": why, "dpos_m": dp.round(4).tolist(), "rot_deg": round(rot, 2)})
                    if ok:
                        leg.update(amount=amount, unit="deg" if is_rot else "m", shrink=shrink, steps=n,
                                   _q0=q0, _q1=q1)
                        break
                if "amount" in leg:
                    break
            if "amount" not in leg:
                leg["skipped"] = "unreachable"
            legs.append(leg)
    return sides, legs


def film(layout, legs, ep):
    sim.reset(layout)
    obj0 = {k: np.asarray(v[:3], float) for k, v in sim.objects().items()}
    sim.recorder = Recorder()
    sim.hold(12)
    fr = lambda: len(sim.recorder.log["step"])
    for leg in legs:
        if "skipped" in leg:
            continue
        side, q0, q1, n = leg["arm"], leg.pop("_q0"), leg.pop("_q1"), leg["steps"]
        s7 = sim.tcp(side); f0 = fr()
        interp(side, q0, q1, n); f1 = fr()
        dp, rot = measured(s7, sim.tcp(side))
        sim.hold(PAUSE); f2 = fr()
        interp(side, q1, q0, n); f3 = fr()
        sim.hold(PAUSE)
        leg.update(out_frames=[f0, f1], back_frames=[f2, f3], filmed_dpos_m=dp.round(4).tolist(), filmed_rot_deg=round(rot, 2),
                   filmed_tracking_ok=bool(tracks(leg["dof"], leg["amount"], dp, rot)[0]),
                   return_error_m=float(np.linalg.norm(sim.tcp(side)[:3] - s7[:3])))
    moved = max(float(np.linalg.norm(np.asarray(v[:3], float) - obj0[k])) for k, v in sim.objects().items())
    return moved


for L in a.layouts:
    ep = os.path.join(a.out, a.task, "calibration", f"layout_{L:03d}")
    if os.path.exists(os.path.join(ep, "metadata.json")):
        continue
    for attempt in range(4):  # RoboDojo sometimes rejects a layout as unstable on reset; retry
        try:
            rng = np.random.default_rng(a.seed * 1000 + L)
            sim.reset(L)
            sides, legs = probe(L, rng)
            moved = film(L, legs, ep)
            done = [l for l in legs if "skipped" not in l]
            if not all(l["filmed_tracking_ok"] and l["return_error_m"] < 0.005 for l in done):
                raise RuntimeError("filmed legs do not match the probe: " + str([(l["arm"], l["dof"]) for l in done if not l["filmed_tracking_ok"]]))
            meta = {"task": a.task, "layout": L, "instruction": sim.instruction, "source": "calibration",
                    "arm_order": sides, "legs": legs,
                    "movement_order": [f"{l['arm']}:{l['dof']}{'+' if l['amount'] > 0 else '-'}" for l in done],
                    "skipped": [f"{l['arm']}:{l['dof']}" for l in legs if "skipped" in l],
                    "rejected_attempts": sum(1 for l in legs for t in l["attempts"] if not t["ok"]),
                    "trans_range_m": TRANS_M, "rot_range_deg": ROT_DEG, "trans_speed_mps": TRANS_SPEED, "rot_speed_dps": ROT_SPEED,
                    "pause_steps": PAUSE, "objects_moved_m": moved, "physics_steps": sim.n_step, "dt": DT}
            T = sim.recorder.save(ep, sim.stream, meta)
            sim.recorder = None
            print("CALIB", json.dumps({"layout": L, "frames": T, "legs": len(done), "skipped": meta["skipped"],
                                       "filmed_ok": sum(l["filmed_tracking_ok"] for l in done), "rejected": meta["rejected_attempts"],
                                       "objects_moved_cm": round(moved * 100, 2)}), flush=True)
        except Exception:
            traceback.print_exc(); sim.recorder = None
            print("CALIB_FAILED", L, "attempt", attempt, flush=True)
            continue
        break
print("CALIB_DONE", flush=True)
sys.stdout.flush(); os._exit(0)
