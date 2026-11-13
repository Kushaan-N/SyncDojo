"""Write the camera parameters of a task to OUT/<task>/cameras.json: per camera the 3x3 intrinsics,
image size and pose. The head camera is fixed in the world; a wrist camera is rigidly mounted on its
arm, so its pose is given relative to that arm's TCP (camera_to_world = TCP_to_world @ camera_to_tcp,
with the TCP pose of every frame in pose.pkl). The mount is measured at two arm poses to confirm it
is rigid, and the head pose on two layouts to confirm it does not move.

  python scripts/dump_cameras.py --task stack_blocks --out OUT"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="stack_blocks")
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from syncdojo import rd_sim
app = rd_sim.launch(a)
import numpy as np
from omegaconf import OmegaConf

sim = rd_sim.Sim(a.task, app)
cm = sim.env.camera_manager


def T_of(pose7):  # xyz + quat wxyz -> 4x4
    T = np.eye(4); T[:3, :3] = rd_sim.qmat(pose7[3:7]); T[:3, 3] = pose7[:3]
    return T


def snap():
    out = {}
    for i, name in enumerate(cm.camera_names[0]):
        out[name] = cm.get_camera_extrinsics(i, 0)
    return out, {s: T_of(sim.tcp(s)) for s in ("left", "right")}


sim.reset(0)
E0, P0 = snap()
moved = {}
for s in ("left", "right"):  # move each arm, then measure its wrist mount again
    t = sim.tcp(s)
    ok, msg = sim.move_tcp(s, [t[0], t[1] + 0.08, t[2] + 0.06], 0.0, 0.0, False)
    moved[s] = bool(ok)
sim.hold(20)
E1, P1 = snap()
sim.reset(1)
E2, _ = snap()

cams = {}
for i, name in enumerate(cm.camera_names[0]):
    cfg = OmegaConf.to_container(cm.camera_config[name], resolve=True)
    K = cm.get_camera_intrinsics(i, 0)
    side = "left" if "left" in name else "right" if "right" in name else None
    c = {"intrinsics": K.tolist(), "config": cfg, "mounted_on": f"{side} arm" if side else "world"}
    if side:
        M0 = np.linalg.inv(P0[side]) @ E0[name]; M1 = np.linalg.inv(P1[side]) @ E1[name]
        c["camera_to_tcp"] = M0.tolist()
        c["mount_check"] = {"arm_moved": moved[side], "tcp_moved_m": float(np.linalg.norm(P1[side][:3, 3] - P0[side][:3, 3])),
                            "max_abs_diff_between_two_arm_poses": float(np.abs(M0 - M1).max())}
    else:
        c["camera_to_world"] = E0[name].tolist()
        c["fixed_check"] = {"max_abs_diff_other_layout": float(np.abs(E0[name] - E2[name]).max()),
                            "max_abs_diff_after_arm_motion": float(np.abs(E0[name] - E1[name]).max())}
    cams[name] = c
doc = {"task": a.task, "conventions": "4x4 camera-to-world / camera-to-TCP, metres; camera frame as returned by the simulator's "
       "camera prim (USD: -Z forward, +Y up); intrinsics for the native resolution in config",
       "video_dirs": {"cam_head": "agentview_rgb", "cam_left_wrist": "wrist_left_rgb", "cam_right_wrist": "wrist_right_rgb"},
       "random_config": OmegaConf.to_container(cm.random_config, resolve=True), "cameras": cams}
os.makedirs(os.path.join(a.out, a.task), exist_ok=True)
with open(os.path.join(a.out, a.task, "cameras.json"), "w") as f:
    json.dump(doc, f, indent=1)
for n, c in cams.items():
    print("CAM", n, c["mounted_on"], c.get("mount_check") or c.get("fixed_check"), "K", np.round(np.array(c["intrinsics"]), 1).tolist(), flush=True)
print("RANDOM", doc["random_config"], flush=True)
print("CAMERAS_DONE", flush=True)
sys.stdout.flush(); os._exit(0)
