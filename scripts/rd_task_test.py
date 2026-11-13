"""Smoke test: load one RoboDojo task headless, save every camera view, print object / end-effector
poses, then plan and execute a small left-arm lift and save the views again.
  python scripts/rd_task_test.py --task stack_blocks --layout 0 --out OUT_DIR
Run inside the RoboDojo container with RoboDojo and RoboDojo/XPolicyLab on PYTHONPATH."""
import argparse, json, os, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="stack_blocks")
p.add_argument("--layout", type=int, default=0)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless, a.enable_cameras = True, True
os.environ.setdefault("ROBODOJO_RUN_ID", "smoke")
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
add_zero_delay_kit_args(a)
# Isaac Sim 6.x defaults to RealTimePathTracing; RaytracedLighting (5.1's mode) denoises cleanly
a.kit_args = ((getattr(a, "kit_args", None) or "") + " --/rtx/rendermode=RaytracedLighting").strip()
app = AppLauncher(a).app

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from env.global_configs import *
import src.eval_client.eval_env as ee
from utils.load_file import load_yaml
from utils.pipeline_utils import *


class NoPolicy:  # stands in for the WebSocket policy client: no policy server is used
    def __init__(self, *args, **kw):
        pass

    def call(self, *args, **kw):
        return None


ee.WsModelClient = NoPolicy
os.makedirs(a.out, exist_ok=True)

ecfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
ecfg.update(task_name=a.task, num_envs=1, device_id=0, eval_batch=False, policy_name="none",
            additional_info="smoke", seed=0, physx_monitor_enabled=False)
c = ecfg["config"]
cfg = OmegaConf.create({
    "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", c["sim"] + ".yml")),
    "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", c["scene"] + ".yml")),
    "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", c["camera"] + ".yml")),
    "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", c["robot"] + ".yml")),
    "task_env": load_yaml(os.path.join(ROOT_DIR, "task", BENCHMARK, "config", a.task + ".yml")),
    "eval_cfg": ecfg,
    "deploy_cfg": {"policy_name": "none", "port": 1, "host": "localhost", "protocol": "ws"},
})
OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
cfg = process_randomization(cfg)
cfg, _ = process_config(cfg, task_name=a.task)
OmegaConf.update(cfg, "camera.default_frequency", ecfg["observation"].get("collect_freq", 0), force_add=True)
cfg.sim.seed = [0]

env = ee.create_eval_env(cfg, app)
env.seed_manager.init_eval() if not hasattr(env.seed_manager, "seed_list") else None
env.reset(seed=[a.layout])


def dump(tag):
    obs = env.get_obs()
    for cam, v in obs.get("vision", {}).items():
        im = np.asarray(v["color"])
        Image.fromarray(im.astype(np.uint8)).save(os.path.join(a.out, f"{tag}_{cam}.png"))
        print(tag, cam, im.shape, "mean %.1f std %.1f" % (im.mean(), im.std()), flush=True)
    st = {}
    for k, v in obs.items():
        if k == "vision":
            continue
        if isinstance(v, dict):  # per-env dict {env_idx: value}; one env here
            v = next(iter(v.values()), None) if v else None
        try:
            st[k] = np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float).round(4).tolist()
        except (TypeError, ValueError):
            st[k] = repr(v)[:200]
    return st


st0 = dump("reset")
lm = env.scene_manager.layout_manager
objs = {}
labels = [l for grp in cfg.task_env.get("Rigid", []) for l in grp.get("select_mode", {}).get("label", [])]
for lab in labels:
    try:
        pos, q = lm.get_instance_pose(0, label=lab, relative=True)
        objs[lab] = {"pos": np.asarray(pos.cpu()).round(4).tolist(), "quat_wxyz": np.asarray(q.cpu()).round(4).tolist()}
    except Exception as e:
        objs[lab] = repr(e)
print("OBJECTS", json.dumps(objs), flush=True)
print("STATE", json.dumps(st0)[:2000], flush=True)

# small planned move: left arm TCP up 5 cm
rm = env.robot_manager
robot = rm.robot_list[0]
try:
    ee_pose = np.asarray(rm.get_link_pose(robot, robot.ee_link_name, is_relative=True)[0], dtype=float).reshape(-1)[:7]
    print("EE start", ee_pose.round(4).tolist(), flush=True)
    tgt = ee_pose.copy(); tgt[2] += 0.05
    plan = rm.planner["x5"].plan_path(rm.get_joint(robot)[0], tgt, real_robot_pose=robot.entity_origin_pose)
    print("PLAN status", plan.get("status"), "len", len(plan.get("position", [])), flush=True)
    steps = rm.plan_ee(0, "left_arm", result=plan)
    rm.control_manager.push([0], [steps])
    for _ in range(len(steps)):
        meta = rm.control_manager.pop([0]); rm.control_robot(meta); env.sim_step(render=False)
    st1 = dump("moved")
    ee1 = np.asarray(rm.get_link_pose(robot, robot.ee_link_name, is_relative=True)[0], dtype=float).reshape(-1)
    print("EE z before %.4f after %.4f" % (ee_pose[2], ee1[2]), flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
# link names + poses, to find the TCP offset from the ee link
key = rm.robot_key[0]
print("LINKS", key.body_names, flush=True)
for ln in key.body_names:
    print("LINKPOSE", ln, np.asarray(rm.get_link_pose(robot, ln, is_relative=True)[0]).round(4).tolist(), flush=True)
print("EE_LINK", robot.ee_link_name, "DELTA", np.asarray(robot.delta_matrix).round(4).tolist(), flush=True)
print("TASK_TEST_DONE", flush=True)
app.close()
