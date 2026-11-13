"""RoboDojo (Isaac Sim 6.2) session for SyncDojo: build one task env, reset to a layout, read the
scene, move the two ARX-X5 arms by TCP pose through CuRobo, and track the task's success check.

Every physics step goes through `Sim.step`, which records the joint targets in effect (the
control stream) and lets an attached recorder capture frames, so any rollout can be replayed.

Launch order matters (Isaac Sim must start before omni imports):
    app = rd_sim.launch(argparser_args)
    sim = rd_sim.Sim("stack_blocks", app)
"""
import os

import numpy as np

DT = 0.004            # physics step (env_cfg/sim/sim_config.yml), 250 Hz
TCP_BIAS = 0.145      # link6 origin -> grasp centre along link6 x (robot_config gripper_bias)
GRIP_STEPS = 60


def launch(a):
    from isaaclab.app import AppLauncher
    a.headless, a.enable_cameras = True, True
    os.environ.setdefault("ROBODOJO_RUN_ID", "syncdojo")
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    add_zero_delay_kit_args(a)
    # Isaac Sim 6.x defaults to RealTimePathTracing; RaytracedLighting (5.1's mode) denoises cleanly
    a.kit_args = ((getattr(a, "kit_args", None) or "") + " --/rtx/rendermode=RaytracedLighting").strip()
    return AppLauncher(a).app


# ---------------------------------------------------------------------------------------- math
def qmul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def tool_quat(yaw_deg=0.0, tilt_deg=0.0):
    """link6 orientation (wxyz) whose x-axis (approach) points down, tilted by `tilt_deg` from
    vertical, then rotated `yaw_deg` about world z. yaw 0: jaws close along world y."""
    pitch = np.deg2rad(90.0 - (tilt_deg or 0.0))
    yaw = np.deg2rad(yaw_deg or 0.0)
    return qmul([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])


def tcp_to_link6(xyz, q):
    return np.asarray(xyz, dtype=float) - TCP_BIAS * qmat(q)[:, 0]


def link6_to_tcp(pose7):
    return np.asarray(pose7[:3], dtype=float) + TCP_BIAS * qmat(pose7[3:7])[:, 0]


# ----------------------------------------------------------------------------------------- sim
class Sim:
    def __init__(self, task, app, robot_cfg="arx_x5.yml"):
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
        import src.eval_client.eval_env as ee
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_randomization, process_config

        class NoPolicy:  # stands in for the WebSocket policy client
            def __init__(self, *args, **kw):
                pass

            def call(self, *args, **kw):
                return None

        ee.WsModelClient = NoPolicy
        ecfg = load_yaml(os.path.join(ENV_CONFIG_PATH, robot_cfg))
        ecfg.update(task_name=task, num_envs=1, device_id=0, eval_batch=False, policy_name="none",
                    additional_info="syncdojo", seed=0, physx_monitor_enabled=False)
        c = ecfg["config"]
        cfg = OmegaConf.create({
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", c["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", c["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", c["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", c["robot"] + ".yml")),
            "task_env": load_yaml(os.path.join(ROOT_DIR, "task", BENCHMARK, "config", task + ".yml")),
            "eval_cfg": ecfg,
            "deploy_cfg": {"policy_name": "none", "port": 1, "host": "localhost", "protocol": "ws"},
        })
        OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=task)
        OmegaConf.update(cfg, "camera.default_frequency", ecfg["observation"].get("collect_freq", 0), force_add=True)
        cfg.sim.seed = [0]
        self.cfg, self.task = cfg, task
        self.env = ee.create_eval_env(cfg, app)
        if not hasattr(self.env.seed_manager, "seed_list"):
            self.env.seed_manager.init_eval()
        self.rm = self.env.robot_manager
        self.lm = self.env.scene_manager.layout_manager
        self.labels = [l for typ in ("Rigid", "Articulation", "Geometry") for grp in (cfg.task_env.get(typ, []) or [])
                       for l in (grp.get("select_mode", {}) or {}).get("label", [])]
        self.arms = {}  # "left"/"right" -> robot, ordered as in robot_list
        for i, r in enumerate(self.rm.robot_list):
            side = "left" if "left" in r.arm_name else "right" if "right" in r.arm_name else ("left", "right")[i]
            self.arms[side] = r
        self.recorder = None
        self.stream = []
        self.n_step = 0

    # ------------------------------------------------------------------ episode
    def reset(self, layout):
        self.env.reset(seed=[int(layout)])
        self.stream, self.n_step = [], 0
        self.recorder = None
        for _ in range(10):
            self.env.sim_step(render=False)
        self.env.run_reward()  # registers the task's success checks; consumed by reward_manager.step
        self.home = {s: self.link6(s) for s in self.arms}
        self.home_q = {s: np.asarray(self.rm.get_joint(r)[0], dtype=float) for s, r in self.arms.items()}
        try:
            mins = [o["bbox_world_min"][2] for o in self.object_info().values() if o.get("bbox_world_min")]
            self.surface_z = float(min(mins)) if mins else 0.765
        except Exception:
            self.surface_z = 0.765
        self.layout = int(layout)
        self.instruction = (self.env.gen_instruction(0) or [""])[0]

    def success(self):
        return bool(self.env.reward_manager.get_reward(final_check=True)[0] >= 1.0)

    # ------------------------------------------------------------------ state
    def link6(self, side):
        r = self.arms[side]
        return np.asarray(self.rm.get_link_pose(r, r.ee_link_name, is_relative=True)[0], dtype=float).reshape(-1)[:7]

    def tcp(self, side):
        p = self.link6(side)
        return np.r_[link6_to_tcp(p), p[3:7]]

    def gripper(self, side):  # normalised opening 0 (closed) .. 1 (open)
        r = self.arms[side]
        v = float(self.rm.get_end_effector_real_val(r)[0][0])
        s = r.gripper_scale
        return (v - s[0]) / (s[1] - s[0])

    def objects(self):
        out = {}
        for lab in self.labels:
            try:
                pos, q = self.lm.get_instance_pose(0, label=lab, relative=True)
            except AttributeError:  # articulations: layout manager's typed lookup misses them
                pos, q = self._pose_any_type(lab)
            if pos is None:
                continue
            arr = lambda x: np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=float).reshape(-1)
            out[lab] = np.r_[arr(pos)[:3], arr(q)[:4]]
        return out

    def _pose_any_type(self, lab):
        inst = self.lm.get_instance_name(0, lab)
        found = self.env.scene_manager.get_objects(env_ids=[0], object_name=inst)
        for obj in found.values():
            if hasattr(obj, "_get_object_transform"):
                pos, rot, _ = obj._get_object_transform()
                return pos, rot
            data = getattr(obj, "data", None)
            if data is not None and hasattr(data, "root_pos_w"):
                origin = self.env.scene_manager.env_origins[0].to(data.root_pos_w.device)
                return data.root_pos_w[0] - origin, data.root_quat_w[0]
        return None, None

    def object_info(self):
        info = {}
        for lab, pose in self.objects().items():
            inst = self.lm.get_instance_name(0, lab)
            size, lo, hi = None, None, None
            try:
                v = self.lm.get_instance_bbox_vertices(inst, 0)
                if v is None:  # articulations: read the box from the layout metadata directly
                    for typ, rec in self.lm.object_records_by_type.items():
                        md = rec.metadata_by_env[0].get(inst) if hasattr(rec, "metadata_by_env") else None
                        bb = ((md or {}).get("geometry") or {}).get("oriented_bbox") or {}
                        if "vertices" in bb:
                            v = bb["vertices"]; break
                if v is not None:
                    v = np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float).reshape(-1, 3)
                    if np.linalg.norm(v.mean(0)) < 0.05 or np.linalg.norm(v.mean(0)[:2] - pose[:2]) > 0.08:
                        v = v @ qmat(pose[3:7]).T + pose[:3]  # local-frame box -> world
                    size = (v.max(0) - v.min(0)).round(4).tolist()
                    lo, hi = v.min(0).round(4).tolist(), v.max(0).round(4).tolist()
            except Exception:
                pass
            try:
                desc = self.lm.get_label_descriptions(0, label=lab)
            except Exception:
                desc = []
            w, x, y, z = pose[3:7]
            yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
            info[lab] = {"xyz": pose[:3].round(4).tolist(), "quat_wxyz": pose[3:].round(4).tolist(),
                         "yaw_deg": round(yaw, 1), "bbox_size_xyz": size, "bbox_world_min": lo, "bbox_world_max": hi,
                         "description": desc[:2] if isinstance(desc, list) else desc}
        return info

    # ------------------------------------------------------------------ stepping
    def _targets(self):
        out = []
        for r in self.rm.robot_list:
            k = self.rm.robot_key[self.rm.robot_list.index(r)]
            d = k.data
            out.append(np.concatenate([d.joint_pos_target[0][r.arm_joint_indices].cpu().numpy(),
                                       d.joint_vel_target[0][r.arm_joint_indices].cpu().numpy(),
                                       d.joint_pos_target[0][r.gripper_joint_indices].cpu().numpy()]))
        return np.concatenate(out).astype(np.float32)

    def step(self):
        rec = self.recorder is not None and self.recorder.wants(self.n_step)
        self.env.sim_step(render=rec)
        self.stream.append(self._targets())
        if self.n_step % 5 == 0:
            self.env.reward_manager.step()
        if rec:
            self.recorder.capture(self, self.n_step)
        self.n_step += 1

    def run_controls(self, steps):
        if not steps:
            return
        self.rm.control_manager.push([0], [steps])
        for _ in range(len(steps)):
            self.rm.control_robot(self.rm.control_manager.pop([0]))
            self.step()

    def hold(self, n):
        """Keep current targets for n physics steps."""
        for _ in range(int(n)):
            self.step()

    def apply_targets(self, row):
        """Set arm + gripper joint targets from one control-stream row (replay)."""
        import torch
        i = 0
        for r in self.rm.robot_list:
            k = self.rm.robot_key[self.rm.robot_list.index(r)]
            na, ng = len(r.arm_joint_indices), len(r.gripper_joint_indices)
            dev = k.data.joint_pos.device
            q = torch.tensor(row[i:i + na], device=dev, dtype=torch.float32).unsqueeze(0); i += na
            dq = torch.tensor(row[i:i + na], device=dev, dtype=torch.float32).unsqueeze(0); i += na
            g = torch.tensor(row[i:i + ng], device=dev, dtype=torch.float32).unsqueeze(0); i += ng
            env_ids = torch.tensor([0], dtype=torch.int32, device=dev)
            k.set_joint_position_target(q, joint_ids=r.arm_joint_indices, env_ids=env_ids)
            k.set_joint_velocity_target(dq, joint_ids=r.arm_joint_indices, env_ids=env_ids)
            k.set_joint_position_target(g, joint_ids=r.gripper_joint_indices, env_ids=env_ids)

    def replay(self, stream):
        for row in stream:
            self.apply_targets(row)
            self.step()

    # ------------------------------------------------------------------ motions
    def plan_link6(self, side, pose7, vertical=False):
        r = self.arms[side]
        planner = self.rm.planner[r.robot_name]
        q0 = self.rm.get_joint(r)[0]
        if vertical:  # hold orientation and x/y, free z: a straight vertical line
            plan = planner.plan_path(q0, np.asarray(pose7, dtype=float), real_robot_pose=r.entity_origin_pose,
                                     constraint_pose=[1, 1, 1, 1, 1, 0])
            if plan.get("status") == "Success":
                return plan
        return planner.plan_path(q0, np.asarray(pose7, dtype=float), real_robot_pose=r.entity_origin_pose)

    def move_tcp(self, side, xyz, yaw_deg=0.0, tilt_deg=0.0, straight=False):
        q = tool_quat(yaw_deg, tilt_deg)
        target = np.r_[tcp_to_link6(xyz, q), q]
        cur = self.link6(side)
        vertical = bool(straight) and np.linalg.norm(target[:2] - cur[:2]) < 0.01
        cur_tcp_z = float(link6_to_tcp(cur)[2])
        low = max(cur_tcp_z, float(xyz[2])) < self.surface_z + 0.06
        if straight and not vertical and low:  # straight line near the table (pushing): 2.5 cm waypoints
            n = int(np.ceil(np.linalg.norm(target[:3] - cur[:3]) / 0.025))
            for k in range(1, n + 1):
                wp = np.r_[cur[:3] + (target[:3] - cur[:3]) * k / n, q]
                plan = self.plan_link6(side, wp)
                if plan.get("status") != "Success":
                    return False, f"straight line blocked at {k}/{n} of the way (unreachable or in collision)"
                self.run_controls(self.rm.plan_ee(0, self.arms[side].arm_name, result=plan))
            err = float(np.linalg.norm(self.link6(side)[:3] - target[:3]))
            return True, f"reached along a straight line (position error {err*100:.1f} cm)"
        plan = self.plan_link6(side, target, vertical=vertical)
        if plan.get("status") != "Success":
            return False, "motion planner found no collision-free path to that pose (unreachable or in collision)"
        self.run_controls(self.rm.plan_ee(0, self.arms[side].arm_name, result=plan))
        err = float(np.linalg.norm(self.link6(side)[:3] - target[:3]))
        return True, f"reached (position error {err*100:.1f} cm)"

    def move_home(self, side):
        """Back to the start configuration: joint-space plan, else pose plan, else lift 8 cm and retry,
        else a straight joint interpolation (the start pose is high and clear of the table)."""
        r = self.arms[side]
        planner = self.rm.planner[r.robot_name]

        def joint_plan():
            return planner.plan_joint(self.rm.get_joint(r)[0], self.home_q[side])
        for attempt in (joint_plan, lambda: self.plan_link6(side, self.home[side])):
            plan = attempt()
            if plan.get("status") == "Success":
                self.run_controls(self.rm.plan_ee(0, r.arm_name, result=plan))
                return True, "back at start pose"
        up = self.link6(side); up[2] += 0.08
        plan = self.plan_link6(side, up)
        if plan.get("status") == "Success":
            self.run_controls(self.rm.plan_ee(0, r.arm_name, result=plan))
            plan = joint_plan()
            if plan.get("status") == "Success":
                self.run_controls(self.rm.plan_ee(0, r.arm_name, result=plan))
                return True, "back at start pose (after lifting)"
        q0 = np.asarray(self.rm.get_joint(r)[0], dtype=float)
        n = 300
        steps = [{self.rm.process_name(r.arm_name): {"position": q0 + (self.home_q[side] - q0) * (i + 1) / n,
                                                     "velocity": np.zeros_like(q0)}} for i in range(n)]
        self.run_controls(steps)
        return True, "back at start pose (direct joint interpolation)"

    def grip(self, side, opening):
        self.run_controls(self.rm.plan_endeffector_joint(0, self.arms[side].arm_name, target_val=float(opening),
                                                         control_step_num=GRIP_STEPS))
        return True, f"gripper now {self.gripper(side):.2f} open"
