"""Frame + state capture in SyncSim's episode layout.

One rollout dir:
    agentview_rgb/video.mp4        RoboDojo head camera (between the arms, looking at the table)
    wrist_left_rgb/video.mp4       left wrist camera
    wrist_right_rgb/video.mp4      right wrist camera
    pose.pkl                       per frame, both arms: TCP pose (xyzw), 4x4 matrix, gripper open
    actions.npy                    [T, 16] per arm TCP xyz + quat xyzw + gripper opening (SyncSim layout)
    joints.npy                     [T, 16] measured arm joints (6) + gripper joints (2), left then right
    object_poses.npz               per frame pose of every task object (xyz + quat wxyz) + names
    stream.npz                     the control stream: joint targets at EVERY physics step (replayable)
    metadata.json                  written by the caller
"""
import json
import os
import pickle

import numpy as np

from .rd_sim import qmat

FRAME_STRIDE = 12          # 250 Hz physics / 12 = 20.8 fps, as in SyncSim
CAMS = {"cam_head": "agentview_rgb", "cam_left_wrist": "wrist_left_rgb", "cam_right_wrist": "wrist_right_rgb"}


def xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def write_video(frames, path, fps, crf=18):
    import imageio_ffmpeg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    gen = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                      output_params=["-crf", str(crf), "-preset", "fast"], macro_block_size=1)
    gen.send(None)
    for f in frames:
        gen.send(np.ascontiguousarray(np.asarray(f, dtype=np.uint8)[..., :3]))
    gen.close()


class Recorder:
    def __init__(self, stride=FRAME_STRIDE, videos=True):
        self.stride, self.videos = stride, videos
        self.frames = {d: [] for d in CAMS.values()}
        self.log = {k: [] for k in ("left", "right", "left_g", "right_g", "left_cmd", "right_cmd", "joints", "objects", "step")}
        self.names = None

    def wants(self, n):
        return n % self.stride == 0

    def capture(self, sim, n):
        if self.videos:
            vis = sim.env.get_obs().get("vision", {})
            for cam, d in CAMS.items():
                if cam in vis:
                    self.frames[d].append(np.asarray(vis[cam]["color"]).astype(np.uint8))
        L = self.log
        for s in ("left", "right"):
            L[s].append(sim.tcp(s))
            L[f"{s}_g"].append(sim.gripper(s))
            r = sim.arms[s]
            k = sim.rm.robot_key[sim.rm.robot_list.index(r)]
            tgt = float(k.data.joint_pos_target[0][r.gripper_joint_indices[0]].cpu())
            sc = r.gripper_scale
            L[f"{s}_cmd"].append((tgt - sc[0]) / (sc[1] - sc[0]))
        J = []
        for s in ("left", "right"):
            r = sim.arms[s]
            k = sim.rm.robot_key[sim.rm.robot_list.index(r)]
            jp = k.data.joint_pos[0]
            J.append(np.concatenate([jp[r.arm_joint_indices].cpu().numpy(), jp[r.gripper_joint_indices].cpu().numpy()]))
        L["joints"].append(np.concatenate(J))
        objs = sim.objects()
        self.names = list(objs)
        L["objects"].append(np.stack([objs[k] for k in self.names]))
        L["step"].append(n)

    def actions(self):
        L = self.log
        lp, rp = np.asarray(L["left"]), np.asarray(L["right"])
        return np.concatenate([lp[:, :3], xyzw(lp[:, 3:7]), np.asarray(L["left_cmd"])[:, None],
                               rp[:, :3], xyzw(rp[:, 3:7]), np.asarray(L["right_cmd"])[:, None]], axis=1).astype(np.float32)

    def save(self, out_dir, stream, meta, fps=250.0 / FRAME_STRIDE):
        os.makedirs(out_dir, exist_ok=True)
        if self.videos:
            for d, fr in self.frames.items():
                if fr:
                    write_video(fr, os.path.join(out_dir, d, "video.mp4"), fps)
        L = self.log
        lp, rp = np.asarray(L["left"]), np.asarray(L["right"])

        def mats(p):
            m = np.tile(np.eye(4), (len(p), 1, 1))
            for i, x in enumerate(p):
                m[i, :3, :3] = qmat(x[3:7]); m[i, :3, 3] = x[:3]
            return m
        pose = {
            "left_gripper_pose": np.concatenate([lp[:, :3], xyzw(lp[:, 3:7])], axis=1),
            "right_gripper_pose": np.concatenate([rp[:, :3], xyzw(rp[:, 3:7])], axis=1),
            "left_gripper_matrix": mats(lp), "right_gripper_matrix": mats(rp),
            "left_gripper_open": np.asarray(L["left_cmd"]), "right_gripper_open": np.asarray(L["right_cmd"]),
            "left_gripper_open_measured": np.asarray(L["left_g"]), "right_gripper_open_measured": np.asarray(L["right_g"]),
            "physics_step": np.asarray(L["step"], dtype=np.int64), "save_stride": int(self.stride), "fps": float(fps),
            "quat_order": "xyzw",
        }
        with open(os.path.join(out_dir, "pose.pkl"), "wb") as f:
            pickle.dump(pose, f)
        np.save(os.path.join(out_dir, "actions.npy"), self.actions())
        np.save(os.path.join(out_dir, "joints.npy"), np.asarray(L["joints"], dtype=np.float32))
        np.savez_compressed(os.path.join(out_dir, "object_poses.npz"), poses=np.asarray(L["objects"], dtype=np.float32),
                            names=np.array(self.names or []), quat_order="wxyz")
        np.savez_compressed(os.path.join(out_dir, "stream.npz"), targets=np.asarray(stream, dtype=np.float32),
                            layout="per arm (left, right): arm q target[6], arm dq target[6], gripper q target[2]",
                            dt=0.004)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        return len(L["step"])
