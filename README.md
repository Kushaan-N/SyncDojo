# SyncDojo

Builds world-model evaluation sets in RoboDojo (bimanual ARX-X5, Isaac Sim) without
expert demonstrations. Claude Opus writes a motion script for each scene; the simulator runs it, RoboDojo's
own success check labels the outcome, and the successful scripts are perturbed into near-miss failures.
Episodes are written in SyncSim's episode layout.

## Pipeline

1. **Collect** (`scripts/collect.py`): for each layout, Opus gets a text description of the scene (objects with
   pose and world bounding box, arm bases and tool positions, table height, the success condition) and returns
   a structured motion script (`move` / `open` / `close` / `wait` per arm). The executor plans each move with
   CuRobo, refuses moves through the table, then opens both grippers and homes both arms. On failure Opus gets
   the execution log and final object poses and revises (up to `--attempts`). Every attempt is saved.
2. **Expand** (`scripts/expand.py`): replays successful (and optionally near-miss) scripts with perturbations:
   `rerun`, `jitter`, `jitter_large`, `noisy_replay`, `grasp_offset`, `yaw_error`, `high_release`,
   `early_open`, `sweep`. Each variant's label comes from the simulator, not from the perturbation type.
3. **Select** (`scripts/select_balanced.py`): builds an exactly 50/50 success/failure set per task,
   round-robin over perturbation kind and layout, as symlinks plus a `manifest.jsonl`.

## Episode layout

```
episode_k/
  agentview_rgb/video.mp4      head camera
  wrist_left_rgb/video.mp4
  wrist_right_rgb/video.mp4
  pose.pkl                     TCP poses (xyzw), gripper measured/commanded
  actions.npy                  [T, 16] per-frame commands
  joints.npy                   [T, 16]
  object_poses.npz             per-object pose over time (wxyz)
  stream.npz                   full-rate joint/gripper control stream (replayable)
  metadata.json                task, layout, instruction, success, script, source, exec log
```

Physics runs at 250 Hz; frames are recorded every 12 steps (20.8 fps).

## Running

Requires a RoboDojo checkout ported to Isaac Sim 6.2 with its CuRobo setup, the `anthropic` Python package,
and `ANTHROPIC_API_KEY` in the environment. `scripts/rd_job.sh` runs collect then expand for one task on one
GPU (see the variables at its top). Smoke tests: `scripts/rd_task_test.py`, `scripts/rd_grasp_test.py`.

```
python scripts/collect.py --task stack_blocks --layouts 0 1 2 3 --stop-after 3 --out runs --headless --enable_cameras
python scripts/expand.py  --task stack_blocks --out runs --per-kind 3 --headless --enable_cameras
python scripts/select_balanced.py --runs runs --task stack_blocks --per-class 30 --out eval
```

Tasks with success text in `syncdojo/executor.py`: `stack_blocks`, `stack_bowls`, `align_blocks`, `push_T`
(`swap_blocks` is defined but its button articulation does not load in the 6.2 port).
