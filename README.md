# SyncDojo

SyncDojo builds **evaluation sets for bimanual robot world models**: episodes of a two-armed robot
(two ARX-X5 arms in the RoboDojo simulator, on Isaac Sim) doing a task, each labelled
success or failure by the simulator, balanced exactly 50/50 per task. A world model is tested on whether
its predictions match what really happened.

No expert demonstrations are used. Claude Opus writes the robot's behaviour, and failures come from
perturbing it. Episodes use the same layout as [SyncSim](https://github.com/Kushaan-N/SyncSim), the
companion training-data generator, so the same loaders work.

## How it works

1. **Opus writes one program per task.** Given the task, its success condition and an example scene as
   text, Opus writes `run(api)`: Python that reads the live object poses and calls skills (`pick`,
   `place`, `move`, ...; `syncdojo/skills.py`). It is tested on a few layouts and revised from the
   execution logs until it passes.
2. **Simulator workers run it.** Persistent simulators (`scripts/worker.py`) serve a shared file queue, so
   any number of workers on any number of GPUs run rollouts in parallel. The program is recorded on every
   layout.
3. **Perturbations make the failures.** Each solved layout is re-run with errors injected inside the
   skills: target jitter, a blind offset or mis-rotated grasp, an early or high release, a collision
   sweep by the free arm, and noisy replays of the recorded joint stream.
4. **Labels come from the simulator.** A rollout is a success only if RoboDojo's success check holds at
   the end of the episode, whatever perturbation was applied.
5. **A balanced set is selected** (`scripts/select_balanced.py`): exactly 50/50 per task, spread over
   perturbation kinds and layouts, with duplicates and no-motion clips left out.

Steps 1-2 are `scripts/program_loop.py`. After the program is written (a few Opus calls, under $1 per
task), more layouts and variants cost only GPU time.

## What an eval set contains

```
<task>/
  manifest.jsonl               episode, success, kind (perturbation), layout, calibration clip
  cameras.json                 intrinsics, head-camera pose, wrist-camera mounts
  calibration/layout_<L>/      per layout: each arm moves each of its 6 DoFs out and back
  episode_<k>/
    agentview_rgb/video.mp4    head camera     (all videos 20.8 fps)
    wrist_left_rgb/video.mp4
    wrist_right_rgb/video.mp4
    pose.pkl                   both arms' TCP poses (xyzw) and gripper opening
    actions.npy                [T, 16] per arm: TCP xyz, quat xyzw, gripper
    joints.npy                 [T, 16] per arm: 6 arm joints, 2 gripper joints
    object_poses.npz           every task object's pose per frame (wxyz)
    stream.npz                 joint targets at every 250 Hz physics step (exact replay)
    metadata.json              task, layout, program, perturbation, execution log, outcome
```

## Running

Needs a RoboDojo checkout with its CuRobo setup, the `anthropic` package and `ANTHROPIC_API_KEY`. On Isaac
Sim 6.2, apply `patches/robodojo_isaacsim62.patch` in the checkout (RoboDojo commit f57e163).

```
# 1. start workers (inside a GPU allocation; several per GPU is fine)
python scripts/worker.py --task stack_blocks --queue Q --headless --enable_cameras

# 2. develop the program on layouts 0-2, then record layouts 0-11 and their perturbed variants
python scripts/program_loop.py --task stack_blocks --queue Q --out runs --dev 0 1 2 --layouts $(seq 0 11)

# 3. calibration clips and camera parameters
python scripts/calibrate.py    --task stack_blocks --layouts $(seq 0 11) --out runs --headless --enable_cameras
python scripts/dump_cameras.py --task stack_blocks --out eval --headless --enable_cameras

# 4. the 50/50 set
python scripts/select_balanced.py --runs runs --task stack_blocks --per-class 50 --out eval
```

`program_loop.py` needs no GPU; it talks to the workers only through the queue directory. Tasks with a
success description in `syncdojo/executor.py`: `stack_blocks`, `stack_bowls`, `align_blocks`, `push_T`.

An earlier mode, one Opus motion script per scene (`scripts/collect.py`, then `scripts/expand.py` for the
perturbations), is kept for reference.
