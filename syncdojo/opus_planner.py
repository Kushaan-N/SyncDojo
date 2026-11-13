"""Opus as the policy: it reads a text description of the scene and writes a motion script.

The script is a short list of end-effector steps for the two arms (move to a pose, open or
close a gripper, wait). The simulator runs it through the motion planner; if the run fails,
`revise` sends Opus what happened (which step failed and why, final object poses, the
success check) and asks for a corrected script. No demonstrations are ever shown.

The API key comes from the environment (ANTHROPIC_API_KEY); it is never logged.
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"


class Step(BaseModel):
    arm: Literal["left", "right"]
    op: Literal["move", "open", "close", "wait"]
    # move: target TCP position in the world frame (m) and the gripper's yaw about world z (deg).
    # The gripper always points straight down (top-down grasp) unless tilt_deg is set.
    xyz: Optional[List[float]] = Field(default=None, description="move only: [x, y, z] in metres, world frame")
    yaw_deg: Optional[float] = Field(default=None, description="move only: gripper yaw about world z")
    tilt_deg: Optional[float] = Field(default=None, description="move only: tilt away from straight down, default 0")
    straight: Optional[bool] = Field(default=None, description="move only: straight-line Cartesian path (for approach/lift)")
    seconds: Optional[float] = Field(default=None, description="wait only: duration")
    note: str = Field(description="one short phrase: what this step does")


class MotionScript(BaseModel):
    reasoning: str = Field(description="2-5 sentences: the plan, which arm does what and why")
    steps: List[Step]


SYSTEM = """You control a bimanual robot in a physics simulator by writing motion scripts.
You never see demonstrations; plan from the scene description alone.

Robot: two ARX-X5 arms with parallel-jaw grippers. Each step moves ONE arm's tool centre point
(TCP, the point between the fingertips) or opens/closes its gripper. Steps run in order; a move
is planned collision-free by a motion planner unless `straight` is set, which moves in a straight
line (use it for the final approach and for lifting). Gripper yaw: at yaw_deg 0 the jaws close
along the world y axis; at yaw_deg 90 they close along world x. The jaws open to about 10 cm. Keep each arm on its own side of the table
unless the task needs it. Approach from above: hover ~8-10 cm above a grasp point, descend with
`straight`, close, then lift with `straight`. Grasp height is about the object's centre height;
never put the TCP below the table surface. Gripper yaw should align the jaws across the object's
narrow side. Release by moving to a hover pose above the target, descending, opening, retreating
upward. Make sure every gripper is open at the end. After your last step the executor automatically
moves both arms back to their start poses (the success check requires it), so do not add that yourself.

For pushing, close the gripper (fingers together make a blunt pusher), lower the TCP to about the
pushed object's mid height beside it, and move with `straight` through the object; plan pushes in
several short strokes and re-approach from another side to rotate an object. To press a button, put
the closed gripper above its top, move down with `straight` until the TCP is about 1 cm below the
top, then move back up with `straight`.

Output only the motion script in the required JSON format."""


def _client():
    # explicit timeout: long (32k-token) non-streaming calls are allowed; a call can take several minutes
    return anthropic.Anthropic(timeout=1800.0, max_retries=4)


def plan(scene_text: str, effort: str = "high") -> tuple[MotionScript, dict]:
    """First attempt. Returns the script and the raw conversation (for revisions/logging)."""
    msgs = [{"role": "user", "content": scene_text}]
    r = _client().messages.parse(model=MODEL, max_tokens=32000, system=SYSTEM, messages=msgs,
                                 output_config={"effort": effort}, output_format=MotionScript)
    _check(r)
    msgs.append({"role": "assistant", "content": r.content})
    return r.parsed_output, {"messages": msgs, "usage": [r.usage.model_dump()]}


def revise(convo: dict, feedback: str, effort: str = "high") -> tuple[MotionScript, dict]:
    """Next attempt after a failed run; `feedback` describes what happened in the simulator."""
    msgs = convo["messages"] + [{"role": "user", "content": feedback}]
    r = _client().messages.parse(model=MODEL, max_tokens=32000, system=SYSTEM, messages=msgs,
                                 output_config={"effort": effort}, output_format=MotionScript)
    _check(r)
    msgs.append({"role": "assistant", "content": r.content})
    return r.parsed_output, {"messages": msgs, "usage": convo["usage"] + [r.usage.model_dump()]}


def _check(r):
    if r.stop_reason == "refusal":
        raise RuntimeError(f"Opus refused: {r.stop_details}")
    if r.stop_reason == "max_tokens":
        raise RuntimeError("Opus hit max_tokens before finishing the script")


def describe_scene(task_text: str, objects: dict, robot: dict, table: dict, success: str) -> str:
    """Scene -> prompt. objects: name -> {xyz, quat_xyzw, size_xyz}; robot: arm -> {base_xyz, tcp_xyz,
    reach_m}; table: {height, x_range, y_range}."""
    return (f"Task: {task_text}\n\nSuccess is checked as: {success}\n\n"
            f"Frame: the scene's world frame, z up, metres. The arms' base positions below show where\n"
            f"the robot stands; it faces the table across the objects.\n"
            f"Table: {json.dumps(table)}\n"
            f"Arms: {json.dumps(robot)}\n"
            f"Objects (position, orientation as quat_wxyz and yaw about z, world-frame bounding box):\n"
            + "\n".join(f"- {n}: {json.dumps(o)}" for n, o in objects.items())
            + "\n\nWrite the motion script.")
