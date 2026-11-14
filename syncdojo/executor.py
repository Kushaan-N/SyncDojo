"""Run an Opus motion script (opus_planner.MotionScript) in a rd_sim.Sim and describe what
happened, in words Opus can act on when revising."""
import numpy as np

from .rd_sim import DT

TABLE_MARGIN = 0.005

SUCCESS_TEXT = {
    "stack_blocks": ("all three blocks form ONE stack: each block resting on the one below, centres within "
                     "about 1.5-2 cm in x and y; afterwards both arms back near their start poses (within 15 cm "
                     "and 20 deg) with grippers open. The stacking order does not matter."),
    "stack_bowls": ("all three bowls upright (rim up, within 45 deg; the bottom bowl within 7 deg), nested in ONE "
                    "stack with centres within 4 cm in x and y; afterwards both arms back near their start poses "
                    "(within 15 cm and 20 deg) with grippers open."),
    "swap_blocks": ("the two blocks (target0, target1) swap mats using the empty third mat, with the button "
                    "(button0) pressed once after EACH of the three moves: (1) move one block to the empty mat, "
                    "press; (2) move the other block to the mat the first one left, press; (3) move the first block "
                    "to the mat the second one left, press. Exactly three presses in total; blocks end within 3 cm of "
                    "their mats. A press means pushing the button top down past half its travel and letting it "
                    "spring back up."),
    "align_blocks": ("the three cubes (cube0, cube1, cube2) end in ONE straight row: every cube centre within 8 mm "
                     "of a common line in x/y, and each cube turned so one of its faces is parallel to that line "
                     "(within 8 deg). The cubes must be pushed - preferably with the set-square tool (label "
                     "'target', a triangular prism you may grasp and slide), or with the closed gripper - and never "
                     "lifted more than 7 mm. Afterwards both arms back near their start poses (within 15 cm and 20 deg)."),
    "sort_nesting_dolls_by_size": ("the five nesting dolls (doll0..doll4) stand upright in ONE row along the world x axis, "
                                   "ordered by size with the SMALLEST at the lowest x and the largest at the highest x (use "
                                   "each doll's bounding box to tell its size): every doll's centre at least 3 cm further in "
                                   "+x than the next smaller doll's, and all five centres within 3.5 cm of each other in y. "
                                   "Dolls may be picked up and placed. Afterwards both arms back near their start poses "
                                   "(within 15 cm and 20 deg)."),
    "classify_objects": ("every object is in the basket of its category: objects whose label starts with cat0 inside "
                         "basket0, cat1 inside basket1, cat2 inside basket2, each resting inside the basket (its centre "
                         "no more than 1 cm above the basket's top rim, i.e. dropped in, not balanced on the edge), and no "
                         "object in a wrong basket. Afterwards both arms back near their start poses (within 15 cm and "
                         "20 deg)."),
    "push_T": ("the T-shaped block (t) is PUSHED with the closed gripper's fingertips or side - grasping, pinching or "
               "carrying the T is not allowed (never close the gripper around it) and it is never lifted more than 1 cm - until it matches the gray T-shaped "
               "pad (target_t): centres within 7 mm in x/y and orientation within 7 deg; afterwards both arms back "
               "near their start poses (within 15 cm and 20 deg)."),
}


def scene_facts(sim):
    objs = sim.object_info()
    tops = [o["bbox_world_min"][2] if o.get("bbox_world_min") else o["xyz"][2] for o in objs.values()]
    table = {"surface_z": round(float(min(tops)), 4) if tops else None}
    arms = {}
    for s, r in sim.arms.items():
        base = np.asarray(r.entity_origin_pose, dtype=float)[:3]
        tcp = sim.tcp(s)
        arms[s] = {"base_xyz": base.round(3).tolist(), "tcp_xyz": tcp[:3].round(3).tolist(),
                   "comfortable_reach_m": 0.55, "gripper_open": round(sim.gripper(s), 2)}
    return objs, arms, table


def object_summary(sim):
    lines = []
    for lab, o in sim.object_info().items():
        lines.append(f"- {lab}: xyz {o['xyz']}, quat_wxyz {o['quat_wxyz']}")
    return "\n".join(lines)


def run_script(sim, script, settle_steps=60):
    """Execute every step; stop at the first step that cannot be planned. Then the arms go home.
    Returns (log lines, failed_step_index or None)."""
    log, failed = [], None
    surface = scene_facts(sim)[2]["surface_z"] or 0.0
    for i, st in enumerate(script.steps):
        side = st.arm
        if st.op == "move":
            if not st.xyz or len(st.xyz) != 3:
                log.append(f"step {i} ({st.note}): invalid move, xyz missing"); failed = i; break
            xyz = list(map(float, st.xyz))
            if xyz[2] < surface + TABLE_MARGIN:
                log.append(f"step {i} ({st.note}): TCP z {xyz[2]:.3f} is below the table surface {surface:.3f}; refused")
                failed = i; break
            ok, msg = sim.move_tcp(side, xyz, st.yaw_deg or 0.0, st.tilt_deg or 0.0, bool(st.straight))
        elif st.op == "open":
            ok, msg = sim.grip(side, 1.0)
        elif st.op == "close":
            ok, msg = sim.grip(side, 0.0)
        elif st.op == "wait":
            sim.hold(max(1, int((st.seconds or 0.5) / DT))); ok, msg = True, "waited"
        else:
            ok, msg = False, f"unknown op {st.op}"
        log.append(f"step {i} [{side} {st.op}] ({st.note}): {'OK' if ok else 'FAILED'} - {msg}")
        if not ok:
            failed = i
            break
    for s in ("left", "right"):
        if sim.gripper(s) < 0.8:
            sim.grip(s, 1.0)
    for s in ("left", "right"):
        ok, msg = sim.move_home(s)
        log.append(f"auto: {s} arm home: {'OK' if ok else 'FAILED'} - {msg}")
    sim.hold(settle_steps)
    return log, failed


def feedback_text(sim, log, failed, success):
    head = "The simulator ran your script." if failed is None else f"Execution stopped at step {failed}."
    return (f"{head}\nExecution log:\n" + "\n".join(log) +
            f"\n\nFinal object poses:\n{object_summary(sim)}\n\nSuccess check: {'PASSED' if success else 'FAILED'}.\n"
            "Figure out what went wrong (reachability, grasp height, yaw, release height, ordering, collisions "
            "between arms) and write a corrected full motion script. Note: the scene was reset to the SAME start "
            "state shown in the first message, not the final poses above.")
