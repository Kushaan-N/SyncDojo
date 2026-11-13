#!/bin/bash
# One GPU job: [grasp test] -> Opus collection -> expansion.
# usage: rd_job.sh TASK LAYOUT_FROM LAYOUT_TO STOP_AFTER [grasp]
# Environment (set before running):
#   RD_BUILD   dir holding RoboDojo/ (Isaac Sim 6.2 port), venv/ and home/
#   RD_SIF     Isaac Sim 6.2 apptainer image
#   OUT        output root for rollouts (default ./opus_runs)
#   RD_BINDS   extra apptainer -B binds (e.g. your data dirs)
#   ANTHROPIC_API_KEY  must be set in the environment
TASK=$1; L0=$2; L1=$3; STOP=$4; GRASP=${5:-}
D=${RD_BUILD:?set RD_BUILD}; RD=$D/RoboDojo; SD=$(cd "$(dirname "$0")/.." && pwd)
SIF=${RD_SIF:?set RD_SIF}; OUT=${OUT:-$PWD/opus_runs}; mkdir -p "$OUT"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
# The NGX bind works around a missing DLSS library on some drivers; drop it if not needed.
run() { apptainer exec --nv -B /usr/lib64/libnvidia-ngx.so.1:/.singularity.d/libs/libnvidia-ngx.so.1 --writable-tmpfs \
  --home $D/home -B "$D" -B "$SD" -B "$OUT" ${RD_BINDS:-} \
  "$SIF" bash -c "cd $RD && export PYTHONPATH=$RD:$RD/XPolicyLab:$SD && $*"; }
if [ -n "$GRASP" ]; then
  echo "=== grasp test"; run timeout 1200 $D/venv/bin/python $SD/scripts/rd_grasp_test.py --task $TASK --layout 0 --out $OUT/grasp_out_$TASK
fi
echo "=== collect $TASK layouts $L0..$L1"
run timeout 6600 $D/venv/bin/python $SD/scripts/collect.py --task $TASK --layouts $(seq -s ' ' $L0 $L1) --stop-after $STOP --out $OUT
echo "=== collect exit $?"
echo "=== expand $TASK"
run timeout 5400 $D/venv/bin/python $SD/scripts/expand.py --task $TASK --out $OUT --per-kind 3 --max-minutes 85
echo "=== expand exit $?"
