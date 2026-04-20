#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run45: Continue from run44 s4000 — same config, 10K more steps
# ═══════════════════════════════════════════════════════════════════
#
# Run44 achieved: pack 43→38%, ss 25→30%, gap 0.62→0.70,
# Rg compaction fixed (76→95%), RMSF reaching 0.46-0.59 Å.
# But 100K basin test showed slow drift still present.
#
# Run45 gives 10K more steps at same LR schedule to see if
# continued rebalancing deepens the basin enough for true
# long-trajectory stability.
#
# Only change: 5000 → 10000 steps, resume from run44 s4000
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

RUN_NAME="run45"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" logs

RESUME="checkpoints/run44/run1/full/step004000.pt"

STEPS=10000
LR=6e-4
LR_FINAL=6e-5
LR_SCHEDULE="cosine"
SCALAR_LR_MULT=1.0

SIGMA_MIN=0.05
SIGMA_MAX=2.0

LAMBDA_NATIVE_DEPTH=0.5
TARGET_NATIVE_DEPTH=-1.0

LAMBDA_GAP=1.0
GAP_T=0.5

LAMBDA_ZSCORE=0.0
TARGET_ZSCORE=3.0

LAMBDA_BALANCE=0.001

LAMBDA_FUNNEL=0.5
FUNNEL_T=0.5
ELT_EVERY=2
ELT_MAX_LEN=512
ELT_BATCH_SIZE=16

LAMBDA_DISCRIM=2.0
DISCRIM_EVERY=2

LAMBDA_FRUST=0.0
LAMBDA_BASIN=0.0
LAMBDA_PACK_C=0.0

LAMBDA_HB_BETA_FLOOR=0.1

VAL_EVERY=500
VAL_SAMPLES=8
LANGEVIN_BETA=50.0
LANGEVIN_STEPS=500

CKPT_EVERY=500

if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Checkpoint not found: ${RESUME}"
    exit 1
fi

echo "════════════════════════════════════════════════════════"
echo "  Run45: Continue from run44 s4000, 10K steps"
echo "  Resume: ${RESUME} (model only, fresh optimizer)"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase full \
    --resume "${RESUME}" --resume-model-only \
    --pdb "${PDB_LIST}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --lr-final "${LR_FINAL}" \
    --scalar-lr-mult "${SCALAR_LR_MULT}" \
    --sigma-min-rad "${SIGMA_MIN}" \
    --sigma-max-rad "${SIGMA_MAX}" \
    --elt-max-len "${ELT_MAX_LEN}" \
    --elt-batch-size "${ELT_BATCH_SIZE}" \
    --elt-every "${ELT_EVERY}" \
    --lambda-funnel "${LAMBDA_FUNNEL}" \
    --funnel-T "${FUNNEL_T}" \
    --lambda-zscore "${LAMBDA_ZSCORE}" \
    --target-zscore "${TARGET_ZSCORE}" \
    --lambda-gap "${LAMBDA_GAP}" \
    --gap-T "${GAP_T}" \
    --lambda-frustration "${LAMBDA_FRUST}" \
    --lambda-native-depth "${LAMBDA_NATIVE_DEPTH}" \
    --target-native-depth "${TARGET_NATIVE_DEPTH}" \
    --lambda-balance "${LAMBDA_BALANCE}" \
    --lambda-discrim "${LAMBDA_DISCRIM}" \
    --discrim-every "${DISCRIM_EVERY}" \
    --discrim-mode max \
    --lambda-basin "${LAMBDA_BASIN}" \
    --lambda-pack-contrastive "${LAMBDA_PACK_C}" \
    --lambda-hb-beta-floor "${LAMBDA_HB_BETA_FLOOR}" \
    --langevin-beta "${LANGEVIN_BETA}" \
    --val-langevin-steps "${LANGEVIN_STEPS}" \
    --val-max-samples "${VAL_SAMPLES}" \
    --validate-every "${VAL_EVERY}" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}" \
    2>&1 | tee logs/run45.log

echo ""
echo "Run45 complete."
