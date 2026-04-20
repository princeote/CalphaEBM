# Phased Training Guide for CalphaEBM

This document describes the recommended phased training strategy for optimal model convergence.

## Overview

The training proceeds in 6 phases, with repulsion split into two passes for optimal results:

1. **Local** - Learn bond lengths and angles
2. **Repulsion (Safety)** - Light calibration to prevent catastrophic clashes
3. **Secondary** - Learn Ramachandran preferences
4. **Repulsion (Final)** - Fine-tune repulsion on realistic geometries
5. **Packing** - Learn residue-residue contact preferences
6. **Full fine-tuning** - All terms trainable with gate ramping

## Why Two Repulsion Phases?

- **Safety pass** (early): Just enough repulsion to prevent steric catastrophes during secondary learning
- **Final calibration** (after secondary): λ_rep set based on realistic conformations, not just local-geometry structures

Monitor these signals during secondary:
- Generation success rate should stay >50%
- Ramachandran correlation should improve steadily
- If either stalls, repulsion may be too strong

## Commands

```bash
# Phase 1: Local term (5-10k steps)
python -m calphaebm.cli.main train \
    --phase local \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --steps 10000 \
    --lr 3e-4

# Phase 2a: Repulsion "Safety Pass" (short, conservative)
# Goal: eliminate catastrophic clashes, not perfect calibration
python -m calphaebm.cli.main train \
    --phase repulsion \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --resume auto \
    --steps 2000 \
    --lr 1e-4

# Phase 2b: Secondary structure (10-15k steps)
python -m calphaebm.cli.main train \
    --phase secondary \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --resume auto \
    --steps 15000 \
    --lr 3e-4

# Phase 2c: Repulsion "Final Calibration" (after secondary)
python -m calphaebm.cli.main train \
    --phase repulsion \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --resume auto \
    --steps 3000 \
    --lr 1e-4

# Phase 3: Packing (10-15k steps)
python -m calphaebm.cli.main train \
    --phase packing \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --resume auto \
    --steps 15000 \
    --lr 3e-4

# Phase 4: Full fine-tuning (20-30k steps) with gate ramping
python -m calphaebm.cli.main train \
    --phase full \
    --pdb train_entities.no_test_entries.txt \
    --ckpt-prefix run1 \
    --resume auto \
    --steps 25000 \
    --lr 1e-4 \
    --ramp-gates \
    --ramp-steps 5000 \
    --ramp-rep-start 1.0 --ramp-rep-end 20.0 \
    --ramp-ss-start 1.0 --ramp-ss-end 8.0 \
    --ramp-pack-start 0.5 --ramp-pack-end 4.0
```

## Notes

- Adjust `--steps` based on convergence monitoring
- The `--resume auto` flag automatically finds the latest checkpoint
- Monitor validation metrics to decide when to move to next phase
