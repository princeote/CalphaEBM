# CalphaEBM: Cα Energy-Based Model for Protein Structure

A physics-based, machine-learned energy function for protein Cα coordinates that achieves native basin stability across diverse protein folds.

## Model

CalphaEBM decomposes the effective free energy into four interpretable terms:

| Term | Parameters | What it captures |
|------|-----------|-----------------|
| **LocalEnergy** | 12,226 | Backbone geometry: 8-residue sliding window MLP over (θ, φ) angles with learned amino acid embeddings |
| **SecondaryEnergy** | 583 | Ramachandran basin potentials (4 basins: helix, sheet, PPII, turn) with sequence-dependent mixture weights + helical and sheet hydrogen bond distances |
| **PackingEnergy** | 222 | Tertiary packing: 5-group coordination statistics with product Gaussian scoring |
| **RepulsionEnergy** | 1 | Excluded volume: PDB-derived repulsive wall with differentiable interpolation |

Total: 13,032 trainable parameters.

All energy terms produce smooth, differentiable forces suitable for Langevin dynamics (MALA) sampling.

## Results

Trained on 2,280 high-quality monomeric protein chains (L=40–512) for 10 rounds of 5,000 steps each, using denoising score matching + contrastive discrimination + energy funnel losses.

### Native basin stability (16 validation proteins × 50K MALA steps × β=100)

| Round | Q | RMSD (Å) | Rg (%) | Acceptance |
|-------|------|----------|--------|------------|
| Best | 0.988 | 2.49 | 100.1 | 70.0% |
| Final (R10) | 0.969 | 2.90 | 98.4 | 71.4% |
| Range (R1–R10) | 0.969–0.988 | 2.49–2.91 | 98.2–100.1 | 66.7–74.4% |

The model maintains native contacts (Q > 0.96) and native compactness (Rg within 2% of crystal) across all 10 training rounds and all 16 validation proteins spanning diverse lengths and fold classes.

### Single-protein tests

- **1YRF** (Villin HP35, L=35, all-α): Q=1.000 for 1M MALA steps, RMSD=2.7 Å

## Training

```bash
# Full training (10 rounds, SLURM)
sbatch scripts/training/run9_full.sh

# Single-protein model test
python scripts/model_test.py \
    --model checkpoints/run9/run9/full-stage/full_round010/step005000.pt \
    --pdb-id 1yrf --n-steps 100000 --beta 100

# TREMD folding test
python scripts/simu/tremd_test.py \
    --model checkpoints/run9/run9/full-stage/full_round008/step005000.pt \
    --pdb-id 1yrf --start-mode extended --minimize \
    --n-replicas 8 --beta 100 --beta-min 5.0 \
    --n-swaps 5000 --steps-per-swap 200
```

## Installation

```bash
git clone https://github.com/aai-research-lab/CalphaEBM.git
cd CalphaEBM
pip install -e .
```

Requires Python 3.9+ and PyTorch.

## Project structure

```
src/calphaebm/
  models/          Energy terms (local, packing, secondary, repulsion)
  training/        DSM + contrastive training loop
  simulation/      MALA, HREMD, TREMD simulators
  analysis/        PDB statistical analysis pipelines
  geometry/        Internal coordinates, pair distances
  evaluation/      Q, RMSD, dRMSD metrics

analysis/          PDB-derived data (basin surfaces, repulsion wall, coordination stats)
checkpoints/       Trained model weights (Run9, 10 rounds)
scripts/           Training, evaluation, and simulation scripts
```

## License

MIT
