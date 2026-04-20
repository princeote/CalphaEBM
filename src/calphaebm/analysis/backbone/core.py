# src/calphaebm/analysis/backbone/core.py

"""Core backbone geometry analysis functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from calphaebm.data.id_utils import normalize_to_entry_ids
from calphaebm.utils.logging import get_logger

from .config import DEFAULT_OUTPUT_DIR
from .data_loader import extract_geometry_from_chains, load_pdb_list
from .plots import (
    plot_bond_length_distribution,
    plot_delta_phi_potential,
    plot_figure_2,
    plot_figure_3,
    plot_phi_phi_correlation,
)

logger = get_logger()


class BackboneAnalyzer:
    """Main class for backbone geometry analysis."""

    def __init__(self, cache_dir: Path, output_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bond_lengths: Optional[np.ndarray] = None
        self.theta_i: Optional[np.ndarray] = None
        self.theta_ip1: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None

    def load_pdb_list(self, pdb_list_path: Path) -> list[str]:
        """
        Load IDs from file. Accepts either entry IDs (1ABC) or entity IDs (1ABC_1).

        We normalize to entry IDs because downstream CIFs are per-entry.
        """
        raw = load_pdb_list(pdb_list_path)
        pdb_ids = normalize_to_entry_ids(raw)
        return pdb_ids

    def extract_geometry(self, pdb_ids: list[str], max_chains: Optional[int] = None) -> None:
        logger.info("Extracting Cα geometry from PDB structures...")

        self.bond_lengths, self.theta_i, self.theta_ip1, self.phi = extract_geometry_from_chains(
            pdb_ids=pdb_ids,
            cache_dir=self.cache_dir,
            max_chains=max_chains,
        )

        logger.info(f"Collected {len(self.bond_lengths)} bond lengths")
        logger.info(f"Collected {len(self.theta_i)} θ_i angles")
        logger.info(f"Collected {len(self.theta_ip1)} θ_{'i+1'} angles")
        logger.info(f"Collected {len(self.phi)} φ angles")

    def run_all_analyses(self, generate_plots: bool = True) -> None:
        """Run all analyses and write all data products.

        Even if generate_plots=False, we still write the .npy artifacts needed downstream
        (theta_edges_deg.npy, phi_edges_deg.npy, delta_phi_energy.npy, etc.).
        """
        if any(x is None for x in (self.bond_lengths, self.theta_i, self.theta_ip1, self.phi)):
            raise RuntimeError("Geometry not extracted yet.")

        plot_bond_length_distribution(self.bond_lengths, self.output_dir, self.data_dir, save_figures=generate_plots)

        theta_all = np.concatenate([self.theta_i, self.theta_ip1])
        plot_figure_2(theta_all, self.phi, self.output_dir, self.data_dir, save_figures=generate_plots)

        plot_figure_3(
            self.theta_i, self.theta_ip1, self.phi, self.output_dir, self.data_dir, save_figures=generate_plots
        )

        plot_phi_phi_correlation(self.phi, self.output_dir, self.data_dir, save_figures=generate_plots)

        plot_delta_phi_potential(self.phi, self.output_dir, self.data_dir, save_figures=generate_plots)

        self._save_summary()

    def _save_summary(self) -> None:
        theta_all = np.concatenate([self.theta_i, self.theta_ip1])
        summary = {
            "bond_lengths": {
                "count": int(len(self.bond_lengths)),
                "mean": float(np.mean(self.bond_lengths)),
                "std": float(np.std(self.bond_lengths)),
                "min": float(np.min(self.bond_lengths)),
                "max": float(np.max(self.bond_lengths)),
            },
            "theta_angles": {
                "count": int(len(theta_all)),
                "mean": float(np.mean(theta_all)),
                "std": float(np.std(theta_all)),
            },
            "phi_angles": {
                "count": int(len(self.phi)),
                "mean": float(np.mean(self.phi)),
                "std": float(np.std(self.phi)),
            },
        }

        if len(self.phi) > 2:
            summary["phi_phi_corr"] = float(np.corrcoef(self.phi[:-1], self.phi[1:])[0, 1])

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def run_backbone_analysis(args) -> int:
    analyzer = BackboneAnalyzer(cache_dir=args.cache_dir, output_dir=args.output_dir)

    pdb_ids = analyzer.load_pdb_list(args.pdb_list)
    logger.info(f"Loaded {len(pdb_ids)} entry IDs (normalized) from {args.pdb_list}")

    analyzer.extract_geometry(pdb_ids, max_chains=args.max_chains)
    analyzer.run_all_analyses(generate_plots=not args.no_plots)

    logger.info(f"\n✅ Backbone analysis complete! Results saved to {analyzer.output_dir}")
    logger.info(f"Data files saved to {analyzer.data_dir}")
    return 0
