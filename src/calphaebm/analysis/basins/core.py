# src/calphaebm/analysis/basins/core.py

"""Core basin analysis functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from calphaebm.data.id_utils import normalize_to_entry_ids
from calphaebm.utils.logging import get_logger

from .clustering import cluster_angles_gmm, cluster_angles_kmeans, get_basin_names
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIRCULAR_PHI,
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_MAX_CHAINS,
    DEFAULT_MAX_PDBS,
    DEFAULT_N_BASINS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_LIST,
    DEFAULT_PLOT_MAX_POINTS,
    DEFAULT_PSEUDOCOUNT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SMOOTH_SIGMA,
    DEFAULT_STANDARDIZE,
    PHI_BINS,
    PHI_MAX,
    PHI_MIN,
    THETA_BINS,
    THETA_MAX,
    THETA_MIN,
)
from .data_loader import load_angle_data
from .plots import plot_basin_comparison, plot_basin_histogram, plot_cluster_scatter

logger = get_logger()


class BasinAnalyzer:
    """Main class for basin analysis."""

    def __init__(
        self,
        cache_dir: Path,
        output_dir: Path,
        n_basins: int = DEFAULT_N_BASINS,
        cluster_method: str = DEFAULT_CLUSTER_METHOD,
        random_state: int = DEFAULT_RANDOM_STATE,
        smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
        circular_phi: bool = DEFAULT_CIRCULAR_PHI,
        standardize: bool = DEFAULT_STANDARDIZE,
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)

        # ✅ Backbone analysis writes its canonical .npy products into output_dir/data/
        # We mirror that convention here so basins can reuse backbone edges without copying/moving.
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.n_basins = int(n_basins)
        self.cluster_method = str(cluster_method)
        self.random_state = int(random_state)
        self.smooth_sigma = float(smooth_sigma)
        self.pseudocount = float(pseudocount)
        self.circular_phi = bool(circular_phi)
        self.standardize = bool(standardize)

        # Ensure plot/output directory exists (plots + summary files go here)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.theta: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None
        self.responsibilities: Optional[np.ndarray] = None
        self.basin_names: Optional[list[str]] = None

        self.load_stats = None
        self.failures: list[tuple[str, str]] = []

        self._saved_energies: list[np.ndarray] = []

    def load_pdb_list(self, pdb_list_path: Path) -> list[str]:
        """
        Load IDs from file. Accepts either entry IDs (1ABC) or entity IDs (1ABC_1).

        We normalize to entry IDs because CIFs are per-entry.
        """
        with open(pdb_list_path, "r") as f:
            raw = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return normalize_to_entry_ids(raw)

    @property
    def _angle_cache_path(self) -> Path:
        return self.data_dir / "angle_cache.npz"

    def load_angle_data(
        self,
        pdb_ids: list[str],
        max_pdbs: Optional[int] = DEFAULT_MAX_PDBS,
        max_chains: Optional[int] = DEFAULT_MAX_CHAINS,
        sample_every: int = DEFAULT_SAMPLE_EVERY,
        verbose: bool = True,
        force_reextract: bool = False,
    ) -> None:
        """Load (θ,φ) angle data, using cache if available.

        On first run: extracts from PDB structures and saves to
        analysis/secondary_analysis/data/angle_cache.npz.
        On subsequent runs: loads from cache instantly.
        Use force_reextract=True to ignore cache and re-extract.
        """
        cache_path = self._angle_cache_path

        if not force_reextract and cache_path.exists():
            logger.info(f"Loading (θ,φ) pairs from cache: {cache_path}")
            data = np.load(cache_path)
            self.theta = data["theta"]
            self.phi = data["phi"]
            self.load_stats = None
            self.failures = []
            logger.info(f"Loaded {len(self.theta)} (θ,φ) pairs from cache")
            return

        logger.info("Extracting (θ,φ) angle data from PDB structures...")
        theta, phi, stats, failures = load_angle_data(
            pdb_ids=pdb_ids,
            cache_dir=self.cache_dir,
            max_pdbs=max_pdbs,
            max_chains=max_chains,
            sample_every=sample_every,
            verbose=verbose,
        )

        self.theta, self.phi = theta, phi
        self.load_stats = stats
        self.failures = failures

        np.savez(cache_path, theta=theta, phi=phi)
        logger.info(f"Saved (θ,φ) cache ({len(theta)} pairs) → {cache_path}")

    def run_clustering(self) -> None:
        """Cluster the angle data into basins."""
        if self.theta is None or self.phi is None:
            raise RuntimeError("Angle data not loaded.")

        logger.info(
            f"Clustering {len(self.theta)} points into {self.n_basins} basins using {self.cluster_method} "
            f"(circular_phi={self.circular_phi}, standardize={self.standardize})..."
        )

        if self.cluster_method == "kmeans":
            self.labels, self.centers, self.responsibilities = cluster_angles_kmeans(
                self.theta,
                self.phi,
                n_clusters=self.n_basins,
                random_state=self.random_state,
                standardize=self.standardize,
                circular_phi=self.circular_phi,
            )
        elif self.cluster_method == "gmm":
            self.labels, self.centers, self.responsibilities = cluster_angles_gmm(
                self.theta,
                self.phi,
                n_components=self.n_basins,
                random_state=self.random_state,
                standardize=self.standardize,
                circular_phi=self.circular_phi,
            )
        else:
            raise ValueError(f"Unknown cluster_method: {self.cluster_method}")

        self.basin_names = get_basin_names(self.centers)

        logger.info("=== Cluster Centers (θ, φ in degrees) ===")
        for k, (theta_c, phi_c) in enumerate(self.centers):
            logger.info(f"  Basin {k} ({self.basin_names[k]}): θ={theta_c:6.1f}°, φ={phi_c:7.1f}°")

    def _get_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute canonical θ and φ bin edges from config constants.

        No file dependency — derived directly from the same binning constants
        used by backbone analysis, so edges are always consistent.
        """
        theta_edges = np.linspace(THETA_MIN, THETA_MAX, THETA_BINS + 1, dtype=np.float32)
        phi_edges = np.linspace(PHI_MIN, PHI_MAX, PHI_BINS + 1, dtype=np.float32)
        return theta_edges, phi_edges

    def save_histograms(self) -> None:
        """Save basin energy surfaces as .npy files (basin_k_histogram.npy)."""
        if self.theta is None or self.phi is None or self.labels is None:
            raise RuntimeError("Must load data and run clustering before saving histograms.")
        if self.basin_names is None:
            raise RuntimeError("Must run clustering before saving histograms (basin_names missing).")

        theta_edges, phi_edges = self._get_edges()
        logger.info(f"Using θ bins: {len(theta_edges)-1}, φ bins: {len(phi_edges)-1}")

        # Save edge files so SecondaryStructureEnergy (BasinPotential) can load them
        np.save(self.data_dir / "theta_edges_deg.npy", theta_edges)
        np.save(self.data_dir / "phi_edges_deg.npy", phi_edges)
        logger.info(f"Saved edge files -> {self.data_dir}/theta_edges_deg.npy, phi_edges_deg.npy")

        energies: list[np.ndarray] = []

        for k in range(self.n_basins):
            mask = self.labels == k
            theta_k = self.theta[mask]
            phi_k = self.phi[mask]

            frac = (len(theta_k) / max(len(self.theta), 1)) * 100.0
            logger.info(f"Basin {k} ({self.basin_names[k]}): {len(theta_k)} points ({frac:.1f}%)")

            hist, _, _ = np.histogram2d(
                theta_k,
                phi_k,
                bins=[theta_edges, phi_edges],
                density=True,
            )

            # Pseudocount to avoid log(0)
            hist = hist + self.pseudocount

            # Smooth with φ periodic wrap if requested
            if self.smooth_sigma > 0:
                try:
                    from scipy.ndimage import gaussian_filter

                    hist = gaussian_filter(hist, sigma=self.smooth_sigma, mode=("nearest", "wrap"))
                except ImportError:
                    logger.warning("scipy not available, skipping smoothing")

            # Normalize to probability mass
            hist = hist / max(hist.sum(), 1e-12)

            # Convert to energy and shift min to 0
            energy = -np.log(hist)
            energy = energy - energy.min()
            energy = energy.astype(np.float32)

            energies.append(energy)

            out_path = self.data_dir / f"basin_{k}_energy.npy"
            np.save(out_path, energy)
            logger.info(f"  Saved energy surface to {out_path}")

        self._saved_energies = energies

    def generate_plots(self, max_points: int = DEFAULT_PLOT_MAX_POINTS) -> None:
        """Generate all plots."""
        if self.theta is None or self.labels is None or self.centers is None or self.basin_names is None:
            logger.warning("No data to plot")
            return

        # Scatter plot in output_dir (human-facing)
        plot_cluster_scatter(
            theta=self.theta,
            phi=self.phi,
            labels=self.labels,
            centers=self.centers,
            basin_names=self.basin_names,
            output_dir=self.output_dir,
            max_points=max_points,
            random_state=self.random_state,
        )

        # Basin hist plots (need canonical edges)
        try:
            theta_edges, phi_edges = self._get_edges()
        except FileNotFoundError as e:
            logger.warning(str(e))
            logger.warning("Skipping basin histogram plots because edges are missing.")
            return

        energies: list[np.ndarray] = []
        for k in range(self.n_basins):
            path = self.data_dir / f"basin_{k}_energy.npy"
            if path.exists():
                energy = np.load(path).astype(np.float32)
                energies.append(energy)
                plot_basin_histogram(
                    energy=energy,
                    theta_edges=theta_edges,
                    phi_edges=phi_edges,
                    basin_idx=k,
                    basin_name=self.basin_names[k],
                    output_dir=self.output_dir,
                )

        if len(energies) > 1:
            plot_basin_comparison(
                energies=energies,
                basin_names=self.basin_names,
                output_dir=self.output_dir,
            )

    def save_summary(self) -> None:
        """Save analysis summary and failures."""
        if self.theta is None or self.labels is None or self.centers is None or self.basin_names is None:
            raise RuntimeError("Analysis incomplete; cannot save summary.")

        counts = [int(np.sum(self.labels == k)) for k in range(self.n_basins)]
        summary = {
            "n_samples": int(len(self.theta)),
            "n_basins": int(self.n_basins),
            "cluster_method": self.cluster_method,
            "random_state": int(self.random_state),
            "circular_phi": bool(self.circular_phi),
            "standardize": bool(self.standardize),
            "smooth_sigma": float(self.smooth_sigma),
            "pseudocount": float(self.pseudocount),
            "basin_names": list(self.basin_names),
            "centers": self.centers.tolist(),
            "counts_per_basin": counts,
            "paths": {
                "output_dir": str(self.output_dir),
                "data_dir": str(self.data_dir),
            },
            "edges": {
                "theta_edges_deg": str(self.data_dir / "theta_edges_deg.npy"),
                "phi_edges_deg": str(self.data_dir / "phi_edges_deg.npy"),
            },
        }

        if self.load_stats is not None:
            summary["load_stats"] = {
                "n_pdbs_attempted": int(self.load_stats.n_pdbs_attempted),
                "n_pdbs_failed": int(self.load_stats.n_pdbs_failed),
                "n_chains_processed": int(self.load_stats.n_chains_processed),
                "n_segments_processed": int(self.load_stats.n_segments_processed),
                "n_pairs_collected": int(self.load_stats.n_pairs_collected),
            }
            summary["n_failures_logged"] = int(len(self.failures))

        summary_path = self.output_dir / "basin_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        if self.failures:
            fail_path = self.output_dir / "failed_pdbs.txt"
            with open(fail_path, "w") as f:
                for pdb_id, reason in self.failures:
                    f.write(f"{pdb_id}\t{reason}\n")
            logger.info(f"Saved failures to {fail_path}")


def run_basin_analysis(args) -> int:
    """Main function to run basin analysis from CLI args."""
    analyzer = BasinAnalyzer(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        n_basins=args.n_basins,
        cluster_method=args.cluster_method,
        random_state=args.random_state,
        smooth_sigma=args.smooth_sigma,
        pseudocount=args.pseudocount,
        circular_phi=args.circular_phi,
        standardize=args.standardize,
    )

    pdb_ids = analyzer.load_pdb_list(args.pdb_list)
    logger.info(f"Loaded {len(pdb_ids)} entry IDs (normalized) from {args.pdb_list}")

    analyzer.load_angle_data(
        pdb_ids=pdb_ids,
        max_pdbs=args.max_pdbs,
        max_chains=args.max_chains,
        sample_every=args.sample_every,
        verbose=not args.quiet,
        force_reextract=args.force_reextract,
    )

    analyzer.run_clustering()
    analyzer.save_histograms()

    if not args.no_plots:
        analyzer.generate_plots(max_points=args.plot_max_points)

    analyzer.save_summary()

    logger.info(f"\n✅ Basin analysis complete! Results saved to {analyzer.output_dir}")
    return 0
