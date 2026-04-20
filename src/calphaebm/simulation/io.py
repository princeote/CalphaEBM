# src/calphaebm/simulation/io.py

"""I/O utilities for saving trajectories in multiple formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mdtraj as md
import numpy as np
import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()


class TrajectorySaver:
    """Save trajectories in multiple formats (DCD, NPY, PT, PDB)."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        sequence: Optional[List[str]] = None,
        topology: Optional[md.Topology] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = sequence
        self.topology = topology
        self.frames: List[np.ndarray] = []
        self.metadata: Dict[str, Any] = {}

    def append(self, R: torch.Tensor) -> None:
        """Add a frame to the trajectory.

        Args:
            R: (B, N, 3) or (N, 3) coordinates.
        """
        if R.dim() == 3:
            R = R[0]  # Take first batch
        self.frames.append(R.cpu().clone().numpy())

    def _create_topology(self, n_atoms: int) -> md.Topology:
        """Create Cα-only topology."""
        if self.topology is not None:
            return self.topology

        top = md.Topology()
        chain = top.add_chain()

        for i in range(n_atoms):
            if self.sequence and i < len(self.sequence):
                resname = self.sequence[i]
            else:
                resname = "ALA"

            residue = top.add_residue(resname, chain, resSeq=i + 1)
            top.add_atom("CA", md.element.carbon, residue)

        return top

    def save_all(self, metadata: Optional[Dict] = None) -> Dict[str, Path]:
        """Save trajectory in all formats.

        Args:
            metadata: Additional metadata to save.

        Returns:
            Dictionary mapping format names to file paths.
        """
        if not self.frames:
            logger.warning("No frames to save")
            return {}

        coords = np.array(self.frames)  # (n_frames, n_atoms, 3)
        n_frames, n_atoms = coords.shape[:2]

        paths = {}

        # 1. NPY format (fast loading for Python)
        npy_path = self.out_dir / "coords.npy"
        np.save(npy_path, coords)
        paths["npy"] = npy_path
        logger.debug(f"Saved NPY: {npy_path}")

        # 2. PyTorch format (for resuming)
        pt_path = self.out_dir / "coords.pt"
        torch.save(torch.from_numpy(coords), pt_path)
        paths["pt"] = pt_path
        logger.debug(f"Saved PT: {pt_path}")

        # 3. MDTraj trajectory (DCD + PDB)
        try:
            top = self._create_topology(n_atoms)
            traj = md.Trajectory(coords, top)

            # DCD format (binary, compressed)
            dcd_path = self.out_dir / "trajectory.dcd"
            traj.save_dcd(str(dcd_path))
            paths["dcd"] = dcd_path
            logger.debug(f"Saved DCD: {dcd_path}")

            # First frame as PDB (for visualization)
            pdb_path = self.out_dir / "frame0.pdb"
            traj[0].save_pdb(str(pdb_path))
            paths["pdb"] = pdb_path
            logger.debug(f"Saved PDB: {pdb_path}")

        except Exception as e:
            logger.warning(f"Failed to save MDTraj formats: {e}")

        # 4. Metadata
        if metadata or self.metadata:
            meta_path = self.out_dir / "metadata.json"
            full_meta = {**self.metadata, **(metadata or {})}

            # Convert non-serializable types
            clean_meta = {}
            for k, v in full_meta.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_meta[k] = v.item()
                elif isinstance(v, (set, tuple)):
                    clean_meta[k] = list(v)
                elif isinstance(v, (Path, torch.device)):
                    clean_meta[k] = str(v)
                else:
                    clean_meta[k] = v

            with open(meta_path, "w") as f:
                json.dump(clean_meta, f, indent=2)
            paths["metadata"] = meta_path
            logger.debug(f"Saved metadata: {meta_path}")

        logger.info(f"Saved trajectory: {n_frames} frames, {n_atoms} atoms")
        return paths

    def get_mdtraj(self) -> Optional[md.Trajectory]:
        """Get trajectory as MDTraj object."""
        if not self.frames:
            return None

        coords = np.array(self.frames)
        top = self._create_topology(coords.shape[1])
        return md.Trajectory(coords, top)


def load_trajectory_npy(path: Union[str, Path]) -> np.ndarray:
    """Load trajectory from NPY file."""
    return np.load(path)


def load_trajectory_pt(path: Union[str, Path]) -> torch.Tensor:
    """Load trajectory from PyTorch file."""
    return torch.load(path, weights_only=False)


def load_trajectory_mdtraj(dcd_path: Union[str, Path], pdb_path: Union[str, Path]) -> md.Trajectory:
    """Load trajectory from DCD+PDB using MDTraj."""
    return md.load_dcd(str(dcd_path), top=str(pdb_path))
