# src/calphaebm/analysis/basins/clustering.py

"""Clustering algorithms for basin analysis.

Key fix: φ is circular, so default clustering uses features [θ, sin(φ), cos(φ)].
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _phi_circular_features(phi_deg: np.ndarray) -> np.ndarray:
    """Convert phi degrees to circular features [sin(phi), cos(phi)]."""
    phi_rad = np.deg2rad(phi_deg.astype(np.float64))
    return np.column_stack([np.sin(phi_rad), np.cos(phi_rad)]).astype(np.float32)


def _build_feature_matrix(theta: np.ndarray, phi: np.ndarray, circular_phi: bool = True) -> np.ndarray:
    """Build feature matrix for clustering."""
    theta = theta.astype(np.float32).reshape(-1, 1)
    phi = phi.astype(np.float32).reshape(-1)

    if circular_phi:
        sc = _phi_circular_features(phi)
        return np.concatenate([theta, sc], axis=1)
    return np.column_stack([theta[:, 0], phi]).astype(np.float32)


def cluster_angles_kmeans(
    theta: np.ndarray,
    phi: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    standardize: bool = True,
    circular_phi: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Cluster angle data using K-means.

    Returns:
        labels: (N,)
        centers: (K,2) in (theta_deg, phi_deg) space (phi recovered via atan2 if circular_phi)
        responsibilities: None
    """
    X = _build_feature_matrix(theta, phi, circular_phi=circular_phi)

    scaler = None
    X_fit = X
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_fit)

    centers_fit = kmeans.cluster_centers_
    centers = centers_fit if scaler is None else scaler.inverse_transform(centers_fit)

    if circular_phi:
        theta_c = centers[:, 0]
        sin_c = centers[:, 1]
        cos_c = centers[:, 2]
        phi_c = np.rad2deg(np.arctan2(sin_c, cos_c))
        centers_theta_phi = np.column_stack([theta_c, phi_c]).astype(np.float32)
    else:
        centers_theta_phi = centers.astype(np.float32)

    return labels, centers_theta_phi, None


def cluster_angles_gmm(
    theta: np.ndarray,
    phi: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
    standardize: bool = True,
    circular_phi: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster angle data using Gaussian Mixture Model.

    Returns:
        labels: (N,)
        centers: (K,2) in (theta_deg, phi_deg) space
        responsibilities: (N,K)
    """
    X = _build_feature_matrix(theta, phi, circular_phi=circular_phi)

    scaler = None
    X_fit = X
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X_fit)

    responsibilities = gmm.predict_proba(X_fit)
    labels = np.argmax(responsibilities, axis=1)

    means_fit = gmm.means_
    means = means_fit if scaler is None else scaler.inverse_transform(means_fit)

    if circular_phi:
        theta_c = means[:, 0]
        sin_c = means[:, 1]
        cos_c = means[:, 2]
        phi_c = np.rad2deg(np.arctan2(sin_c, cos_c))
        centers_theta_phi = np.column_stack([theta_c, phi_c]).astype(np.float32)
    else:
        centers_theta_phi = means.astype(np.float32)

    return labels, centers_theta_phi, responsibilities


def get_basin_names(centers: np.ndarray) -> list[str]:
    """Assign heuristic names based on (theta, phi) centers (degrees).

    Regions based on Cα Ramachandran map in torsions() convention:
      Helix:  θ ∈ [80,110],  φ ∈ [−75, −20]              — α-helix core
      Sheet:  θ ∈ [110,150], φ ∈ [110,180] ∪ [−180,−170] — β-strand
      Coil:   everything else
    """
    names: list[str] = []
    for theta_c, phi_c in centers:
        if 80 <= theta_c <= 110 and -75 <= phi_c <= -20:
            names.append("Helix")
        elif 110 <= theta_c <= 150 and (phi_c >= 110 or phi_c <= -170):
            names.append("Sheet")
        elif 80 <= theta_c <= 110 and -130 <= phi_c <= -75:
            names.append("PPII")
        elif 100 <= theta_c <= 130 and 20 <= phi_c <= 80:
            names.append("Turn")
        else:
            names.append("Coil")
    return names
