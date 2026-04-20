"""
plot_hremd_fes.py — FES matching Noe-group / Majewski et al. style.
x=dRMSD, y=Q, viridis_r, 1kT contour lines, white background.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter


def load_fes_dir(fes_dir, pdb_id, replicas=None):
    data = {}
    for f in sorted(Path(fes_dir).glob(f"{pdb_id}_hremd_*.npz")):
        tag = f.stem.split("_")[-1]
        idx = 0 if tag == "target" else int(tag.replace("rep", ""))
        if replicas is None or idx in replicas:
            d = np.load(f)
            data[idx] = {k: d[k] for k in d.files}
            data[idx]["tag"] = tag
    return dict(sorted(data.items()))


def compute_fes(Q, dRMSD, n_bins=20, kT=1.0, dR_range=None, smooth=1.2):
    """
    Compute F = -kT log P on a 2D grid.
    Empty bins get F = F_max (not NaN) so Gaussian smoothing works everywhere.
    """
    if dR_range is None:
        dR_range = (0.0, max(float(np.percentile(dRMSD, 98)), 4.0))

    H, Qe, dRe = np.histogram2d(Q, dRMSD, bins=n_bins, range=[(0.0, 1.0), dR_range])

    # Replace zeros with a small pseudocount so log is defined everywhere
    H_smooth = H + 0.1
    F = -kT * np.log(H_smooth)
    F -= F.min()

    # Gaussian smooth
    F = gaussian_filter(F, sigma=smooth)
    F -= F.min()

    Qc = 0.5 * (Qe[:-1] + Qe[1:])
    dRc = 0.5 * (dRe[:-1] + dRe[1:])

    # Build a visited mask: bins with at least 1 real count
    visited = H > 0
    visited = gaussian_filter(visited.astype(float), sigma=smooth) > 0.05

    return F, Qc, dRc, visited


def plot_fes(fes_dir, pdb_id, out_path=None, replicas=None, n_bins=20, smooth=1.2, F_max=6.0):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
        }
    )

    data = load_fes_dir(fes_dir, pdb_id, replicas)
    if not data:
        print(f"No FES data found in {fes_dir} for {pdb_id}")
        return

    # Debug: print data ranges
    for idx, d in data.items():
        Q = d["Q"]
        dR = d["dRMSD"]
        print(f"  rep{idx}: n={len(Q)}  Q=[{Q.min():.3f},{Q.max():.3f}]  " f"dR=[{dR.min():.2f},{dR.max():.2f}]")

    n = len(data)
    all_dR = np.concatenate([d["dRMSD"] for d in data.values()])
    dR_max = float(max(np.percentile(all_dR, 97), 4.0))

    norm = mcolors.Normalize(vmin=0, vmax=F_max)
    cmap = plt.cm.viridis_r
    levels = np.arange(0, F_max + 0.5, 1.0)

    PANEL = 2.4
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(PANEL * n + 0.75, PANEL + 0.65),
        facecolor="white",
    )
    plt.subplots_adjust(left=0.12, right=0.87, bottom=0.14, top=0.87, wspace=0.10)
    if n == 1:
        axes = [axes]

    last_pcm = None

    for col, ((idx, d), ax) in enumerate(zip(data.items(), axes)):
        Q = d["Q"].astype(float)
        dRMSD = d["dRMSD"].astype(float)
        g_ss = float(d["gate_ss"])
        g_pk = float(d["gate_pack"])
        is_target = idx == 0

        F, Qc, dRc, visited = compute_fes(Q, dRMSD, n_bins=n_bins, dR_range=(0.0, dR_max), smooth=smooth)

        # Clip F at F_max for display
        F_plot = np.clip(F, 0, F_max)
        # Mask truly unvisited regions (far extrapolation)
        F_masked = np.where(visited, F_plot, np.nan)
        F_masked = np.ma.masked_invalid(F_masked)

        best_Q = float(Q.max())
        best_dR = float(dRMSD[np.argmax(Q)])
        folded = best_Q >= 0.95 and best_dR < 3.0

        # Smooth fill
        pcm = ax.pcolormesh(dRc, Qc, F_masked, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        last_pcm = pcm

        # Contour lines every 1 kT
        try:
            ax.contour(dRc, Qc, F_masked, levels=levels, colors="white", linewidths=0.55, alpha=0.55)
        except Exception:
            pass

        # Native state star
        ax.plot(0.0, 1.0, "*", ms=10, color="white", mec="#444", mew=0.5, zorder=10, ls="none")

        # Threshold lines
        ax.axhline(0.95, color="white", lw=0.55, ls=(0, (5, 4)), alpha=0.7)
        ax.axvline(3.0, color="white", lw=0.55, ls=(0, (5, 4)), alpha=0.7)

        ax.set_xlim(0, dR_max)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.tick_params(which="both", labelsize=8, top=True, right=True)

        ax.set_xlabel("dRMSD (Å)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Q", fontsize=9)
        else:
            ax.set_yticklabels([])

        label = "Target" if is_target else f"Rep {idx}"
        fold_tag = " [folded]" if folded else ""
        ax.set_title(f"{label}{fold_tag}", fontsize=8.5, pad=4, fontweight="bold" if is_target else "normal")

        ax.text(
            0.97,
            0.97,
            f"ss={g_ss:.2f} | pk={g_pk:.2f}",
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
            color="white",
            family="monospace",
        )

        ax.text(
            0.97,
            0.03,
            f"Q*={best_Q:.2f}  dR*={best_dR:.1f} Å",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            color="white",
        )

        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    # Colorbar
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.020, pos.height])
    cb = fig.colorbar(last_pcm, cax=cax, extend="max")
    cb.set_label("F (kT)", fontsize=8.5, labelpad=4)
    cb.set_ticks(np.arange(0, F_max + 0.1, 1.0))
    cb.ax.tick_params(labelsize=7.5, width=0.6, length=2.5, direction="in")
    cb.outline.set_linewidth(0.6)

    fig.suptitle(f"{pdb_id.upper()} — HREMD free energy surfaces", fontsize=10, fontweight="bold", y=0.97)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fes-dir", required=True)
    p.add_argument("--pdb-id", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--replicas", nargs="+", type=int, default=None)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--smooth", type=float, default=1.2)
    p.add_argument("--f-max", type=float, default=6.0)
    args = p.parse_args()
    plot_fes(args.fes_dir, args.pdb_id, args.out, args.replicas, args.n_bins, args.smooth, args.f_max)


if __name__ == "__main__":
    main()
