"""
plot_band.py
    Python module that plots a publication-ready band structure plot from the outputs of a non-SCF band structure
    Abinit run.

    TODO
        - Make it a script runnable in command line with inputs 
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

from electron_defect_interaction.io.abinit_io import get_band, get_pot


def plot_graphene_bands(wfk, title, shift=True):
    bohr_to_ang = 0.529177210903
    ha_to_ev = 27.211386245988

    kpt_path_red, eigs, fermi_energy = get_band(wfk)
    
    if shift:
        K = np.array([2/3, 1/3, 0.0])
        iK = np.linalg.norm(kpt_path_red - K, axis=1).argmin()
        fermi = eigs[3:5, iK].mean()
    
    else:
        fermi=0

    eigs_ev = (eigs - fermi) * ha_to_ev  # (nband, nkpt)

    # 1D curvilinear distance along the path (ok in reduced coords for plotting)
    dk = np.linalg.norm(np.diff(kpt_path_red, axis=0), axis=1)
    ds = np.concatenate([[0.0], np.cumsum(dk)])

    # Target symmetry points (labels, reduced coordinates)
    sym_red = [
        (r'$\Gamma$', np.array([0.0, 0.0, 0.0])),
        ('K',         np.array([2/3, 1/3, 0.0])),
        ('M',         np.array([1/2, 0.0, 0.0])),
        (r'$\Gamma$', np.array([0.0, 0.0, 0.0])),
    ]
    K = np.array([2/3, 1/3, 0.0])
    iK = np.linalg.norm(kpt_path_red - K, axis=1).argmin()


    # Snap each target to the nearest point in the path
    idxs = [np.linalg.norm(kpt_path_red - coord, axis=1).argmin()
            for _, coord in sym_red]

    # Remove accidental duplicates while preserving order
    tick_pos, tick_lab = [], []
    last = None
    for (lab, _), i in zip(sym_red, idxs):
        if (last is None) or (i != last):
            tick_pos.append(ds[i])
            tick_lab.append(lab)
            last = i

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

    for n in range(eigs_ev.shape[0]):
        ax.plot(ds, eigs_ev[n, :], linewidth=1.2)

    # Vertical lines at symmetry points
    for x in tick_pos:
        ax.axvline(x, color='k', lw=0.6, alpha=0.5)

    # Aesthetics
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.axhline(0, color='k', lw=0.6, alpha=0.6)
    ax.set_xlim(ds[0], ds[-1])
    # ax.set_ylim(-20, 6)
    ax.set_title(str(title))
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    plt.show()



