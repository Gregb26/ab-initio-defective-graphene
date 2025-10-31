"""
plot_band.py

Python scripts that plots a publication-ready band structure plot from the outputs of a non-SCF band structure
Abinit run.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

from electron_defect_interaction.utils.get_utils import get_band

# Useful conversion factors
bohr_to_ang = 0.529177210903
ha_to_ev = 27.211386245988

filepath = "data/graphene/unit_cell/5x5x1/abinit/sp2/graphene_w90_5x5x1o_DS3_GSR.nc"
kpt_path_red, eigs, fermi_energy = get_band(filepath)

# pi/pi* bands index: 3, 4 for graphene
eigs = eigs[3:5,:]
eigs_ev = (eigs - fermi_energy) * ha_to_ev # transform to eV and set fermi energy to zero

# Build 1d curvilinear path in kspace
dk = np.linalg.norm(np.diff(kpt_path_red, axis=0), axis=1) # distance between each kpoint
ds = np.concatenate([[0.0], np.cumsum(dk)])

# High symmetry points in  kspace the band structure is plotted along
sym_red = [
    (r'$\Gamma$', np.array([0.0, 0.0, 0.0])),
    ('K',         np.array([2/3, 1/3, 0.0])),
    ('M',         np.array([1/2, 0.0, 0.0])),
    (r'$\Gamma$', np.array([0.0, 0.0, 0.0])),
]

tol= 1e-12
labels, coords = zip(*sym_red)

# Finding the indices of the high symmetry points in the kpoint path
match_idx = [
    np.flatnonzero(np.all(np.isclose(kpt_path_red, coord, atol=tol, rtol=0.0), axis=1))
    for coord in coords
]

# Removing duplicate indices
sym_ids = np.unique(np.concatenate(match_idx))

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = plt.gca()

for n in range(eigs_ev.shape[0]):
    ax.plot(ds, eigs_ev[n,:], 'k')

for i in sym_ids:
    ax.axvline(ds[i], color='k', lw=0.6, alpha=0.5)

ax.set_xticks([ds[i] for i in sym_ids])
ax.set_xticklabels(labels)
ax.axhline( 0, color='k', lw=0.6, alpha=0.5)
ax.set_xlim(ds[0], ds[-1])
ax.set_ylim(-20, 6)
ax.set_title("Pristine Graphene Unit Cell Band Structure - 20,20,1 MP Grid")
plt.xlabel("Kpoint path")
plt.ylabel("Eigenvalues (eV)")
plt.savefig("graphene_20x20x1_ebands.png", dpi=300)
plt.show()


