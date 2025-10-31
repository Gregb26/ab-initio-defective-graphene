"""
pw_utils.py
    Python module containing helper functions to perform operations of planewaves quantities.
"""

import numpy as np
from netCDF4 import Dataset

def make_Cdicts_for_k(C_nk, G_red_uc, npw, ik, band_indices):
    """
    Build per-band dictionaries at k-index ik: {(h,k,l) -> C_{band,k}(G)}.
    Useful to lookup the value of C_nk(G) given G in reduced coordinates.

    Inputs:
    -------
        C_nk: [nband, nkpt, npwmax] array:
            Array with the wavefunction coefficients. nband is the band index, npkt is the kpoint index and npwk is the coefficients index.
            For example C_nk[0,0, :] gives the npw_k coefficients of the wavefunction for the first band, at the first kpoint.
        G_red_uc: [nkpt, npwmax, 3] array of ints:
            Reduced G-vectors in the unit cell, in units of the reciprocal lattice vectors.
            For example G_red_uc[0, :, :] gives the (h,k,l) reduced coordinates of the planewaves at the first kpoint.
        npw: [nkpt] array of ints:
            Number of active planewaves for a given kpoint. 
        ik: int:
            Index of the k-point to process.
        band_indices: list of ints:
            List of band indices to include in the output dictionaries.

    Returns:
    --------
        Cdicts: list of dicts:
            List of dictionaries, one per band in band_indices, mapping reduced integer coordinates of a G vector (h,k,l) to  a coefficient C_{nk}(G).
            The order of the list corresponds to the order of band_indices.
    """
    # Number of active planewaves at this k-point
    npw_k = int(npw[ik])

    Gk = G_red_uc[ik, :npw_k, :] # (npw_k, 3), ints
    Cdicts = []
    for ib in band_indices:
        coeffs = C_nk[ib, ik, :npw_k]          # (npw_k,)
        d = {tuple(map(int, g)): coeffs[i] for i, g in enumerate(Gk)}

        Cdicts.append(d)

    return Cdicts  # list in same order as band_indices

def mask_invalid_G(nG):
    """
    Returns a boolean mask that selects only the active recripocal lattice vector G for each k-point.

    Inputs:
        nG: (nkpt, ) array of int
            Number of active G for each kpoint
    """

    nG_max = np.max(nG)
    id_G_valid = np.arange(nG_max)[None, :] # (1, nG_max)
    keep = id_G_valid < nG[:, None] # (nkpt, nG_max)

    return keep