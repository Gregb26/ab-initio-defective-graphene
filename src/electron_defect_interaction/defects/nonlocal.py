"""
nonlocal.py
    Python modules containing the functions to compute the non-local part of the electron-defect scattering matrix.

TODO: 
    - Generalize to supercells with different atoms (right now assumes one atom type)
    - Generalize to atoms with different l quantum number (right now assums a single l since one atom type)
    - Generalize to atoms with different number of projector channels per l (now assums 2 for l=0,1)
"""

import numpy as np

from electron_defect_interaction.utils.lattice import red_to_cart
from electron_defect_interaction.io.abinit_io import *
from electron_defect_interaction.utils.planewaves import mask_invalid_G
from electron_defect_interaction.io.pseudo_io import get_psps, fq_from_fr


def build_K_vectors(k_red, G_red, keep, B_uc):
    """
    Compute the K=k+G vectors, their norm and their unit vectors.

    Inputs:
        k_red: (nkpt, 3) array of ints:
            kpoints in reduced coordinates.
        G_red: (nkpt, nG_max, 3) array of ints:
            Reciprocal lattice vectors in reduced coordinates
        mask: (nkpt, nG_max) array of boolean:
            Boolean mask that selects only active G vectors for a given k 
            i.e. those that satisfy 0.5*(2pi)^2(k+G)^2 < ecut
        B_uc: (3, 3) array of floats:
            Columns are the primitive reciprocal lattice vectors B[:,i] = B_i
        
    Returns
        K: (nkpt, nG_max, 3) array of floats:
            K=k+G vectors in cartesian coords
        K_norm: (nkpt, nG_max) array of floats:
            Norm of K=k+G vectors
        K_hat: (nkpt, nG_max, 3):
            Unit vectors in the direction of the K=k+G vectors.
    """

    # Compute K=k+G in cartesian coordinates
    K_red = k_red[:, np.newaxis, :] + G_red
    Ks = red_to_cart(K_red, B_uc)
    K = np.where(keep[..., np.newaxis], Ks, 0.0)

    # Compute norm of K=k+G vectors
    norms = np.linalg.norm(K, axis=2) # (nkpt, nG_max)
    K_norm = np.where(keep, norms, 0.0) # (nkpt, nG_max)

    # Compute unit vectors 
    valid_Ks = keep & (K_norm > 0) # selects Ks with valig G and non-zero norm (to avoid division by zero)
    K_hat = np.zeros_like(K) # (nkpt, nG_max, 3)
    K_hat = np.divide(K, K_norm[:, :, None], out=K_hat, where=valid_Ks[:,:,None]) # (nkpt, nG_max, 3)

    return K, K_norm, K_hat

def compute_phase(K, tau_as):
    """
    Compute phase exp(i tau_{a}\\cdot K)

    Inputs:
        K: (nkpt, nG_max, 3):
            K=k+G vectors in cartesian coordinates
        tau_as: (natom, 3):
            Atomic positions of atom s of type a in cartesian coordinates

    Returns:
        phase_ksg: (nkpt, natom, nG_max)      
    """

    dot = np.einsum("kgd, sd -> ksg", K, tau_as, optimize = True)
    phase_ksg = np.exp(-1j * dot)

    return phase_ksg

def compute_angular_part(K_hat, lmax):
    """
    Compute the spherical harmonics Y_l^m(K_hat)

    Inputs:
        K_hat: (nkpt, nG_max) array of floats:
            Norm of the K=k+G vectors
        lmax: float
            Maximum orbital angular momentum quantum number
        
    Returns:
        Y_kglm (nkgpt, nG_max, lmax, 2lmax+1) array of complex
            Spherical harmonics at l, m, phi and theta, where phi and theta are the spherical angles of the unit vectors K_hat
    """

    nkpt, nG_max, _ = K_hat.shape
    # compute angles in K
    theta = np.arccos(np.clip(K_hat[...,2], -1.0, 1.0)) # theta = arccos(Kz) (0, pi)
    phi = np.arctan2(K_hat[..., 1], K_hat[..., 0]) + np.pi # phi = arctan(Ky/Kx) (0, 2pi)

    Y_kglm = np.zeros((nkpt, nG_max, lmax+1, 2*lmax+1), dtype=complex)

    from scipy.special import sph_harm_y
    for l in range(lmax+1):
        for m in range(-l, l+1):
            mi = m + lmax
            Y_kglm[..., l, mi] = sph_harm_y(l, m, theta, phi) # (nkpt, nG_max, lmax, 2lmax+1)

    return Y_kglm

def compute_M_NL(uc_wfk_path, sc_wfk_path, psp8_path):
    """
    Computes the non local part of the electron-defect interaction matrix.
    
    Inputs:
        uc_wfk_path: str
            Path to the ABINIT WFK.nc output file for the wavefunctions of the unit cell.
        sc_wfk_path:
            Path to the ABINIT WFK.nc output file for the wavefunctions of the supercell.
        sc_psps_path:
            Path to the ABINIT PSPS.nc output file for the pseudopotentials of the supercell

    Returns:
        M_NL: (nband, nkpt, nband, nkpt) array of complex:
            Electron-defect interaction matrix
    """
    from scipy.interpolate import CubicSpline

    # Get non local part of the pseudopotentials 
    ekb_li, fr_li, rgrid = get_psps(psp8_path) # KB energies (lmax+1, imax+1), KB radial projectors in real space (lmax+1, imax+1, mmax), grid on which the projectors are defined (mmax, )
    lmax = ekb_li.shape[0] - 1 # maximum orbital angular momentum present

    # Get necessary unit cells quantities
    C_nkg, nG = get_C_nk(uc_wfk_path) # planewave coefficients (nband, nkpt, nG_max) and number of active G per k (nkpt)
    G_red = get_G_red(uc_wfk_path) # reciprocal lattice vectors in reduced coords of the unit cell (nkpt, nG_max, 3)
    k_red = get_kpt_red(uc_wfk_path) # kpoints in reduced coords of the unit cell (nkpt, 3)
    B_uc, _ = get_B_volume(uc_wfk_path) # primitive reciprocal lattice vectors of the unit cell B[:,i] = b_i
    ecut = get_ecut(uc_wfk_path) # planewave energy cutoff used in the calculation

    assert ecut == get_ecut(sc_wfk_path), 'Must use the same ecut in both unit cell and supercell calculations!'

    # Get necessary super cell quantities
    x_red = get_x_red(sc_wfk_path) # atomic positions in reduced coords of the supercell (natom, 3)
    A_sc, Omega_sc = get_A_volume(sc_wfk_path) # primitive lattice vectors and cell volume of the supercell
    tau_s = red_to_cart(x_red, A_sc) # atomoic positions of atoms in the supercell in cartesian coords 

    # Compute boolean mask that selects only the active recripocal lattice vector G for each k-point and mask invalid C's (pad with zeros)
    keep = mask_invalid_G(nG)
    C_nkg = np.where(keep, C_nkg, 0.0) # (nband, nkpt, nG_max)

    # Compute K=k+G vectors
    K, K_norm, K_hat = build_K_vectors(k_red, G_red, keep, B_uc) # (nkpt, nG_max, 3), (nkpt. nG_max), (nkpt, nG_max, 3)

    # Transform the radial from factor to q space
    qmax = 2*np.sqrt(2*ecut) # choose a qmax two times the recommended for safety ...
    q = np.linspace(0, qmax, 10000)
    fq_li = fq_from_fr(rgrid, fr_li, q)

    # Interpolate radial form factors to be able to evaluate them at K=|k+G| vectors
    Fq_li = CubicSpline(q, fq_li, axis=-1, extrapolate=False)   
    assert np.max(K_norm) < qmax, 'Maximum |K|=|k+G| must be within the qgrid'
    F_likg = Fq_li(K_norm) # (lmax+1, imax+1, nkpt, nG_max)

    # Compute phases
    phase_ksg = compute_phase(K, tau_s) # (nkpt, natom, nG_max)

    # Compute angular part
    Y_kglm = compute_angular_part(K_hat, lmax) # (nkpt, nG_max, lmax+1, 2lmax+1)

    # Compute overlaps by summing over all G vectors
    B_nkslim = (4*np.pi / np.sqrt(Omega_sc)) * np.einsum("nkg, likg, kglm, ksg -> nkslim ", np.conj(C_nkg), F_likg, Y_kglm, phase_ksg, optimize=True) # (nkpt, nG_max, ntypat, natom, nkb)
    B_jpslim_conj = np.conj(B_nkslim)

    # Compute matrix elements by summing over l, i, s (atoms) and m
    M_NL = np.einsum("li, nkslim, jpslim -> nkjp ", ekb_li, B_nkslim, B_jpslim_conj)

    return M_NL 