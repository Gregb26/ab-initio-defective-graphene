"""
non_local.py

    Python module that contains functions for computing non-local electron-defect interaction matrix elements.
"""

import numpy as np
from scipy.special import sph_harm

def build_beta(qG, lmn, eval_B, omega, eps, dtype=np.complex64):
    """
    Build the beta projectors in G-space: \\beta_i(q) = f_i(|q|) * Y_{l_i, m_i}(\\hat{q})} for all channels i.

    Inputs
    ------
        qG : (npw, 3) array
            k+G grid for every k in Bohr^{-1}
        lmn : (nkb, 3) array of ints
            Orbital angular momentum quantum numbers (l, m, n) for the spherical harmonics. nkb is the number of KB channels
        eval_B : callable |q| (npw) -> (npw, nkb):
            Function that computes the radial factors f_i(|q|) given a q vector.
        eps: float
            Small value to avoid division by zero to handle |q| = 0 cases.

        Outputs
        -------
        beta : (npw, nkb) complex array
            The beta projectors in G-space: beta[G, i] = \\beta_i(q_G)

        """

    npw = qG.shape[0] # number of k+G points
    l = lmn[:, 0].astype(int) # (nkb,)
    m = lmn[:, 1].astype(int) # (nkb,)
    n = lmn[:, 2].astype(int) # (nkb,)

    # Getting spherical angles (theta and phi) of \hat{q}
    q = np.linalg.norm(qG, axis=1) # |q|
    cos_theta = np.divide(qG[:, 2], q, out=np.ones_like(q), where=q > eps) # cos(theta) = qz/|q|
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # theta in [0, pi]
    phi = np.arctan2(qG[:, 1], qG[:, 0]) # phi in (-pi, pi]

    # Computing the radial factors
    f_q = eval_B(q) # (NG, nkb)

    # Computing the spherical harmonics
    Y = np.stack([sph_harm(m_i, l_i, phi, theta) for l_i, m_i in zip(l, m)], axis=1) # (npw, nkb)

    # Handle q=0, only l=0 contributes 
    if np.any(q < eps):
        Y[(q < eps)[:, None] & (l[None, :] > 0)] = 0.0 

    prefecator = ((-1j) ** l )[None, :] / (np.sqrt(omega) * 4*np.pi) # (1, nkb)
    # Combine to compute beta projectors 
    beta = f_q * Y # (npw, nkb)

    # Testing
    if np.any(q < eps):
        q0 = (q < eps)
        col_l0   = (l == 0)
        col_lgt0 = (l > 0)
        atol = 1e-5 if np.dtype(dtype) == np.complex64 else 1e-10

        # β = 0 for ℓ>0
        assert np.allclose(beta[np.ix_(q0, col_lgt0)], 0.0, atol=atol)

        # β = f(0)/√(4π) for ℓ=0
        ref = f_q[np.ix_(q0, col_l0)] / np.sqrt(4*np.pi)
        assert np.allclose(beta[np.ix_(q0, col_l0)], ref, rtol=1e-6, atol=atol)
    
    return beta

def build_projector_overlaps(qG_k, tau, beta_k, C_nk): 
    """
    Compute channel by atoms overlaps: U_i^{(s)}(k,n) = sum_G [ e^{-i (k+G)·tau_s} * beta_i(k+G) * C_{nk}(G) ].

    Inputs:
    -------
        qG : array_like
            k+G grid for every k.
        tau : array_like
            Atomic positions in cartesian coordinates
        beta : array_like
            The beta projectors in G-space.
        C_nk : array_like
            Planewave coefficients.

    Outputs:
    --------
        U : array_like
            The channel by atom overlaps.
    """

    phase = np.exp(-1j * (qG_k @ tau.T)).astype(beta_k.dtype)   # (NG_k, Nat)

    return np.einsum('Gi,nG,Gs->nis', beta_k.conj(), C_nk, phase, optimize=True)

def build_nonlocal_ed_matrix(qG, npw, C_nk, kpt_id, band_id, tau, idlmn, eval_B, omega, D):
    """
    Compute the non-local electron-defect interaction matrix elements M_{k,k′}[n,m] = Σ_{i,s} D_i U_k[n,i,s] U_{k′}[m,i,s]^*.
    Inputs:
    -------
        qG: ndarray of shape (nkpt, npw, 3)
            Unit cell k+G grid 
        npw: ndarray of shape (nkpt,)
            Number of plane waves for each k-point
        C_nk: ndarray of shape (nband, nkpt, npw)
            Planewave coefficients of the wavefunctions of the unit cell
        kpt_id: list
            Indices of the kpoints for which to compute the matrix
        band_id: list
            Indices of the bands for which to compute the matrix
        tau: ndarray of shape (nchannels, natoms, nbands)
            Atomic positions of atoms in the supercell

    Outputs:
    --------
        M: list of {"ik": int, "ikp": int, "M": (nband, nband) ndarray}
            List of dictionaries containing the k-point indices and the corresponding matrix elements
    """

    nband = len(band_id)
    nkpt = C_nk.shape[1]

    # Precompute betas
    betas= [build_beta(qG[ik, :npw[ik], :], idlmn, eval_B, omega=omega, eps=1e-12) for ik in range(nkpt)]

    # Cache U_k per k so we don't have to recompute for each kp
    U_cache = {}
    for ik in kpt_id:
        npw_k = int(npw[ik]) # number of active planewaves for that k
        qG_k = qG[ik, :npw_k, :] # (npw_k, 3)
        Ck = C_nk[band_id, ik, :npw_k] # (nband, npw_k)
        U_cache[ik] = build_projector_overlaps(qG_k, tau, betas[ik], Ck)

    # Build blocks M_{k,k′}[n,m] = Σ_{i,s} D_i U_k[n,i,s] U_{k′}[m,i,s]^*
    M_ed = []
    for ik in kpt_id:
        Uk = U_cache[ik]
        for ikp in kpt_id:
            Ukp = U_cache[ikp]
            M = np.einsum('nis, i, mis->nm', Uk, D, Ukp.conj(), optimize=True)
            M_ed.append({"ik": int(ik), "ikp": int(ikp), "M": M})

    return M_ed