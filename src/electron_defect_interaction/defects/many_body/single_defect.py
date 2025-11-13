"""
single_defect.py
    Python modules containing functions to compute various many body quantities in the single defect limit.
"""

import numpy as np

# Non interacting Green's function
def compute_G0(eps, eigs, eta=1e-6):
    """
    Computes the unperturbed Green's function G^{(0)}_{n\bm{k}}(\varepsilon) = [\varepsilon - \varepsilon_{n\bm{k}}+i\\eta]^{-1}

    Inputs:
        eps: float or array of floats (Ne,)
            energy at which to evaluate the Green's function
        eigs: (nband, nkpt) array of floats
            eigenvalues of the unperturbed system (prisitine unit cell)
        eta: float
            Broadening factor. \\eta -> 0^+ for retarded Green's function
    Returns:
        G0: (Ne, nband, nkpt) array of complex
            Unperturbed Green's function for each band and kpoint.
    """

    eps = np.array(eps) # (Ne, )

    G0 = 1.0 / (eps[..., np.newaxis, np.newaxis] - eigs[np.newaxis, ...]+ 1j*eta) # (Ne, nband, nkpt)

    return G0

# T-matrix
def compute_T(eps, eigs, M, eta=1e-6):
    """
    Computes the T matrix: \bm{T}(\varepsilon)= [\bm{1}-\bm{M}\bm{G}^{(0)}(\varepsilon)]^{-1}\bm{M}
    Inputs:
        eps: float or array of floats (Ne,)
            Energy at which to evaluate the T matrix
        eigs: (nband, nkpt) array of floats
            Eigenvalues of the unperturbed system (pristine unit cell)
        M: (nband, nkpt, nband, nkpt) array of complex
            Electron-defect scattering matrix
        eta: float
            Broadening factor: eta -> 0^+ gives the retarded Green's function
    Returns: 
        T (Ne, nband, nkpt, nband, nkpt) array of complex
            T-matrix for all bands and kpoints.
    """

    nband, nkpt = eigs.shape
    N = nband * nkpt

    G0 = compute_G0(eps, eigs, eta) # (nband, nkpt)
    Ne = G0.shape[0]

    # MG0[e,n,k,n',k'] = M[n,k,n',k'] * G0[ne, n',k']
    MG0 = M[np.newaxis, ...] * G0[:,np.newaxis, np.newaxis, :, :] # (Ne, nband, nkpt, nband, nkpt)

    M_mat = M.reshape(N,N) # (N, N)
    M_mat = M_mat[np.newaxis, ...]  # (1, N, N) broadcast over NE
    MG0_mat = MG0.reshape(Ne,N,N) # (Ne, N,N)

    I = np.eye(N, dtype=complex) # (N,N)
    I = I[np.newaxis, ...] # (1, N, N) broadcast over Ne

    # Solve (I-MG0)T = M for each energy
    T_mat = np.linalg.solve(I-MG0_mat, M_mat) # (Ne,N,N)
    T = T_mat.reshape(Ne, nband, nkpt, nband, nkpt) # (Ne, nband, nkpt, nband, nkpt)

    return T

# Interacting Green's function
def compute_G(eps, eigs, M, eta=1e-6):
    """
    Computes the interacting Green's function G_{\bm{kk}'}(\varepsilon) = G_{\bm{k}}^{(0)}(\varepsilon)\\delta_{\bm{kk}'}+G^{(0)}_{\bm{k}}(\varepsilon)T_{\bm{kk}'}(\varepsilon)G^{(0)}_{\bm{k}'}(\varepsilon)
    for a single defect, using the T-matrix formalism.
    Inputs
        eps: float or array of floats
            Energy at which to evaluate the interacting Green's function
        eigs: (nband, nkpt) array of floats
            Eigenvalues of the unperturbed system (pristine unit cell)
        M: (nband, nkpt, nband, nkpt) array of complex
            Electron-defect scattering matrix
        eta: float
            Broadening factor: eta -> 0^+ gives the retarded Green's function
    Returns: G (Ne, nband, nkpt, nband, nkpt) array of complex
        Interacting Green's function for a single defect for all bands and kpoints
    """
    nband, nkpt = eigs.shape
    N = nband * nkpt

    G0 = compute_G0(eps, eigs, eta) # (Ne, nband, nkpt)
    Ne = G0.shape[0]

    # Build diagonal G0 as (Ne, N, N)
    G0_flat = G0.reshape((Ne, N))
    I = np.eye(N, dtype=complex) # (N,N)
    G0_mat = G0_flat[..., np.newaxis] * I[np.newaxis, ...] # (Ne, N, N)

    T = compute_T(eps, eigs, M, eta) # (Ne, nband, nkpt, nband, nkpt)
    T_mat = T.reshape(Ne, N,N)

    G_mat = G0_mat + G0_mat @ T_mat @ G0_mat # (Ne, N, N)
    G = G_mat.reshape(Ne, nband, nkpt, nband, nkpt)

    return G 
