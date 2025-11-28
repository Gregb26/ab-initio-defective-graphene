"""
wannier_interpolation.py
    Python module containing functions to interpolate objects computed on a coarse kpoint grid onto a fine kpoint grid
    via Maximally Localized Wannier Functions. 
"""

import numpy as np

def wannier_gauge_transform(Mbk, U, U_dis=None):
    """
    Rotates an object Mb in Bloch gauge into Wannier gauge using the Wannier gauge matrix U (or V=U_dis @ U in the case of entangled bands)
    Inputs:
        Mb: (nband, nkpt, nband, nkpt) array of complex
            Object in Bloch gauge to rotate. This can be e.g. a Hamiltonian or a scattering matrix
        U: (nkpt, nband, nwann) or (nkpt, nwann, nwann) array of complex
            Wannier gauge matrix. First shape if no entangled bands (U_dis is None). Second shape if entangled bands (U_dis is not None).
        U_dis: (nkpt, nband, nwann) array of complex
            Disentanglement matrix. Nband is the number of Bloch bands and nwann is the number of Wannier functions (bands to disentangle)
    Returns:
        Mwk: (nwann, nkpt, nwann, nkpt)
            Object in the Wannier gauge: Mw = V^dag Mb V
    """

    nkpt, nwann, _ = U.shape
    # Entangled case, rotation matrix is V = U_dis @ U
    if U_dis is not None:
        nband = U_dis.shape[1]
        V = np.zeros((nkpt, nband, nwann), dtype=complex)
        for ik in range(nkpt):

            V[ik] = U_dis[ik] @ U[ik] # (nkpt, nb, nw)

            # testing
            with np.errstate(all='ignore'):
                P = V[ik] @ V[ik].conj().T
                assert np.allclose(P, P.conj().T, atol=1e-10)     
                assert np.allclose(P @ P, P, atol=1e-8)           
                assert np.allclose(np.trace(P), V.shape[-1], 1e-10)

    else:
        V = U # (nkpt, nband, nwann) with nwann = nband
    
    # Rotate Mb from Bloch gauge to Wannier gauge Mw
    Mwk = np.zeros((nwann, nkpt, nwann, nkpt), dtype=complex)
    for ik in range(nkpt):
        Vk_h = V[ik].conj().T # Hermitian conjugate
        for ikp in range(nkpt):

            Mwk[:, ik, :, ikp] = Vk_h @ Mbk[:, ik, :, ikp] @ V[ikp] # (nwann, nkpt, nwann, nkpt)
    
    return Mwk

def reciprocal_to_real_double_FT(Mwk, k_red, MP_grid):
    """
    Transforms the object Mw from reciprocal space to real space using a double Fourier transform.
    Inputs:
        Mw: (nw, nk, nw, nk) array of complex
            Object to transform to real space, nk is the number of kpoints.
        MP_grid: tuple of ints
            Monkhorst-Pack grid used to define the kpoint grid. Assuming no symmetry reduction has been applied! This is important! 
    Returns
        Mwr: (nw, nR, nw, nR) array of complex
            Fourier transform of Mk, nR is the number of R, inferred from the Monkhorst-Pack grid.
    """

    nk = Mwk.shape[1]

    # Build the grid in real space based on the kpoint grid
    N1, N2, N3 = MP_grid
    r1 = np.arange(N1); r2 = np.arange(N2); r3 = np.arange(N3)
    rr1, rr2, rr3 = np.meshgrid(r1, r2, r3)
    R = np.stack((rr1, rr2, rr3), axis=-1) # (N1, N2, N3, 3)
    R = R.reshape(N1*N2*N3, 3) # (nR, 3)

    # Compute phase
    phase_kp = np.exp(2j*np.pi * (k_red @ R.T))
    phase_k = phase_kp.conj()

    # Double sum over k and k'
    Mwr = np.einsum('kr, nkNK, KR -> nrNR', phase_kp, Mwk, phase_k, optimize=True)  / nk **2 # (n, nR, n, nR)

    return Mwr, R