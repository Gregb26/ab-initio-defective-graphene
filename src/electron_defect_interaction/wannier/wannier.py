"""
wannier_interpolation.py
    Python module containing functions to interpolate objects computed on a coarse kpoint grid onto a fine kpoint grid
    via Maximally Localized Wannier Functions. 
"""

import numpy as np

def wannier_gauge_transform(Mb, U, U_dis=None):
    """
    Rotates an object Mb in Bloch gauge into Wannier gauge using the Wannier gauge matrix U (or V=U_dis @ U in the case of entangled bands)
    Inputs:
        Mb: (nband, nkpt, nband, nkpt) array of complex
            Object in Bloch gauge to rotate. This can be e.g. a Hamiltonian or a scattering matrix
        U: (nkpt, nwann, nwann) array of complex
            Wannier gauge matrix. If there are no entangled bands, nwann = nband
        U_dis: (nkpt, nband, nwann) array of complex
            Disentanglement matrix. Nband is the number of Bloch bands and nwann is the number of Wannier functions (bands to disentangle)
    Returns:
        Mw: (nwann, nkpt, nwann, nkpt)
            Object in the Wannier gauge: Mw = V^dag Mb V
    """

    nkpt, nwann, _ = U.shape
    # Entangled case, rotation matrix is V = U_dis @ U
    if U_dis is not None:
        nband = U_dis.shape[1]
        V = np.zeros((nkpt, nband, nwann), dtype=complex)
        for ik in range(nkpt):

            V[ik] = U_dis[ik] @ U[ik] # (nkpt, nwann, nwann)

            # testing
            with np.errstate(all='ignore'):
                P = V[ik] @ V[ik].conj().T
                assert np.allclose(P, P.conj().T, atol=1e-10)     
                assert np.allclose(P @ P, P, atol=1e-8)           
                assert np.allclose(np.trace(P), V.shape[-1], 1e-10)

    else:
        V = U # (nkpt, nband, nwann) with nwann = nband
    
    # Rotate Mb from Bloch gauge to Wannier gauge Mw
    Mw = np.zeros((nwann, nkpt, nwann, nkpt), dtype=complex)
    for ik in range(nkpt):
        Vk_h = V[ik].conj().T
        for ikp in range(nkpt):

            Mw[:, ik, :, ikp] = Vk_h @ Mb[:, ik, :, ikp] @ V[ikp] # (nwann, nkpt, nwann, nkpt)
    
    return Mw