"""
local_sc.py

    Python module containing functions to compute local electron-defect interaction matrix elements using supercell potentials.

"""

import numpy as np
from electron_defect_interaction.utils.pw_utils import *
from electron_defect_interaction.utils.fft_utils import *

def compute_S_correlation(ik, ikp, band_n, band_m, G_red_uc, C_nk, npw, tol_S=1e-14): 
    """
    Compute plane wave overlap S(k,k',ΔG) = Σ_G C_{nk}(G) C_{n'k'}^*(G+ΔG), by using the convolution theorem.

    Inputs:
    -------
        ik: int:
            Index of the initial k-point.
        ikp: int: int:
            Index of the final k-point.
        bands: list of int:
            List of band indices to include in the output dictionaries.
        G_red_uc: [nkpt, npwmax, 3] array of ints:
             Reciprocal lattice vectors of the unit cell in reduced coordinates.
        C_nk: [nband, nkpt, npwmax] array:
            Array with the wavefunction coefficients. nband is the band index, npkt is the kpoint index and npwk is the coefficients index.
            For example C_nk[0,0, :] gives the npw_k coefficients of the wavefunction for the first band, at the first kpoint.
        npw: [nkpt] array of ints:
            Number of active planewaves for a given kpoint.

    Returns:
    --------
        S: [n1,n2,n3] array of complex:
            Overlap of planewaves on ΔG grid.
        offset_k: [3] array of int:
            Offset applied to k-point ik to place its G-vectors in the array A.
        offset_kp: [3] array of int:
            Offset applied to k-point ikp to place its G-vectors in the array A.
        A: [M,3] array of int:
            Indices where S is non-zero.
    """

    from collections import defaultdict
    S = defaultdict(complex)
 
    D_k = make_Cdicts_for_k(C_nk, G_red_uc, npw, ik, [band_n])[0]
    D_kp = make_Cdicts_for_k(C_nk, G_red_uc, npw, ikp, [band_m])[0]

    # Get tight bounding boxes
    gk = np.row_stack(np.array(list(D_k.keys()) ))
    gk_min = np.min(gk, axis=0); gk_max = np.max(gk, axis=0)    

    gkp = np.row_stack(np.array(list(D_kp.keys())))
    gkp_min = np.min(gkp, axis=0); gkp_max = np.max(gkp, axis=0)

    # Linear correlation size
    Nk = gk_max - gk_min + 1
    Nkp = gkp_max - gkp_min + 1
    Nconv = Nk + Nkp - 1
   
    # Place data with offsets
    offset_k = -gk_min
    offset_kp = -gkp_min

    # Padded FFT friendly size per axis
    primes = (2,3,5,7)
    Nfft = tuple(next_good_fft_len(int(n), primes=primes, force_odd=False) for n in Nconv)

    Ak = np.zeros(tuple(Nfft), dtype=np.complex128); Akp = np.zeros(tuple(Nfft), dtype=np.complex128)

    for key, C_nkG in D_k.items():
        idx = tuple(np.array(key) + offset_k)
        Ak[idx] += C_nkG

    for key, C_nkpG in D_kp.items():
        idx = tuple(np.array(key) + offset_kp)
        Akp[idx] += C_nkpG
    
    # FFTs
    A_k = np.fft.fftn(Ak, norm="forward").astype(np.complex128)
    A_kp = np.fft.fftn(Akp, norm="forward").astype(np.complex128)

    S = np.fft.ifftn(np.conj(A_kp) * A_k, norm="forward").astype(np.complex128)

    A = np.argwhere(np.abs(S) > tol_S)

    N = np.prod(S.shape)
    S = S * N  # undo FFT normalization

    return S, A, offset_k, offset_kp

def build_Q_uc(A, offset_k, offset_kp, q):
    """
    Compute the vectors Q = q + ΔG = k'-k + G'' - G in reduced coordinates of the unit cell.

    Inputs:
    -------
        A: [M,3] array of int:
            Indices where S is non-zero, in reduced coordinates.
        offset_k: [3] array of int:
            Offset applied to k-point ik to place its G-vectors in the array A.
        offset_kp: [3] array of int:
            Offset applied to k-point ikp to place its G-vectors in the array A.
        q: [3] array of float:
            Reduced coordinates of the vector k' - k in the unit cell.

    Returns:
    --------
        Q: [M,3] array of float:
            Vectors Q = q + ΔG in reduced coordinates of the unit cell.
        DeltaG: [M,3] array of int:
            The ΔG = G'' - G vectors in reduced coordinates of the unit cell.
    """
    
    Delta0 = (np.array(offset_kp) - np.array(offset_k)).astype(int)
    DeltaG = A - Delta0[None, :]

    Q = np.asarray(q, float)[None, :] + DeltaG

    return Q, DeltaG

def uc_to_sc_reduced(q_uc, N_diag):
    """
    Convert reduced coordinates in the unit cell to reduced coordinates in the supercell.

    Inputs:
    -------
        q_uc: [M,3] array of float:
            Reduced coordinates in the unit cell.
        N_diag: [3] array of int:
            Diagonal elements of the integer matrix that relates the supercell lattice vectors to the unit cell lattice vectors.
            A_sc = N @ A_uc.

    Returns:
    --------
        q_sc: [M,3] array of float:
            Reduced coordinates in the supercell.
    """

    N = np.diag(N_diag)
    q_sc = q_uc @ N.T

    return q_sc

def compute_signed_modes(q_red, ngfft, eps_ongrid=1e-12):
    """
    Map reduced q to signed integer FFT modes.
    """

    q_red = wrap_half_open(q_red)  # ensure in (-0.5, 0.5]

    n1, n2, n3 = ngfft
    u = q_red * np.array([n1, n2, n3], float)
    m = np.rint(u).astype(int)
    offgrid = np.any(np.abs(u - m) > eps_ongrid, axis=1)

    # Nyquist tie-break
    if n1 % 2 == 0:
        m[:, 0][np.isclose(u[:, 0], n1/2, atol=eps_ongrid)] = -n1 // 2
    if n2 % 2 == 0:
        m[:, 1][np.isclose(u[:, 1], n2/2, atol=eps_ongrid)] = -n2 // 2
    if n3 % 2 == 0:
        m[:, 2][np.isclose(u[:, 2], n3/2, atol=eps_ongrid)] = -n3 // 2

    return m, offgrid

def compute_local_M_sc(S, A, offset_k, offset_kp, k_red, ik, ikp,
                       V_sc_G, ngfft_sc, N_diag):
    """
    Compute the local electron-defect interaction matrix elements using the supercell potential.

    Inputs:
    -------
        S: [N1,N2,N3] array of complex:
            Overlap of planewaves on \\Delta_G grid used to build A.
        A: [M,3] array of int:
            Indices where S is non-zero, in reduced coordinates.
        offset_k: [3] array of int:
            Offset applied to k-point ik to place its G-vectors in the array A.
        offset_kp: [3] array of int:
            Offset applied to k-point ikp to place its G-vectors in the array A.
        k_red: [nkpt,3] array of float:
            Reduced coordinates of the k-points in the unit cell.
        ik: int:
            Index of the initial k-point.
        ikp: int:
            Index of the final k-point.
        V_sc_G: [n1,n2,n3] array of complex:
            Defect potential in reciprocal space, computed in a supercell.
        ngfft_sc: tuple of int:
            FFT grid size of the supercell.
        N_diag: [3] array of int:
            Diagonal elements of the integer matrix that relates the supercell lattice vectors to the unit cell lattice vectors.
            A_sc = N @ A_uc.
    """

    # Momentum transfer in reduced coordinates of the unit cell for k and k'
    q = k_red[ikp] - k_red[ik]
    # Build Q = q + ΔG and ΔG in reduced coordinates of the unit cell
    Q_uc, DeltaG = build_Q_uc(A, offset_k, offset_kp, q)

    # Convert to reduced coordinates of the supercell and wrap
    Q_sc = uc_to_sc_reduced(Q_uc, N_diag)
    Q_sc = wrap_half_open(Q_sc)

    # Map to signed integer FFT modes of the supercell
    m, offgrid = compute_signed_modes(Q_sc, ngfft_sc)

    # Get potential at these modes
    i1, i2, i3 = modes_to_fft_indices(m, ngfft_sc)
    Vq = V_sc_G[i1, i2, i3]

    # Pick out non-zero S and form matrix element
    s = S[A[:, 0], A[:, 1], A[:, 2]]
    M = np.sum(s * Vq)

    return M, offgrid

def compute_local_M_sc_all_pairs(bands, k_red, G_red_uc, C_nk, npw,
                      V_sc_G, ngfft_sc, N_diag,
                      tol_S=1e-14):
    """
    Build the full local e–defect matrix tensor:
        M[i_n, ik, i_m, ikp]  (complex128)

    bands    : list[int]    bands to include (e.g. [4,5])
    k_red    : (nkpt,3)     k-points in UC reduced coords
    G_red_uc : (...)        UC G-vectors per k (as in your code)
    C_nk     : (nbnd,nkpt, npwmax) PW coeffs
    npw      : (nkpt,)      number of active PWs per k
    V_sc_G   : (n1,n2,n3)   supercell FFT of ΔV (with your (Ω_sc/Ω_uc) factor)
    ngfft_sc : tuple[int]   (n1,n2,n3)
    N_diag   : (3,) ints    supercell multiples along a1,a2,a3
    tol_S    : float        support threshold for S
    use_uc_norm : bool      if True, multiply by N=prod(N_diag)

    Returns
    -------
    M : (Nb, Nk, Nb, Nk) complex128
    """

    Nb = len(bands)
    Nk = k_red.shape[0]
    M = np.zeros((Nb, Nk, Nb, Nk), dtype=np.complex128)

    for i_n, n in enumerate(bands):
        for ik in range(Nk):
            for i_m, m in enumerate(bands):
                for ikp in range(Nk):

                    # Compute pair-specific plane wave overlap S
                    S, A, offset_k, offset_kp = compute_S_correlation(
                        ik, ikp, n, m, G_red_uc, C_nk, npw, tol_S=tol_S)
                    
                    # Compute Q = q + ΔG = k'-k+ G'' - G in reduced coords of the unit cell
                    q = k_red[ikp] - k_red[ik]
                    Q_uc, DeltaG = build_Q_uc(A, offset_k, offset_kp, q)

                    # Convert to reduced coordinates of the supercell and wrap + index of supercell FFT grid
                    Q_sc = uc_to_sc_reduced(Q_uc, N_diag)
    
                    modes, offgrid = compute_signed_modes(Q_sc, ngfft_sc, eps_ongrid=1e-12)

                    i1, i2, i3 = modes_to_fft_indices(modes, ngfft_sc)


                    if offgrid.any():
                        print(f"Warning: some Q points off-grid for bands {n},{m} and k-points {ik},{ikp}")
                        Vq = Vq.copy()
                        Vq[offgrid] = 0.0

                    # Get potential at this mode
                    Vq = V_sc_G[i1, i2, i3]

                    # Pick out non-zero S and form matrix element
                    s_vals = S[A[:, 0], A[:, 1], A[:, 2]]
                    M_val = np.sum(s_vals * Vq)
                    M[i_n, ik, i_m, ikp] = M_val

    return M