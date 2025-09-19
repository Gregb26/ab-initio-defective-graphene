"""
local_old.py

    Module for computing local electron-defect interaction matrix elements using plane-wave DFT data. Old code, kept for reference. DO NOT USE, use local_sc.py instead.
"""

import numpy as np
from electron_defect_interaction.utils.fft_utils import *
from electron_defect_interaction.utils.pw_utils import *
from electron_defect_interaction.utils.lattice_utils import *
from electron_defect_interaction.utils.pot_utils import *

def build_coeff_cube(Cdict_band, ngfft, map1, map2, map3):
    """
    Build a 3D complex array (cube) of shape ngfft, placing the coefficients C_{nk}(G)
    at the correct FFT grid points as specified by the mapping arrays map1, map2, map3.
    Inputs:
    -------
        Cdict_band: dict mapping reduced integer coordinates of a G vector (h,k,l) to  a coefficient C_{nk}(G)
        ngfft: tuple of 3 ints, the FFT grid dimensions
        map1, map2, map3: arrays mapping reduced G indices to FFT grid indices
    
    Returns:
    --------
        cube: 3D complex array of shape ngfft, with C_{nk}(G) placed at the correct FFT grid points
    """ 
    cube = np.zeros(ngfft, dtype=np.complex128)

    for (h,k,l), c in Cdict_band.items():
        cube[map1[h], map2[k], map3[l]] = c

    return cube

def build_local_ed_matrix_for_k_kp(Cdicts_k, K_r, ngfft, map1, map2, map3, Cdicts_kprime=None, voxel_vol=1.0):
    # assemble cubes in G
    cubes_k  = [build_coeff_cube(d, ngfft, map1, map2, map3) for d in Cdicts_k]
    cubes_kp = cubes_k if Cdicts_kprime is None else \
               [build_coeff_cube(d, ngfft, map1, map2, map3) for d in Cdicts_kprime]

    # G → r
    psis_k   = [np.fft.ifftn(c)            for c in cubes_k ]
    psis_kp  = [np.fft.ifftn(c)            for c in cubes_kp]

    # apply local potential in real space
    accs_r   = [K_r * psi for psi in psis_kp]

    nb = len(psis_k)
    M  = np.empty((nb, nb), dtype=np.complex128)
    for n in range(nb):
        for m in range(nb):
            M[n, m] = voxel_vol * np.vdot(psis_k[n], accs_r[m])
    return M


def build_diagonal_local_ed_matrix(WFK, band_indices, ngfft, Omega_uc, V_ed_G, S):
    """
    Function that computes the local electron-defect interaction matrix blocks M_{nm}(k,k) for a given set of band indices at each k-point,
    using the defect potential in reciprocal space V_ed_G computed in a supercell and the integer matrix S relating supercell and unit cell lattice vectors.
    This function only computes the diagonal blocks (k=k').    

    Inputs:
    -------
        WFK: str:
            Path to the WFK.nc file.
        band_indices: list of ints:
            List of band indices to include in the output matrices.
        ngfft: tuple of 3 ints:
            The FFT grid dimensions.
        V_ed_G: [n1,n2,n3] array of complex:
            Defect potential in reciprocal space, computed in a supercell.
        S: [3,3] array:
            Integer matrix that relates the supercell lattice vectors to the unit cell lattice vectors.
            A_sc = S @ A_uc
    Returns: 
    --------
        results: list of tuples:
            list of (ik, M_block) for each k, where M_block has shape (nbands, nbands).
    """

    voxel_vol = Omega_uc / np.prod(ngfft)
    # Precompute index maps and kernel FFT once
    map1, map2, map3 = build_maps_from_ngfft(ngfft)
    K = build_kernel_uc_from_supercell(V_ed_G, S, ngfft)
    F_K = np.fft.fftn(K)

    C_nk, npw = get_C_nk(WFK)  # shape (nband, nkpt, npwmax) 
    G_red_uc = get_G(WFK)  # (nkpt, npwmax, 3)

    nkpt = G_red_uc.shape[0]
    results = []

    for ik in range(nkpt):
        # Build per-band dicts at this k
        Cdicts_k = make_Cdicts_for_k(C_nk, G_red_uc, npw, ik, band_indices)
        # Assemble the local matrix block at this k
        M_block = build_local_ed_matrix_for_k_kp(Cdicts_k, F_K, ngfft, map1, map2, map3, voxel_vol) # Cdicts_kprime=None: diagonal case only
        results.append((ik, M_block))
        
    return results

def build_general_local_ed_matrix(
    WFK_uc,                  # path to unit-cell WFK
    V_ed,              # isolated ΔV(r) on ngfft grid (Ha)
    ngfft,                   # (n1,n2,n3)
    A_uc, B_uc,              # direct & reciprocal (Bohr, 1/Bohr). B_uc satisfies A_uc.T @ B_uc = 2π I
    Omega_sc, Omega_uc,      # volumes (Bohr^3)
    band_indices,            # list of band indices, e.g. [ib_pi, ib_pistar]
    k_pairs=None,            # list of (ik, ikp). If None → all pairs with same set (cartesian Δk computed)
    cache_delta_k=True,      # reuse kernels for repeated Δk
    round_tol=12             # rounding tolerance for Δk caching in decimal places
):
    """
    Function that computes the local electron-defect interaction matrix blocks M_{nm}(k, k') for a given set of band indices at each k-point pair (k,k'),
    using the defect potential in real space V_ed(r) computed in a supercell and the unit cell lattice vectors A_uc, B_uc.
    This function can compute both diagonal (k=k') and off-diagonal (k≠k') blocks.

    Inputs:
    -------
        WFK_uc: str:
            Path to the WFK.nc file of the unit cell.
        V_ed: [n1,n2,n3] array of float:
            Real-space defect potential ΔV(r) on the supercell FFT grid (Hartree).
            If periodic images are undesired, apply windowing first.            
        ngfft: tuple of 3 ints:
            The FFT grid dimensions.
        A_uc: [3,3] array:
            Direct lattice vectors of the unit cell (Bohr).
        B_uc: [3,3] array:
            Reciprocal lattice vectors of the unit cell (1/Bohr).
            Satisfies A_uc.T @ B_uc = 2π I
        Omega_sc: float:
            Volume of the supercell (Bohr^3).
        Omega_uc: float:
            Volume of the primitive unit cell (Bohr^3).
        band_indices: list of ints:
            List of band indices to include in the output matrices.
        k_pairs: list of tuples (ik, ikp):
            List of k-point index pairs to include in the output matrices.
            If None, defaults to diagonal blocks only (ik, ik).
        cache_delta_k: bool:
            Whether to cache and reuse kernels for repeated Δk values.
        round_tol: int:
            Rounding tolerance in decimal places for Δk caching to avoid floating-point noise.      
    
    Returns:
    --------
        results: list of dicts:
            list of {"ik": ik, "ikp": ikp, "M": M_block} for each k-point pair, where M_block has shape (nbands, nbands).   
    """

    voxel_vol = Omega_uc / np.prod(ngfft)

    # Precompute index maps (mapping from reduced G indices to FFT grid indices) once
    map1, map2, map3 = build_maps_from_ngfft(ngfft)


    # Read plane-wave data once
    C_nk, npwarr = get_C_nk(WFK_uc) # (nband, nkpt, npwmax), (nkpt,)
    G_red_uc     = get_G(WFK_uc)    # (nkpt, npwmax, 3)
    kred         = get_kpt_red(WFK_uc)  # (nkpt,3), reduced coords
    nkpt         = kred.shape[0]

    # Choose kpoints to process
    # If none are given, default to diagonal blocks only
    if k_pairs is None:
        # by default, only diagonal blocks (ik,ik) — change to outer product if you want all pairs
        k_pairs = [(ik, ik) for ik in range(nkpt)]

    # Build per-k Cdicts once 
    Cdicts_by_k = {}
    for ik in sorted(set([ik for ik,_ in k_pairs] + [ikp for _,ikp in k_pairs])):
        Cdicts_by_k[ik] = make_Cdicts_for_k(C_nk, G_red_uc, npwarr, ik, band_indices)

    # Kernel cache over Δk to avoid recomputing identical kernels, to be implemented if needed
    kernel_cache = {}

    results = []
    for (ik, ikp) in k_pairs:
        # Δk in cartesian (1/Bohr): (kred_kp - kred_k) · B_uc
        delta_k_cart = (kred[ikp] - kred[ik]) @ B_uc   # shape (3,)

        # Cache key
        if cache_delta_k:
            key = tuple(np.round(delta_k_cart, round_tol))
        else:
            key = None

        if cache_delta_k and (key in kernel_cache):
            F_K_dk = kernel_cache[key]
        else:
            F_K_dk = build_kernel_uc_from_supercell_for_delta_k(V_ed, delta_k_cart, ngfft, A_uc, Omega_sc, Omega_uc)
            if cache_delta_k:
                kernel_cache[key] = F_K_dk

        # Build blocks
        Cdicts_k  = Cdicts_by_k[ik]
        Cdicts_kp = Cdicts_by_k[ikp]

        # Compute the matrix block M_{nm}(k,k')
        M_block = build_local_ed_matrix_for_k_kp(Cdicts_k, F_K_dk, ngfft, map1, map2, map3, voxel_vol, Cdicts_kprime=Cdicts_kp)
        results.append({"ik": ik, "ikp": ikp, "M": M_block})

    return results 