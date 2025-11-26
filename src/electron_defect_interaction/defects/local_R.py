"""
local_R.py
    Python module to compute the local part of the electron-defect scattering matrix by directly evaluating the 
    integral in real space. Surprisingly fast.
"""

import numpy as np

from electron_defect_interaction.io.abinit_io import *
from electron_defect_interaction.wavefunctions.fold_wfk_to_sc import compute_psi_nk_fold_sc
from tqdm import tqdm

def compute_M_L_r(uc_wfk_path, sc_wfk_path, sc_p_pot_path, sc_d_pot_path, bands, subtract_mean=True):
    """
    Computes the local part of the electron-defect interaction matrix in real space.

    Inputs:
        uc_wfk_path: str
            Path to the ABINIT WFK.nc output file for the wavefunctions of the unit cell
        sc_wfk_path:
            Path to the ABINIT WFK.nc output file for the wavefucntions of the pristine super cell
        sc_p_pot_path: str
            Path to the ABINIT POT.nc output file for the local potential of the pristine super cell
        sc_d_pot_path: str
            Path to the ABINIT POT.nc output file for the local potential of the defective super cell
        bands: list of ints:
            Band indices at which to compute the matrix elements
    """

    # Get necessary unit cells quantities
    C_nkg, nG = get_C_nk(uc_wfk_path) # planewave coeffs (nband, nkpt, nG_max) and number of active G per k (nkpt, )
    G_red = get_G_red(uc_wfk_path) # reciprocal lattice vectors in reduced coords of unit cell (nkpt, nG_max, 3)
    k_red = get_k_red(uc_wfk_path) # kpoints in reduced coords of unit cell (nkpt, 3)
    A_uc, _ = get_A_volume(uc_wfk_path) # primitive lattice vectors of the unit cell A[:, i]=a_i

    C_nkg = C_nkg[bands, ...]
    nband, nkpt, _ = C_nkg.shape

    # Get necessary super cell quantities
    A_sc, Omega_sc = get_A_volume(sc_wfk_path) # primitive lattice vectors and cell volume of the supercell
    Vp, _ = get_pot(sc_p_pot_path, subtract_mean); Vp = Vp.transpose(2,1,0) # pristine supercell local potential
    Vd, _ = get_pot(sc_d_pot_path, subtract_mean); Vd = Vd.transpose(2,1,0) # defective supercell local potential
    ngfft = Vp.shape # FFT grid shape

    # Compute defect potential
    Ved = Vd - Vp

    # Supercell scaling factor
    Ndiag = tuple(np.diag(np.rint(A_sc @ np.linalg.pinv(A_uc))))

    # Compute unict wavefunctions unfolded onto supercell from unit cell planewave coefficients
    print('Computing wavefunctions')
    psi = compute_psi_nk_fold_sc(C_nkg, nG, G_red, k_red, Omega_sc, Ndiag, ngfft, bands)

    dV = Omega_sc / np.prod(ngfft)

    M_L = np.zeros((nband, nkpt, nband, nkpt), dtype=complex)

    total = (nband * nkpt) ** 2
    next_mark = 5
    
    print('Computing local matrix elements')
    count = 0
    for ib in range(nband):
        for ik in range(nkpt):
            for ibp in range(nband):
                for ikp in range(nkpt):
                    psi_nk = psi[ib, ik, ...]
                    psi_npkp = psi[ibp, ikp, ...]

                    M_L[ibp, ikp, ib, ik] = np.vdot(psi_npkp, Ved * psi_nk) * dV

                    count+=1
                    p = (count) / total * 100
                    if p >= next_mark:
                        print(f"Computed {next_mark:.0f}% ({count}/{total} matrix elements)")
                        next_mark += 5

    return M_L

def compute_M_L_r_BLAAS(uc_wfk_path, sc_wfk_path, sc_p_pot_path, sc_d_pot_path, subtract_mean=True, pristine=False):
    """
    Computes the local part of the electron-defect interaction matrix in real space.

    Inputs:
        uc_wfk_path: str
            Path to the ABINIT WFK.nc output file for the wavefunctions of the unit cell
        sc_wfk_path:
            Path to the ABINIT WFK.nc output file for the wavefunctions of the pristine super cell
        sc_p_pot_path: str
            Path to the ABINIT POT.nc output file for the local potential of the pristine super cell
        sc_d_pot_path: str
            Path to the ABINIT POT.nc output file for the local potential of the defective super cell
        bands: list of ints:
            Band indices at which to compute the matrix elements
    """

    # Get necessary unit cells quantities
    C_nkg, nG = get_C_nk(uc_wfk_path) # planewave coeffs (nband, nkpt, nG_max) and number of active G per k (nkpt, )
    G_red = get_G_red(uc_wfk_path) # reciprocal lattice vectors in reduced coords of unit cell (nkpt, nG_max, 3)
    k_red = get_k_red(uc_wfk_path) # kpoints in reduced coords of unit cell (nkpt, 3)
    A_uc, _ = get_A_volume(uc_wfk_path) # primitive lattice vectors of the unit cell A[:, i]=a_i
    nband, nkpt, _ = C_nkg.shape

    # Get necessary super cell quantities
    A_sc, Omega_sc = get_A_volume(sc_wfk_path) # primitive lattice vectors and cell volume of the supercell
    Vp, _ = get_pot(sc_p_pot_path, subtract_mean); Vp = Vp.transpose(2,1,0) # pristine supercell local potential
    Vd, _ = get_pot(sc_d_pot_path, subtract_mean); Vd = Vd.transpose(2,1,0) # defective supercell local potential
    ngfft = Vp.shape # FFT grid shape

    # Compute defect potential
    if pristine:
        Ved = Vp
    else:
        Ved = Vd - Vp

    # Supercell scaling factor
    Ndiag = tuple(np.diag(np.rint(A_sc @ np.linalg.pinv(A_uc))))

    # Compute unict wavefunctions unfolded onto supercell from unit cell planewave coefficients
    print('Computing wavefunctions')
    psi = compute_psi_nk_fold_sc(C_nkg, nG, G_red, k_red, Omega_sc, Ndiag, ngfft)

    R = np.prod(ngfft)

    psi_r = psi.reshape(nband, nkpt, R) # (nband, nkpt, R)
    Ved_r = Ved.reshape(R) # (R,)
    Bk = nband * nkpt
    dV = Omega_sc / np.prod(ngfft)
    Psi = np.ascontiguousarray(psi_r.reshape(Bk, R)) # (Bk, R)
 
    def accumulate_M(Psi, Ved_r, dV, chunk=100_000):
        # Psi: (BK, R) complex128; Ved_r: (R,) real/complex
        BK, R = Psi.shape
        M = np.zeros((BK, BK), dtype=np.complex128)

        with tqdm(total=R, unit="r", desc="Accumulating M") as pbar:
            for start in range(0, R, chunk):
                sl = slice(start, min(start + chunk, R))
                PsiB = Psi[:, sl]                              # (BK, r)
                M += dV * ((PsiB.conj() * Ved_r[sl]) @ PsiB.T) # (BK, BK)
                pbar.update(sl.stop - sl.start)
        return M
    with np.errstate(all='ignore'):
        M_L = accumulate_M(Psi, Ved_r, dV)
    print('Done!')
    
    return M_L.reshape(nband, nkpt, nband, nkpt).transpose(2,1,0,3)