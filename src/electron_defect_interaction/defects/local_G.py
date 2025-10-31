"""
local_G.py
    Python module used to compute the local part of the electron-defect scattering matrix in reciprocal space. 
    Quite a bit slow due to the number of FFTs to do ...
"""

from electron_defect_interaction.io.abinit_io import *
from electron_defect_interaction.utils.fft_utils import map_G_to_fft_grid

def compute_M_L_G(uc_wfk_path, sc_wfk_path, sc_p_pot_path, sc_d_pot_path, bands, subtract_mean=True):
    """
    Computes the local part of the electron-defect interaction matrix in reciprocal space

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

    # Get necessary unit cell quantities
    C_nkg, nG = get_C_nk(uc_wfk_path) # planewave coefficients (nband, nkpt, nG_max) and active G per k (nkpt,)
    G_red = get_G_red(uc_wfk_path) # reciprocal lattice vectors in reduced coords of the unit cell (nkpt, nG_max, 3)
    k_red = get_kpt_red(uc_wfk_path) # kpoints in reduced coords of the unit cell (nkpt, 3)
    A_uc, _ = get_A_volume(uc_wfk_path) # primitive lattice vectors of the unit cell A[:, i] = a_i
    _, nkpt, _ = C_nkg.shape

    nband = len(bands)

    # Get necessary super cell quantities
    A_sc, Omega_sc = get_A_volume(sc_wfk_path) # primite lattice vectors and volume of supercell

    Vp, _ = get_pot(sc_p_pot_path, subtract_mean); Vp = Vp.transpose(2,1,0) # pristine local potential
    Vd, _ = get_pot(sc_d_pot_path, subtract_mean); Vd = Vd.transpose(2,1,0) # defective local potential
    ngfft = Vp.shape # FFT grid shape

    # Embbed unit cell coefficients on supercell FFT grid
    map_dict_x, map_dict_y, map_dict_z = map_G_to_fft_grid(ngfft)
    Nx, Ny, Nz = ngfft
    Ndiag = tuple(np.diag(np.rint(A_sc @ np.linalg.pinv(A_uc))))
    Ndiag = np.array(Ndiag, dtype=int)
    u_nk = np.zeros((nband, nkpt, Nx, Ny, Nz), dtype=complex)

    for ik in range(nkpt):
        nG_k = nG[ik]
        Gk_sc = G_red[ik, :nG_k, :] * Ndiag[np.newaxis, :]

        # Map G vectors to FFT grid indices using mapping computed above
        jx = np.array([map_dict_x[int(G_sc_x)] for G_sc_x in Gk_sc[:, 0]], dtype=np.int64)
        jy = np.array([map_dict_y[int(G_sc_y)] for G_sc_y in Gk_sc[:, 1]], dtype=np.int64)
        jz = np.array([map_dict_z[int(G_sc_z)] for G_sc_z in Gk_sc[:, 2]], dtype=np.int64)

        # Place coefficients on grid
        C_ng = C_nkg[:, ik, :nG_k]
        C_grid = np.zeros((nband, Nx, Ny, Nz), dtype=complex)
        C_grid[:, jx, jy, jz] = C_ng

        # IFFT to get u_nk
        u_nk[:, ik, ...] = np.fft.ifftn(C_grid, axes=(1,2,3))
        
    def compute_S(u_nk, ibp, ikp, ib, ik):
        """
        Computes S(G) = sum_G' C_n'k'^*(G+G')C_nk(G') by doing a cross-correlation
        """
        return np.fft.fftn(np.conj(u_nk[ibp, ikp]) * u_nk[ib, ik])

    Vp_G = np.fft.fftn(Vp)
    M = np.zeros((nband, nkpt, nband, nkpt), dtype=complex)

    for ikp, kp in enumerate(k_red):
        for ibp in range(nband):
            for ik, k in enumerate(k_red):
                for ib in range(nband):

                    # Compute shift
                    Delta_k = kp - k
                    Delta_k_sc = Delta_k * Ndiag
                    m = np.rint(Delta_k_sc).astype(int)

                    Vshift = np.roll(np.roll(np.roll(Vp_G, m[0], 0), m[1], 1), m[2], 2)

                    M[ibp, ikp, ib, ik] = np.sum(Vshift * compute_S(u_nk, ibp, ikp, ib, ik))