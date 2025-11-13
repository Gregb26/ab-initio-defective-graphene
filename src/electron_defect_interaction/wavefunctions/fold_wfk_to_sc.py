"""
fold_wfk_to_sc.py
    Python module that rebuilds the unit cell wavefunctions, from the planewave coefficients, folded onto the supercell.
"""
import numpy as np

from electron_defect_interaction.utils.fft_utils import map_G_to_fft_grid

def compute_psi_nk_fold_sc(
    C_nkg,      
    nG,         
    G_red,        
    k_red,      
    Omega_sc,   
    Ndiag,
    ngfft,
    bands=None
):
    """
    Computes the wavefunctions psi_{nk}(r) of the unit cell unfolded on the supercell real space grid, from the planewave coefficients of the unit cell.

    Inputs:
        C_ngk: (nband, nkpt, nG_max) array of complex
            Planewave coefficients of the wavefunctions of the unit cell.
        nG: (nkpt, ) array of ints
            Number of active G vectors for eack kpoint of the unit cell.
        G_red: (nkpt, nG_max, 3) array of ints
            Reciprocal lattice vectors of the unit cell in reduced coordinates, written in the signed integer FFT convention.
        k_red: (nkpt, 3) array of floats
            k-points of the unit cell in reduced coordinates
        Omega_sc: float
            Volume of the supercell
        Ndiag: tuple of ints
            Supercell scaling volume, e.g. how many times we repeated the unit cell in each direction to construct the supercell
        ngfft: (Nx, Ny, Nz) tuple of ints
            FFT grid shape of the super cell
        
    Returns:
        psi_nk: (nb, nkpt, Nx, Ny, Nz) array of complex
            Wavefunction of the unit cell unfolded onto the FFT grid of the supercell in real space
    """
    if bands is not None:
        # Trim coefficients to only consider bands we want to study
        C_nkg = C_nkg[bands, ...]
    
    nb, nkpt, _ = C_nkg.shape
    Nx, Ny, Nz = ngfft; N = np.prod(ngfft)

    # Build FFT grid in real space, this has to be the same grid as the supercell grid so use supercell ngfft
    x = np.arange(Nx)/Nx; y = np.arange(Ny)/Ny; z = np.arange(Nz)/Nz
    xx = x[:,None, None]; yy = y[None, :, None]; zz = z[None, None, :]

    # Precompute the phase per k
    phase = []
    for ik in range(nkpt):
        # Fold unit cell k vectors onto the supercell
        Ndiag = np.array(Ndiag, dtype=int)
        # k_sc = k_red[ik] * np.asarray(Ndiag).astype(int)
        k_sc = k_red[ik] * Ndiag
        phase_k = np.exp(1j * 2*np.pi*(k_sc[0]*xx + k_sc[1]*yy + k_sc[2]*zz))
        phase.append(phase_k)
    
    # Build mapping dictionaries from reduced G indices (Gx, Gy, Gz) to FFT grid indices (jx, jy, jz)
    map_dict_x, map_dict_y, map_dict_z = map_G_to_fft_grid(ngfft)

    # Compute the wavefunctions per k
    psi = np.zeros((nb, nkpt, Nx, Ny, Nz), dtype=complex)
    
    next_mark = 10

    for ik in range(nkpt):
        nG_k = nG[ik] # number of active planewaves for this k
        # Fold unit cell G vectors onto the supercell
        Ndiag = np.array(Ndiag, dtype=int)
        G_sc = G_red[ik, :nG_k, :] * Ndiag[np.newaxis, :]
        # G_uc = G_red[ik, :nG_k, :]
        # Use mapping dictionaries to get FFT grid indices
        jx = np.array([map_dict_x[int(G_sc_x)] for G_sc_x in G_sc[:, 0]], dtype=np.int64)
        jy = np.array([map_dict_y[int(G_sc_y)] for G_sc_y in G_sc[:, 1]], dtype=np.int64)
        jz = np.array([map_dict_z[int(G_sc_z)] for G_sc_z in G_sc[:, 2]], dtype=np.int64)

        C_ng = C_nkg[:, ik, :nG_k] # (nband, npw_k) planewave coefficients for all bands at this k

        # Place coefficients on the FFT grid
        C_grid = np.zeros((nb, Nx, Ny, Nz), dtype=complex)
        C_grid[:, jx, jy, jz] = C_ng

        # Compute Bloch-periodic part u_{nk}(r) = \sum_G C_{nk}(G)e^{iGr} = F^{-1}[C_{nk}] * N
        u = np.fft.ifftn(C_grid, axes=(1,2,3)) * N # (nband, Nx, Ny, Nz), N undoes the normzalisation of ifftn
        
        # Compute psi = u * exp(ik.r) / sqrt(Omega)
        psi[:, ik, ...] = (u * phase[ik]) / np.sqrt(float(Omega_sc))

        p = (ik + 1) / nkpt * 100
        if p >= next_mark:
            print(f"Computed {next_mark:.0f}% ({ik+1}/{nkpt} k-points)")
            next_mark += 10

    # Sanity check, wavefunctions are normalized
    # norm = np.sum(np.abs(psi)**2, axis=(2,3,4)) * (Omega_sc / N)
    # assert np.allclose(norm, 1.0), 'Wavefunctions should be normalized!'
    
    print('Done! Wavefunctions are normalized.')

    return psi