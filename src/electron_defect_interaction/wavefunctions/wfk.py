"""
wfk.py
    Python module that rebuilds the wavefunction from the planewave coefficients on a uniform grid in real space.
    Automatically chooses the correct FFT grid given the maximum number of planewaves, following Abinit's convention.
"""

import numpy as np
from electron_defect_interaction.utils.fft_utils import map_G_to_fft_grid, fft_grid_from_G_red

def compute_psi_nk(
    C_nkg,  
    nG,       
    G_red,       
    k_red,  
    Omega,     
):
    """
    Computes the wavefunction psi_{nk}(r) on a uniform grid in real space consistent with the number of G vectors, from
    the planewave coefficients C_{nk}(G). 

    Inputs:
        C_nkg: (nband. nkpt, nG_max) array of complex:
            Planewave coefficients of the wavefunctions, indexed by the reciprocal lattice vectors G.
        nG: (nkpt, ) array of ints
            Number of active G vectors for each kpoint.
        G_red: (nkpt, nG_max, 3) array of ints
            Reciprocal lattice vectors in reduced coordinates written in the "signed integer FFT convention" 
            i.e. -Ni/2 <= G_i < N_i/2 for a grid with N_i points in the i'th direction.
        k_red: (nkpt, 3) array of floats
            k-points in reduced coordinates
        Omega: float
            Cell volume
        
    Returns:
        psi_nk: (nband, nkpt, Nx, Ny, Nz) array of complex
            Wavefunctions for all bands and all kpoints in a uniform grid in real space with N=NxNyNz points. The grid shape
            is determined according to Abinit's double grid convention. It matches ngfft as computed by Abinit.
        ngfft: tuple of ints:
            FFT grid shape
    """
    
    nband, nkpt, _ = C_nkg.shape

    # Build the FFT grid from the G vectors using Abinit's  convention (double grid)
    ngfft = fft_grid_from_G_red(G_red, nG)
    Nx, Ny, Nz = ngfft; N = np.prod(ngfft)

    # Build FFT grid
    x = np.arange(Nx)/Nx; y = np.arange(Ny)/Ny; z = np.arange(Nz)/Nz
    xx = x[:,None, None]; yy = y[None, :, None]; zz = z[None, None, :]

    # Build mapping from G_red = (Gx, Gy, Gz) to FFT grid index (jx, jy, jz) to place coefficients on FFT grid
    map_dict_x, map_dict_y, map_dict_z = map_G_to_fft_grid(ngfft)

    # Precompute phase per k
    phase = []
    for ik in range(nkpt):
        k = k_red[ik]
        phase_k = np.exp(1j * 2*np.pi*(k[0]*xx + k[1]*yy + k[2]*zz))
        phase.append(phase_k)
    
    # Compute the wavefunctions per k
    psi = np.zeros((nband, nkpt, Nx, Ny, Nz), dtype=complex)
    
    percent_marks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    next_mark_i = 0

    for ik in range(nkpt):
        nG_k = nG[ik] # number of active planewaves for this k
        Gk = G_red[ik, :nG_k, :] # (npw_k, 3) reciprocal lattice vectors for this k

        # Use mapping dictionaries to get FFT grid indices
        ix = np.array([map_dict_x[int(Gk_i)] for Gk_i in Gk[:, 0]])
        iy = np.array([map_dict_y[int(Gk_i)] for Gk_i in Gk[:, 1]])
        iz = np.array([map_dict_z[int(Gk_i)] for Gk_i in Gk[:, 2]])

        C_ng = C_nkg[:, ik, :nG_k] # (nband, npw_k) planewave coefficients for all bands at this k

        # Place coefficients on the FFT grid
        C_grid = np.zeros((nband, Nx, Ny, Nz), dtype=complex)
        C_grid[:, ix, iy, iz] = C_ng

        # Compute Bloch-periodic part u_{nk}(r_red) = \sum_G C_{nk}(G)e^{iGr} = F^{-1}[C_nk]*N on the grid
        u = np.fft.ifftn(C_grid, axes=(1,2,3)) * N # (nband, Nx, Ny, Nz), N undoes the normalization introduced by Numpys IFFT
        
        # Compute psi = u * exp(ik.r) / sqrt(Omega)
        psi[:, ik, ...] = (u * phase[ik]) / np.sqrt(Omega)

        p = (ik + 1) / nkpt * 100
        if next_mark_i < len(percent_marks) and p >= percent_marks[next_mark_i]:
            print(f"Computed {percent_marks[next_mark_i]}% of wavefunctions, ({ik+1}/{nkpt}) kpoints")
            next_mark_i += 1

    # Sanity check, wavefunctions are nornmalized
    norm = np.sum(np.abs(psi)**2, axis=(2,3,4)) * (Omega / N)
    assert np.allclose(norm, 1.0), 'Wavefunctions should be normalized!'

    print('Done! Wavefunctions are normalized.')

    return psi, ngfft

