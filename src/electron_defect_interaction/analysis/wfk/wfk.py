"""
wfk.py

    Python module that contains functions to compute the wavefunctions on a uniform grid from the planewave coefficients.
    
"""

import numpy as np
from electron_defect_interaction.utils.fft_utils import *

def compute_psi_nk(
    C_nk,      # (nband, nkpt, npw_max) complex
    npw,       # (nkpt,) int
    G_red,     # (nkpt, npw_max, 3) ints
    kpts_red,  # (nkpt, 3) floats (reduced)
    Omega,     # float, cell volume (Bohr^3)
    Nx=None, Ny=None, Nz=None,      # FFT grid; if None, infer via grid_from_gmax
    *, complex64=False, primes=(2,3,5,7)
):
    """
    Computes psi[n,k,x,y,z] on a uniform grid in [0,1)^3 for all bands and k.
    Uses your build_maps_from_ngfft (index maps) and grid_from_gmax (FFT-friendly sizes).
    Vectorized over bands per k.
    """
    
    nband, nkpt, _, = C_nk.shape
    dtype = np.complex64 if complex64 else np.complex128

    # Build the FFT grid if not provided
    if Nx is None or Ny is None or Nz is None:

        # Get the maximum G-vector in each direction, over all kpoints, for active planewaves
        Gmax = np.array([0,0,0], dtype=int)

        for ik in range(nkpt):
            npw_k = npw[ik] # number of active planewaves for this k
            Gk = np.asarray(G_red[ik, :npw_k, :], dtype=int) # (npw_k, 3) reciprocal lattice vectors for this k
            Gmax = np.maximum(Gmax, np.max(np.abs(Gk), axis=0)) # maximum reciprocal lattice vector over all k
        
        Nx, Ny, Nz = grid_from_Gmax(Gmax, primes=primes) # FFT-friendly grid sizes

    ngfft = (Nx, Ny, Nz)
    N = np.prod(ngfft)

    # Build mapping dictionaries from reduced G indices (h,k,l) to FFT grid indices
    map1, map2, map3 = build_maps_from_ngfft(ngfft)

    # Precompute fraction grids and per-k Bloch phases
    x = np.arange(Nx)/Nx; y = np.arange(Ny)/Ny; z = np.arange(Nz)/Nz
    xx = x[:,None, None]; yy = y[None, :, None]; zz = z[None, None, :]

    phase = []
    for ik in range(nkpt):
        kx, ky, kz = map(float, kpts_red[ik])
        phase_k = np.exp(1j * 2*np.pi*(kx*xx + ky*yy + kz*zz))
        phase.append(phase_k.astype(dtype, copy=False))
    
    # Compute the wavefunctions per k
    psi = np.empty((nband, nkpt, Nx, Ny, Nz), dtype=dtype)
    
    count = 0

    for ik in range(nkpt):
        npw_k = npw[ik] # number of active planewaves for this k
        Gk = np.asarray(G_red[ik, :npw_k, :], dtype=int) # (npw_k, 3) reciprocal lattice vectors for this k

        # Use mapping dictionaries to get FFT grid indices
        ix = np.array([map1[int(Gk_i)] for Gk_i in Gk[:, 0]], dtype=np.int64)
        iy = np.array([map2[int(Gk_i)] for Gk_i in Gk[:, 1]], dtype=np.int64)
        iz = np.array([map3[int(Gk_i)] for Gk_i in Gk[:, 2]], dtype=np.int64)

        C = C_nk[:, ik, :npw_k].astype(dtype, copy=False) # (nband, npw_k) planewave coefficients for all bands at this k

        # Place coefficients on the FFT grid
        C_grid = np.zeros((nband, Nx, Ny, Nz), dtype=dtype)
        C_grid[:, ix, iy, iz] = C

        # To normalize the IFFT
        Gx_max = np.max(np.abs(Gk[:,0])); Gy_max = np.max(np.abs(Gk[:,1])); Gz_max = np.max(np.abs(Gk[:,2]))
        Nx_min = 2*Gx_max + 1; Ny_min = 2*Gy_max + 1; Nz_min = 2*Gz_max + 1
        N_min = Nx_min * Ny_min * Nz_min

        # Compute Bloch-periodic part u_{nk}(r_red) on the grid
        u = np.fft.ifftn(C_grid, axes=(1,2,3)) * N # (nband, Nx, Ny, Nz)
        
        # Compute psi = u * exp(ik.r) / sqrt(Omega)
        psi[:, ik, ...] = (u * phase[ik]) / np.sqrt(float(Omega))
        
        count += 1
        print("Computed ", count, "out of", nkpt, "k-points")

    return psi, ngfft

