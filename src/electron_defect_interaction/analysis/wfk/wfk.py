import numpy as np

from electron_defect_interaction.utils.fft_utils import *

def compute_psi_nk_FFT(C_nk, G_red, npw, kpts_red, Omega, n, k, Nx, Ny, Nz, use_complex64=False):
    """
    Fast IFFT reconstruction of the wavefunction \\psi_{nk}(r) on a uniform grid (u,v,w) in [0,1)^3

    Inputs
    ------
        C_nk: [nband, nkpt, npw_max] array of complex:
            Complex planewave coefficients of the basis set.
        G_red: [nkpt, npw_max, 3] array of ints:
            Reduced interger planewave indicies for each k.
        npw: [nkpt] array of ints:
            Active number of planewave for each k.
        kpts_red: [nkpt, 3] array of floats:
            Kpoints in reduced coordinates.
        Omega: float:
            Unit cell volume (Bohr^3).
        n,k: ints:
            Band and kpoint indices
        Nx, Ny, Nz: ints:
            FFT grid size.
        use_complex64: bool:
            Memory saver. Use False for complex128.

    Returns
    -------
        psi: [Nx, Ny, Nz] array of complex:
            Wavefunction on fractional grid r = u a1 + v a2 + w a3.
        (Nx, Ny, Nz): tuple:
            Grid shape, for normalization
    """

    # Active planewaves
    npw_k = npw[k]
    C = C_nk[n, k, :npw_k]
    C = np.asarray(C, dtype=np.complex64 if use_complex64 else np.complex128)

    Gk = np.asarray(G_red[k, :npw_k, :], dtype=np.int64)

    # Place coefficients on the FFT index grid
    Cgrid = np.zeros((Nx,Ny,Nz), dtype=C.dtype)
    ix = np.mod(Gk[:,0], Nx)
    iy = np.mod(Gk[:,1], Ny)
    iz = np.mod(Gk[:,2], Nz)
    Cgrid[ix, iy, iz] = C

    # Cell periodic part u_{nk}(r_red)
    u = np.fft.ifftn(Cgrid) * (Nx*Ny*Nz) # Numpy's FFT returns (1/N)*\sum C ep{i*2\pi G.r}, multiply by N to cancel the 1/N

    # Build Bloch phase on fractional grid and normalize
    u = u.astype(np.complex64 if use_complex64 else np.complex128, copy=False)
    u_grid = u  # (Nx,Ny,Nz)


    x = np.arange(Nx)/Nx; y = np.arange(Ny)/Ny; z = np.arange(Nz)/Nz
    xx = x[:,None, None]; yy = y[None, :, None]; zz = z[None, None, :]

    kx,ky,kz = np.asarray(kpts_red[k], float)
    bloch = np.exp(1j * 2*np.pi*(kx*xx + ky*yy + kz*zz)).astype(Cgrid.dtype, copy=False)

    psi = (u_grid * bloch) / np.sqrt(Omega, dtype=np.float64)

    return psi, (Nx,Ny,Nz)

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

