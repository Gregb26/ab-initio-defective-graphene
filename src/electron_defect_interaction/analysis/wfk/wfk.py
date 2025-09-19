import numpy as np

from electron_defect_interaction.utils.fft_utils import *

def compute_psi_nk_FFT(C_nk, G_red, npw, kpts_red, Omega, n, k, Nx, Ny, Nz, use_complex64=True):
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
            FFT grid size. If None, pick 2*Gmax+1 .
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
        C_nk, # [nband, nkpt, npw_max] array of complex
        G_red, # [nkpt, npw_max, 3] array of ints
        npw, # [nkpt] array of ints
        kpts_red, # [nkpt, 3] array of floats
        Omega, # cell volume (Bohr^3)
        Nx, Ny, Nz # FFT grid size
):
    "Computes psi[n,k,x,y,z] on a uniform grid (Nx,Ny,Nz) in [0,1)^3 for all bands n and kpoints k"

    psi = np.empty((C_nk.shape[0], C_nk.shape[1], Nx, Ny, Nz), dtype=np.complex128)

    # precompute fraction grids and per-k Bloch phases
    x = np.arange(Nx)/Nx; y = np.arange(Ny)/Ny; z = np.arange(Nz)/Nz
    xx = x[:,None, None]; yy = y[None, :, None]; zz = z[None, None, :]  

    bloch_k = []
    for k in range(kpts_red.shape[0]):
        kx,ky,kz = np.asarray(kpts_red[k], float)
        bloch = np.exp(1j * 2*np.pi*(kx*xx + ky*yy + kz*zz))
        bloch_k.append(bloch)

    C_grid = np.zeros((Nx,Ny,Nz), dtype=np.complex128)
    for k in range(kpts_red.shape[0]):
        npw_k = npw[k]
        Gk = np.asarray(G_red[k, :npw_k, :], dtype=np.int64)
        ix = np.mod(Gk[:,0], Nx)
        iy = np.mod(Gk[:,1], Ny)
        iz = np.mod(Gk[:,2], Nz)

        for n in range(C_nk.shape[0]):
            C_grid.fill(0)
            npw_k = npw[k]
            Gk = np.asarray(G_red[k, :npw_k, :])
            ix = np.mod(Gk[:,0], Nx); iy = np.mod(Gk[:,1], Ny); iz = np.mod(Gk[:,2], Nz)

            # For each banch, place coefficients, IFFt, apply Bloch phase, normalize
            for n in range(C_nk.shape[0]):
                C = np.asarray(C_nk[n, k, :npw_k])

                # Place coefficients on the FFT index grid
                C_grid[ix, iy, iz] = C

                # IFFT on cell periodic part u_{nk}(r_red)
                u = np.fft.ifftn(C_grid) * (Nx*Ny*Nz)
             
                # Apply Bloch phase and normalize
                psi[n,k] = (u * bloch_k[k]) / np.sqrt(Omega)

                C_grid[ix, iy, iz] = 0 # reset for next band

    return psi

