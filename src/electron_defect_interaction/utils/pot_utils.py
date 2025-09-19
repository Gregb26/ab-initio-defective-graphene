"""
pot_utils.py

    Python module containing helper functions to extract pseudopotential and potential information from Abinit output files.
"""

import numpy as np
from netCDF4 import Dataset

def _hermite_eval(qgrid, y, dy, q):
    """C1 cubic Hermite interpolation on a 1D grid (value+derivative)."""
    q = np.asarray(q)
    q = np.clip(q, qgrid[0], qgrid[-1])           # clamp to grid
    k = np.searchsorted(qgrid, q, side="right") - 1
    k = np.clip(k, 0, len(qgrid)-2)

    x0, x1 = qgrid[k], qgrid[k+1]
    h = x1 - x0
    t = (q - x0) / h

    y0, y1 = y[k],   y[k+1]
    m0, m1 = dy[k]*h, dy[k+1]*h

    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    return h00*y0 + h10*m0 + h01*y1 + h11*m1

def get_KB(psps_path, species=0):
    """
    Read KB coefficients and radial projector splines from ABINIT PSPS.nc file.

    Returns
    -------
    D      : (nkb,)         KB diagonal coefficients D_i [Hartree]
    qgrid  : (nq,)          |q| grid [Bohr^{-1}]
    B_q    : (nq, nkb)      projector values f_i(q) on grid (no interpolation)
    indlmn : (nkb, >=3)     channel metadata per i: (l, m, n, ...)
    eval_B : callable       eval_B(|q|) -> (..., nkb) via Hermite interpolation
    """
    with Dataset(psps_path, "r") as nc:
        # bare-minimum presence check
        ekb   = np.array(nc["ekb"][:])      # (npsp, nkb)
        ffspl = np.array(nc["ffspl"][:])    # (npsp, *, *, 2) with values/derivatives
        ind   = np.array(nc["indlmn"][:])   # (npsp, nkb, >=3)

        D = ekb[species].astype(float)      # (nkb,)
        indlmn = ind[species].astype(int)   # (nkb, ...)

        # detect axes in ffspl[species]: find axis==2 (val/der), axis==nkb, remaining==nq
        F = ffspl[species]                  # shape like (nkb, nq, 2) or (nq, nkb, 2)
        ax2  = int([a for a,s in enumerate(F.shape) if s == 2][0])
        axkb = int([a for a,s in enumerate(F.shape) if s == D.shape[0] and a != ax2][0])
        axnq = int([a for a in range(F.ndim) if a not in (ax2, axkb)][0])

        # reorder to (nq, nkb, 2)
        F = np.moveaxis(F, (axnq, axkb, ax2), (0, 1, 2))
        B_val = F[..., 0]                   # (nq, nkb)
        B_der = F[..., 1]                   # (nq, nkb)

        # simple q-grid fetch: take any 1D float var whose length==nq
        nq = B_val.shape[0]
        qgrid = None
        for v in nc.variables.values():
            arr = np.array(v)
            if arr.ndim == 1 and arr.shape[0] == nq and np.issubdtype(arr.dtype, np.floating):
                qgrid = arr.astype(float)
                break
        if qgrid is None:
            raise KeyError("Could not find a 1D q-grid of length nq in PSPS.nc.")

    def eval_B(absq):
        """Evaluate all channels f_i(|q|) at arbitrary |q| via Hermite interpolation."""
        absq = np.asarray(absq)
        out = np.empty(absq.shape + (B_val.shape[1],), dtype=float)
        for i in range(B_val.shape[1]):
            out[..., i] = _hermite_eval(qgrid, B_val[:, i], B_der[:, i], absq)
        return out

    return D.astype(float), qgrid, B_val.astype(float), indlmn, eval_B



def get_pot(filepath, subtract_mean=True, spin_channel=0):
    """
    Load ABINIT POT.nc 'vtrial' (local KS potential), return V(r) on the FFT grid

    Inputs
    -------
        filepath: str:
            Filepath of the potential output file. Must be a POT.nc file.

        A_sc: [3,3] array:
            Primitive lattice vectors of the direct lattice of the supercell.

        Omega_uc: float:
           Volume (in Bohr^3) of the unit cell.

        subtract_mean: bool:
            If True: Remove constant offset before FFT (recommended).

        spin_channel : int
            0 = charge channel. If spin-polarized data are present, pick 0.

    Returns
    -------
        V_r: [n1,n2,n3] array of floats:
            Total local Kohn-Sham potential in real space.

        ngfft: tuple (n1,n2,n3):
            FFT grid size.
    """

    # Read file with netCDF4
    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc["vtrial"]  # typically (nspden, n1, n2, n3) or (1,n1,n2,n3,1)
        V = np.asarray(var[:], dtype=np.float64)
        dims = var.shape

    # Drop known singleton axes and select charge channel -> (n1,n2,n3)
    # Handles both (1,n1,n2,n3) and (1,n1,n2,n3,1)
    if V.ndim == 5:
        # (nspden, n1, n2, n3, 1)
        V = V[spin_channel, :, :, :, 0]
    elif V.ndim == 4:
        # (nspden, n1, n2, n3)
        V = V[spin_channel]
    elif V.ndim == 3:
        # Already (n1, n2, n3)
        pass
    else:
        raise ValueError(f"Unexpected vtrial shape {dims}")

    n1, n2, n3 = V.shape
    ngfft = (int(n1), int(n2), int(n3))

    if subtract_mean:
        V = V - V.mean()


    return V, ngfft

####################################
# Old code kept for reference      #
####################################

def build_kernel_uc_from_supercell(V_ed_G, S, ngfft):
    """
    Function that turns the defect potential V_ed(G) computed in a supercell into a kernel that can be convolved
    with the wavefunctions in the unit cell.

    Inputs:
    -------
        V_ed_G: [n1,n2,n3] array of complex:
            Defect potential in reciprocal space, computed in a supercell.

        S: [3,3] array:
            Integer matrix that relates the supercell lattice vectors to the unit cell lattice vectors.
            A_sc = S @ A_uc

        ngfft: tuple (n1,n2,n3):
            FFT grid size of the supercell.
    
    Returns:
    --------
        K_uc: [n1,n2,n3] array of complex:
            Defect potential kernel in the unit cell, on the FFT grid.
    """
    
    n1, n2, n3 = ngfft

    # Build fft index grids in unshifted FFT convention
    q1 = (np.fft.fftfreq(n1)*n1).astype(np.int64)
    q2 = (np.fft.fftfreq(n2)*n2).astype(np.int64)
    q3 = (np.fft.fftfreq(n3)*n3).astype(np.int64)

    H, K, L = np.meshgrid(q1, q2, q3, indexing="ij")

    # Map to unit cell shifts
    ST = np.array(S, dtype=np.int64).T
    Qx = ST[0,0]*H + ST[0,1]*K + ST[0,2]*L
    Qy = ST[1,0]*H + ST[1,1]*K + ST[1,2]*L
    Qz = ST[2,0]*H + ST[2,1]*K + ST[2,2]*L

    # Modulo reduction to FFt grid
    i = (Qx % n1).astype(np.int64)
    j = (Qy % n2).astype(np.int64)
    k = (Qz % n3).astype(np.int64)
    
    # Accumulate into unit cell kernel
    K_uc = np.zeros(ngfft, dtype=np.complex128)
    np.add.at(K_uc, (i, j, k), V_ed_G)

    # Fix gauge freedom
    K_uc[0,0,0] = 0.0

    return K_uc

def build_kernel_uc_from_supercell_for_delta_k(V_ed, delta_k_cart, ngfft, A_uc, Omega_sc, Omega_uc):
    """
    Construct the Fourier-space defect kernel for a given momentum transfer Δk.

    Implements:
        F_K(Δk; G) = (1/Ω_uc) ∫_{Ω_sc} d^3r V_ed(r) e^{-i (Δk+G)·r}

    Inputs
    ------
    V_ed : ndarray, shape (n1,n2,n3)
        Real-space defect potential ΔV(r) on the supercell FFT grid (Hartree).
        If periodic images are undesired, apply windowing first.

    delta_k_cart : array_like, shape (3,)
        Momentum transfer Δk = k' - k in Cartesian coordinates (1/Bohr).

    ngfft : tuple of ints (n1,n2,n3)
        FFT grid dimensions.

    A_uc : ndarray, shape (3,3)
        Direct lattice matrix of the *unit cell* in Bohr.
        Columns are the lattice vectors a1, a2, a3.

    Omega_sc : float
        Volume of the supercell (Bohr^3).

    Omega_uc : float
        Volume of the primitive unit cell (Bohr^3).

    Returns
    -------
    F_K_dk : ndarray, shape (n1,n2,n3), complex
        Fourier-transformed defect kernel on the FFT grid.
        Suitable for convolution with plane-wave coefficients at k and k'.

    Notes
    -----
    * Uses np.fft.fftn(..., norm="forward"), so the FFT already divides by N=n1*n2*n3.
    * The prefactor (Ω_sc / Ω_uc) ensures the correct normalization for matrix elements.
    * The (0,0,0) Fourier component is set to zero (constant shift in potential is irrelevant).
    """


    # Build unifrom grid in real space of fractional coordinates inside the unit cell
    n1,n2,n3 = ngfft # FFT grid size
    x = np.arange(n1)/n1; y = np.arange(n2)/n2; z = np.arange(n3)/n3
    X,Y,Z = np.meshgrid(x,y,z, indexing="ij") # Array of shape (n1,n2,n3) with values in [0,1)

    # Convert to cartesian coordinates
    # r = X*a1 + Y*a2 + Z*a3  (A_uc columns = a1,a2,a3 in Bohr)
    frac = np.stack((X,Y,Z), axis=-1) # (n1,n2,n3,3)
    R = frac @ A_uc.T  # (n1,n2,n3,3)

    # Compute phase factor e^{-i Δk·r} on each point of the grid
    phase = np.exp(-1j * np.tensordot(R, delta_k_cart, axes=([3],[0])))

    # Compute the FFT of the product ΔV(r) * phase 
    # Note: np.fft.fftn(..., norm="forward") divides by N = n1*n2*n3
    # Multiplying by (Omega_sc/Omega_uc) gives the correct normalization for matrix elements
    F_K_dk = np.fft.fftn(V_ed * phase, norm="forward") * (Omega_sc / Omega_uc)

    # Fix gauge freedom
    F_K_dk[0,0,0] = 0.0

    return F_K_dk

