"""
defect_me.py

    Python module containing the functions to compute the local and non-local parts of the electron-defect interaction matrix.
"""

# IMPORTS

import numpy as np

from electron_defect_interaction.utils.get_utils import *
from electron_defect_interaction.utils.lattice_utils import *
from electron_defect_interaction.analysis.wfk import compute_psi_nk
from electron_defect_interaction.utils.fft_utils import *

# GENERAL

def load_unitcell_quantities(uc_path):
    """
    Loads unit cell quantities needed to compute the non-local part of the interaction matrix elements

    Inputs:
        uc_path: str:
            Path to the ABINIT WFK.nc outputfile for the wavefunctions of the unit cell.
    
    Returns:
        C_nk: (nband, nkpt, nG_max) array of complex
            Planewave coefficients of the wavefunctions
        nG: (nkpt, ) array of ints
            Number of active G vectors per kpoints. I.e. the number of G that satisfy 0.5*(2pi)^2(k+G)^2 for a given k.
        G_red: (nkpt, nG_max, 3) array of ints
            Reciprocal lattice vectors of the unit cell in reduced coordinates
        k_red: (nkpt, 3) array of ints
            k-points of the unit cell in reduced coordinates
        A_uc: (3,3) array of floats:
            Columns are the primitive lattice vectors of unit cell:  A[:,i] = a_i
        Omega_uc: float
            Volume of the unit cell
        B_uc: (3,3)
            Colums are the primitive reciprocal lattice vectors of the unit cell: B[:,i] = b_i

    """
    # planewave coefficients and number of active G per k-point
    C_nk, nG = get_C_nk(uc_path) # (nband, nkpt, nG_max), (nkpt, )
    # reciprocal lattice vectors
    G_red = get_G_red(uc_path) # (nkpt, nG_max, 3)
    # k-point grid
    k_red = get_kpt_red(uc_path)
    # primitive lattice vector and unit cell volume
    A_uc, Omega_uc = get_A_volume(uc_path)
    # primitive reciprocal lattice vectors
    B_uc, _ = get_B_volume(uc_path)
    # ecut
    ecut = get_ecut(uc_path)

    return C_nk, nG, G_red, k_red, A_uc, Omega_uc, B_uc, ecut

def load_supercell_quantities(sc_path):
    """
    Load supercell quantities needed to compute the electron-defect matrix elements.

    Inputs:
        sc_path: str
            Path to the ABINIT WFK.nc outputfile for the wavefunctions of the supercell
    
    Returns:
        tau_s: (natom, 3) array of ints
            Atomic positions of the atoms in the supercell in cartesian coordinates
        A_sc: (3, 3) array of floats 
            Columns are the primitive lattice vectors of the supercell: A[:,i] = a_i
        Omega_sc: float
            Supercell volume
    """
    
    # atomic positions in reduced coordinates
    x_red = get_x_red(sc_path) # (natoms, 3)
    A_sc, Omega_sc = get_A_volume(sc_path)
    # convert to cartesian coords 
    tau_s = red_to_cart(x_red, A_sc) # (natom, 3)

    return tau_s, A_sc, Omega_sc


# NON-LOCAL PART

def fq_from_fr(r, fr_li, q):
    """
    Transforms the radial form factors fr_li in real space to reciprocal space on a q grid. This is done via a Hankel transofrmation:
    fq_li = int dr r fr_il(r) j_l(qr), where j_l(qr) is the spherical Bessel function of order l.

    Inputs:
        r: (mmax, ) array floats
            Grid in real space
        fr_li: (lmax+1, imax+1, mmax) array of floats
            Radial form factors on a 1d grid of size mmax in real space.
        q: (mqff) array of floatsL:
            1d grid in recirprocal space. qmax is chosen such that qmax = 2*sqrt(2*ecut) to make sure to resolve all K=|k+G| vectors
        
    Returns:
        fq_li: (lmax+1, imax+1, mqff) array of floats
            Radial form factors on a 1d grid of size mqff in reciprocal space. To be interpolated and evaluated at K=|k+G| vectors.
    """
    from scipy.special import spherical_jn
    from scipy.integrate import simpson

    lmax = fr_li.shape[0] - 1
    imax = fr_li.shape[1] - 1
    mqff = q.size

    qr = r[:, np.newaxis] * q[np.newaxis, :] # (mmax, mqff)
    fq_li = np.zeros((lmax+1, imax+1, mqff))

    for l in range(lmax + 1):
       jl = spherical_jn(l, qr) # (mmax, mqff)
       fr_i = fr_li[l, :, :] # (imax+1, mmax)
       integrand = (fr_i * r[np.newaxis, :])[:, :, np.newaxis] *jl[np.newaxis, :, :] # (imax+1, mmax, mqff)

       fq_li[l, :, :] =  simpson(integrand, x=r, axis=1) # integrate over
    
    return fq_li # (lmax+1, imax+1, mqff)

def build_K_vectors(k_red, G_red, keep, B_uc):
    """
    Compute the K=k+G vectors, their norm and their unit vectors.

    Inputs:
        k_red: (nkpt, 3) array of ints:
            kpoints in reduced coordinates.
        G_red: (nkpt, nG_max, 3) array of ints:
            Reciprocal lattice vectors in reduced coordinates
        mask: (nkpt, nG_max) array of boolean:
            Boolean mask that selects only active G vectors for a given k 
            i.e. those that satisfy 0.5*(2pi)^2(k+G)^2 < ecut
        B_uc: (3, 3) array of floats:
            Columns are the primitive reciprocal lattice vectors B[:,i] = B_i
        
    Returns
        K: (nkpt, nG_max, 3) array of floats:
            K=k+G vectors in cartesian coords
        K_norm: (nkpt, nG_max) array of floats:
            Norm of K=k+G vectors
        K_hat: (nkpt, nG_max, 3):
            Unit vectors in the direction of the K=k+G vectors.
    """

    # Compute K=k+G in cartesian coordinates
    K_red = k_red[:, np.newaxis, :] + G_red
    Ks = red_to_cart(K_red, B_uc)
    K = np.where(keep[..., np.newaxis], Ks, 0.0)

    # Compute norm of K=k+G vectors
    norms = np.linalg.norm(K, axis=2) # (nkpt, nG_max)
    K_norm = np.where(keep, norms, 0.0) # (nkpt, nG_max)

    # Compute unit vectors 
    valid_Ks = keep & (K_norm > 0) # selects Ks with valig G and non-zero norm (to avoid division by zero)
    K_hat = np.zeros_like(K) # (nkpt, nG_max, 3)
    K_hat = np.divide(K, K_norm[:, :, None], out=K_hat, where=valid_Ks[:,:,None]) # (nkpt, nG_max, 3)

    return K, K_norm, K_hat

def compute_phase(K, tau_as):
    """
    Compute phase exp(i tau_{a}\\cdot K)

    Inputs:
        K: (nkpt, nG_max, 3):
            K=k+G vectors in cartesian coordinates
        tau_as: (natom, 3):
            Atomic positions of atom s of type a in cartesian coordinates

    Returns:
        phase_ksg: (nkpt, natom, nG_max)      
    """

    dot = np.einsum("kgd, sd -> ksg", K, tau_as, optimize = True)
    phase_ksg = np.exp(-1j * dot)

    return phase_ksg

def compute_angular_part(K_hat, lmax):
    """
    Compute the spherical harmonics Y_l^m(K_hat)

    Inputs:
        K_hat: (nkpt, nG_max) array of floats:
            Norm of the K=k+G vectors
        lmax: float
            Maximum orbital angular momentum quantum number
        
    Returns:
        Y_kglm (nkgpt, nG_max, lmax, 2lmax+1) array of complex
            Spherical harmonics at l, m, phi and theta, where phi and theta are the spherical angles of the unit vectors K_hat
    """

    nkpt, nG_max, _ = K_hat.shape
    # compute angles in K
    theta = np.arccos(np.clip(K_hat[...,2], -1.0, 1.0)) # theta = arccos(Kz) (0, pi)
    phi = np.arctan2(K_hat[..., 1], K_hat[..., 0]) + np.pi # phi = arctan(Ky/Kx) (0, 2pi)

    Y_kglm = np.zeros((nkpt, nG_max, lmax+1, 2*lmax+1), dtype=complex)

    from scipy.special import sph_harm_y
    for l in range(lmax+1):
        for m in range(-l, l+1):
            mi = m + lmax
            Y_kglm[..., l, mi] = sph_harm_y(l, m, theta, phi) # (nkpt, nG_max, lmax, 2lmax+1)

    return Y_kglm

def compute_M_NL(uc_wfk_path, sc_wfk_path, psp8_path):
    """
    Computes the non local part of the electron-defect interaction matrix.
    
    Inputs:
        uc_wfk_path: str
            Path to the ABINIT WFK.nc output file for the wavefunctions of the unit cell.
        sc_wfk_path:
            Path to the ABINIT WFK.nc output file for the wavefunctions of the supercell.
        sc_psps_path:
            Path to the ABINIT PSPS.nc output file for the pseudopotentials of the supercell

    Returns:
        M_NL: (nband, nkpt, nband, nkpt) array of complex:
            Electron-defect interaction matrix
    """
    from scipy.interpolate import CubicSpline

    # Get non local part of the pseudopotentials of the supercell
    ekb_li, fr_li, rgrid = get_psps(psp8_path)
    lmax = ekb_li.shape[0] - 1

    # Get necessary unit cells quantities
    C_nkg, nG, G_red, k_red, _, Omega_uc, B_uc, ecut = load_unitcell_quantities(uc_wfk_path)
    # Get necessary super cell quantities
    tau_as, _, Omega_sc = load_supercell_quantities(sc_wfk_path)

    # Compute boolean mask that selects only the active recripocal lattice vector G for each k-point and mask invalid C's (pad with zeros)
    keep = mask_invalid_G(nG)
    C_nkg = np.where(keep, C_nkg, 0.0) # (nband, nkpt, nG_max)

    # Compute K=k+G vectors
    K, K_norm, K_hat = build_K_vectors(k_red, G_red, keep, B_uc)

    # Transform the radial from factor to q space
    qmax = ecut
    q = np.linspace(0, qmax, 10000)
    fq_li = fq_from_fr(rgrid, fr_li, q)

    # Interpolate radial form factors to be able to evaluate them at K=|k+G| vectors
    Fq_li = CubicSpline(q, fq_li, axis=-1, extrapolate=False)   
    assert np.max(K_norm) < qmax, 'Maximum |K|=|k+G| must be within the qgrid'
    F_likg = Fq_li(K_norm) # (lmax+1, imax+1, nkpt, nG_max)

    # Compute phases
    phase_ksg = compute_phase(K, tau_as) # (nkpt, natom, nG_max)

    # Compute angular part
    Y_kglm = compute_angular_part(K_hat, lmax) # (nkpt, nG_max, lmax+1, 2lmax+1)

    # Compute overlaps by summing over all G vectors
    B_nkslim = (4*np.pi / np.sqrt(Omega_sc)) * np.einsum("nkg, likg, kglm, ksg -> nkslim ", np.conj(C_nkg), F_likg, Y_kglm, phase_ksg, optimize=True) # (nkpt, nG_max, ntypat, natom, nkb)
    B_jpslim_conj = np.conj(B_nkslim)

    # Compute matrix elements by summing over l, i, s (atoms) and m
    M_NL = np.einsum("li, nkslim, jpslim -> nkjp ", ekb_li, B_nkslim, B_jpslim_conj)

    return M_NL

# LOCAL PART

# REAL SPACE ROUTE

def compute_psi_nk_fold_sc(
    C_nkg,      
    nG,         
    G_red,        
    k_red,      
    Omega_sc,   
    Ndiag,
    ngfft,
    bands
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
        bands: list of ints
            Specific band index at which to compute the wavefunctions. Reduces computational load by only considering bands we want to study
        
    Returns:
        psi_nk: (nb, nkpt, Nx, Ny, Nz) array of complex
            Wavefunction of the unit cell unfolded onto the FFT grid of the supercell in real space
    """
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
    map_dict_x, map_dict_y, map_dict_z = map_G_to_grid(ngfft)

    # Compute the wavefunctions per k
    psi = np.zeros((nb, nkpt, Nx, Ny, Nz), dtype=complex)
    
    percent_marks = [25, 50, 75, 100]
    next_mark_i = 0

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
        if next_mark_i < len(percent_marks) and p >= percent_marks[next_mark_i]:
            print(f"Computed {percent_marks[next_mark_i]}% of wavefunctions, ({ik+1}/{nkpt}) kpoints")
            next_mark_i += 1

    # Sanity check, wavefunctions are normalized
    norm = np.sum(np.abs(psi)**2, axis=(2,3,4)) * (Omega_sc / N)
    assert np.allclose(norm, 1.0), 'Wavefunctions should be normalized!'
    
    print('Done! Wavefunctions are normalized.')

    return psi
    
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
    C_nkg, nG, G_red, k_red, A_uc, Omega_uc, _ , _ = load_unitcell_quantities(uc_wfk_path)
    _, nkpt, _ = C_nkg.shape

    nband = len(bands)
    # Get necessary super cell quantities
    _, A_sc, Omega_sc = load_supercell_quantities(sc_wfk_path)
    Vp, _ = get_pot(sc_p_pot_path, subtract_mean); Vp = Vp.transpose(2,1,0)
    Vd, _ = get_pot(sc_d_pot_path, subtract_mean); Vd = Vd.transpose(2,1,0)
    ngfft = Vp.shape

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

    marks = [25, 50, 75, 100]
    next_i = 0
    
    print('Computing local matrix elements')
    count = 0
    for ib in range(nband):
        for ik in range(nkpt):
            for ibp in range(nband):
                for ikp in range(nkpt):
                    psi_nk = psi[ib, ik, ...]
                    psi_npkp = psi[ibp, ikp, ...]

                    M_L[ibp, ikp, ib, ik] = np.vdot(psi_npkp, Ved * psi_nk) * dV

                    count += 1
                    p = count / total * 100.0
                    if next_i < len(marks) and p >= marks[next_i]:
                        print(f"Computed {marks[next_i]}% of matrix elements ({count}/{total})")
                        next_i += 1

    return M_L

# RECIPROCAL SPACE ROUTE 

def compute_M_L_g(uc_wfk_path, sc_wfk_path, sc_p_pot_path, sc_d_pot_path, bands, subtract_mean=True):
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

    C_nkg, nG, G_red, k_red, A_uc, Omega_uc, _ , _ = load_unitcell_quantities(uc_wfk_path)
    _, nkpt, _ = C_nkg.shape

    nband = len(bands)
    # Get necessary super cell quantities
    _, A_sc, Omega_sc = load_supercell_quantities(sc_wfk_path)
    Vp, _ = get_pot(sc_p_pot_path, subtract_mean); Vp = Vp.transpose(2,1,0)
    Vd, _ = get_pot(sc_d_pot_path, subtract_mean); Vd = Vd.transpose(2,1,0)
    ngfft = Vp.shape

    # Embbed unit cell coefficients on supercell FFT grid
    map_dict_x, map_dict_y, map_dict_z = map_G_to_grid(ngfft)
    Nx, Ny, Nz = ngfft; N = np.prod(ngfft)
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
