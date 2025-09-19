"""
get.py

Helper functions that extract useful parameters from Abinit's output files for further postprocessing.

Currently supported quantities are:
    - C_nk: coefficients of the planewave basis set (from WFK.nc file),
    - KB: Kleinman-Bylander coefficients and projectors (from PSPS.nc file),
    - POT: Total local Kohn-Sham potential (V_H + V_XC + V_PSPS) (from POT.nc file),
    - KPT: kpoints in reduced coordinates (from WFK.nc file),
    - A: Direct primtive lattice vectors (from WFK.nc file),
    - B: Reciprocal primitive lattice vectors (from WFK.nc file),
    - G: Reciprocal lattice vectors in reduced coordinates (from WFK.nc file),
    - NPW: Number of active planewaves (from WFK.nc file),

The keyword "iomode 3" is necessary to output the files in netCDF format (.nc)
"""

# imports
from netCDF4 import Dataset
import numpy as np

##################################################
# Extracting planewave basis coefficients        #
##################################################

def get_C_nk(filepath):
    """
    Function that reads the coefficients of wavefunctions computed by DFT using Abinit, and stores them
    into an array for post-processing.

    Inputs
    ------
        filepath: str:
            Filepath of the wavefunction output file computed by Abinit. Must be a WFK.nc file.

    Returns
    -------
        C_nk: [nband, nkpt, npw_k] array:
            Array with the wavefunction coefficients. nband is the band index, npkt is the kpoint index and npwk is the coefficients index.
            For example C_nk[0,0, :] gives the npw_k coefficients of the wavefunction for the first band, at the first kpoint.
        npw: [npw] array of ints:
            Number of active planewaves for a given kpoint.
    """

    # Read file with netCDF4
    with Dataset(filepath, 'r') as nc:
        nc.set_auto_mask(False)
        var = nc["coefficients_of_wavefunctions"][:] # shape (nspin, nkpt, nband, npw_k, nsppol, 2)
        npw = np.array(nc["number_of_coefficients"][:],dtype=int)
      
    # Unmask
    if np.ma.isMaskedArray(var):
        var = var.filled(0.0)
    arr = np.asarray(var)

    # Squeeze singleton spin/spinor axes
    arr = np.squeeze(arr) # shape (nkpt, nband, npw_k, 2)

    # Extract real and imaginary parts of the coefficients
    real = arr[..., 0]
    imag = arr[..., 1]

    # Reconstruct coefficients
    coeffs = real + 1j*imag
    C_nk = np.transpose(coeffs, (1,0,2)) # shape (nband, nkpt, npw_k)

    return C_nk, npw

##################################################
# Extracting KB coefficients and projectors      #
##################################################

def get_KB(filepath):
    """
    Function that reads the Kleinman-Bylander (KB) coefficients and projectors from pseudopotentials and stores them for post-processing
    (computing non-local matrix elements).

    Inputs
    ------
        Filepath: str:
            Filepath of the pseupopotential output file. Must be a *PSPS.nc file.

    Returns
    -------
        D: [nkb] array:
            KB coefficients, nkb is the number of channels.

        B_q: [nqpt, nkb] array:
            KB prokectors, nqpt is the qpoint index, e.g. [q, i] gives B_i(q).
    """

    # read the file with netCDF4
    psps = Dataset(filepath, 'r')

    # extract coefficients and projectors
    D = psps.variables["ekb"][0] # [nkb
    B = psps.variables["ffspl"][0] # [nkb, 0:value, 1:derivative, nqpt]

    nkb = B.shape[0]  # number of KB projectors
    nqpt = B.shape[2]  # number of for which B_i(q) is evaluated

    # arrange the projectors in an intuitive way
    B_q = np.zeros((nqpt, nkb ), dtype=complex)
    for i in range(nkb):
        B_q[:, i] = B[i, 0, :]

    # printing stuff
    indlmn = np.array(psps.variables['indlmn'][0])
    for i in range(nkb):
        l = indlmn[i, 0]
        n_proj = indlmn[i, 2]
        D_i = D[i]
        B_i = B_q[:, i]
        print(f"Coefficient {i}: l = {l}, n_proj = {n_proj}, D = {D_i}")
        print(f"Projector {i}: l = {l}, n_proj = {n_proj}, B = {B_i}")

    return D, B_q

##################################################
# Extracting Local Kohn-Sham Potential           #
##################################################

def get_POT(filepath, Omega_sc, subtract_mean=True, spin_channel=0):
    """
    Load ABINIT POT.nc 'vtrial' (local KS potential), return V(r) on the FFT grid
    and its Fourier transform V(G) with the normalization used for matrix elements.

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

        V_G: [n1,n2,n3] array of complex:
            Total local Kohn-Sham potential in reciprocal space.

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

    # NumPy forward FFT divides by N; multiply by Ω_sc / Ω_uc
    V_G = np.fft.fftn(V, norm="forward") * Omega_sc 

    return V, V_G, ngfft

##################################################
# Extracting Kpoint Grid                         #
##################################################

def get_KPT(filepath):
    """
    Function that extracts the kpoint grid in reduced coordinates from Abinit's output files.

    Inputs
    ------
     filepath: str:
        Filepath of the output file.

    Returns
    -------
        kpts_red [nkpts, 3] array of floats:
            Kpoints in reduced coordinates.
    """

    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc.variables["reduced_coordinates_of_kpoints"]

        kpts_red = np.asarray(var)

    return kpts_red

##################################################
# Extracting Kohn-Sham Eigenvalues               #
##################################################

def get_EIG(filepath):
    """"
    Function that extracts the Kohn-Sham eigenvalues from Abinit's output files.

    Inputs
    ------
        filepath: str:
            Filepath containined eigenvalues.

    Returns
    -------
        eigs: [nabnd, nkpts] array of floats:
            Kohn-Sham eigenvalues at each kpoint.
    """

    # Read file with netCDF4
    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc.variables["Eigenvalues"][:] # shape [nspin, nkpt, nband]

        eigs = np.asarray(var, dtype=np.float64)
        eigs = np.transpose(np.squeeze(eigs), (1,0)) # shape [nband, nkpt]

    return eigs

##################################################
# Extracting Direct Primitive lattice vectors    #
##################################################

def get_A_volume(filepath):
    """
    Function that extracts the primitive lattice vectors of the direct lattice from Abinit output files and computes the volume of the primitive cell.

    Inputs
    ------
        filepath: str:
            File containing information.
    Returns:
        A: [3, 3] array of floats:
            Array containing the primitive lattice vectors as columns.

        Omega: float:
            Volume of the cell.
    """

    # Read file with netCDF4
    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc["primitive_vectors"][:]

        A = np.asarray(var, dtype=np.float64).T

    # Compute volume of cell
    Omega = np.abs(np.linalg.det(A))

    return A, Omega


##################################################
# Extracting Primitive Reciprocal Lattice Vectors#
##################################################

def get_B_volume(filepath):
    """
    Function that computes the primitive lattice vectors of the reciprocal lattice from the primitive lattice vectors of the
    direct lattice and the volume of the cell in reciprocal space. If the direct lattice vectors A are those of the primitive cell,
    this volume is the volume of the first Brillouin Zone.

    Inputs
    -------
        filepath: str:
            Filepath of the output file containing direct primitive lattice vectors.

    Returns
    -------
        B: [3,3] array of floats:
            Array containing the reciprocal primitive lattice vectors as columns.

        Omega_G: float:
            Volume of the cell in reciprocal space.
    """

    A, Omega = get_A_volume(filepath)
    # Get direct primitive lattice vectors
    a1 = A[:,0]; a2 = A[:,1]; a3 = A[:,2]

    # Compute reciprocal primitive lattice vectors
    b1 = 2*np.pi * np.cross(a2,a3)/Omega
    b2 = 2*np.pi * np.cross(a3,a1)/Omega
    b3 = 2*np.pi * np.cross(a1, a2)/Omega

    # Arrange them in columns
    B = np.column_stack((b1, b2, b3))

    # Compute cell volume in reciprocal space
    Omega_G = np.abs(np.linalg.det(B))

    return B, Omega_G

##################################################
# Extracting Reciprocal Lattice Vectors          #
##################################################

def get_G(filepath):
    """"
    Function that extracts the reciprocal lattice vectors G in reduced coordinates from Abinit's output files.

    Inputs
    ------
        filepath: str:
            File containing the information the reciprocal vectors.

    Outputs
    -------
        G_red: [nkpt, npw_k, 3] array of ints:
            Array containing the reciprocal lattice vectors for every kpoint and plane wave coefficient.
    """

    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc["reduced_coordinates_of_plane_waves"][:]

        G_red = np.asarray(var, dtype=np.int64) # shape (nkpt, npw_k, 3)
    return G_red

def get_npw(filepath):
    """
    Function that extracts the numer of active planewaves for each kpoint from Abinit's output files.

    Inputs
    ------
        filepath: str:
            File containing the information.

    Returns
    -------
        npw_k: [nkpt] array of ints:
            Number of active planewaves for each kpoint.
    """

    # Read file with netCDF4
    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc["number_of_coefficients"][:]

        npw_k = np.asarray(var, dtype=np.int64) # shape (nkpt)

    return npw_k

filepath = "../data/graphene/pristine_uc/abinit/20x20x1_gs/graphene_p_uc_gso_DS2_GSR.nc"

def get_band(filepath):
    """
    Function that extracts the necessary parameters to plot a band structure from Abinit output files.

    Inputs
    ------
        filepath: str:
            Path to the output file containing the band structure calculation. Must be a GSR.nc file.

    Returns
    -------
        kpt_path: (ngkpt, 3) array of floats:
            Kpoint path the band structure was computed on.

        eigs: (nband, nkpt) array of floats:
            Eigenvalues at each point on the kpoint path.

        fermi_energy: float:
            Fermi energy.
    """

    with Dataset(filepath, mode='r') as nc:
        nc.set_auto_mask(False)

        # Get kpoint path
        kpt_path= nc.variables["reduced_coordinates_of_kpoints"][:]
        kpt_path = np.asarray(kpt_path, dtype=np.float64) # shape (nkpt, 3)

        # Get eigenvalues
        eigs = nc.variables["eigenvalues"][:]
        eigs = np.asarray(eigs, dtype=np.float64) # shape (nsppol, nkpt, nband)
        eigs = np.transpose(np.squeeze(eigs), (1,0)) # shape (nband, nkpt)

        # Get Fermi energy
        fermi_energy = nc.variables["fermi_energy"][:]
        fermi_energy = float(fermi_energy) # you can guess the shape

    return kpt_path, eigs, fermi_energy


