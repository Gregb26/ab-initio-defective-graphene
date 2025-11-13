"""
abinit_io.py
    Helper functions that extract useful quantities from Abinit's output files for further postprocessing.

    Currently supported quantities are:
        TODO

    The keyword "iomode 3" is necessary to output the files in netCDF format (.nc)
"""

# imports
from netCDF4 import Dataset
import numpy as np

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

def get_G_red(filepath):
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

def get_eigenvalues(filepath, shift_Fermi=False):
    """
    Function that reads the Kohn-Sham eigenvalues computed by Abinit and stores them in an array for postprocessing.

    Inputs:
    -------
        filepath: str:
            Filepath of the eigenvalues output file computed by Abinit. Must be a WFK.nc file.

    Returns:
    -------
        eigenvalues: [nband, nkpt] array:
            Array with the Kohn-Sham eigenvalues. nband is the band index and nkpt is the kpoint index.
            For example eigenvalues[0,0] gives the eigenvalue of the first band at the first kpoint.
    """

    with Dataset(filepath, mode='r') as nc:
        nc.set_auto_mask(False)
        vals = nc["eigenvalues"][:]
        fermi_level = nc["fermi_energy"][:]

        if shift_Fermi:
            
            vals -= fermi_level

        eigenvalues = np.transpose(np.squeeze(np.asarray(vals))) # (nband, nkpt)

    return eigenvalues

def get_k_red(filepath):
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

def get_x_red(filepath):
    """
    Function that extracts the reduced coordinates of the atoms in the cell from Abinit's outputfiles

    Inputs
    ------
        filepath: str:
            Filepath of the output file containing reduced coordinates of atoms.

    Outputs
    -------
        x_red: [natom, 3] array of floats:
            Array containing the reduced coordinates of the atoms.
    """
    with Dataset(filepath, "r") as nc:
        nc.set_auto_mask(False)
        var = nc["reduced_atom_positions"][:]

        x_red = np.asarray(var, dtype=np.float64)

    return x_red

def get_typat(filepath):
    """
    Function that extracts the atom types present in the cell

    Inputs:
        filepath: str
            Filepath of the Abinit output file containing the atom types (WFK.nc works)
    Outputs: 
        typat: (natom, ) of ints
            typat[i] is the atom type of atom i 
    """
    with Dataset(filepath, "r") as nc:
        typat = np.asarray(nc.variables["atom_species"][:]).astype(int)

    return typat

def get_ecut(path):
    """
    Get the ecut used in the calculation from WFK.nc file.
    """
    with Dataset(path, 'r') as nc:
        ecut = np.asarray(nc.variables["ecut_eff"][:]).astype(float)

    return ecut

