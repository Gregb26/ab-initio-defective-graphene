"""
lattice_utils.py

    Python module containing helper functions to extract lattice information from Abinit output files.
"""

import numpy as np
from netCDF4 import Dataset

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

def get_kpt_red(filepath):
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