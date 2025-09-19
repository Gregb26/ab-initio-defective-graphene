"""
pw_utils.py

    Python module containing helper functions to extract plane-wave information from Abinit output files.
"""

import numpy as np
from netCDF4 import Dataset

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

def get_eigenvalues(filepath, shift_Fermi=True):
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
        if shift_Fermi:
            fermi_level = nc["fermi_energy"][:]
            vals -= fermi_level

        eigenvalues = np.transpose(np.squeeze(np.asarray(vals))) # (nband, nkpt)

    return eigenvalues

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

def build_maps_from_ngfft(ngfft):
    """
    Build mapping dictionaries from reduced G indices (h,k,l) to FFT grid indices, for each of the three dimensions.

    Inputs:
    -------
        ngftt: tuple of 3 ints:
            The FFT grid dimensions
    
    Returns:
    --------
        map1, map2, map3: dicts:
            Mapping dictionaries from reduced G indices (h,k,l) to FFT grid indices, for each of the three dimensions.

    """    

    n1, n2, n3 = ngfft

    # Generate sequence of FFT indices in the unshifted FFT convention
    q1 = (np.fft.fftfreq(n1)*n1).astype(int)
    q2 = (np.fft.fftfreq(n2)*n2).astype(int)
    q3 = (np.fft.fftfreq(n3)*n3).astype(int)

    # Build mapping dictionaries
    map1 = {int(h):i for i,h in enumerate(q1)}
    map2 = {int(k):j for j,k in enumerate(q2)}
    map3 = {int(l):m for m,l in enumerate(q3)}

    return map1, map2, map3

def make_Cdicts_for_k(C_nk, G_red_uc, npw, ik, band_indices):
    """
    Build per-band dictionaries at k-index ik: {(h,k,l) -> C_{band,k}(G)}.
    Useful to lookup the value of C_nk(G) given G in reduced coordinates.

    Inputs:
    -------
        C_nk: [nband, nkpt, npwmax] array:
            Array with the wavefunction coefficients. nband is the band index, npkt is the kpoint index and npwk is the coefficients index.
            For example C_nk[0,0, :] gives the npw_k coefficients of the wavefunction for the first band, at the first kpoint.
        G_red_uc: [nkpt, npwmax, 3] array of ints:
            Reduced G-vectors in the unit cell, in units of the reciprocal lattice vectors.
            For example G_red_uc[0, :, :] gives the (h,k,l) reduced coordinates of the planewaves at the first kpoint.
        npw: [nkpt] array of ints:
            Number of active planewaves for a given kpoint. 
        ik: int:
            Index of the k-point to process.
        band_indices: list of ints:
            List of band indices to include in the output dictionaries.

    Returns:
    --------
        Cdicts: list of dicts:
            List of dictionaries, one per band in band_indices, mapping reduced integer coordinates of a G vector (h,k,l) to  a coefficient C_{nk}(G).
            The order of the list corresponds to the order of band_indices.
    """
    # Number of active planewaves at this k-point
    npw_k = int(npw[ik])

    Gk = G_red_uc[ik, :npw_k, :] # (npw_k, 3), ints
    Cdicts = []
    for ib in band_indices:
        coeffs = C_nk[ib, ik, :npw_k]          # (npw_k,)
        d = {tuple(map(int, g)): coeffs[i] for i, g in enumerate(Gk)}

        Cdicts.append(d)

    return Cdicts  # list in same order as band_indices