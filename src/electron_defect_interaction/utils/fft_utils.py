"""
fft_utils.py

    Python modules that contains helper functions for fast fourier transforms (FFT) operations.
"""

import math
import numpy as np

from electron_defect_interaction.utils.pw_utils import mask_invalid_G

def is_fft_friendly(n: int, primes=(2, 3, 5, 7)) -> bool:
    """
    Function that checks if an integer n is FFT-friendly, i.e., if n factors are all in `primes`.

    Inputs:
    -------
        n : int
            The integer to check.
        primes : tuple of int, optional
            Tuple of allowed prime factors. Default is (2, 3, 5, 7).

    Returns:
    -------
        bool
            True if n is FFT-friendly, False otherwise.
    """

    if n < 1:
        return False
    
    for p in primes:
        # if n is divisible by p, divide by p 
        while n % p == 0:
            n = int(n/p)
        
        # if n factors only 'primes' remainder is 1 after all divisions
        if n == 1:          
            return True
        
    return n == 1

def next_good_fft_len(n_min: int, primes=(2, 3, 5, 7), force_odd=False) -> int:
    """
    If n_min is not FFT-friendly, return the next larger integer that is, i.e. the next larger integer that only factors into `primes`.
    If force_odd is True, only consider odd integers, for symmetrical FFT grids.

    Inputs:
    -------
        n_min : int
            Minimum integer to consider.
        primes : tuple of int, optional
            Tuple of allowed prime factors. Default is (2, 3, 5, 7).
        force_odd : bool, optional
            If True, only consider odd integers. Default is False.
    
    Returns:
    -------
        int
            The next FFT-friendly integer >= n_min.
    """
    n = int(math.ceil(n_min))
    if force_odd and n % 2 == 0:
        n += 1

    # If already friendly, return immediately
    if is_fft_friendly(n, primes):
        return n

    # add to n until it is fft friendly
    step = 2 if force_odd else 1
    while True:
        n += step
        if is_fft_friendly(n, primes):
            return n

def fft_grid_from_G_red(G_red, nG, primes=(2, 3, 5)):
    """


    Inputs:
    -------
        G_red: (nkpw, nG_max, 3) array of ints
            Reciprocal lattice vectors in reduced coordinates for all kpoints
        nG: (nkpt, ) array of ints
            Number of active G_red for each kpoint.
        primes : tuple of int, optional
            Tuple of allowed prime factors. Default is (2, 3, 5).
    Returns:
    -------
        ngfft: tuple of int
            Minimal FFT double grid sizes (Nx, Ny, Nz) on which to place the G vectors to avoid aliasing.
    """

    # Get a boolean mask to remove invalid G vectors

    keep, nG_max = mask_invalid_G(nG)
    G_red = np.where(keep[..., np.newaxis], G_red, 0.0) # set invalid G's to zero to get correct min and max values

    # Round up Gmax components to nearest integer
    Gmax = np.max(G_red, axis=(0,1))
    Gmin = np.min(G_red, axis=(0,1))

    # Nyquist requires at least Gmax - Gmin + 1 points to place G on a grid without aliasing
    dG = Gmax - Gmin
    Gx, Gy, Gz = tuple(dG)

    # Place coefficients on a double grid to avoid aliasing during products or convolution of wavefunctions
    # Round up to next fft friendly integer for fft speed (integers that only factors into 'primes')
    return (next_good_fft_len((2*Gx+1), primes, force_odd=False),
            next_good_fft_len((2*Gy+1), primes, force_odd=False),
            next_good_fft_len((2*Gz+1), primes, force_odd=False))

def modes_to_fft_indices(m_signed, ngfft):
    """
    Converts signed FFT mode indices in [-n/2, ..., n/2-1] to unsigned FFT grid indices in [0..n-1] for numpy array indexing.
        In mathematics, Fourier modes are usually represented in the signed format: m = (m1, m2, m3) with mi in [-ni/2, ..., ni/2-1], symmetric about 0
        In numpy FFTs, the array indices are in the unsigned format: i = (i1, i2, i3) with mi in [0, ..., ni-1].
    
    Inputs:
    -------
        m_signed : (N, 3) numpy.ndarray
            Array of signed FFT mode indices.
        ngfft : tuple of int
            Number of FFT grid points along each direction (n1, n2, n3).
    
    Returns:
    -------
        i1, i2, i3 : numpy.ndarray
            Arrays of unsigned FFT grid indices corresponding to the input signed mode indices.
    """

    n1, n2, n3 = map(int, ngfft)
    m = m_signed.copy()
    
    # ensure +n/2 → -n/2 for even sizes (choose negative representative)
    if n1 % 2 == 0: m[:,0][m[:,0] ==  n1//2] = -n1//2
    if n2 % 2 == 0: m[:,1][m[:,1] ==  n2//2] = -n2//2
    if n3 % 2 == 0: m[:,2][m[:,2] ==  n3//2] = -n3//2

    # convert to unsigned indices
    i1 = (m[:,0] % n1); i2 = (m[:,1] % n2); i3 = (m[:,2] % n3)
    return i1, i2, i3

def K_from_fft_indices(ngfft, B_sc):
    """
    Function that computes the reciprocal lattice vectors K from the FFT grid indices in cartresian coordinates.
    K = iB1 + jB2 + kB3
    where i, j, k are the FFT grid indices and the Bi's are the primitive reciprocal lattice vectors of the supercell.

    Inputs:
    -------
    ngfft : tuple of int
        Number of FFT grid points along each direction (n1, n2, n3).
    B_sc : (3, 3) numpy.ndarray
        Primitive reciprocal lattice vectors of the supercell in the columns of the array.
    """

    n1, n2, n3 = ngfft
    # Fractional frequencies
    f1 = np.fft.fftfreq(n1); f2 = np.fft.fftfreq(n2); f3 = np.fft.fftfreq(n3)
    
    # Meshgrid in fractional coordinates
    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing='ij')

    K = F1[..., None] * B_sc[:, 0] + F2[..., None] * B_sc[:, 1] + F3[..., None] * B_sc[:, 2]

    return K

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
