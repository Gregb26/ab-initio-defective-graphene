"""
fft_utils.py
    Python modules that contains helper functions FFT operations.
"""

import math
import numpy as np

from electron_defect_interaction.utils.planewaves import mask_invalid_G

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

    keep = mask_invalid_G(nG)
    G_red = np.where(keep[..., np.newaxis], G_red, 0.0) # set invalid G's to zero to get correct min and max values

    # Round up Gmax components to nearest integer
    Gmax = np.max(G_red, axis=(0,1)); print('max', Gmax)
    Gmin = np.min(G_red, axis=(0,1)); print('min', Gmin)

    # Nyquist requires at least Gmax - Gmin + 1 points to place G on a grid without aliasing
    dG = Gmax - Gmin
    Gx, Gy, Gz = tuple(dG)

    # Place coefficients on a double grid to avoid aliasing during products or convolution of wavefunctions
    # Round up to next fft friendly integer for fft speed (integers that only factors into 'primes')
    return (next_good_fft_len((2*Gx+1), primes, force_odd=False),
            next_good_fft_len((2*Gy+1), primes, force_odd=False),
            next_good_fft_len((2*Gz+1), primes, force_odd=False))


def map_G_to_fft_grid(ngfft):
    """
    Build mapping from reciprocal lattice vectors G in reduced coordinates, written in the signed mode convention, to an array index.
    This is the mapping G -> FFT grid index. The mapping is as follows:
        if G_i >= 0 then j_i = G_i
        if G_i < 0 then j_i = G_i + N_i, where N_i is the number of points in the grid.

    Inputs:
        ngftt: tuple of 3 ints:
            The FFT grid dimensions
    
    Returns:
        map_dict_x, map_dict_y, map_dict: dicts:
            Mapping dictionaries from reduced G indices (Gx, Gy, Gz) to FFT grid indices, for each of the three dimensions.
            map_dict_i[G_i] = j_i.

    """    

    Nx, Ny, Nz = ngfft

    # Mapping is given by np.fft.fftfreq
    map_x = (np.fft.fftfreq(Nx)*Nx).astype(int)
    map_y = (np.fft.fftfreq(Ny)*Ny).astype(int)
    map_z = (np.fft.fftfreq(Nz)*Nz).astype(int)

    # Build mapping dictionaries: map_dict_i[G_i] = j_i
    map_dict_x = {G_x:j_x for j_x,G_x in enumerate(map_x)}
    map_dict_y = {G_y:j_y for j_y,G_y in enumerate(map_y)}
    map_dict_z = {G_z:j_z for j_z,G_z in enumerate(map_z)}

    return map_dict_x, map_dict_y, map_dict_z
