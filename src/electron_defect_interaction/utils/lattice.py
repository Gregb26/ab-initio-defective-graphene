"""
lattice_utils.py
    Python module containing helper functions to perform operations on lattice quantities
"""

import numpy as np

def red_to_cart(x_red, X):
    """
    Converts the vector x in reduced coordinates to cartesian coordinates using the scaling matrix X:

    Inputs:
    -------
        x_red: (.., nd) array:
            Vector in reduced coordinates to convert to cartesian coordinates
        
        X: (nd, nd) array:
            Primitive vectors to use to convert to cartesian coordinates. For real space use A where A[:,i] is a_i. For 
            reciprocal space use B where B[:, i] is b_i

    Returns:
    --------
        x: (nd, ):
            Vector in cartesian coordinates
    """

    x_red = np.asarray(x_red, dtype=np.float64)   # (..., nd)
    X     = np.asarray(X,     dtype=np.float64)   # (nd, nd)

    x = np.einsum('...j,ij->...i', x_red, X, optimize=True)

    return x

def monkhorst_pack_grid(ngkpt, signed=True):
    """
    Generates a uniform (no symmetry reduction) Monkhorst-Pack kpoint grid in reduced coordinates from the tuple ngkpt. Returns the same grid
    as Abinit if signed=True
    Inputs:
        ngkpt: tuple (N1, N2, N3) of ints
            Number of kpoints in each direction
        signed: Bool:
            If True, fold [0,1) -> [-0.5, 0.5). Signed convention Abinit uses. Default is True.
    Returns:
        k_grid: (nk, 3) array of floats
            kpoint grid in reduced coordinates
    TODO: implement shift
    """

    N1, N2, N3 = ngkpt
    N = np.prod(ngkpt)
    k_grid = np.zeros((N,3), dtype=float)
    
    ik = 0
    for i3 in range(N3):
        k3 = i3 / N3
        for i2 in range(N2):
            k2 = i2 / N2
            for i1 in range(N1):
                k1 = i1 / N1
                k_grid[ik, :] = (k1, k2, k3)
                ik += 1
    if signed:
        k_grid = ((k_grid + 0.5) % 1.0) - 0.5 # fold [0, 1) -> [-0.5, 0.5)

    return k_grid


