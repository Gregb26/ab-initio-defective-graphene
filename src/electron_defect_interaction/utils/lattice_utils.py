"""
lattice_utils.py

    Python module containing helper functions to extract lattice information from Abinit output files.
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