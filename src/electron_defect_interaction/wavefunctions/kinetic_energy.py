"""
kinetic_energy.py
    Python module that computes the kinetic energy of the wavefunctions given its planewave coefficients.
"""

from electron_defect_interaction.io.abinit_io import *
from electron_defect_interaction.utils.lattice import red_to_cart

def compute_Tk(wfk_path):
    """
    Computes the kinetic energy of the wavefunction T_nk = 0.5 sum_G |C_nk(G)|^2|k+G|^2 given planewave coefficients

    Inputs:
        wfk_path: str
            Path to wavefunction WFK.nc output file
    
    Returns
        T_nk: (nband, nkpt) array of floats
            Kinetic energy of state at band n and kpoint k.
    """
    
    C_nkg, nG = get_C_nk(wfk_path) # pw coeff (nband, nkpt, nG_max) and number of active G per k (nkpt, )
    G_red = get_G_red(wfk_path) # reciprocal lattice vectors in reduced coords (nkpt, nG_max, 3)
    k_red = get_k_red(wfk_path) # kpoints in reduced coors (nkpt, 3)
    B_uc, _ = get_B_volume(wfk_path) # primitive reciprocal lattice vectors B[:,i] = b_i

    # Construct K=k+G vectors in cartesian coords
    K_red = k_red[:, np.newaxis, :] + G_red # (nkpt, nG_max, 3)
    K = red_to_cart(K_red, B_uc) # (nkpt, nG_max, 3)

    nband, nkpt, _ = C_nkg.shape
    # Compute kinetic energies per k
    T_nk = np.zeros((nband, nkpt))
    
    for ik in range(nkpt):
        nG_k = nG[ik] # number of active planewaves for that k
        K2 = np.einsum('gd, gd -> g', K[ik, :nG_k, :], K[ik, :nG_k, :])
        T_nk[:, ik] = 0.5 * np.dot(np.abs(C_nkg[:, ik, :nG_k])**2, K2)
    
    return T_nk