"""
wannier_io.py
    Python module containing functions that extracts information from Wannier90 output files.
"""

import numpy as np

def read_w90_mat(w90_path):
    """
    Extracts the unitary matrix U, from a Wanier90 .mat outputfile, needed to go from the Bloch basis to the Wannier basis. 

    Inputs:
        w90_path: str,
            Path to Wannier90 .mat output file.
    Returns:
        mat: (N,J) array of complex
            Unitary U transformation matrix to go from Bloch basis to Wannier basis
        k_red: (nkpt, 3) array of floats
            kpoints in reduced coordinates. Should (must) match the ones given by Abinit
        N: int
            Number of wannerized bands
        J: int
            Not sure yet
    """

    from pathlib import Path

    with Path(w90_path).open() as f:

        date = f.readline()

        nkpt, nwann, nband = map(int, f.readline().split() )# number of kpoints, number of wannier functions, number of bloch bands. For U, Nw = Nb

        f.readline()

        mat = np.zeros((nkpt, nband, nwann), dtype=complex)
        k_red = np.zeros((nkpt, 3), dtype=float)

        # nkpt blocks with first row being the kpt the next N*J lines being real and imag part of the matrix in column major order
        for ik in range(nkpt):

            k_red[ik,:] = np.array([float(kpt) for kpt in f.readline().split()])

            block = np.zeros((nband*nwann), dtype=complex)
            for i in range(nwann*nband):
                Re, Im = map(float, f.readline().split())
                block[i] = Re + 1j*Im
            
            mat[ik] = block.reshape((nband,nwann), order='F')
            f.readline()


    # testing unitarity
    #for ik in range(mat.shape[0]):
       # assert np.allclose(mat[ik].conj().T @ mat[ik], np.eye(mat[ik].shape[1]), atol=1e-8), 'U matrix must be unitary'

    return mat, k_red