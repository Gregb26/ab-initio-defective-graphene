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
        mat: (nkpt, nband, nwann) array of complex
            Unitary U transformation matrix to go from Bloch basis to Wannier basis. If U is not a disentanglement matrix: nwann=nband. If 
            U is a disentanglement matrix: nwann < nband.
        k_red: (nkpt, 3) array of floats
            kpoints in reduced coordinates. Should (must) match the ones given by Abinit
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
    for ik in range(mat.shape[0]):
       assert np.allclose(mat[ik].conj().T @ mat[ik], np.eye(mat[ik].shape[1]), atol=1e-8), 'U matrix must be unitary'

    return mat, k_red

def check_hermicity_HR(HR, R):
    """
    Checks the Hermicity of the Hamiltonian written in the Wannier tight binding basis. The condition is
    H(R) = H^\\dag(-R).
    Inputs
        HR: (nrpts, nw, nw) array of complex
            Hamiltonien written in Wannier tight binding basis
        R: (nrpts, 3) array of floats
            R vectors computed by Wannier90 used in the tight binding basis
    """

    # Map R = (n1, n2, n3) to index in array R 
    R_tuples = [tuple(r) for r in R]
    R_index  = {r: i for i, r in enumerate(R_tuples)}

    for i, Rs in enumerate(R):
        R_ = tuple(-Rs) # -R

        # some R may be unpaired, skip those
        j = R_index.get(R_)
        if j is None:
            continue
        H = HR[i]
        H_h = HR[j].conj().T

        if not np.allclose(H, H_h, atol=1e-10):
            print(f"Hermiticity fails for R = {R_}, -R index = {j}")
            raise TypeError('Hamiltonian not Hermitian')
    
    return True

def read_w90_HR(w90_path):

    from pathlib import Path
    with Path(w90_path).open() as f:

        date = f.readline()

        # Lattice vectors in angstrom
        a1 = np.fromstring(f.readline(), sep=' ')
        a2 = np.fromstring(f.readline(), sep=' ')
        a3 = np.fromstring(f.readline(), sep=' ')

        A = np.column_stack((a1, a2, a3)) # A[:,i] = a_i

        nw = int(f.readline()) # number of wannier functions

        nrpts = int(f.readline()) # number of inequivalent R points used by Wannier90 

        # skip the R point degeneracy list
        count = 0
        while count < nrpts:
            line = f.readline().split()
            count += len(line)

        HR = []
        R = []
        for _ in range(nrpts):

            f.readline() # blank space

            r = np.array([int(n) for n in f.readline().split()]) # R vectors
            hr = np.zeros((nw, nw), dtype=complex) # allocate matrix

            for _ in range(nw*nw):
                m_str, n_str, Re_str, Im_str = f.readline().split()
                m = int(m_str) - 1 # convert to 0 based
                n = int(n_str) - 1
                hr[m,n] = float(Re_str) + 1j*float(Im_str)
            
            R.append(r)
            HR.append(hr)

        HR = np.array(HR)
        R = np.array(R) 

        if check_hermicity_HR(HR,R):
            return HR, R
