"""
pseudo_io.py
    Python module containing functions to extract the relevant information from .psp8 pseudopotential files. Currently extracts
    the nonlocal part of the pseudopotential, KB energies and KB radial projectors in real space, and transforms the radial
    projectors to reciprocal space via a Hankel transformation. To be used to evaluate non-local matrix elements.

    TODO
        - Generalize get_psps to handle different different kind of .psp8 files (currentely tailored to carbon only)
        - Could maybe add a module to get_psps to read the local part of the pseudopotential as well
"""

import numpy as np

def get_psps(psp8):
    """
    Reads input .psp8 input file to extract KB energies and radial form factors. The radial form factor are read from
    the input file in real space and should be transformed to q space via a Hankel transform for non-local matrix elements
    evaluation. Currentely, only .psp8 files with atoms with a maximum orbital quantum number l=1 are supported. 

    Input:
        psp8: str
            Path to input pseudopotential file. MUST BE A .PSP8 file.

    Outpus:
        ekb_li: (lmax+1, imax+1) array of floats:
            KB energies.
        fr_li: (lmax+1, imax+1, mmax) array of floats
            Radial form factors on a 1d grid of size mmax in real space.
        
    """
    def ffloat(s):
        """
        Convert Fortran's exponential "D" with Python's exponential "E".
        """
        return float(s.replace("D", "E"))
    
    from pathlib import Path

    with Path(psp8).open("r") as f:
        lines = [line.rstrip("\n") for line in f]

        # Header
        # First line is title
        title = lines[0]
        print(".psp8 file: ", title)
        # Second line is zatom, zion and date
        parts = lines[1].split()
        zatom, zion, pspd = map(float, parts[:3])

        # Third line is pspcod, pspxc, lmax, lloc, mmax, r2well
        parts = lines[2].split()
        pspcod, pspxc, lmax, lloc, mmax, r2well = map(int, parts[:6])

        # Fourth line is rchrg, fchrg, qchrg
        parts = lines[3].split()
        rchrg, fchrg, qchrg = map(float, parts[:3])

        # Fifth channel is nproj, i.e. how many projector channels per l for l = 0, ..., lmax
        parts = lines[4].split()
        nproj_l = list(map(int, parts[:5]))

        proj_li = []
        for l, n in enumerate(nproj_l):
            for i in range(n):
                proj_li.append((l,i)) # (angular momentum l, projector index i)

        # First data block
        # First line is l=0 and ekb_ii for i=0,1
        parts = lines[6].split()
        assert int(parts[0]) == 0
        ekbl0 = list(map(ffloat, parts[1:3]))
        # Data: first column is radial grid index, 2nd is radial grid meshpoint, 3rd i=0 KB projector for l=0, 4th is i=1 KB projector for l=0
        rgrid0 = np.zeros(mmax); fl0i0 = np.zeros(mmax); fl0i1 = np.zeros(mmax)
        for k in range(mmax):
            parts = lines[7+k].split()
            rgrid0[k], fl0i0[k], fl0i1[k] = map(ffloat, parts[1:4])

        # Second data block
        # First line is l=1 and ekb_ii for i=0,1
        parts = lines[7+mmax].split()
        assert int(parts[0]) == 1
        ekbl1 = list(map(ffloat, parts[1:3]))
        # Data: first column is radial grid index, 2nd is radial grid meshpoint, 3rd i=0 KB projector for l=1, 4th is i=1 KB projector for l=1
        rgrid1 = np.zeros(mmax); fl1i0 = np.zeros(mmax); fl1i1 = np.zeros(mmax)
        for k in range(mmax):
            parts = lines[8 + mmax + k].split()
            rgrid1[k], fl1i0[k], fl1i1[k] = map(ffloat, parts[1:4])
        
        assert np.allclose(rgrid0, rgrid1), 'rgrids must be the same'

        ekb_li = np.array([[ekbl0[0], ekbl0[1]], [ekbl1[0], ekbl1[1]]])
        fr_li = np.array([[fl0i0, fl0i1],[fl1i0, fl1i1]])
    
    return ekb_li, fr_li, rgrid0

def fq_from_fr(r, fr_li, q):
    """
    Transforms the radial form factors fr_li in real space to reciprocal space on a q grid. This is done via a Hankel transofrmation:
    fq_li = int dr r fr_il(r) j_l(qr), where j_l(qr) is the spherical Bessel function of order l.

    Inputs:
        r: (mmax, ) array floats
            Grid in real space
        fr_li: (lmax+1, imax+1, mmax) array of floats
            Radial form factors on a 1d grid of size mmax in real space.
        q: (mqff) array of floatsL:
            1d grid in recirprocal space. qmax is chosen such that qmax = 2*sqrt(2*ecut) to make sure to resolve all K=|k+G| vectors
        
    Returns:
        fq_li: (lmax+1, imax+1, mqff) array of floats
            Radial form factors on a 1d grid of size mqff in reciprocal space. To be interpolated and evaluated at K=|k+G| vectors.
    """
    from scipy.special import spherical_jn
    from scipy.integrate import simpson

    lmax = fr_li.shape[0] - 1
    imax = fr_li.shape[1] - 1
    mqff = q.size

    qr = r[:, np.newaxis] * q[np.newaxis, :] # (mmax, mqff)
    fq_li = np.zeros((lmax+1, imax+1, mqff))

    for l in range(lmax + 1):
       jl = spherical_jn(l, qr) # (mmax, mqff)
       fr_i = fr_li[l, :, :] # (imax+1, mmax)
       integrand = (fr_i * r[np.newaxis, :])[:, :, np.newaxis] *jl[np.newaxis, :, :] # (imax+1, mmax, mqff)

       fq_li[l, :, :] =  simpson(integrand, x=r, axis=1) # integrate over
    
    return fq_li # (lmax+1, imax+1, mqff)

