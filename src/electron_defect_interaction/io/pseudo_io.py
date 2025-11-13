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

def read_psp8(path):
    """
    Reads a pseudopotential .psp8 input file and extracts the nonlocal part of the pseudopotential: the KB energies and the radial
    form factors tabulated on a grid in real space. TODO: extract the local part of the pseudopotential.

    Inputs:
        path: str
            Path to the .psp8 pseudopotential. File must have .psp8 extension.
    
    Returns:
        ekb_li: (lmax+1, imax) array of floats
            Kleinman-Bylander energies, i.e. the strength of the associated Kleinman-Bylander projector.
        fr_li: (lmax+1, imax, mmax) array of floats
            Radial form factors of the Kleinman-Bylander projectors, i.e. the radial part only, scaled by r: 
            F_il(r, theta, phi) = f_il(r) Y_l^m(theta, phi) / r, tabulated on a grid in real space
        rgrid: (mmax, ) array of floats
            Grid in real space on which the radial form factors are defined
        lmax: int,
            Maximum orbital angular momentum quantum number present in the file.
        imax: int,
            Maximum number of projection channels for an l present in the file
    """

    def ffloat(s):
        """
        Convert Fortran's exponential "D" with Python's exponential "E".
        """
        return float(s.replace("D", "E"))

    from pathlib import Path

    with Path(path).open("r") as f:
        # Extracting useful info from header
        title = f.readline()
        print("Pseudopotential file: ", title)

        f.readline() # zatom, zion, pspd

        lmax, _ ,mmax = map(int, f.readline().split()[2:5]) # maximum orbital angular momentum quantum number and number of grid points

        f.readline() # rchrg, fchrg, qchrg
        
        nproj = np.array(f.readline().split()[:lmax+1], dtype=int) # (lmax+1, ) nproj[i] = number of channels for i'th l
        imax = np.max(nproj)

        f.readline() # extension switch

        # Next is the non-local part of the pseudopotential
        # There lmax+1 blocks with ekb_li, and fr_li tabulated on a real space grid
        ekb_li = np.zeros((lmax+1, imax)) # KB energies
        fr_li = np.zeros((lmax+1, imax, mmax)) # radial form factors of non-local part of pseudopotentail tabulated on real space grid

        for l in range(lmax+1):

            ekb = f.readline().split()
            ekb_li[l,:] = np.array([ffloat(e) for e in ekb[1:3]])

            fr = np.zeros((mmax, 2+imax))
            for ri in range(mmax):
                
                fr[ri,:] = np.array([ffloat(line) for line in f.readline().split()])
            
            fr_li[l, ...] = fr[:, 2:].T

        rgrid = fr[:, 1]

        # Next is the local part of the pseudopotential
        f.readline()

        # TODO    

    return ekb_li, fr_li, rgrid

def fq_from_fr(r, fr_li, q):
    """
    Transforms the radial form factors fr_li in real space to reciprocal space on a q grid. This is done via a Hankel transofrmation:
    fq_li = int dr r fr_il(r) j_l(qr), where j_l(qr) is the spherical Bessel function of order l.

    Inputs:
        r: (mmax, ) array floats
            Grid in real space
        fr_li: (lmax+1, imax, mmax) array of floats
            Radial form factors on a 1d grid of size mmax in real space.
        q: (mqff) array of floats:
            1d grid in recirprocal space. qmax is chosen such that qmax = 2*sqrt(2*ecut) to make sure to resolve all K=|k+G| vectors
        
    Returns:
        fq_li: (lmax+1, imax, mqff) array of floats
            Radial form factors on a 1d grid of size mqff in reciprocal space. To be interpolated and evaluated at K=|k+G| vectors.
    """
    from scipy.special import spherical_jn
    from scipy.integrate import simpson

    lmax = fr_li.shape[0] - 1
    imax = fr_li.shape[1] 
    mqff = q.size

    qr = r[:, np.newaxis] * q[np.newaxis, :] # (mmax, mqff)
    fq_li = np.zeros((lmax+1, imax, mqff))

    for l in range(lmax + 1):
       jl = spherical_jn(l, qr) # (mmax, mqff)
       fr_i = fr_li[l, :, :] # (imax, mmax)
       integrand = (fr_i * r[np.newaxis, :])[:, :, np.newaxis] *jl[np.newaxis, :, :] # (imax, mmax, mqff)

       fq_li[l, :, :] =  simpson(integrand, x=r, axis=1) # integrate over r
    
    return fq_li # (lmax+1, imax, mqff)

