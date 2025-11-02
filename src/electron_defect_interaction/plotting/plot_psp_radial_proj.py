"""
plot_psp_radial_proj.py
    Python module that plots the radial projectors of the nonlocal part of the pseudopotentials.

    TODO
        - make plots taht are nicer to look at 
"""

import matplotlib.pyplot as plt

def plot_radial_form_factors(rgrid, fr_li, space='real', qgrid=None):
    """
    Plots the radial form factors on a grid in real or reciprocal space as given by the keyword argumen 'space'. 
    Inputs
        rgrid: (mmax, ) array of floats:
            Grid on which to plot the radial form factors
        fr_li: (lmax+1, imax, mmax) array of floats
            Radial form factors on a grid in real space
        space: str,
            Space on which to plot. The two options are: real (plot in real space) or reciprocal (plot in reciprocal space). Default is 'real'
        q: (mqff) array of floats:
            1d grid in recirprocal space. qmax should be chosen such that qmax = 2*sqrt(2*ecut) to make sure to resolve all K=|k+G| vectors. Default is None
    """

    if space == 'real':
        f = fr_li
        grid = rgrid
    
    if space == 'reciprocal':
        from electron_defect_interaction.io.pseudo_io import fq_from_fr

        if qgrid is None:
            raise TypeError('Must specify qgrid to plot in reciprocal space')

        f = fq_from_fr(rgrid, fr_li, qgrid)
        print(f.shape)
        grid = qgrid
    
    else:
        raise ValueError("Unsupported space. Valid entries are 'real' and 'recirpocal'.")

    for l in range(f.shape[0]):
        for i in range(f.shape[1]):
            plt.plot(grid, f[l,i], label=" l: " + str(l)+" i " + str(i))

    plt.legend()
    plt.show()
