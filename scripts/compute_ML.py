import numpy as np
import sys

from electron_defect_interaction.defects.local_R import compute_M_L_r_BLAAS

# Parse arguments from command line
wfk_uc = str(sys.argv[1])
wfk_sc = str(sys.argv[2])
Vp_sc = str(sys.argv[3])
Vd_sc = str(sys.argv[4])

ML = compute_M_L_r_BLAAS(
    wfk_uc,
    wfk_sc,
    Vp_sc,
    Vd_sc,
    subtract_mean=False,
    pristine=True
)

np.save("ML", ML)