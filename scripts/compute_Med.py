import numpy as np
import sys

from electron_defect_interaction.defects.local_R import compute_M_L_r_BLAAS
from electron_defect_interaction.defects.non_local import compute_M_NL

def main():

    # Paths
    wfk_uc = ""
    wfk_p_sc = ""
    wfk_d_sc = ""
    pot_p_sc = ""
    pot_d_sc = ""
    psp8 = ""

    # Local matrix
    print('Computing local part...')

    ML = compute_M_L_r_BLAAS(
        wfk_uc,
        wfk_p_sc,
        pot_p_sc,
        pot_d_sc,
        subtract_mean=False,
        pristine=False
    )

    # Non local matrix
    print('Computing non local part...')

    MNLp = compute_M_NL(
        wfk_uc,
        wfk_p_sc,
        psp8
    )

    MNLd = compute_M_NL(
        wfk_uc,
        wfk_d_sc,
        psp8
    )

    MNL = MNLd - MNLp

    # Total matrix
    M = ML + MNL

    # Save output
    np.save("Med", M)

if __name__ == "__main__":
    main()