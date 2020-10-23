

import numpy as np
from . import BLADE_run


def BLADE_wrapper(X, stdX, Y, Alpha, Alpha0, Kappa0, SY,
                Init_Fraction, Rep, Crit='E_step', fsel=0.5):
    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]
    
    Mu0 = X
    
    logY = np.log(Y+10e-6)
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * SY
    Omega_Init = stdX
    Beta0 = Alpha0 * stdX
    
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = X


    setting = {'Alpha': Alpha, 'Alpha0': Alpha0,
            'Beta0': Beta0, 'Kappa0': Kappa0, 'SigmaY': SY, 'Rep':Rep}
    
    out = BLADE_run(logY, SigmaY, Mu0,
                    Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction)

    setting['E-step'] = out.E_step(out.Nu, out.Beta, out.Omega)


    return out, setting


