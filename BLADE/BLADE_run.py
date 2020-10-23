
import numpy as np
from . import BLADE
from numpy import transpose as t

def BLADE_run(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, 
        Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction):

    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + t(Init_Fraction) * 10
    obs = BLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
            Nu_Init, Omega_Init, Beta_Init, fix_Nu=True, fix_Omega=True)

    obs.Optimize()
   
    obs.Fix_par['Nu'] = False
    obs.Fix_par['Omega'] = False

    obs.Optimize()

    return obs


