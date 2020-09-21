
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse
from Deconvolution_CVB import LogNormalDeconv_CVB as BLADE 
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import itertools
import time
from numpy import transpose as t

def BLADE_run(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, 
        Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction):

    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + t(Init_Fraction) * 10
    obs = BLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
                Nu_Init, Omega_Init, Beta_Init, fix_Nu=False, fix_Omega=True)

    obs.Optimize()
    #while not obs.log:
    #    obs.Beta = np.random.gamma(shape=1, size=(Nsample, Ncell)) 
    #    obs.Optimize()


    #M = obs.ExpQ(obs.Nu, obs.ExpF(obs.Beta), obs.Omega)
    #VarY = obs.VarQ(obs.Nu, obs.ExpF(obs.Beta), obs.Omega)

    #obs.SigmaY = np.maximum(np.sqrt(np.log(
    #                np.sqrt( np.exp(np.log(VarY) - 2*M) + 0.5*0.5) + 0.5   
    #            )), 0.000001) + SY
    
    obs.Fix_par['Nu'] = False
    obs.Fix_par['Omega'] = False

    obs.Optimize()
    #while not obs.log:
    #    obs.Beta = Beta_Init + np.random.gamma(shape=1, size=(Nsample, Ncell))
    #    obs.Nu = Nu_Init + np.random.normal(0, 0.1, obs.Nu.shape)
    #    obs.Omega = Omega_Init + np.square(np.random.normal(0, 0.1, obs.Omega.shape))
    #    obs.Optimize()

    return obs

def BLADE_wrapper(X, stdX, Y, Alpha, Alpha0, Kappa0, SY,
                Init_Fraction, Rep, Crit='E_step', fsel=0.5):
    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]
    
    Mu0 = X
    
    logY = np.log(Y+10e-6)
    #Alpha = np.ones((Nsample, Ncell)) * Alpha
    #SigmaY = np.tile(np.maximum(np.std(logY,1), SY)[:, np.newaxis], [1, Nsample])
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * SY
    #SigmaY = np.ones(logY.shape) * SY
    Omega_Init = stdX
    Beta0 = Alpha0 * stdX
    #Omega_Init = np.square(np.random.normal(0, 0.1, size=(Ngene, Ncell))) + 0.01
    
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = X


    setting = {'Alpha': Alpha, 'Alpha0': Alpha0,
            'Beta0': Beta0, 'Kappa0': Kappa0, 'SigmaY': SY, 'Rep':Rep}

    
    out = BLADE_run(logY, SigmaY, Mu0,
                    Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction)

    setting['E-step'] = out.E_step(out.Nu, out.Beta, out.Omega)


    return out, setting


def NuSVR_jobs(X, Y, Nus, sample):
    sols = [NuSVR(kernel='linear', nu=nu).fit(np.exp(X),Y[:,sample]) for nu in Nus]
    RMSE = [mse(sol.predict(np.exp(X)), Y[:,sample]) for sol in sols]
    return sols[np.argmin(RMSE)]


def BLADE_framework(X, stdX, Y, Ind_Marker, Ind_sample, Alphas, Alpha0s, Kappa0s, SYs,
        Nrep, Njob, Nrepfinal, fsel):

    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]

    X_small = X[Ind_Marker,:]
    Y_small = Y[Ind_Marker,:][:,Ind_sample]
    stdX_small = stdX[Ind_Marker,:]

    Nmarker = Y_small.shape[0]
    Nsample_small = Y_small.shape[1]

    if Nmarker < Ngene:
        print("start optimization using marker genes: " + str(Nmarker) +\
            " genes out of " + str(Ngene) + " genes.")
    else:
        print("all of " + str(Ngene) + " genes are used for optimization.")

    if Nsample_small < Nsample:
        print("Number of samples used: " + str(Nsample_small) + " out of " + str(Nsample) + " samples.")
    else:
        print("All samples are used during the optimization.")


    print('Initialization with Support vector regression')
    SVRcoef = np.zeros((Ncell, Nsample))
    Selcoef = np.zeros((Nmarker, Nsample))
    Nus = [0.25, 0.5, 0.75]

    sols = Parallel(n_jobs=Njob, verbose=10)(
            delayed(NuSVR_jobs)(X_small, Y[Ind_Marker,:], Nus, i)
            for i in range(Nsample)
            )

    for i in range(Nsample):
        Selcoef[sols[i].support_,i] = 1
        SVRcoef[:,i] = np.maximum(sols[i].coef_,0)

    Init_Fraction = SVRcoef
    for i in range(Nsample):
        Init_Fraction[:,i] = Init_Fraction[:,i]/np.sum(SVRcoef[:,i])

    if fsel > 0:
        Ind_use = Selcoef.sum(1) > Nsample * fsel
        print( "SVM selected " + str(Ind_use.sum()) + ' genes out of ' + str(len(Ind_use)) + ' genes')
    else:
        print("No feature filtering is done (fsel = 0)")
        Ind_use = np.ones((Nmarker)) > 0


    start = time.time()
    outs = Parallel(n_jobs=Njob, verbose=10)(
            delayed(BLADE_wrapper)(X_small[Ind_use,:], stdX_small[Ind_use,:], Y_small[Ind_use,:], 
                a, a0, k0, sY, Init_Fraction[:,Ind_sample], Rep=rep, fsel=fsel)
                for a, a0, k0, sY, rep in itertools.product(
                    Alphas, Alpha0s, Kappa0s, SYs, range(Nrep)
                    )
            )

    outs, setting = zip(*outs)
    cri = [obs.E_step(obs.Nu, obs.Beta, obs.Omega) for obs in outs]
    best_obs = outs[np.argmax(cri)]
    best_set = setting[np.argmax(cri)]

    names = ['a_' + str(sett['Alpha']) + '_a0_' + str(sett['Alpha0']) +\
            '_k0_' + str(sett['Kappa0']) + '_sY_' + str(sett['SigmaY']) + '_rep_' + str(sett['Rep'])\
            for sett in setting]

    All_out = dict(zip(names, outs))
    
    end = time.time()
    elapsed = end - start
    print("Done optimization, elapsed time (min): " + str(elapsed/60))

    # run on entire cohort
    if Nsample_small < Nsample or Ngene < Nmarker:
        print("Start inferring per-sample gene expression levels using the entire genes and samples")

        logY = np.log(Y+10e-6) 
        Nu_Init = np.zeros((Nsample, Ngene, Ncell))
        for i in range(Nsample):
            Nu_Init[i,:,:] = X
        Omega_Init = np.square(np.random.normal(0, 0.1, size=(Ngene, Ncell))) + 0.01
        SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample])*best_set['SigmaY']

        Beta0 = best_set['Alpha0'] * np.square(stdX)

        final_obs = Parallel(n_jobs=Njob, verbose=10)(
            delayed(BLADE_wrapper)(X, stdX, Y, best_set['Alpha'], best_set['Alpha0'], best_set['Kappa0'], 
                best_set['SigmaY'], Init_Fraction, Rep=rep, fsel=fsel)
                for rep in range(Nrepfinal)
            )

        outs, setting = zip(*final_obs)
        cri = [obs.E_step(obs.Nu, obs.Beta, obs.Omega) for obs in outs]

        final_obs = outs[np.argmax(cri)]
        #final_obs = LNVI(logY, SigmaY, X, best_set['Alpha'], best_set['Alpha0'], Beta0, best_set['Kappa0'],
        #    Nu_Init = Nu_Init, Omega_Init=Omega_Init, Beta_Init = Beta_Init, fix_Beta = False)

        #final_obs.Optimize()

        #while not final_obs.log:
        #    final_obs.Beta = Beta_Init + np.random.gamma(shape=1, size=(Nsample, Ncell))
        #    final_obs.Nu = Nu_Init
        #    final_obs.Optimize()


    else:
        final_obs = best_obs

    return final_obs, best_obs, best_set, All_out
            
