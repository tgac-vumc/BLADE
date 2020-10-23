from numba import jit, njit
import numpy as np
from numpy import transpose as t
import scipy.optimize
from scipy.special import loggamma
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import polygamma

from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse



### Variational parameters Q(X|Nu, Omega) ###
# Nu: Nsample by Ngene by Ncell
# Ometga : Ngene by Ncell


# Beta : Nsample by Ncell
# ExpQ : Nsample by Ngene by Ncell


#@njit(parallel = True, fastmath = True)
@njit
def ExpF(Beta, Ncell):
    #NSample by Ncell (Expectation of F)
    output = np.empty(Beta.shape)
    for c in range(Ncell):
        output[:,c] = Beta[:,c]/np.sum(Beta, axis=1)
    return output
    #return Beta/np.tile(np.sum(Beta, axis=1)[:,np.newaxis], [1, Ncell])


#@njit(parallel = True, fastmath = True)
@njit
def ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Expected value of Y)

    ExpB = ExpF(Beta, Ncell) # Nsample by Ncell

    out = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            out[:,i] = out[:,i] + ExpB[i,c] * np.exp(Nu[i,:,c] + 0.5*np.square(Omega[:,c]))

    return out 

#@njit(parallel = True, fastmath = True)
@njit
def VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    # Nsample Ncell Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)


    # Ngene by Nsample by Ncell by Ncell
    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = np.exp(Nu[i,:,k] + Nu[i,:,l] + \
                        0.5*(np.square(Omega[:,k]) + np.square(Omega[:,l])))

   
    #Ngene by Nsample
    #VarTerm = np.dot(np.exp(2*Nu+2*np.square(Omega)), t(VarB+np.square(Btilda)))\
    #    - np.dot(np.exp(2*Nu + np.square(Omega)), t(np.square(Btilda)))

    VarTerm = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            VarTerm[:,i] = VarTerm[:,i] + \
                np.exp(2*Nu[i,:,c] + 2*np.square(Omega)[:,c])*(VarB[i,c] + np.square(Btilda[i,c])) \
                    - np.exp(2*Nu[i,:,c] + np.square(Omega[:,c]))*(np.square(Btilda[i,c]))

    # Ngene by Ncell
    CovTerm = np.zeros((Ngene, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,i] = CovTerm[:,i] + CovX[:,i,l,k] * CovB[i,l,k]
    
     
    return VarTerm + CovTerm


#@njit(parallel = True, fastmath = True)
@njit
def Estep_PY(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    a = Var / Exp / Exp
                            
    return np.sum(
            -0.5 / np.square(SigmaY) * (a + np.square((Y-np.log(Exp)) - 0.5 * a))
            )


def grad_Nu_python(Y, SigmaY, Nu, Nu0, Beta, Sigma, Omega, Ngene, Ncell, Nsample):
    # return Ngene by Ncell
    grad_PX = -(Nu - Nu0) / np.square(Sigma)

    # Ngene by Nsample (Variance value of Y)
    a0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = t(np.tile(ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))
    Var = t(np.tile(VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))
    #Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    #Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    CovB = - np.tile(Btilda[:,:,np.newaxis], [1,1,Ncell])\
            * t(np.tile(Btilda[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))\
            / np.tile(1+B0[:,np.newaxis,np.newaxis], [1,Ncell,Ncell])
    CovB = t(np.tile(CovB[:,:,:,np.newaxis], [1,1,1, Ngene]), (3,0,1,2))
    # Ngene by Nsample by Ncell by Ncell


    CovX = np.tile((Nu+0.5*np.square(Omega))[:,:,np.newaxis], [1,1,Ncell])
    CovX = np.exp(CovX + t(CovX, (0,2,1)))
    CovX = t(np.tile(CovX[:,:,:,np.newaxis],[1,1,1,Nsample]), (0,3,1,2))
    # Ngene by Nsample by Ncell by N cell
    CovTerm = CovX * CovB

    # Ngene by Ncell by Nsample
    CovTerm = 2*t(np.sum(CovTerm, axis=(3))- \
            np.diagonal(CovTerm, axis1=2, axis2=3), (0,2,1))
    # Ngene by Ncell by Nsample
    g_Exp = t(np.tile(Btilda[:,:,np.newaxis], [1,1,Ngene]), (2,1,0)) \
            * np.exp(np.tile((Nu+0.5*np.square(Omega))[:,:,np.newaxis], [1,1,Nsample]))

    # Ngene by Ncell by Nsample
    g_Var = np.tile(2*np.exp(2*Nu+2*np.square(Omega))[:,:,np.newaxis], [1,1,Nsample]) \
            * t(np.tile((Btilda*(1-Btilda)/(np.tile(B0[:,np.newaxis],[1,Ncell])+1)+np.square(Btilda))[:,:,np.newaxis], [1,1,Ngene]), (2,1,0))\
            - np.tile(2*np.exp(2*Nu+np.square(Omega))[:,:,np.newaxis], [1,1,Nsample])\
            * t(np.tile(np.square(Btilda)[:,:,np.newaxis], [1,1,Ngene]), (2,1,0))\
            + CovTerm

    # Ngene by Ncell by Nsample
    a = (g_Var*Exp - 2*g_Exp*Var) / np.power(Exp, 3)

    b = - (\
            (t(np.tile(Y[:,:,np.newaxis], [1,1,Ncell]), (0,2,1)) \
            - np.log(Exp)) - Var/(2*np.square(Exp))\
            ) * (2*g_Exp/Exp + a)

    grad_PY = - np.sum( 0.5 / t(np.tile(np.square(SigmaY)[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))\
            * (a + b), axis=2) 

    return grad_PX + grad_PY

           
@njit
def grad_Nu(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample):
    # return Nsample by Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample

    Diff = np.zeros((Ngene, Ncell))
    ExpBetaN = Beta0 + Nsample*(Nsample+3)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)
        Diff = Diff + (Nu[i,:,:] - NuExp) / Nsample


    Nominator = np.empty((Nsample, Ngene, Ncell)) 
    for i in range(Nsample):
        Nominator[i,:,:] = Nu[i,:,:] - NuExp - Diff + Kappa0 / (Kappa0+Nsample) * (NuExp - Mu0)
   
    grad_PX = - AlphaN * Nominator / ExpBetaN


    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)


    #ExpX = np.exp(Nu + 0.5*np.square(Omega))
    ExpX = np.empty(Nu.shape) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]


    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    #VarX = np.exp(2*Nu + 2*np.square(Omega))
    VarX = np.empty(Nu.shape) 
    for i in range(Nsample):
        VarX[i,:,:] = np.exp(2*Nu[i,:,:] + 2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 2*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]/Exp*Var) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])


    grad_PY = np.zeros((Nsample, Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,:,c] = -np.transpose( 0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]))

    return grad_PX + grad_PY

#@njit(parallel = True, fastmath = True)
@njit
def grad_Omega(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample):
    # Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample
    ExpBetaN = Beta0 + Nsample*(Nsample+3)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    Nominator = - AlphaN * Nsample*(Nsample+3)*Omega + Kappa0 /(Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN


    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)


    #ExpX = np.exp(Nu + 0.5*np.square(Omega))
    ExpX = np.exp(Nu) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]*Omega[:,l]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]*Omega[:,c]


    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    #VarX = np.exp(2*Nu + 2*np.square(Omega))
    VarX = np.exp(2*Nu)
    for i in range(Nsample):
        VarX[i,:,:] = VarX[i,:,:] * np.exp(2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 4*Omega[:,c]*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*Omega[:,c]*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]*Var/Exp) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])


    grad_PY = np.zeros((Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,c] = np.sum(-0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]), axis=1)


    # Q(X) (fourth term)
    grad_QX =  - Nsample / Omega

#    return  grad_PX 
    return grad_PX + grad_PY - grad_QX

#@njit(parallel = True, fastmath = True)
@njit
def g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
    ExpX = np.exp(Nu)
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega)) #Nsample by Ngene by Ncell
    B0mat = np.empty(Beta.shape)
    for c in range(Ncell):
        B0mat[:,c] =Beta[:,c]/np.square(B0)
    #B0mat = np.dot(B0mat, t(ExpX)) # Nsample by Ngene
    tmp = np.empty((Nsample, Ngene))
    for i in range(Nsample):
        tmp[i,:] = np.dot(B0mat[i,:], t(ExpX[i,:,:]))
    B0mat = tmp

    g_Exp = np.empty((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Exp[s,c,:] = t(ExpX[s,:,c] / B0[s]) - B0mat[s,:]


    return g_Exp

def g_Exp_Beta_python(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    g_Exp = t(np.tile(np.exp(Nu + 0.5*np.square(Omega))[:,:,np.newaxis], [1,1,Nsample]), (2,1,0))\
                /np.tile(B0[:,np.newaxis, np.newaxis], [1,Ncell, Ngene])\
            - t(np.tile(np.sum( # Nsample Ncell Ngenes
                t(np.tile(np.exp(Nu + 0.5*np.square(Omega))[:,:,np.newaxis], [1,1,Nsample]), (2,1,0))\
                *np.tile((Beta/np.tile(np.square(B0)[:,np.newaxis], [1,Ncell]))[:,:, np.newaxis], [1,1, Ngene]), axis=1
                )[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))
    return g_Exp


#@njit(parallel = True, fastmath = True)
@njit
def g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):

    B0Rep = np.empty(Beta.shape) # Nsample by Ncell
    for c in range(Ncell):
        B0Rep[:,c] = B0

    aa = (B0Rep - Beta)*B0Rep*(B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa/(np.power(B0Rep,3) * np.square(B0Rep + 1))
    aa = aa + 2*Beta*(B0Rep - Beta)/np.power(B0Rep,3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (np.power(B0Rep,3) * np.square(B0Rep + 1))
    aaNotT = aaNotT + 2*Beta*(0 - Beta)/np.power(B0Rep,3)
    
    ExpX2 = 2*Nu #Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX2[i,:,:,] = np.exp(ExpX2[i,:,:] + 2*np.square(Omega))

    g_Var = np.zeros((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = t(ExpX2[s,:,c]) * aa[s,c]

       
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                for s in range(Nsample):
                    g_Var[s,i,:] = g_Var[s,i,:] + t(ExpX2[s,:,j])* aaNotT[s,j]


    B_B02 = Beta / np.square(B0Rep) # Beta / (Beta0^2) / Nsample by Ncell
    B0B0_1 = B0Rep * (B0Rep + 1) # Beta0 (Beta0+1) / Nsample by Nell
    B2_B03 = np.square(Beta) / np.power(B0Rep, 3) # Beta^2 / (Beta0^3) / Nsample by Ncell
        
    ExpX = np.empty(Nu.shape)
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(2*Nu[i,:,:]+np.square(Omega))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = g_Var[s,c,:] - 2 * t(ExpX[s,:,c]) * B_B02[s,c]

    
    #Dot = np.dot(B2_B03, t(ExpX)) # Nsample by Ngene
    Dot = np.zeros((Nsample, Ngene))
    for i in range(Nsample):
        for c in range(Ncell):
            Dot[i,:] = Dot[i,:] + B2_B03[i,c] * ExpX[i,:,c]
        #Dot[i,:] = np.dot(B2_B03[i,:], t(ExpX[i,:,:]))

    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + 2*Dot
    
    # Ngene by Nsample by Ncell by N cell
    ExpX = np.empty((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))
    CovX = np.empty((Nsample, Ngene, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[i,:,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

                
    gradCovB = np.empty((Nsample, Ncell, Ncell))
    B03_2_B03_B0_1 = (3*B0 + 2) / np.power(B0,3) / np.square(B0+1)
    for l in range(Ncell):
        for k in range(Ncell):
            gradCovB[:,l,k] = Beta[:,l] * Beta[:,k] * B03_2_B03_B0_1

    # Nsample by Ncell by Ncell by Ngene
    CovTerm1 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    CovTerm2 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / np.square(B0B0_1) # Nsample by Ncell
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                if l != k:
                    CovTerm1[i,l,k,:] = gradCovB[i,l,k]*CovX[i,:,l,k]
                    CovTerm2[i,l,k,:] = B_B0_1_B0B0_1[i,l]*CovX[i,:,l,k]


    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + np.sum(np.sum(CovTerm1, axis=1), axis=1)
    g_Var = g_Var - 2*np.sum(CovTerm2, axis=1)

       
    return g_Var


#@njit(parallel = True, fastmath = True)
@njit
def g_PY_Beta(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY

#@njit(parallel = True, fastmath = True)
@njit
def g_PY_Beta(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY

#@njit(parallel = True, fastmath = True)
@njit(fastmath = True)
def Estep_PX(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = np.sum(Nu, 0)/Nsample # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5*Nsample # Posterior Alpha

    ExpBetaN = Beta0 + Nsample*(Nsample+3)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    return np.sum(- AlphaN * np.log(ExpBetaN))




def g_PY_Beta_python(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):
    # Ngene by Ncell by Nsample
    Exp = t(np.tile(ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))
    Var = t(np.tile(VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = (g_Var * t(Exp, (2,1,0)) - 2*g_Exp*t(Var, (2,1,0)))/t(np.power(Exp,3), (2,1,0))
    b = - (\
            t(np.tile(Y[:,:,np.newaxis], [1,1,Ncell]), (1,2,0))\
            - t(np.log(Exp+0.000001), (2,1,0))- t(Var/(2*np.square(Exp)), (2,1,0))\
        )*(2*g_Exp/t(Exp, (2,1,0)) + a)

    grad_PY = -np.sum( 0.5 / t(np.tile(np.square(SigmaY)[:,:,np.newaxis], [1,1,Ncell]), (1,2,0)) \
           * ( a + b ), axis=2)
 
   
    return grad_PY

def g_Var_Beta_python(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):

    B0Rep = np.tile(B0[:,np.newaxis], [1, Ncell]) # Nsample by Ncell

    aa = (B0Rep - Beta)*B0Rep*(B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa/(np.power(B0Rep,3) * np.square(B0Rep + 1))
    aa = aa + 2*Beta*(B0Rep - Beta)/np.power(B0Rep,3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (np.power(B0Rep,3) * np.square(B0Rep + 1))
    aaNotT = aaNotT + 2*Beta*(0 - Beta)/np.power(B0Rep,3)
        
    g_Var = t(np.tile(np.exp(2*Nu+2*np.square(Omega))[:,:,np.newaxis],[1,1,Nsample]), (2,1,0)) \
            * np.tile(aa[:,:,np.newaxis], [1,1,Ngene])
    
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                g_Var[:,i,:] = g_Var[:,i,:] + (t(np.tile(np.exp(2*Nu+2*np.square(Omega))[:,:,np.newaxis], [1,1,Nsample]), (2,1,0)))[:,j,:] \
                        *(np.tile(aaNotT[:,:,np.newaxis], [1,1,Ngene]))[:,j,:]


    B_B02 = Beta / np.tile(np.square(B0)[:,np.newaxis], [1, Ncell]) # Beta / (Beta0^2) / Nsample by Ncell
    B0B0_1 = np.tile((B0 * (B0 + 1))[:,np.newaxis], [1, Ncell]) # Beta0 (Beta0+1) / Nsample by Nell
        
    g_Var = g_Var \
            - 2 * t(np.tile(np.exp(2*Nu+np.square(Omega))[:,:,np.newaxis],[1,1,Nsample]),(2,1,0)) \
            * np.tile(B_B02[:,:,np.newaxis], [1,1,Ngene])

    g_Var = g_Var\
            + t(np.tile(np.sum( # Nsample by Ncell by Ngene
                2*t(np.tile(np.exp(2*Nu+np.square(Omega))[:,:,np.newaxis],[1,1,Nsample]), (2,1,0)) \
                * np.tile((np.square(Beta) / np.tile(np.power(B0, 3)[:,np.newaxis],[1, Ncell]))[:,:,np.newaxis], [1,1,Ngene]),\
                    axis=1)[:, :, np.newaxis], [1,1,Ncell]), (0,2,1))

    # Ngene by Nsample by Ncell by N cell
    CovX = np.tile((Nu + 0.5 *np.square(Omega))[:,:,np.newaxis], [1,1,Ncell])
    CovX = np.exp(CovX + t(CovX, (0,2,1)))
    CovX = t(np.tile(CovX[:,:,:,np.newaxis],[1,1,1,Nsample]), (0,3,1,2))
                 
    gradCovB = np.tile(Beta[:,:,np.newaxis], [1,1,Ncell])
    gradCovB = (gradCovB * t(gradCovB, (0,2,1))) * np.tile(((3*B0 + 2) / np.power(B0, 3) / np.square(B0 + 1))[:,np.newaxis,np.newaxis], [1,Ncell,Ncell])
    # Beta^kBeta^l(3B0+2)/(B0^3 (B0+1)^2) / Nsample by Ncell by Ncell
    
    # Nsample by Ncell by Ncell by Ngene
    CovTerm1 = t(CovX, (1,2,3,0))*np.tile(gradCovB[:,:,:,np.newaxis],[1,1,1,Ngene])
    CovTerm2 = t(CovX, (1,2,3,0)) \
            * np.tile((Beta*np.tile((B0+1)[:,np.newaxis], [1,Ncell])/np.square(B0B0_1))[:,:,np.newaxis,np.newaxis], [1,1,Ncell, Ngene])

    # Adding covariance terms
    for i in range(Ncell):
        CovTerm1[:,i,i,:] = 0
        CovTerm2[:,i,i,:] = 0

    g_Var = g_Var\
            + t(np.tile(np.sum(CovTerm1, axis=(1,2))[:,:,np.newaxis], [1,1,Ncell]), (0,2,1))\
            - (2*np.sum(CovTerm2, axis=(1)))
       
    return g_Var


class BLADE:
    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,\
            Alpha0=None, Beta0=None, Kappa0=None,\
            Nu_Init=None, Omega_Init=1, Beta_Init=None, \
            fix_Beta = False, fix_Nu=False, fix_Omega=False):
        self.Y = Y
        self.Ngene, self.Nsample = Y.shape
        self.Fix_par = {
            'Beta': fix_Beta,
            'Nu': fix_Nu,
            'Omega': fix_Omega
        }

        # process input variable
        if not isinstance(Mu0, np.ndarray):
            self.Ncell = Mu0
            self.Mu0 = np.zeros((self.Ngene, self.Ncell))
        else:
            self.Ncell = Mu0.shape[1]
            self.Mu0 = Mu0

        if isinstance(SigmaY, np.ndarray):
            self.SigmaY = SigmaY
        else:
            self.SigmaY = np.ones((self.Ngene, self.Nsample))*SigmaY

        if isinstance(Alpha, np.ndarray):
            self.Alpha = Alpha
        else:
            self.Alpha = np.ones((self.Nsample, self.Ncell))*Alpha

        if isinstance(Omega_Init, np.ndarray):
            self.Omega = Omega_Init
        else:
            self.Omega = np.zeros((self.Ngene, self.Ncell)) + Omega_Init
 
        if Nu_Init is None:
            self.Nu = np.zeros((sefl.Nsample, self.Ngene, self.Ncell))
        else:
            self.Nu = Nu_Init
       
        if isinstance(Beta_Init, np.ndarray):
            self.Beta = Beta_Init
        else:
            self.Beta = np.ones((self.Nsample, self.Ncell))*Beta_Init

        if isinstance(Alpha0, np.ndarray):
            self.Alpha0 = Alpha0 
        else:
            self.Alpha0 = np.ones((self.Ngene, self.Ncell))*Alpha0

        if isinstance(Beta0, np.ndarray):
            self.Beta0 = Beta0 
        else:
            self.Beta0 = np.ones((self.Ngene, self.Ncell))*Beta0

        if isinstance(Kappa0, np.ndarray):
            self.Kappa0 = Kappa0 
        else:
            self.Kappa0 = np.ones((self.Ngene, self.Ncell))*Kappa0


   

    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = np.dot(np.exp(Nu), t(F))
        return np.sum(np.square(self.Y-Ypred))

    def ExpF(self, Beta):
        #NSample by Ncell (Expectation of F)
        return ExpF(Beta, self.Ncell)

    def ExpQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    def VarQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)


    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        return -(np.sum(loggamma(self.Alpha)) - np.sum(loggamma(self.Alpha.sum(axis=1)))) + \
            np.sum((self.Alpha-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell])))

    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample*np.sum(np.log(Omega))

    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        return -(np.sum(loggamma(Beta)) - np.sum(loggamma(Beta.sum(axis=1))))+ \
            np.sum((Beta-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell]))
                )


    def grad_Nu(self, Nu, Omega, Beta): 
        # return Ngene by Ncell
        return grad_Nu(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample)

#    def grad_Nu_python(self, Nu, Nu0, Beta): 
        # return Ngene by Ncell
#        return grad_Nu_python(self.Y, self.SigmaY, Nu, Nu0, Beta, self.Sigma, self.Omega, self.Ngene, self.Ncell, self.Nsample)


    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega(self.Y, self.SigmaY, Nu, Omega, Beta,
                self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)


    def grad_Beta(self, Nu, Omega, Beta):
        # return Nsample by Ncell
        B0 = np.sum(self.Beta, axis=1)

        grad_PY = g_PY_Beta(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)

        grad_PF = (self.Alpha-1)*polygamma(1,Beta) - \
            np.tile(np.sum((self.Alpha-1)*np.tile(polygamma(1,B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        grad_QF = (Beta-1)*polygamma(1, Beta) - \
            np.tile(np.sum((Beta - 1) * np.tile(polygamma(1, B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        #return grad_PF
        return grad_PY + grad_PF - grad_QF


    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta)
        QX = self.Estep_QX(Omega)
        QF = self.Estep_QF(Beta)
        #return PY
        return PX+PY+PF-QX-QF

           

    def Optimize(self):
            
            # loss function
        def loss(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

            if self.Fix_par['Nu']:
                Nu = self.Nu
            if self.Fix_par['Beta']:
                Beta = self.Beta
            if self.Fix_par['Omega']:
                Omega = self.Omega

            return -self.E_step(Nu, Beta, Omega)

        # gradient function
        def grad(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)
            
            if self.Fix_par['Nu']:
                g_Nu = np.zeros(Nu.shape)
            else:
                g_Nu = -self.grad_Nu(Nu, Omega, Beta)
            
            if self.Fix_par['Omega']:
                g_Omega = np.zeros(Omega.shape)
            else:
                g_Omega = -self.grad_Omega(Nu, Omega, Beta)
            
            if self.Fix_par['Beta']:
                g_Beta = np.zeros(Beta.shape)
            else:
                g_Beta = -self.grad_Beta(Nu, Omega, Beta)

            g = np.concatenate((g_Nu.flatten(), g_Omega.flatten(), g_Beta.flatten()))

            return g


        Init = np.concatenate((self.Nu.flatten(), self.Omega.flatten(), self.Beta.flatten()))
        bounds = [(-np.inf, np.inf) if i < (self.Ncell*self.Ngene*self.Nsample) else (0.0000001, 100) for i in range(len(Init))]


        out = scipy.optimize.minimize(
                fun = loss, x0 = Init, bounds = bounds, jac = grad,
                options = {'disp': False},
                method='L-BFGS-B')

#        x,f,d = out
        params = out.x

        self.Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
        self.Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
        self.Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                        self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

        self.log = out.success

