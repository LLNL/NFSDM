"""
Copyright (c) 2022, Lawrence Livermore National Security, LLC.  All rights 
reserved.  LLNL-CODE-839524
MIT License
This work was produced at the Lawrence Livermore National Laboratory (LLNL) 
under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department 
of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the 
operation of LLNL.  See license for disclaimers, notice of U.S. Government 
Rights and license terms and conditions.
@authors: Dana McGuffin, Donald Lucas
"""

"""
Routines to analyze & convert particle size distribution to super-droplets and vice-versa
sd2psdf_param - fit bimodal logrnormal distribution to super-droplets
lognorm - lognormal distribution function
init - initialize super-droplets from nucleated rate and size
dNdR2dN - convert dN/dlogR to dN (useful if integrating for total number)
dN2dNdR - convert dN to dN/dlogR
"""

import numpy as np
import random
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from lmfit import models, minimize

# ========== Superdroplets --> lognormal size distribution function ==========
# ========= Estimate (height, center, width) = (N, lnS, lnDpg) ===============
# ===  psd_i( Dp ) = N_i/(lnS_i*sqrt(2*pi)*Dp)*exp[-(lnDp - lnDpg_i)^2/(2*lnS_i^2)] ====
# ============ return (N, S, Dpg) for two modes in a bimodal distribution ====
def sd2psdf_param(SDr, SDc, figs=False):
    # SDr = radius of each superdroplet (nm)
    # SDc = concentration of real particles (1/cm3)
    Nsd = np.sum( ~np.isnan(SDr) )
    Dp = SDr[:Nsd]*2.
    Nd = SDc[:Nsd]
    
    n, bins = np.histogram(np.log(Dp),weights=Nd, bins=50)
    bin_avg = np.sqrt( np.exp(bins[:-1])*np.exp(bins[1:]) )
    # plt.figure()
    # plt.step(np.log10(bin_avg), n )
    
    
    dpm = np.average(Dp,weights=Nd) 
    if dpm < 1.:
        dpm_init = [np.log(np.min(Dp)), np.log(np.max(Dp) )]
        S_init = [np.log(1.1), np.log(1.8)]
        N_init = [np.sum(Nd), np.sum(Nd)*1e-4]
    else:
        dpm_init = [np.log(np.min(Dp)), np.log(dpm )]
        S_init = [np.log(1.5), np.log(1.5)]
        N_init = [n[0], np.sum(Nd)]
    
    bimodal = (models.LognormalModel(prefix='m1_',nan_policy='omit') + 
           models.LognormalModel(prefix='m2_',nan_policy='omit'))
    unimodal = (models.LognormalModel(prefix='m1_',nan_policy='omit') )
    params = bimodal.make_params(m1_center=dpm_init[0], m1_amplitude=N_init[0], m1_sigma=S_init[0],
                                  m2_center=dpm_init[1], m2_amplitude=N_init[1], m2_sigma=S_init[1])
    params['m1_center'].min = np.min(np.log(Dp) )
    params['m1_amplitude'].min = 0.
    params['m1_sigma'].min = np.log(1.+1e-4)
    params['m1_center'].max = np.max(np.log(Dp) )
    params['m1_sigma'].max = np.log(2.5)
    params['m2_center'].min = np.min(np.log(Dp) )
    params['m2_amplitude'].min = 0.
    params['m2_sigma'].min = np.log(1.+1e-4)
    params['m2_center'].max = np.max(np.log(Dp) )
    params['m2_sigma'].max = np.log(2.5)
    
    params_1m = unimodal.make_params(m1_center=dpm_init[0], m1_amplitude=N_init[0], m1_sigma=S_init[0])
    params_1m['m1_center'].min = np.min(np.log(Dp) )
    params_1m['m1_amplitude'].min = 0.
    params_1m['m1_sigma'].min = np.log(1.)
    params_1m['m1_center'].max = np.max(np.log(Dp) )
    params_1m['m1_sigma'].max = np.log(2.5)
   
    residual_val = np.nan
    residual_val1 = np.nan
 
    result_2m = bimodal.fit(n, params=params, x=bin_avg, verbose=True)
    if result_2m.success:
        err_bnd = result_2m.eval_uncertainty(sigma=1)
        m1_amp = result_2m.params['m1_amplitude'].value
        m1_cen = result_2m.params['m1_center'].value
        m1_sig = result_2m.params['m1_sigma'].value
        m2_amp = result_2m.params['m2_amplitude'].value
        m2_cen = result_2m.params['m2_center'].value
        m2_sig = result_2m.params['m2_sigma'].value
        residual_val = np.sum( abs(result_2m.residual ) )
        print(result_2m.fit_report() )
        Dpg = (np.exp(m1_cen), np.exp(m2_cen) )
        sigma = (np.exp(m1_sig), np.exp(m2_sig) )
        N = (m1_amp, m2_amp)
        
    result_1m = unimodal.fit(n, params=params_1m, x=bin_avg, verbose=True)
    if result_1m.success:
        err_bnd1 = result_2m.eval_uncertainty(sigma=1)
        m1_amp1 = result_1m.params['m1_amplitude'].value
        m1_cen1 = result_1m.params['m1_center'].value
        m1_sig1 = result_1m.params['m1_sigma'].value
        m2_amp1 = 0.
        m2_cen1 = 0.
        m2_sig1 = 0.
        residual_val1 = np.sum( abs(result_1m.residual ) )
        print(result_1m.fit_report() )
        
    if ( (residual_val1 < residual_val)&(~np.isnan(residual_val1)) ):
        print('**One mode!')
        Dpg = (np.exp(m1_cen1), np.exp(m2_cen1) )
        sigma = (np.exp(m1_sig1), np.exp(m2_sig1) )
        N = (m1_amp1, m2_amp1)
        err_bnd = err_bnd1
    elif np.isnan(residual_val):
        print( '**One mode!')
        Dpg = (np.exp(m1_cen1), np.exp(m2_cen1) )
        sigma = (np.exp(m1_sig1), np.exp(m2_sig1) )
        N = (m1_amp1, m2_amp1)
        err_bnd = err_bnd1
    elif np.isnan( residual_val1):
        figs = False
    else:
        print('**Two modes!')
    
    if figs:
        plt.figure()
        plt.hist(np.log10(Dp), weights=Nd, bins=50, density=False, color='r', 
                 alpha=0.5, label='super-droplet hist')
        plt.scatter( np.log10(Dp),Nd,c='r', label='Super-droplet data')
        x = np.logspace( -1, 4, 1000)
        model = bimodal.eval(m1_amplitude=N[0], m1_center=np.log(Dpg[0]), 
                             m1_sigma=np.log(sigma[0]), m2_amplitude=N[1],
                             m2_center=np.log(Dpg[1]), m2_sigma=np.log(sigma[1]), x=bin_avg)       
        plt.fill_between( np.log10(bin_avg), model-err_bnd, model+err_bnd, color='b',
                          alpha=0.4, label='1-$\sigma$ uncertainty')
        model = bimodal.eval(m1_amplitude=N[0], m1_center=np.log(Dpg[0]), 
                             m1_sigma=np.log(sigma[0]), m2_amplitude=N[1],
                             m2_center=np.log(Dpg[1]), m2_sigma=np.log(sigma[1]), x=x)  
        plt.plot( np.log10(x), model, color='b', label='fit')
        plt.title( 'Best fit1 ($d_{pm},N_{tot}, S$) = (%3.2f nm, %3.2e $cm^{-3}$,%3.2f) \n Best fit2 ($d_{pm},N_{tot}, S$) = (%3.2f nm, %3.2e $cm^{-3}$,%3.2f)'
                  %(Dpg[0], N[0],sigma[0],Dpg[1],N[1],sigma[1]) )
                    
        plt.legend()
        plt.xlabel('$ log_{10}(D_p [nm]) $')
        plt.ylabel('Number concentration ($cm^{-3}$)')
        plt.yscale('log')
        plt.ylim(1e-2,1e15)
        
    return Dpg, sigma, N, residual_val

# ============= Initialize size distribution with lognorm parameters =======================
def lognorm( Nr, r, rdm, sig, Nt):
    # Nmode - number of lognormal distributions to include
    Nmode = np.shape( Nt )[0]
    # Nr - number of samples from lognormal distribution
    n = np.zeros( (Nr,Nmode))
    for i in range(Nmode):
        n[:,i] =  Nt[i]/( np.sqrt(2*np.pi)*np.log(sig[i]) )*np.exp( -(np.log(r)-
                     np.log(rdm[i]))**2/(2*(np.log(sig[i]))**2) )
    
    return np.sum( n, axis=1 ) 
# ============= Initialize size distribution as exponential =======================
def init( Ntot, rdm, sig, Rd_range):
    x = np.logspace( np.log10(Rd_range[0]), np.log10(Rd_range[1]), 100 )
    ndist_ln = lognorm(100, x, rdm, sig, Ntot ) #dN/dlnR
    ndist = 2.303*ndist_ln #natural log to base-10 log
    return ndist, x 

def dNdR2dN( dNdlgR, Rd):
    b0 = np.sqrt(Rd[1:]*Rd[:-1])
    b_ = np.hstack( (Rd[0]**2/b0[0], b0) ) # left boundary
    b1 = np.hstack( (b0, Rd[-1]**2/b0[-1]) )  # right boundary
    dlgR = np.log10(b1/b_) # calculate the delta\log10(Dp) 
    return dNdlgR*dlgR
def dN2dNdR( dN, Rd):
    b0 = np.sqrt(Rd[1:]*Rd[:-1])
    b_ = np.hstack( (Rd[0]**2/b0[0], b0) ) # left boundary
    b1 = np.hstack( (b0, Rd[-1]**2/b0[-1]) )  # right boundary
    dlgR = np.log10(b1/b_) # calculate the delta\log10(Dp) 
    return dN/dlgR
    
"""
Copyright (c) 2022, Lawrence Livermore National Security, LLC.  All rights 
reserved.  LLNL-CODE-839524
MIT License
This work was produced at the Lawrence Livermore National Laboratory (LLNL) 
under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department 
of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the 
operation of LLNL.  See license for disclaimers, notice of U.S. Government 
Rights and license terms and conditions.
@authors: Dana McGuffin, Donald Lucas
"""
