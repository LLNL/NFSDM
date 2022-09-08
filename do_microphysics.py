# -*- coding: utf-8 -*-
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
Created on Fri Mar  5 13:18:00 2021

Wrapper for particle microphysics
do_coag - Coagulation routine
do_cond - Condensation routine
do_nuc - Nucleation routine
nuc_start - move forward in time until nucleation forms first particles

@author: mcguffin1
"""


import numpy as np
import matplotlib.pyplot as plt
import thermo_properties as dat
import condensation as cond, nucleation as nuc
import cooling, coagulation as coag
from numba import jit

# @jit(forceobj=True)
def do_coag( tcurr, Rd, mult, age, Td, dt_all, dt_coag, j, base_seed, Ykt,  
            rho, t_plot, Rdiff, Kcoag, Rdcoag1, Rdcoag2, Volinit, rho_air,
            T_incr, ti, scale ):
    
    Nsd = len( Rd )
    Rdout = Rd
    multout = mult
    ageout = age
    Tdout = Td
    Ncoag = int( dt_all/dt_coag )
    t = int( (ti-1)*Ncoag + 1 )
    for i in range( Ncoag ):
        j += 1
        Nsdi = Nsd
        ## Get list of super-droplet pairs
        Lsd = coag.coag_pairs( Nsd, j, base_seed )
        # Do coagulation
        Tcoag = cooling.fireball_T(Ykt, tcurr + (i+1)*dt_coag) + T_incr
        # current cloud volume (cm3)
        Vol=cooling.Vol_empirical(Ykt, tcurr+(i+1)*dt_coag, rho_air)*1e6
        mult1, Rd1, age1, Td1, dR, Kcoag1, Rdcoag = coag.mcm_coag(Lsd, multout, ageout, 
                                              Rdout, Tdout, Nsd, dt_coag,Vol, Tcoag, 
                                              base_seed, rho, scale)
        Rdout = Rd1
        multout = mult1
        Tdout = Td1
        ageout = age1
        Nsd = len( Rdout )
        # dR = np.zeros( (Nsd,) )
        tc = (tcurr + (i+1)*dt_coag)*np.ones( (Nsdi,))
        t_plot[t+i,:Nsdi] = tc
        Rdiff[t+i,:Nsdi] = np.nansum(dR*multout)/(dt_coag*np.sum(multout) )
    return Rdout, multout, age1, Td1, j, t_plot, Rdiff 
# @jit(forceobj=True)
def do_cond(tcurr, Rd, mult, Td, mass_vap, dt_all, dt_cond, Ykt, rho_l, MWv, 
            Hv, Tb, Rv, Dva, MWa, Tm, Lw, itr_max, Rdiff, Volinit, rho_air, 
            T_incr, debug_flg, logname, ti, spec_name, scale ):
    Ncond = int( dt_all/dt_cond )
    Nsd = len( Rd )
    yplot = np.zeros( (Nsd,Ncond+1) )
    yplot[:,0] = Rd
    tplot = np.zeros( (Nsd,Ncond+1) )
    Tsd = np.zeros( (Nsd,Ncond+1) )
    Tsd[:,0] = Td
    dCv = np.zeros( (Nsd,))
    deltaMass = np.zeros( (Ncond+1,))
    Cond_intgt = np.zeros( (Ncond+1,))
    t = int( (ti-1)*Ncond +1 )
    
    ## loop over condensation steps
    for i in range( Ncond ):
        
        T = cooling.fireball_T(Ykt, tcurr + (i+1)*dt_cond) + T_incr
        V=cooling.Vol_empirical(Ykt, tcurr+(i+1)*dt_cond, rho_air)*1e6
        Pa = (mass_vap/V)*8314.46/MWv*T #kPa
        # print( Pa )
        Cpv = dat.spec_heat(dat.Cp_A[ spec_name ], dat.Cp_B[ spec_name ], 
                      dat.Cp_C[ spec_name ], dat.Cp_D[ spec_name ], 
                            dat.Cp_E[ spec_name ], T )
        Pe = cond.flat_sat_pressure( Hv, Tb, T)*1e-3 #kPa
        sigma = dat.surf_tens(rho_l, MWv, Tb, T) # (N/m) ...3.81e-8 (J/cm2)
        vap_press_in = 2*sigma/(Rv*rho_l*1e3) # K-m
        Rd_up, T_up, dCv = cond.condevap_aerosol(mult, yplot[:,i], Tsd[:,i], Dva, MWv, MWa, rho_l, 
                                               Pa, Pe, dt_cond, T,Tm, vap_press_in, 
                                               Cpv, Lw, itr_max, debug_flg, logname, scale )
        # Rd_up = yplot[:,i]
        # dT = np.zeros( (Nsd,))
        yplot[:,i+1] = Rd_up #, axis=1 ) (nm)
        tplot[:,i+1] = (tcurr + (i+1)*dt_cond)*np.ones( (Nsd,)) #, axis=1 )
        Tsd[:,i+1] = T_up #K
        # deltaConc[i+1] = np.sum(dCv*mult)/Vol #g/cm3
        deltaMass[i+1] = np.sum( 4*np.pi*rho_l/(3)*mult*( (yplot[:,i+1]*1e-7)**3 - 
                                                (yplot[:,i]*1e-7)**3) ) #(g)
        Cond_intgt[i+1] = deltaMass[0:i].sum()/V
        mass_vap = mass_vap - deltaMass[i+1]
        Rdiff[t+i,:Nsd] = np.nansum((Rd_up-yplot[:,i])*mult)/(dt_cond*np.sum(mult))   
    
    tv = tplot[:,-1]
    
        
    return Rd_up, mass_vap, tv, T_up, deltaMass, Rdiff

def do_nuc(Rd, mult, age, Td, mass_vap, Nsd, T, rho_l, MWv, Tb, Hv, Vol, dt_all, 
           dMp_nuc, Rdiff, debug_flg, logname, scale):
    eps = 1e-6 # small number (g)
    mult_prev = mult
    Rdprev = Rd
    conc_vap = mass_vap/Vol
     
    sigma = dat.surf_tens(rho_l, MWv, Tb, T) # (N/m) ...3.81e-8 (J/cm2)    
    Jnuc, Rd_nuc = nuc.classic_homog(MWv, sigma, rho_l, T, Hv, Tb, conc_vap, Vol)
    Nnuc = Jnuc*dt_all*scale # number of particles
    Rnuc, mult_nuc, age_nuc, Tnuc = nuc.nuc2sd( Nnuc, Rd_nuc, 1, T) # (nm, # particles)
    mass_nuc = mult_nuc*rho_l*4./3.*np.pi*(Rnuc*1e-7)**3 # (g)
    if mass_nuc > 0.:
        if mass_nuc > mass_vap:
            mass_nuc = mass_vap - eps
            mult_nuc = mass_nuc/( rho_l*4./3.*np.pi*(Rnuc*1e-7)**3 )
        if debug_flg:
            f = open(logname, 'a')
            f.write('## NUC : radius %f nm, nucleation rate %3.2e #/cm3/s\n' %(Rnuc,
                            Jnuc/Vol*scale))
                   
        k = np.argmin( (Rd-Rnuc) ) 
        if( ( abs(Rd[k]-Rnuc)<= 1.2*min(Rd) )&(mult[k]<=10*Vol ) ): #
            # print( 'add to existing SD')
            prev_mass = mult[k]*rho_l*(Rd[k]*1e-7)**3*4*np.pi/3
            # Rd[k] = np.sqrt( Rd[k]*Rnuc )
            Rd[k] = ( (mult[k]*Rd[k]**3 + mult_nuc*Rnuc**3)/(mult[k]+mult_nuc) )**(1/3)
            Td[k] = ( (mult[k]*Td[k] + mult_nuc*Tnuc)/(mult[k]+mult_nuc) )
            mult[k] += mult_nuc
            mass_nuc = mult_nuc*rho_l*(Rnuc*1e-7)**3*4*np.pi/3 #mult[k]*rho_l*(Rd[k]*1e-7)**3*4*np.pi/3 -prev_mass
            if debug_flg:
                f.write( '##  NUC : %i particles added to SD no. %i\n' %(mult_nuc,k)) 
        else:
            
            Rd = np.append(Rd, Rnuc)
            mult = np.append(mult, mult_nuc)
            age = np.append( age, age_nuc )
            Td = np.append( Td, Tnuc )
            Nsd += 1
            mass_nuc = mult_nuc*rho_l*(Rnuc*1e-7)**3*4*np.pi/3
            if debug_flg:
                f.write('## NUC : add superdroplet..')
                f.write( '##  NUC : created %i particles in new SD no. %i\n' %(mult_nuc,Nsd-1)) 
        if debug_flg:
           f.close()
    mass_vap = mass_vap - mass_nuc # vapor lost to nucleating particles
    dMp_nuc = np.append( dMp_nuc, mass_nuc )
    Rdiff = np.append( Rdiff, Rnuc/dt_all )
    return Rd, mult, age, Td, mass_vap, dMp_nuc, Rdiff
# @jit(forceobj=True)
def nuc_start(tstart, N_min, dt_nuc, dt_cond, Nsd, Ykt, mass_vap, Hv, Tb, MWv,  
              rho_l, Volinit, rho_air, scale, plt_flag=False):
    
    
    
    T = cooling.fireball_T(Ykt, tstart)

    pre_time = tstart
    Vol=cooling.Vol_empirical(Ykt, tstart, rho_air)*1e6
    conc_vap = mass_vap/Vol #g/cm3
    pre_S = 8314.46*conc_vap*T/(cond.flat_sat_pressure(Hv,Tb, T)*1e-3*MWv)
    pre_T = T
    pre_V = Vol
    # rd_err = 100
    # while rd_err >= 1:
    Nucrate = 0.
    while( (Nucrate*Vol*dt_nuc/Nsd <= N_min ) | (np.isnan(Nucrate)) ):
        tstart = tstart + dt_cond*10
        Vol=cooling.Vol_empirical(Ykt, tstart, rho_air)*1e6
        conc_vap = mass_vap/Vol #g/cm3
        T = cooling.fireball_T(Ykt, tstart)
        # conc_vap = conc_vap*Tinit/T
        Pe = cond.flat_sat_pressure( Hv, Tb, T)*1e-3 #kPa
        S = conc_vap*8314.46/MWv*T/Pe
        sigma = dat.surf_tens(rho_l, MWv, Tb, T) # (N/m) ...3.81e-8 (J/cm2)
        I, rd_nuc = nuc.classic_homog(MWv, sigma, rho_l, T, Hv, Tb, conc_vap, Vol)
        Nucrate = I/Vol*scale
        Tinit = T
        pre_time = np.append( pre_time, tstart)
        pre_S = np.append( pre_S, S)
        pre_T = np.append( pre_T, T)
        pre_V = np.append( pre_V, Vol)
        
        
    tval = np.linspace(0.5,tstart, 1000)
    N = len( tval )
    Temp = cooling.fireball_T(Ykt, tval)
    
    conc_vap0 = conc_vap
        
    print( 'saturation pressure (kPa) and supersaturation ratio', Pe, S-1)
    
    if plt_flag:
        fig, ax = plt.subplots()
        iend = N
        
        plt.plot( tval[:iend], Temp[:iend] )
        plt.plot( tstart, Tinit, '*')
        ax2 = ax.twinx()
        
        ax2.semilogy( pre_time, pre_V, 'r' )
        ax2.set_ylabel('Volume (cm^3)', color='red')
        # plt.xscale('linear')
        plt.xlabel('Time (seconds)')
        ax.set_ylabel('Fireball Temperature (K)')
        
    tf_nuc = dt_nuc+tstart
    Nnuc = I*dt_nuc*scale
    print( 'Nucleation of %3.2e nm radius at %3.2e cm-3s-1 for %3.2e g' %(rd_nuc
                        *1e9, Nucrate, Nnuc*np.pi*(rd_nuc)**3*rho_l*4./3.) )

    R0, mult, age, Td = nuc.nuc2sd( Nnuc, rd_nuc, Nsd, Tinit )
    mass_nuc = np.sum(mult)*rho_l*4./3.*np.pi*(R0[0]*1e-7)**3 # (g)
    conc_vap = conc_vap - mass_nuc/Vol # vapor lost to nucleating particles
    
    #print( '%3.2e cm^{-3} nucleated of %3.2e nm particles' %(Nnuc,R0[0]))
    
    pre_TS = (pre_time, pre_T, pre_S, pre_V) 
    mass_vap -= mass_nuc
    return R0, mult, age, Td, tf_nuc, mass_nuc, conc_vap0, mass_vap, pre_TS
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