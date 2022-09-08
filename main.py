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

import numpy as np, matplotlib.pyplot as plt
import thermo_properties as dat
import condensation as cond , sizedist as psd
import figs, cooling, do_microphysics as sdm
from IPython import get_ipython
import time, logging
from os import getcwd
from datetime import date, datetime
import adapt_superdroplets as fixSD
import sys
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
# import matplotlib as mpl
# mpl.use('Agg')
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')

start = time.time()

    
def do_main(scenario_name, spec, Ykt, shell_mass, save_path, save_data, t_final, scale_condnuc, scale_coag):
    ## Log diagnostics?
    debug = True
    
    Nsd0= 20# number of super-droplets
    Nmax = 100
    Nmin = 50 
    dt_cond = 1e-2 # time step (s)
    dt_coag = 1e-2
    dt_all = 0.1
    dt_nuc = 1e-3 # time step (s)g
    tstart = 0.5 # (sec) starting time that approximately corresponds to 3600 K temperature
    # t_final = 10 #*60.
    
    make_figs = True
    save_figs = True
    merge_sds = True
    save_str = scenario_name
    logfile = save_path+date.today().isoformat()+'_'+save_str+'.log'
    rho_air = 1.02 # density at height of burst (kg/m3)
    kB = 1.381e-23 # m2-kg/s2-K
    
    itr_max = 25 # maximum number of iterations for condensation solver
    N_min = 1 # minimum multiplicity of a Super-droplet
    
    V_fire=cooling.Vol_empirical(Ykt, tstart, rho_air) #m3
    conc_vap = shell_mass/V_fire*1e-3 # g/cm3 at Tinit temperature
    T = cooling.fireball_T(Ykt, tstart)
    Tinit = cooling.fireball_T(Ykt, tstart)
    
    #### properties ##############################################################
    MWv = dat.MW[ spec ] 
    Rv = 8.314*1e3/MWv # specific gas constant species (J/kg/K)
    Hv = dat.Hv[ spec ]*1e3 # (J/mol)
    Lw = Hv/MWv # latent heat vaporization (J/g) 
    rho_l = dat.Rho[ spec ] #liquid density (g/cm3) Al2O - 3.97 g/cm3
    K = 2.55e-2 # coefficient of thermal conductivity of air (J/m s K)
    Tm = dat.Tm[ spec ]
    # D = 2.52e-5 # molecular diffusion coefficient from Shima code (cm2/s)
    MWa = 28.97 #g/mol air
    Tb = dat.Tb[ spec ] # boil temperature (K)
    mwsol = dat.MW[ spec ] #molecular weight of solute (g/mol)
    ## temperature-dependent
    sigma = dat.surf_tens(rho_l, MWv, Tb, T) # (N/m) ...3.81e-8 (J/cm2)
    Cpv = dat.spec_heat(dat.Cp_A[ spec ], dat.Cp_B[ spec ], dat.Cp_C[ spec ], dat.Cp_D[ spec ], 
                       dat.Cp_E[ spec ], T )
    Dva = 3.*kB**1.5*(6.02e23)**0.5/(8.*101325*(2.*np.pi)**0.5*(2.*3.711e-10)**2 )*(
            1e3)**0.5*1e4*T**1.5*np.sqrt( 1/MWv + 1/MWa ) #diffusion coefficient (cm2/s)
    vap_press_in = 2*sigma/(Rv*rho_l)*10. # K-m
    #if debug:
    f = open(logfile, 'w')
    f.write('---------------------------------------------------\n')
    f.write('START microphysics at %s\n' %datetime.now().strftime("%Y-%m-%d %H:%M:%S") )
    f.write('Yield: %f kt & Mass: %f kg\n' %(Ykt,shell_mass) )
    f.write('Diffusion coeff='+str(Dva)+' cm2/s\n')
    f.close()
    
    
    
    #### Initializing super-droplets  #######################################################
    
    # get volume (cm3)
    Vol=cooling.Vol_empirical(Ykt, tstart, rho_air)*1e6
    
    #### Find when nucleation starts during cooling ################################
    shell_mass_g = shell_mass*1e3
    Nsd = Nsd0
    R0, mult, age, Td, tf_nuc, mass_nuc, conc_vap0, mass_vap, pre_TSV = sdm.nuc_start(
                                tstart, N_min, dt_nuc, dt_cond, Nsd, Ykt, shell_mass_g, 
                                Hv, Tb, MWv, rho_l, V_fire, rho_air,
                                scale_condnuc, plt_flag=False)
    (pre_time, pre_T, pre_S, pre_Vol) = pre_TSV
    tstart = tf_nuc - dt_nuc
    tcurr = tf_nuc
    dMp_nuc = mass_nuc
    dMv_nuc = mass_nuc
    part_mass = mass_nuc
    mult_init = mult
    dRnuc = np.nanmean(R0)/dt_nuc
    # get volume (cm3)
    Vol=cooling.Vol_empirical(Ykt, tcurr, rho_air)*1e6
    SD_numb = mult/Vol
    
    #### Initializing arrays ######################################################
    
    Vcm3=cooling.Vol_empirical(Ykt, tstart, rho_air)*1e6
    
    base_seed = 0
    #base_seed = int( datetime.now().strftime('%H%M%S') )
    if debug:
        f = open(logfile, 'a')
        f.write('# Base for random seed = %i\n' %base_seed )
        f.close()
    Cond_intgt_plot = 0.
    Nuc_intgt_vap = -dMv_nuc/Vcm3
    conc_vap_init_plot = np.nan
    Kcoag = tcurr*np.ones( (10000,))+np.nan
    R1 = tcurr*np.ones( (10000,))+np.nan
    R2 = tcurr*np.ones( (10000,))+np.nan
    dRcond = np.nan
    dRcoag = np.nan
    Rd = R0
    
    mass_vap1 = mass_vap
    Ntot = np.sum( mult/Vol)
    if np.sum( mult) == 0:
        dpm = 0.
        SAdpm = 0.
        Vdpm = 0.
        dp_median = 0.
    else:
        dpm = np.average( R0*2., weights=mult)
        SAdpm = np.average( R0*2., weights=mult*R0**2)
        Vdpm = np.average( R0*2., weights=mult*R0**3)
        wq = DescrStatsW(data=Rd*2., weights=mult )
        dp_median = wq.quantile(probs=0.5,return_pandas=False)[0]
    dMp_coag = 0.
    dMp_cond = 0.
    dMv_cond = 0.
    
    
    Vol_init = Vcm3
    TK = cooling.fireball_T(Ykt, tstart)
    Pe = cond.flat_sat_pressure( Hv, Tb, TK)*1e-3 #kPa
    SS = (mass_vap/Vcm3)*8314.46/MWv*TK/Pe
    j=0
    Ta_diff = 0. # change in cooling curve due to condensation
    
    Nt = int( t_final/dt_all + 2 )
    Ny = Nmax+1
    Rd_plot = np.zeros( (Nt,Ny) ) + np.nan
    mult_plot = np.zeros( (Nt,Ny) ) + np.nan
    age_plot = np.zeros( (Nt,Ny) ) + np.nan
    t_plot = np.zeros( (Nt,Ny) ) + np.nan
    Tsd_plot = np.zeros( (Nt,Ny) ) + np.nan
    dTd_plot = np.zeros( (Nt,Ny) ) + np.nan
    t_plot_coag = np.zeros( (int(Nt*dt_all/dt_coag + 1),Ny) ) + np.nan
    dRcoag = np.zeros( (int(Nt*dt_all/dt_coag + 1),Ny) ) + np.nan
    dRcond = np.zeros( (int(Nt*dt_all/dt_coag + 1),Ny) ) + np.nan
    t = 0
    mult_plot[t,:Nsd] = mult/Vcm3
    Rd_plot[t,:Nsd] = R0
    age_plot[t,:Nsd] = age
    t_plot[t,:] = tcurr*np.ones( (Ny,) )
    t_plot_coag[t,:] = tcurr*np.ones( (Ny,) )
    
    not_finished = True
    #### Master time-loop
    while not_finished: #T > 300: #
        t += 1
    
        if t > 2:
            if Cond_intgt_plot[-1]/shell_mass_g > 0.1:
                delta_vap = ( Cond_intgt_plot[-1]-Cond_intgt_plot[-2])/Cond_intgt_plot[-2]
                delta_dp = ( dpm[-1]-dpm[-2])/dpm[-2]
                not_finished = ( abs(delta_vap) > 1e-6 )
                f = open( logfile, 'a')
                f.write('# MAIN! at t= %f seconds, %3.2e fractional change in Dp\n' %(tcurr, delta_dp) )
                f.close()
            else:
                not_finished = True
        before_end  = (tcurr-tstart <= t_final)
        if not before_end:
            not_finished = False
        #### Get Temperature and saturation ratio   
        V=cooling.Vol_empirical(Ykt, tcurr, rho_air)*1e6
    
        Vcm3 = np.append( Vcm3, V )
        Tval = cooling.fireball_T(Ykt, tcurr)
        TK = np.append( TK, Tval )
        conc_vap = mass_vap/Vcm3[-1]
        SS = np.append( SS, 8314.46*conc_vap*TK[-1]/(cond.flat_sat_pressure(Hv,
                                   Tb, TK[-1])*1e-3*MWv))
        
        ### Check if superdroplets need to be merged #######################
        if merge_sds:
            merged = fixSD.merge(Rd, mult, age, Td, Nmax, Nmin, debug, logfile, tcurr)
            (Rd, mult, age, Td, merge_flg) = merged
        Nsd = len( Rd )
           
        #### nucleation  ###################################################
        mass_prev = mass_vap
        mult_prev = mult
        Rdprev = Rd
    
        T = cooling.fireball_T(Ykt, tcurr+dt_all) + Ta_diff
        Vol=cooling.Vol_empirical(Ykt, tcurr+dt_all, rho_air)
        Vcm3_t = Vol*1e6
        Dva = 3.*kB**1.5*(6.02e23)**0.5/(8.*101325*(2.*np.pi)**0.5*(2.*3.711e-10)**2 )*(
                 1e3)**0.5*1e4*T**1.5*np.sqrt( 1/MWv + 1/MWa ) #diffusion coefficient (cm2/s)
        
        if debug:
            f = open(logfile, 'a')
            f.write('# DO Nucleation! at t= %f seconds\n' %tcurr )
            f.close()
        
        Rdout, mult, age, Td, mass_vap, dMp_nuc, dRnuc = sdm.do_nuc(Rd, mult, age, Td, mass_vap, Nsd, 
                            T, rho_l, MWv, Tb, Hv, Vcm3_t, dt_all, dMp_nuc, dRnuc, 
                            debug, logfile, scale_condnuc )
        if ( np.isnan( mass_vap)or(mass_vap<0) ):
            print(tcurr,' debug nuc')
        Rd = Rdout
        Nsd = len( Rd )
        
        
        dMv_nuc = np.append( dMv_nuc,mass_vap-mass_prev ) #(g)
        
        
        #### Condensation rate solved with Implicit Euler ################
        # ## Nusselt number and saturation vapor pressure over curved surface changes over iterations
        Rdprev = Rd
        mass_vap_init = mass_vap
        
        Rdout, mass_vap, tv, Tdout, Cond_intgt, dRcond = sdm.do_cond(tcurr, Rd, mult, Td, mass_vap,
                                dt_all, dt_cond, Ykt, rho_l, MWv, Hv, Tb, Rv, 
                                Dva, MWa, Tm, Lw, itr_max, dRcond, V_fire, rho_air, 
                                Ta_diff, debug, logfile, t, spec,
                                scale_condnuc )
        
        # Ta_diff += np.nansum( -dTemp )
        
        if ( np.isnan( mass_vap)or(mass_vap<0) ):
            print(tcurr,' debug cond')
        Rd = Rdout
        Td = Tdout
        Nsdc = len( Rd )
        dMp_cond = np.append( dMp_cond, np.sum(mult*( (Rd*1e-7)**3-(
                           Rdprev*1e-7)**3))*rho_l*4*np.pi/3)
       
        
        dMv_cond = np.append( dMv_cond, mass_vap-mass_vap_init)
        
        
        #### Coagulation MCM ###################################################
        Rdprev = Rd
        mult_prev = mult
        Rdout, mult, age,Td, j, t_plot_coag, dRcoag = sdm.do_coag(tcurr, 
                      Rd, mult, age, Td, dt_all, dt_coag, j, base_seed, Ykt, rho_l, 
                      t_plot_coag, dRcoag, Kcoag, R1, R2, V_fire, rho_air, 
                                Ta_diff, t, scale_coag )
        Rd = Rdout
        Nsd = len( Rd )
        dMp_coag = np.append( dMp_coag, (np.sum(mult*(Rd*1e-7)**3)-np.sum(mult_prev*(
                           Rdprev*1e-7)**3) )*rho_l*4*np.pi/3)
        
        
        tcurr += dt_all
        age += dt_all
        part_mass = np.append( part_mass, np.sum(mult*rho_l*4./3.*np.pi*
                                                         (Rd*1e-7)**3) )#(g)
        mass_vap1 = np.append( mass_vap1, mass_vap )
        
        # get temperture to compare with super-droplets
        Tval = cooling.fireball_T(Ykt, tcurr)
        
        # total number particles
        Ntot = np.append( Ntot, np.sum(mult/Vcm3_t))
        # number mean diameter
        dpm = np.append( dpm, np.average(Rd*2., weights=mult) )
        SAdpm = np.append( SAdpm, np.average(Rd*2., weights=mult*Rd**2) )
        Vdpm = np.append( Vdpm, np.average(Rd*2., weights=mult*Rd**3) )
        wq = DescrStatsW(data=Rd*2., weights=mult )
        dp_median = np.append( dp_median, wq.quantile(probs=0.5,return_pandas=False)[0] )
        
        # create arrays to plot
        Rd_plot[t,:Nsd] = Rd
        mult_plot[t,:Nsd] = mult/Vcm3[-1]
        age_plot[t,:Nsd] = age
        t_plot[t,:Nsd] = tv[:Nsd]
        Tsd_plot[t,:Nsd] = Td
        dTd_plot[t,:Nsd] = Td - Tval
            
        conc_vap_init_plot = np.append( conc_vap_init_plot, mass_vap_init/Vcm3[-1] )
        if np.size( Cond_intgt_plot)>1:
            Cond_intgt_plot = np.append( Cond_intgt_plot, Cond_intgt.sum()+
                                        Cond_intgt_plot[-1] )
        else:
            Cond_intgt_plot = np.append( Cond_intgt_plot, Cond_intgt.sum()+
                                        Cond_intgt_plot )  
        if np.size( Nuc_intgt_vap)>1:
            Nuc_intgt_vap = np.append( Nuc_intgt_vap, (-dMv_nuc[-1]
                                      +Nuc_intgt_vap[-1]) )
        else:
            Nuc_intgt_vap = np.append( Nuc_intgt_vap, (-dMv_nuc[-1] +Nuc_intgt_vap) )
    Nf = len( mass_vap1 )
    save_strngs = spec+'_'+save_str+'_Nsd%i_%i'%(Nsd0,Nsd)
    if make_figs:
        figs.make_fig_SanityCheck(t_plot[:Nf,0], mass_vap1, part_mass, shell_mass_g, 
                                 [dMp_nuc, dMv_nuc], [dMp_cond, dMv_cond], dMp_coag,  
                                 MassBal=True, Particles=True, Vapor=True)
        T1 = [pre_time, pre_T]
        T2 = [t_plot[:Nf,0], TK[:Nf] ]
        S1 = [pre_time, pre_S]
        S2 = [t_plot[:Nf,0], SS[:Nf] ]
        fig, ax = plt.subplots()
        figs.make_fig_TS( ax, tstart, T1, T2, S1, S2, 'Seconds', 'Temperature (K)', 
                         'Saturation ratio')
        if save_figs:
            fig.set_size_inches(6,3)
            fig.savefig( save_path+'SDevolution_cooling_S_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        V1 = [pre_time, pre_Vol]
        V2 = [t_plot[:Nf,0], Vcm3[:Nf]*1e-6]
        fig, ax = plt.subplots()
        figs.make_fig_TV( ax, tstart, T2, V2, 'Time (seconds)', 'Temperature (K)', 
                         'Volume (m$^3$)', log=False)
        if save_figs:
            fig.savefig( save_path+'SDevolution_cooling_expansion_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        fig, ax = plt.subplots()
        figs.make_fig_NtotDpm( ax, t_plot[:Nf,0], Ntot, dpm, 'Seconds', 'Number of particles ($cm^{-3}$)',
                              'Number mean diameter (nm)')
        if save_figs:
            fig.savefig( save_path+'SDevolution_Ntot_nmd_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        
        # #### Plot time-series for radius, Temperature, mass vapor condensed ######
        # fig, ax = plt.subplots()
        # str_title ='Superdroplet evolution'
        # str_label = 'Radius (nm)'
        # figs.make_fig_cond( ax, t_plot ,  Rd_plot, Nsd ,str_title, logy=True, ylab=str_label)
        
        # fig, ax = plt.subplots()
        # str_title ='Superdroplet ageing'
        # str_label = 'Age (sec)'
        # figs.make_fig_cond( ax, t_plot ,  age_plot, Nsd ,str_title, logy=False, ylab=str_label)
        
        fig, ax = plt.subplots()
        figs.make_fig_NtotDpm( ax, t_plot[:Nf,0], np.nanmean(dTd_plot,axis=1), np.nanstd(dTd_plot,axis=1), 
                              'Time (seconds)', '$T_d - T_{\infty} $ Mean (K)',
                              '$T_d - T_{\infty} $ Standard Deviation (K)', log=False)
        if save_figs:
            fig.savefig( save_path+'SDevolution_dropletTemp_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        fig, ax = plt.subplots()
        ax.plot( t_plot[:Nf,0] ,  Cond_intgt_plot/shell_mass_g, label='Condensed' )
        ax.plot( t_plot[:Nf,0] ,  Nuc_intgt_vap/shell_mass_g, label='Nucleated' )
        ax.plot( t_plot[:Nf,0] ,  (Nuc_intgt_vap+Cond_intgt_plot)/shell_mass_g, 
                 'k--', label='Total' )
        plt.ylabel('Cumulative mass fraction of vapor lost')
        plt.xlabel('Time (seconds)')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlim(0, 45)
        plt.ylim(1e-6, 1)
        if save_figs:
            fig.set_size_inches(6,3)
            fig.savefig( save_path+'condensed_frac_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        fig, ax = plt.subplots()
        figs.make_fig_radius( ax, t_plot[:Nf,:] ,  Rd_plot*2., mult_plot, dpm, Nsd, 
                             'Particle diameter (nm)', 1e-1, 1e9, weight='Number' )
        fig.set_size_inches(6,3)
        if save_figs:
            fig.savefig( save_path+'PNSDevolution_sorted_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        fig, ax = plt.subplots()
        figs.make_fig_radius( ax, t_plot[:Nf,:] ,  Rd_plot*2., mult_plot, SAdpm, Nsd, 
                             'Particle diameter (nm)', -1,-1, weight='Area' )
        if save_figs:
            fig.savefig( save_path+'PSaSDevolution_sorted_%s.png'%save_strngs, 
                           dpi=220, bbox_inches='tight')
        fig, ax = plt.subplots()
        figs.make_fig_radius( ax, t_plot[:Nf,:] ,  Rd_plot*2., mult_plot, Vdpm, Nsd, 
                             'Particle diameter (nm)', -1,-1, weight='Volume' )
        if save_figs:
            fig.savefig( save_path+'PVSDevolution_sorted_%s.png'%save_strngs, 
                           dpi=220, bbox_inches='tight')
        fig, ax = plt.subplots()
        figs.make_fig_radius( ax, t_plot[:Nf,:] ,  Rd_plot*2., mult_plot, dpm, Nsd, 
                             'Particle diameter (nm)', 1e-1, 1e9, weight='Number',
                             mmd=Vdpm )
        if save_figs:
            fig.set_size_inches(6,3)
            fig.savefig( save_path+'PNSDevolution_mmd_nmd_sorted_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        # fig, ax = plt.subplots()
        # str_lab1 ='Coagulation'
        # str_lab2 ='Condensation'
        # str_lab3 = 'Nucleation'
        # str_label = 'Particle growth rate (nm/s)'
        # Nt = len( t_plot_coag[:,0])
        # figs.make_fig_diag(ax, t_plot_coag[:,0] ,  dRcoag, dRcond[:Nt], t_plot[:,0], dRnuc, 
        #                    Nsd, str_lab1, str_lab2, str_lab3, logy=True, ylab=str_label )
        # ax.set_ylim(1e-3)
        # if save_figs:
        #    fig.savefig( save_path+'diagnostic_rates_%s.png'%save_strngs, dpi=220,
        #                     bbox_inches='tight')
        # figs.make_fig_cond( ax, t_plot_coag[:,0] ,  dRcoag, Nsd, str_title1,
        #                    logy=True, ylab=str_label )
        
        # fig, ax = plt.subplots()
        # figs.make_fig_cond( ax, t_plot_coag[:,0] ,  dRcond, Nsd, str_title2, logy=True, 
        #                    ylab=str_label )
        
        ### Get lognormal distribution parameters
        log_params = psd.sd2psdf_param(Rd_plot[Nf-1,:], mult_plot[Nf-1,:], figs=True)
        if save_figs:
            plt.savefig( save_path+'lognormalFit_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
        
        figs.make_hist_init_final(2.*Rd_plot[0,:], mult_plot[0,:], Rd_plot[Nf-1,:]*2., mult_plot[Nf-1,:] )
        if save_figs:
            plt.savefig( save_path+'Histograms_%s.png'%save_strngs, dpi=220, bbox_inches='tight')
    
    end = time.time()
    print('Elapsed time = %s min' %((end-start)/60))
    
    f = open(logfile, 'a')
    f.write('---------------------------------------------------\n')
    f.write('FINISHED microphysics at %s\n' %datetime.now().strftime("%Y-%m-%d %H:%M:%S") )
    f.write(' overall %s minutes to simulate %f minutes' %((end-start)/60, (tcurr-tstart)/60.) )
    f.close()
    
    out = pd.Series( {'Mass_kg':[], 'Yield_kt':[], 'Temperature_K':[], 
                      'Volume_cm3':[], 'Dp_nm':[], 'Ntot_cm3':[],
                      'SDradius_nm':[], 'SDmult_cm3':[], 'time_sec':[],
                      'vapor_g':[], 'vaporCondensed_frac':[],'SDevolution_dp':[],
                      'SDevolution_numb':[], 'SDevolution_time':[], 'N_sd':[],
                      'Nsteps':[]} )
    out['Mass_kg'] = shell_mass
    out['Yield_kt'] = Ykt
    out['Temperature_K'] = TK
    out['Volume_cm3'] = Vcm3
    out['Dpm_nm'] = dpm
    out['Dpg_nm'] = dp_median
    out['Ntot_cm3'] = Ntot
    out['SDradius_nm'] = Rd
    out['SDmult_cm3'] = mult/Vcm3[-1]
    out['time_sec'] = t_plot[:,0]
    out['vapor_g'] = mass_vap1
    out['vaporCondensed_frac'] = Cond_intgt_plot/shell_mass_g
    out['SDevolution_dp'] = Rd_plot*2
    out['SDevolution_numb'] = mult_plot
    out['SDevolution_time'] = t_plot
    out['N_sd'] = Nsd
    out['Nsteps'] = Nf
    print('Finished writing output!! %3.2e kg ; %3.2e kt' %(shell_mass, Ykt) ) 
    out.to_pickle(save_path+'Out_%s.pkl' %save_strngs, compression='zip')
    if save_data:
        out.to_pickle(save_path+'Out_%s.pkl' %save_strngs, compression='zip')
    return out

if __name__ == "__main__":
    name_str = sys.argv[1]
    Ykt = float( sys.argv[2] )
    shell_mass = float( sys.argv[3] )
    save_path = sys.argv[4]
    t_final = float( sys.argv[5] )
    condnuc = float( sys.argv[6] )
    coag = float( sys.argv[7] )
    save_data=bool( sys.argv[8] )
    species_name = sys.argv[9] 
    do_main(name_str,species_name, Ykt, shell_mass, save_path, save_data, t_final, condnuc, coag)


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