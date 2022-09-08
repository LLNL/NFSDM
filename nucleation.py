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
Routines for nucleation:
classic_homog - calculate nucleation rate and nucleated particle size
nuc2sd - convert nucleation rate and size to super-droplets

@author: mcguffin1
"""


import numpy as np
import coagulation as coag, condensation as cond, cooling
import thermo_properties as dat
#from numba import jit 

#@jit(nopython=True)
def classic_homog(MW, sigma, rho_l, T, Hv, Tb, conc_vap, dV):
    ## INPUTS ######################################
    # MW: molecular weight (g/mol)
    # sigma: surface tension (N/m -- kg/s2)
    # sigma = sigma*1e4
    # rho_l: density of condensate (g/cm3)
    rho_l = rho_l*1e3 #(kg/m3)
    # T: temperature (K)
    # Hv: latent heat of vaportization (J/mol)
    # Tb: boiling point (K)
    # conc_vap: mass of vapor (g/cm3)
    # dV: box volume (cm3)
    ###################################################
    
    #### unit conversion and calculation of parameters
    m = MW/6.022e23*1e-3 # kg/mlc
    M_vap = conc_vap*dV*1e-3 #kg
    N0 = M_vap/m # no. of mlc
    Rv = 8.31446/MW*1e3 #Pa-m3/kg-K
    kB = 1.381e-23 # m2-kg/s2-K
    p = conc_vap*8314.46/MW*T #kPa  R: kPa-cm3/(mol-K)
    # pe: saturation vapor pressure (flat surface) (kPa)
    pe = cond.flat_sat_pressure( Hv, Tb, T)*1e-3 #kPa
    S = p/pe # --
    ###################################################
    if np.log(S) < 0.:
        rd_nuc = 0.
        I = 0.
    else:
        ## critical radius of nucleated cluster #######
        rd_crit = 2.*sigma*m/(rho_l*kB*T*np.log(S) ) # meters
        # nucleate clusters with i*+1 monomers
        i_crit = np.floor( 4*np.pi/3*rd_crit**3*rho_l/m )
        rd_nuc = ((i_crit*m/rho_l)*3/(4*np.pi))**(1/3)
    
    	### Calculating nucleation rate #####################################
    	## Number of clusters according to stat thermo Boltzmann distbn
        Nembryo = N0*np.exp( -4.*np.pi*sigma*rd_nuc**2/(3*kB*T) )
    	## Nucleation rate of stable clusters with probability of stabilization
        I = 2*pe*1e3/(rho_l*np.sqrt(2*np.pi*Rv*T) )*np.sqrt(sigma/(kB*T) )*Nembryo
    	# I: #/s
    	###################################################################
    return I, rd_nuc 

def nuc2sd( nucnumb, nucsize, Nsd, Temp):
    phi = np.random.uniform(low=0., high=1.)
    if phi >= 0.5:
        dN = np.floor( nucnumb/Nsd )
    else:
        dN = np.ceil( nucnumb/Nsd )
    Rd = np.ones( (Nsd,))*nucsize*1e9 #nm
    mult = np.ones( (Nsd,) )*dN # no. real particles
    age = np.zeros( (Nsd,) )
    Td = np.ones( (Nsd,) )*Temp
    
    return Rd, mult, age, Td

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
