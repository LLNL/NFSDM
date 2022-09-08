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
Routines for Coagulation of particles due to their Brownian motion

Functions defined below:
sigmoid - a sigmoid function
gaussian - a Gaussian function
air_viscosity - air viscosity as a function of T
air_molwt - air molecular weight as a function of T
air_kt - air thermal conductivity as a function of T
air_cp - air specific heat as a function of T
coag_coeff - coagulation coefficient of two particles at T
cont_corr - continuum correction factor
coag_pairs - create list of super-droplet pairs for MCM
mcm_coag - Monte carlo method (MCM) for SDM coagulation
mcm_coag_test - Monte carlo method (MCM) for verifying code with constant coagulation kernel

@author: mcguffin1
"""
import numpy, math

"Sigmoid function"
def sigmoid(x, c, d):
    y = (x-c)/d
    return ( math.exp(y) / ( math.exp(y)+ math.exp(-y))  )

"Gaussian function"
def gaussian(x,c,d):
    y = -( (x-c) / d )**2
    return math.exp(y)

def air_viscosity(tk):
    "Air viscosity kg/m/s"
    
    "parameters"
    a_mu = numpy.array([2.4206e-4, 1.2681e-4, -3.4926e-4, -1.2445e-5, -4.6583e-6])
    c_mu = numpy.array([7968.7,2428.4,13037,29022,45190])
    d_mu = numpy.array([4197.4,2511.6,2983.4,3230,9661.5])

    mu_sum = 0.
    for i in range(len(a_mu)):
        mu_sum = mu_sum + a_mu[i]*sigmoid(tk,c_mu[i],d_mu[i])
        
    mu = mu_sum + (-8.2881e-4 + 1573.6*gaussian(tk,842.03,2.9207e-4))/(2+tk**.94898)
    return mu

def air_molwt(tk):
    "Air mean molar mass (g/mol)"
    "parameters"
    a = numpy.array([1.0355,1.2139,2.3122,7.2030,10.085,4.0978])
    c = numpy.array([61980.,43921.,28653.,13676.,6521.3,3384.7])
    d = numpy.array([-11923.,-7495.6,-5004.1,-3025.8,-1201.1,-561.79])
    
    mw_sum = 0.
    for i in range(len(a)):
        mw_sum = mw_sum + a[i]*sigmoid(tk, c[i], d[i])
        
    mw = mw_sum + 2.9059
    
    mw = 40.
    return mw

def air_kt(tk):
    "Air thermal conductivity (W/K/m)"
    a_j = [1.6027, 3.5183, 0.51834, 0.046393]
    c_j = [14327., 6830., 3486.3, 1295.4]
    D_j = [3174., 1252.3, 770.33, 1065.9]
    a5 = -7.3543e3
    c5 = 1.7290e6
    D5 = -1.5675
    
    lambda_sum = 1./(a5 + c5*tk**D5 )
    for i in range( len(c_j)):
        lambda_sum += a_j[i]*gaussian(tk,c_j[i],D_j[i])
        
    return lambda_sum

def air_cp(tk):
    "Air specific heat (J/kg/K)"
    a_j = [1.466, 2.0586, 1.9846, 0.51666, 2.7110, 3.9774, 3.9040, 4.9202, 5.918]
    c_j = [7822.6, 27693., 88546., 3547.4, 6931.9, 14830., 29169., 46703., 65610.]
    D_j = [9413.3, 1932.2, 5839.2, 668.06, 1148.7, 2877.2, 4046., 5044.4, 10886.]
    
    Cp_cal_g = 0.
    for i in range( 3, 9):
        Cp_cal_g += a_j[i]*gaussian(tk, c_j[i], D_j[i])
        
    for i in range(3):
        Cp_cal_g += a_j[i]*sigmoid(tk, c_j[i], D_j[i])
        
    return Cp_cal_g*4.184*1e3

"Coagulation Kernel calculation from Don Lucas:"
def coag_coeff(dp1, dp2, mu, molwt,rho, tk):

    # Pressure (Pa)
    press = 101325.
    # Air mean free path (m) :
    lamb_air = 2.*mu/(press*numpy.sqrt(8.*molwt*1.e-3/(numpy.pi*8.314*tk)))

    boltz = 1.38e-23             # J/K

    rho1 = rho * (1.e3)          # kg/m3
    rho2 = rho * (1.e3)          # kg/m3

    m1 = (rho1*numpy.pi*dp1**3)/6.
    m2 = (rho2*numpy.pi*dp2**3)/6.

    knud1 = 2*lamb_air/dp1
    knud2 = 2*lamb_air/dp2
    mspd1 = numpy.sqrt(8*boltz*tk/(numpy.pi*m1))
    mspd2 = numpy.sqrt(8*boltz*tk/(numpy.pi*m2))

    D1pre = (boltz*tk/(3.*numpy.pi*mu*dp1))
    D1num = 5. + 4.*knud1 + 6.*knud1**2 + 18.*knud1**3
    D1den = 5. - knud1 + (8. + numpy.pi)*knud1**2
    D1 = D1pre*D1num/D1den

    D2pre = (boltz*tk/(3.*numpy.pi*mu*dp2))
    D2num = 5. + 4.*knud2 + 6.*knud2**2 + 18.*knud2**3
    D2den = 5. - knud2 + (8. + numpy.pi)*knud2**2
    D2 = D2pre*D2num/D2den

    ell1 = 8.*D1/(numpy.pi*mspd1)
    ell2 = 8.*D2/(numpy.pi*mspd2)
    gee1 = numpy.sqrt(2)*( ((dp1 + ell1)**3 - (dp1**2 + ell1**2)**1.5)/
                          (3.*dp1*ell1) - dp1 ) #
    gee2 = numpy.sqrt(2)*( ((dp2 + ell2)**3 - (dp2**2 + ell2**2)**1.5)/
                          (3.*dp2*ell2) - dp2 ) #

    kterm1 = 2.*numpy.pi*(D1+D2)*(dp1 + dp2)
    kterm2 = (dp1 + dp2)/(dp1 + dp2 + 2*numpy.sqrt(gee1**2 + gee2**2))
    kterm3 = 8.*(D1 + D2)/(numpy.sqrt(mspd1**2 + mspd2**2)*(dp1 + dp2))

    return 1.e6*kterm1/(kterm2 + kterm3)

"Calculation of beta, continuum correction factor, to adjust condensation rate"
"  using approach by Fuchs and Sutugin (1971)"
def cont_corr(dp, tk, molwt, Da, alpha):
    # knudsen number using Fuchs & Sutugin's mean free path
    cbar = numpy.sqrt( 8*8.314*tk/(numpy.pi*molwt)*1.e3) #m/s
    kn= 6.*Da*1.e-4/cbar/dp 
    
    return 0.75*alpha*(1+kn)/( kn**2 + kn + 0.283*kn*alpha + 0.75*alpha )

def coag_pairs( Nsd, sd, base_sd ):
    # list of super-droplets
    I = numpy.arange(0,Nsd)
    # Use different random seed for each function call
    numpy.random.seed(sd+base_sd) 
    # random permutation
    I = numpy.random.permutation(I)
    # random list of pairs
    Npair = int( numpy.floor( Nsd/2) )
    L = numpy.zeros( (Npair,2), dtype=numpy.int )
    ct = 0
    for s in range(Npair):
        L[s,] = [int(I[ct]), int(I[ct+1]) ]
        ct += 2
        
    return L

def mcm_coag(L, mult, age, Rd, Td, Nsd, dt, dV, T, base_sd, rho, scale):
    # L : list of random super-droplet pair indices
    # mult : multiplicity [#]
    # Rd : radius array [nm]
    # Td : particle temperature [K]
    # dt : time step [s]
    # dV : volume [cm3]
    # T : temperature [K]
    # rho: density [g/cm3]
    
    mu = air_viscosity(T)
    molwt = air_molwt(T)
    
    dR = numpy.zeros( (Nsd,) )
    remv_sd = numpy.empty( 0, int)
    Kall = []
    rd1 = []
    rd2 = []
    #evaluate if coagulation should happen for each pair
    for a in range( len(L) ):
        # Use different random seed for each pair of super-droplets 
        numpy.random.seed(a+base_sd) # + base_seed, i.e. yyyymmdd
        # Get random number from uniform distribution over [0,1)
        phi = numpy.random.uniform(low=0., high=1.)
        #get particle indices
        j = L[a][ numpy.argmax( mult[L[a]] ) ] # larger multiplicity
        k = L[a][ numpy.where( L[a]!=j ) ]
        rd1.append( Rd[j] )
        rd2.append( Rd[k] )
        # coagulation kernel
        Kcoag_Br = coag_coeff(Rd[j]*2*1e-9, Rd[k]*2*1e-9, mu,molwt,rho, T)*scale
        rho_kgm3 = rho*1e3
        # probability
        p = mult[j]*Kcoag_Br*dt/dV*Nsd*(Nsd-1)/( 2*len(L) )
        if phi < (p - numpy.floor(p) ):
            gamma = numpy.floor(p) +1
        else:
            gamma = numpy.floor(p)
        if gamma != 0:
            # print( 'Coagulation!')
            gamma_ = min( gamma, numpy.floor(mult[j]/mult[k]) )
            if (mult[j]-gamma_*mult[k]) > 0:
                dR[k] = ((gamma_*Rd[j]**3 + Rd[k]**3)**(1/3) - Rd[j])
                Td[k] = (Td[j]*gamma_ + Td[k])/(gamma_+1.)
                mult[j] = mult[j]-gamma_*mult[k]
                Rd[k] = (gamma_*Rd[j]**3 + Rd[k]**3)**(1/3)
                age[k] = max( age[k], age[j])
                # print( 'xi_j %i xi_k %i with gamma frac %3.2f' %(mult[j],mult[k], gamma_) )
                # mass[k] = gamma_*mass[j] + mass[k]
            elif (mult[j]-gamma_*mult[k]) == 0:
                dR[k] = ((gamma_*Rd[j]**3 + Rd[k]**3)**(1/3) - Rd[j])
                Td[j] = (Td[j] + Td[k])/2.
                Td[k] = Td[j]
                mult[j] = numpy.floor(mult[k]/2)
                mult[k] = mult[k] - numpy.floor(mult[k]/2)
                Rd[j] = (gamma_*Rd[j]**3 + Rd[k]**3)**(1/3)
                Rd[k] = Rd[j]
                age[j] = max( age[k], age[j])
                
                # mass[k] = gamma_*mass[j] + mass[k]
                # mass[j] = mass[k]
            if mult[j] == 0:
                remv_sd = numpy.append( remv_sd, int(j) )
                print( 'all of superdroplet % i scavenged!' %j )
                
        if len( remv_sd)> 0:
            mult_out = numpy.delete(mult, remv_sd )
            Rd_out = numpy.delete( Rd, remv_sd )
            age_out = numpy.delete( age, remv_sd )
            Td_out = numpy.delete( Td, remv_sd )
        else:
            mult_out = mult
            Rd_out = Rd
            age_out = age
            Td_out = Td
            
    Kall = numpy.asarray( Kall )
    rd1 = numpy.asarray( rd1 )
    rd2 = numpy.asarray( rd2 )
    rdcoag = (rd1, rd2)
          
    return mult_out, Rd_out, age_out, Td_out, dR, Kall, rdcoag
def mcm_coag_test(L, mult, age, Rd, Nsd, dt, dV, Kcoag, base_sd):
    # L : list of random super-droplet pair indices
    # mult : multiplicity [#]
    # Rd : radius array [nm]
    # dt : time step [s]
    # dV : volume [cm3]
    # T : temperature [K]
    
    remv_sd = []
    #evaluate if coagulation should happen for each pair
    for a in range( len(L) ):
        numpy.random.seed(a+base_sd) # + base_seed, i.e. yyyymmdd
        # Get random number from uniform distribution over [0,1)
        phi = numpy.random.uniform(low=0., high=1.)
        #get particle indices
        j = L[a][ numpy.argmax( mult[L[a]] ) ] # larger multiplicity
        k = L[a][ numpy.where( L[a]!=j ) ]
        # probability
        p = mult[j]*Kcoag*(dt/dV)*( Nsd*(Nsd-1)/2)/(Nsd/2)
        if phi < (p - numpy.floor(p) ):
            gamma = numpy.floor(p) +1
        else:
            gamma = numpy.floor(p)
        if gamma != 0:
            # print( 'Coagulation!')
            gamma_ = min( gamma, numpy.floor(mult[j]/mult[k]) )
            if (mult[j]-gamma_*mult[k]) > 0:
                mult[j] = mult[j]-gamma_*mult[k]
                Rd[k] = (gamma_*Rd[j]**3 + Rd[k]**3)**(1/3)
                age[k] = max( age[k], age[j])
                # print( 'xi_j %i xi_k %i with gamma frac %3.2f' %(mult[j],mult[k], gamma_) )
                # mass[k] = gamma_*mass[j] + mass[k]
            elif (mult[j]-gamma_*mult[k]) == 0:
                mult[j] = numpy.floor(mult[k]/2)
                mult[k] = mult[k] - numpy.floor(mult[k]/2)
                Rd[j] = (gamma_*Rd[j]**3 + Rd[k]**3)**(1/3)
                Rd[k] = Rd[j]
                age[j] = max( age[k], age[j])
                # print( 'xi_j %i xi_k %i with %i collisions' %(mult[j],mult[k], 
                #              numpy.floor( mult[k]/2)) )
                # mass[k] = gamma_*mass[j] + mass[k]
                # mass[j] = mass[k]
            if mult[j] == 0:
                remv_sd = numpy.append( remv_sd, j)
                print( 'all of superdroplet % i scavenged!' %j )
            if mult[k] == 0:
                remv_sd = numpy.append( remv_sd, k)
                print( 'all of superdroplet % i scavenged!' %k )
                
        if len( remv_sd)> 0:
            mult_out = numpy.delete(mult, remv_sd )
            Rd_out = numpy.delete( Rd, remv_sd )
            age_out = numpy.delete( age, remv_sd )
        else:
            mult_out = mult
            Rd_out = Rd
            age_out = age
          
    return mult_out, Rd_out, age_out

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