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

Routines to do condensation:
kelvin_vap_pressure - vapor pressure over curved surface
dPdR - derivative of Kelvin vapor pressure with respect to particle radius
flat_sat_pressure - vapor pressure over flat surface
Nusselt - dimensionless nusselt number for mass transfer
dNudR - derivative of Nusselt number with respect to particle radius
condevap_aerosol - routine for condensation to aerosol with implicit Euler integration
condevap_simple - routine for condensation in simple test problem (no Kelvin effect)
condevap_analytical - analytical solution to simple test problem
condevap_aerosol_explicit - routine for condensation to aerosol with explicit Euler integration

"""
import numpy as np
import coagulation as coag

#================== Property calculation fcns ==========================
def kelvin_vap_pressure( Pflat, r, T, factor ):
    # print( 'Kelvin effect over %3.2e m radius' %r )
    Pd = Pflat*np.exp( factor/(r*T))
    return Pd

def dPdR( Pflat, r, T, factor):
    Pd = Pflat*np.exp( factor/(r*T))
    return -factor/(r**2*T)*Pd

def flat_sat_pressure( Hv, Tboil, T):
    # Hv in J/mol, T's in Kelvin
    return np.exp( Hv/8.314*(1/Tboil - 1/T))*101325 #Pa

#================= Non-continuum correction =========================
def Nusselt( rdi, MWv, MWa, T, Dva, Pd, Pa):
    # Pa, Pd : Pascal
    R = 8.314
    Dva = Dva/1e4 # m2/s
    Td = T
    Ta = T
    Tm = 1./3.*(2.*Td + Ta)
    # rdi : radius in meters
    Numm_1 = Pd/np.sqrt(2*np.pi*R/MWv*Td)
    Numm_2 = Pa/np.sqrt(2*np.pi*R/MWa*Ta)
    Numm = 2*rdi*R/MWv*Tm/(Dva*(Pd-Pa))*(Numm_1-Numm_2)
    return Numm

def dNUdR( rdi, MWv, MWa, T, Dva, Pd, Pa, dPdR):
    R = 8.314
    Dva = Dva/1e4 # m2/s
    Td = T
    Ta = T
    Tm = 1./3.*(2.*Td + Ta)
    # rdi : radius in meters
    b1 = 1./np.sqrt(2*np.pi*R/MWv*Td)
    Numm_1 = Pd*b1
    b2 = 1./np.sqrt(2*np.pi*R/MWa*Ta)
    Numm_2 = Pa*b2
    a = R/(MWv*1e3)*Tm/Dva
    dNummdR = (2.*a/(Pd-Pa) - 2.*rdi*a*dPdR/(Pd-Pa)**2)*(Numm_1-Numm_2) + 2.*rdi*a/(Pd-Pa)*b1*dPdR
    return dNummdR

# ==================== condensation routines =====================================

# Implicit Euler applied to condensation equation
def condevap_aerosol(mult, Rd,T_sd, Dva,MWv, MWa, dens_p, P_a, P_s, dt, T, Tm, pd_input,
                     cpv, Lv, itr_max, debug_flg, logname, scale ):
    # a = D*Mw/(dens_particle*R*T)
    # dens_p : particle density g/cm3
    # Dva: vapor diffusion coefficient cm2/s
    # P_a: ambient vapor partial pressure kPa
    # P_s: saturation vapor pressure / pressure at particle surface) kPa
    # pd_input : for Kelvin effect K-m
    # Cpv : heat capacity J/kg-K
    # Lv : latent heat vaporization J/g
    # dconc = change in vapor concentration
    if debug_flg:
        f = open(logname, 'a')
    
    # parameters for heat transfer
    dens_m3 = dens_p*1e6
    kt = coag.air_kt(T) #67.68*1e-3 #W/m-K
    Penv = 101325 #air pressure (Pa)
    cpa = coag.air_cp(T) #1005.2 #J/kg-K
    Ra = 287 #J/kg-K
    Rv = 8.314*1.e3/MWv # specific gas constant species (J/kg/K)
    R = 8314.46 #kPa-cm3/mol-K
    
    Nsd = len( Rd )
    dvapor_mass = np.zeros( (Nsd,))
    dTemp = np.zeros( (Nsd,))
    Rd_out = np.zeros( (Nsd,))
    for i in range( Nsd):
        crd = Rd[i]*1e-9 # (m)
        new_rd = Rd[i]*1e-9 # (m)
        # Newton-Raphson iterations
        rdi = new_rd
        Rd_iters = np.zeros( (itr_max,))
        Rd_iters[0] = rdi*1e9 #(nm)
    
        for k in range(itr_max-1):
            n = k+1
            rd2 = rdi*rdi #m2
            # surface saturation pressure over curved surface
            P_d = kelvin_vap_pressure( P_s, rdi, T, pd_input ) #kPa   ...*1e-3
            #Nusselt numb for mass transport
            Nu_m = Nusselt( rdi, MWv, MWa, T, Dva, P_d, P_a) #(-)
            # Nusselt numb for heat transport = 2...
            # Nusselt number
            Nu = 2.*Nu_m/(2.+Nu_m) 
            a = Dva*MWv/(2.*dens_p*R*T)*1e-4 # coefficients (m2/s-kPa)
            I = a*Nu*(P_a-P_d)*scale  # Growth rate * radius (m2/s)
            #no evaporation
            if (I<0):
                I = 0
            
            # Solve for radius-squared by finding zero of function:
            #  f = rdi(t+1)**2 - rdi(t)**2 - dt*a*Nu(rdi)*[P_a - P_d(rdi)]
            #  f = rd2 - oldRterm - Rterm...        
            dRterm = dt*I # change in radius squared with current guess
            oldRterm = crd*crd # radius squared at previous time step
            
            ## For Newton-Raphson, need derivative of f wrt rd2 (ddRterm)...
            # derivative of surface pressure with current guess (kPa/m2) 
            dPd = dPdR(P_s, rdi, T, pd_input )*1e-3
            # derivative of Nu_m with current guess (1/m2)
            dNum = dNUdR( rdi, MWv, MWa, T, Dva, P_d, P_a, dPd) 
            # derivative of Nusselt number with current guess (1/m2)
            dNu = 4./(2.+Nu_m)**2*dNum 
            
            #derivative of function wrt radius-squared (-)
            ddRterm = -1. + 2.*dt*a*(dNu*(P_a-P_d) - Nu*dPd)
        
            dtmp = rd2 - ( oldRterm - rd2 + 2.*dRterm )/ ddRterm

            #print( 'iter #'+str(n)+': r^2= '+str(dtmp) )
            if( dtmp <=0.): dtmp = rdi*rdi*1e-4
            
            diff  = abs( rdi - np.sqrt(dtmp) )/rdi
            rdi = np.sqrt( dtmp )
            Rd_iters[n] = rdi*1e9 #(nm)
            
            # Check if converged: radius changed by less than 1% of original val
            if diff >= 1e-10:
                if debug_flg:
                    if I > 0:
                        f.write('## COND : iteration %i for SD no. %i fractional change in radius %e\n' 
                                %(k,i, diff))
                        f.write( '## COND : previous radius (nm) %f and new radius %f\n'
                                %(Rd_iters[n-1], Rd_iters[n]) )
            else:
                if debug_flg:
                    if I > 0:
                        f.write('## COND : converged on iteration %i for SD no. %i\n' 
                                %(k,i))
                        f.write( '## COND : previous radius (nm) %f and new radius %f\n'
                                %(Rd_iters[n-1], Rd_iters[n]) )
                break
            
        Rd_out[i] = rdi*1e9 #(nm)
        drdt = (rdi- crd)/dt
        # calculate heat rate for delta T:
        Num_h = 2*rdi/kt*( P_a*1e3*(cpv - 0.5*Rv)/np.sqrt(2*np.pi*Rv*T) + Penv*
                          (cpa - 0.5*Ra)/np.sqrt(2*np.pi*Ra*T) )
        Nuh = 2*Num_h/(2+Num_h) 
        dTemp[i] = drdt*2*rdi*dens_m3*Lv/(Nuh*kt) #K
        
        dT = ( 4.*np.pi/3.*(rdi**3-crd**3)*dens_m3/dt )*Lv/(2.*np.pi*rdi*kt*Nuh) 
        # calculate mass condensed
        dvapor_mass[i] = dRterm*4.*np.pi*rdi*dens_m3 # g
    if debug_flg:
        f.close()
    T_sd = T + dTemp
    return Rd_out, T_sd, dvapor_mass


def condevap_simple( Rd, Dva,MWv, MWa, dens_p, P_a, P_s, dt, T, Tm, pd_input,
                     cpv, Lv, itr_max ):
    # a = D*Mw/(dens_particle*R*T)
    # dens_p : particle density g/cm3
    # Dva: vapor diffusion coefficient cm2/s
    # P_a: ambient vapor partial pressure kPa
    # P_s: saturation vapor pressure / pressure at particle surface) kPa
    # pd_input : for Kelvin effect K-m
    # Cpv : heat capacity J/kg-K
    # Lv : latent heat vaporization J/g
    # dconc = change in vapor concentration
    R = 8314.46 #kPa-cm3/mol-K
    Nsd = len( Rd )
    dvapor_mass = np.zeros( (Nsd,))
    dTemp = np.zeros( (Nsd,))
    Rd_out = np.zeros( (Nsd,))
    for i in range( Nsd):
        crd = Rd[i]*1e-9 # (m)
        new_rd = Rd[i]*1e-9 # (m)
        # Newton-Raphson iterations
        rdi = new_rd
        Rd_iters = np.zeros( (itr_max,))
        Rd_iters[0] = rdi*1e9 #(nm)
    
        for k in range(itr_max-1):
            n = k+1
            rd2 = rdi*rdi #m2
            # Simplify by neglecting Kelvin effect 
            P_d = P_s
            #Nusselt numb for mass transport
            Nu_m = Nusselt( rdi, MWv, MWa, T, Dva, P_d, P_a) #(-)
            # Nusselt numb for heat transport = 2...
            # Nusselt number
            Nu = 2.*Nu_m/(2.+Nu_m) 
            a = Dva*MWv/(2.*dens_p*R*T)*1e-4 # coefficients (m2/s-kPa)
            I = a*Nu*(P_a-P_d)  # Growth rate * radius (m2/s)
            #no evaporation
            if (I<0):
                I = 0
            
            # Solve for radius-squared by finding zero of function:
            #  f = rdi(t+1)**2 - rdi(t)**2 - dt*a*Nu(rdi)*[P_a - P_d(rdi)]
            #  f = rd2 - oldRterm - Rterm...        
            dRterm = dt*I # change in radius squared with current guess
            oldRterm = crd*crd # radius squared at previous time step
            
            ## For Newton-Raphson, need derivative of f wrt rd2 (ddRterm)...
            # derivative of surface pressure with current guess (kPa/m2) 
            dPd = dPdR(P_s, rdi, T, pd_input )*1e-3
            # derivative of Nu_m with current guess (1/m2)
            dNum = dNUdR( rdi, MWv, MWa, T, Dva, P_d, P_a, dPd) 
            # derivative of Nusselt number with current guess (1/m2)
            dNu = 4./(2.+Nu_m)**2*dNum 
            
            #derivative of function wrt radius-squared (-)
            ddRterm = -1. + 2.*dt*a*(dNu*(P_a-P_d) - Nu*dPd)
        
            dtmp = rd2 - ( oldRterm - rd2 + 2.*dRterm )/ ddRterm

            #print( 'iter #'+str(n)+': r^2= '+str(dtmp) )
            if( dtmp <=0.): dtmp = rdi*rdi*1e-4
            
            rdi = np.sqrt( dtmp )
            Rd_iters[n] = rdi*1e9 #(nm)

        Rd_out[i] = rdi*1e9 #(nm)
        drdt = (rdi- crd)/dt
        # calculate heat rate for delta T:
        dens_m3 = dens_p*1e6
        kt = 67.68*1e-3 #W/m-K
        Penv = 101325 #Pa
        cpa = 1005.2 #J/kg-K
        Ra = 287 #J/kg-K
        Rv = 8.314*1.e3/MWv # specific gas constant species (J/kg/K)
        Num_h = 2*rdi/kt*( P_a*1e3*(cpv - 0.5*Rv)/np.sqrt(2*np.pi*Rv*T) + Penv*
                          (cpa - 0.5*Ra)/np.sqrt(2*np.pi*Ra*T) )
        Nuh = 2*Num_h/(2+Num_h) #1.99
        dTemp[i] = -drdt*2*rdi*dens_m3*Lv/(Nuh*kt) #K
        # calculate mass condensed
        dvapor_mass[i] = dRterm*4.*np.pi*rdi*dens_m3 # g
    return Rd_out, dTemp, dvapor_mass
def condevap_analytical( t, t0, Rd0, Rdt, Dva,MWv, MWa, dens_p, P_a, P_s, dt, T, Tm, pd_input,
                     cpv, Lv, itr_max ):
    # a = D*Mw/(dens_particle*R*T)
    # dens_p : particle density g/cm3
    # Dva: vapor diffusion coefficient cm2/s
    # P_a: ambient vapor partial pressure kPa
    # P_s: saturation vapor pressure / pressure at particle surface) kPa
    # pd_input : for Kelvin effect K-m
    # Cpv : heat capacity J/kg-K
    # Lv : latent heat vaporization J/g
    # dconc = change in vapor concentration
    R = 8314.46 #kPa-cm3/mol-K
    Nsd = len( Rd0 )
    Rd_out = np.ones( (Nsd,))
    d_out = np.ones( (Nsd,))
    
    for i in range( Nsd):
        di = 2.*Rdt[i]*1e-9 #meters
        d0 = 2.*Rd0[i]*1e-9
        P_d = P_s
        #Nusselt numb for mass transport
        Nu_m = Nusselt( di/2., MWv, MWa, T, Dva, P_d, P_a) #(-)
        # Nusselt number
        Nu = 2.*Nu_m/(2.+Nu_m) 
        a = Dva*MWv/(2.*dens_p*R*T)*1e-4 # coefficients (m2/s-kPa)
        I = 4.*a*Nu*(P_a-P_d)  # Growth rate * radius (m2/s) 
        d_out[i] = np.sqrt( d0**2 + 2.*I*(t-t0) )
      
    Rd_out = d_out/2.*1e9 # nm
            
    return Rd_out

# Explicit Euler applied to condensation equation 
def condevap_aerosol_explicit( Rd, Dva,MWv, MWa, dens_p, P_a, P_s, dt, T, Tm, pd_input, itr_max ):
    # a = D*Mw/(dens_particle*R*T)
    #(D: vapor diffusion coefficient, pa: ambient vapor partial pressure,
    # ps: saturation vapor pressure / pressure at particle surface)
    R = 8314.46 #kPa-cm3/mol-K
    Nsd = len( Rd )
    for i in range( Nsd):
        rdi = Rd[i]*1e-6 # (m)
        P_d = kelvin_vap_pressure( P_s, rdi, T, pd_input )#kPa 
        Nu_m = Nusselt( rdi, MWv, MWa, T, Dva, P_d, P_a)
        Nu = 2.*Nu_m/(2.+Nu_m)
        a = Dva*MWv/(2.*dens_p*R*T)*1e-4 # m2/s-kPa
        I = a*Nu*(P_a-P_d) 
        rdi = np.sqrt( rdi**2 + 2*dt*I )
        Rd[i] = rdi*1e6 #(um)     
    return Rd
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