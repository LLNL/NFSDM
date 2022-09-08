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
Created on Wed Jan 27 17:15:15 2021

Routines for cloud boundary conditions:
fireball_T - fireball / cloud temperature at time t for system yield W
fireballT_time - time t at which fireball / cloud reaches T
NathansJGR_T - fireball / cloud temperature at time t used in 1970 paper by Nathans et al
NathansJGR_V - fireball / cloud volume used in 1970 paper by Nathans et al
Vol_glasstone - cloud explansion at time t according to Glasstone book
Vol_empirical - empirical cloud expansion at time t with system yield W

@author: mcguffin1
"""
import numpy as np

# Hillendahl Cooling used in doi:10.1016/j.gca.2017.12.011 (Weisz, D G, et al, 2017)
# # dT/dt = -776*W^0.1*t^-1.34 ~ 3e-11*W^-.3*T^4
def fireball_T(W, t ):
    # W: yield (kt)
    # t: time (sec)
    # dT/dt: cooling rate (K/s)
    # dTdt =  -776.*W**0.1*t**(-1.34)
    # dTdt = 3E-11*W**(-0.3)*T**4
    # Solve above two equations for temperature
    return ( 776./3e-11*W**0.4*t**(-1.34) )**(.25)

# Get time at specified temperature
def fireballT_time(W, T ):
    # W: yield (kt)
    # t: time (sec)
    # dT/dt: cooling rate (K/s)
    # dTdt =  -776.*W**0.1*t**(-1.34)
    # dTdt = 3E-11*W**(-0.3)*T**4
    # Solve above two equations for temperature
    return ( 3e-11/776. * W**(-0.4) * T**4 )**(-1/1.34)

def NathansJGR_T(W, t):
    # W: yield (kt)
    # t: time (sec)
    # return temperature (K)
    return 5.95e3*W**(-0.2)*np.exp(-0.796*W**(-0.413)*t)

def NathansJGR_V(W, t):
    # W: yield (kt)
    # t: time (sec)
    # return volume (m3)
    return 3.58e11*W*np.exp( 2.1*W**(-0.413)*t)*1e-6

def Vol_glasstone( t):
    # t : time in seconds
    # return volume m3
    # R : radius in meters
    R = 471*t/60. + 211.4
    return (4./3.)*np.pi*(R)**3

def Vol_empirical( W, t, rho):
    # W : yield (kt)
    # t : time (sec)
    # rho : density at HOB (kg/m3)
    # return V : volume (m3)
    scale_fact = (W/rho)**(1/3)
    tscaled = t/scale_fact
    tscaled1 = 4.8
    t0 = 0.00487*(W/rho)**(2/5)*15. #asymptotic time
    V0 = 4.1888*(67.4*scale_fact)**3# asymptotic volume (m3)
    t1 = tscaled1*scale_fact
    V1 = np.exp( 1.5198*np.log(tscaled1) + 13.854 )*(scale_fact**3)
    Vout = V0
    if t > t0:
        Vout += (V1-V0)*(t-t0)/(t1-t0)
    if tscaled > tscaled1:
        Vscaled = np.exp( 1.5198*np.log(tscaled) + 13.854 )
        Vout = Vscaled*(scale_fact**3)
    return Vout
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