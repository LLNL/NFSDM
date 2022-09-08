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
""" Properties of species to represent fallout

Routines for properties:
spec_heat - specific heat for temperature T
surf_tens - surface tension of condensate for temperature T

"""
# molar mass (g/mol)
MW = {'FeO':71.85, 'SrO':103.62 }

# density (g/cm3)
# FeO - Cornell and Schwertmann (2003) = CS03
# SrO - Yaws, Carl L.. (2008) Yaws' Handbook of Physical Properties
#        fro Hydrocarbons and Chemicals: Inorganic Compounds = Yaws 
Rho = {'FeO':5.9,'SrO':5.1 }

# melt temperature (K)
# FeO - CS03, SrO - Yaws
Tm = {'FeO':1377.+273., 'SrO':2531.+273. }

# boil temperature (K)
# FeO - CS03, SrO - Shick, H. L. (1966) Thermodynamics of certain 
#       refractory compounds (pg 111)
Tb = {'FeO':2512.+273., 'SrO':4500. }

# enthalpy vapor (kJ/mol)
# FeO - CS03, SrO - Moore, G. E., Allison, H. W., and Struthers, J. D.. 
#     (1950) The Vaporization of Strontium Oxide. J. Chem. Phys. 18, 1572
Hv = {'FeO':230.3, 'SrO':527. }

#specific heat [J/kg-K]
def spec_heat(A,B,C,D,E, T):
    return A+B*T+C*T**2+D*T**3+E/T**2
# FeO & SrO: Chase, M.W., Jr., NIST-JANAF Thermochemical Tables, fourth
#            edition, J. Phys. Chem. Ref. Data, Monograph 9, 1998, 1-1951
Cp_A = {'FeO':68.2,'SrO':66.94 }
Cp_B = {'FeO':0., 'SrO':1e-7 }
Cp_C = {'FeO':0., 'SrO':-3e-8 }
Cp_D = {'FeO':0., 'SrO':2e-9 }
Cp_E = {'FeO':0., 'SrO':2e-7 }

def surf_tens(dens, molwt, Tb, Ta):
    # density g/cm3
    # molecular weight g/mol
    # boiling point K
    # ambient temperature K
    ## Kou, H., et al. (2019) Temperature-dependent coefficient of
    ##   surface tension prediction model without arbitraty parameters.
    ##   Fluid Phase Equilibria, 484, 53-59. 
    sigma = 2.1e-7*1e4*(Tb*1.5 -Ta)*dens/molwt # N/m
    return sigma 
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