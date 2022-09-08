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
Created on Tue Aug 31 10:37:32 2021

Routine to merge super-droplets if the number exceeds the limit

@author: mcguffin1
"""
import numpy as np


def merge(Rsd, Msd, age, Tsd, Nm, Nk, debug, filename, tval):
    # Rsd: radius of superdroplets (nm)
    # Msd: multiplicity of superdroplets (#)
    # age: age of superdroplets (sec)
    # Tsd: temperature "" (K)
    # Nm : threshold for allowable number of superdroplets
    # Nk : threshold for number of smallest superdroplets to keep
    
    # current number of superdroplets
    Nsd = len( Rsd)
    
    # superdroplets to return
    Nsdf = min( Nsd, Nm )
    Nmerge =int( np.ceil( (Nsd - Nsdf)*2 ) )
    Rf = np.zeros( (Nsdf,) )
    Mf = np.zeros( (Nsdf,) )
    age_f = np.zeros( (Nsdf,) )
    Tf = np.zeros( (Nsdf,) )
   
    # print( 'Merge? %i Nsd' %Nsd )
    
    # only merge if we have more than the threshold 
    if Nmerge > 0:
        indx = np.argsort( Rsd )
        Rsort = Rsd[indx] # superdroplets from smallest to largest
        Msort = Msd[indx]
        age_sort = age[indx]
        Tsort = Tsd[indx]
        
        #Fill arrays with smallest particles not being merged
        Nf = Nsd - Nmerge
        
        Rf[:Nf] =  Rsort[:Nf]
        Mf[:Nf] =  Msort[:Nf]
        age_f[:Nf] =  age_sort[:Nf]
        Tf[:Nf] = Tsort[:Nf]
        Rmerge = Rsort[Nf:] # list to merge
        Mmerge = Msort[Nf:]
        age_merge = age_sort[Nf:]
        Tmerge = Tsort[Nf:]
        f = open(filename, 'a')
        f.write('MERGING @ %f seconds-----\n' %tval )
    
        k1 = 0 # index of superdroplet 1 to merge
        k2 = 1 # index of superdroplet 2 to merge
        n = Nf # index of final superdroplet
        for i in range( int( np.ceil(Nmerge/2)) ):
            if ((Rmerge[k2]-Rmerge[k1])/(Rmerge[k1]+Rmerge[k2]) <= 0.05):
                w1 = Mmerge[k1]/(Mmerge[k1]+Mmerge[k2])
                w2 = 1. - w1
                Rf[n] = ( Rmerge[k1]**3*w1 + Rmerge[k2]**3*w2 )**(1/3)
                Mf[n] = Mmerge[k1] + Mmerge[k2]
                age_f[n] = w1*age_merge[k1] + w2*age_merge[k2]
                Tf[n] = w1*Tmerge[k1] + w2*Tmerge[k2]
                f.write('Merge SD %i and %i to no. %i\n' %(k1,k2,n) )
                f.write('%e & %e = %e multiplicity\n' %(Mmerge[k1],Mmerge[k2],Mf[n]) )
                f.write('%f & %f --> %f nm radius\n' %(Rmerge[k1],Rmerge[k2],Rf[n]) )
                n += 1
                k1+= 1 
                k2 += 1
                merged = True
            else:
                # find a different pair that can be merged
                dR = np.diff( Rsort[Nk:] )/Rsort[Nk:-1]
                indx = np.argmin( dR )
                if dR[indx]<= 0.05:
                    n = Nk+indx
                    Rf[:n] =  Rsort[:n]
                    Mf[:n] =  Msort[:n]
                    age_f[:n] =  age_sort[:n]
                    Tf[:n] = Tsort[:n]
                    k1 = Nk+indx
                    k2 = Nk+indx+1
                    w1 = Msort[k1]/(Msort[k1]+Msort[k2])
                    w2 = 1. - w1
                    Rf[n] = ( Rsort[k1]**3*w1 + Rsort[k2]**3*w2 )**(1/3)
                    Mf[n] = Msort[k1] + Msort[k2]
                    age_f[n] = w1*age_sort[k1] + w2*age_sort[k2]
                    Tf[n] = w1*Tsort[k1] + w2*Tsort[k2]
                    Rf[n+1:] =  Rsort[k2+1:]
                    Mf[n+1:] =  Msort[k2+1:]
                    age_f[n+1:] =  age_sort[k2+1:]
                    Tf[n+1:] = Tsort[k2+1:]
                    merged = True
                    f.write('Merge SD %i and %i to no. %i\n' %(k1,k2,n) )
                    f.write('%e & %e = %e multiplicity\n' %(Msort[k1],Msort[k2],Mf[n]) )
                    f.write('%f & %f --> %f nm radius\n' %(Rsort[k1],Rsort[k2],Rf[n]) )
                else:
                    merged = False
                    Rf = Rsd
                    Mf = Msd
                    age_f = age
                    Tf = Tsd
            
        
        f.close()
        
    else:
        Rf = Rsd
        Mf = Msd
        age_f = age
        Tf = Tsd
        merged = False
        
    
    return Rf, Mf, age_f, Tf, merged
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