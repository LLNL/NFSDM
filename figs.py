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

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import matplotlib.ticker as ticker
import numpy as np
import coagulation as coag


#=========================== plotting fcns ====================
def make_hist_init_final(Dp0, Nd0, Dpf, Ndf ):
    plt.figure()
    plt.hist(np.log10(Dp0), weights=Nd0/np.nansum(Nd0), bins=10, density=False, color='b', 
                 alpha=0.5, label='Initial')
    plt.hist(np.log10(Dpf), weights=Ndf/np.nansum(Ndf), bins=30, density=False, color='g', 
                 alpha=0.5, label='Final')
    plt.xlabel('Particle diameter (nm)', fontsize=14)
    plt.ylabel('Fraction of particles', fontsize=14)
    # plt.yscale('log')
    # plt.ylim(0,1)
    plt.xticks(ticks=[-1,0,1,2,3], labels=['0.1', '1', '10', '$10^2$', '$10^3$'], fontsize=12 )
def make_fig_cond( ax, x, y, Nsd, text, logy, ylab ):
    ax.set_ylabel(ylab) #$\mu m$
    ax.set_xlabel('Time (s)')
    ax.plot( x,y )
    ax.margins( x=0, y=0)
    plt.title( text )
    plt.grid(which='minor', axis='both')
    #plt.legend(loc='best', fontsize='small')
    if logy:
        plt.yscale( 'log')
def make_fig_radius(ax, x, y, Nconc, yavg, Nsd, xlab, vmin, vmax, weight='Number',
                    mmd=None):
    ax.set_ylabel(xlab, fontsize=16) #$\mu m$
    ax.set_xlabel('Time (sec)', fontsize=16)
    # cm_subsection = numpy.linspace(0., 1., Nsd )
    # color_vals = [cm.viridis(x) for x in cm_subsection ]

    # Choose colormap
    cmap = plt.cm.YlGnBu #PuBuGn #viridis
    
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    cmap1 = colors.ListedColormap(cmap(np.arange(cmap.N)))
    
    # Set alpha
    my_cmap[:,-1] = np.logspace(-1, 0, cmap.N)
    
    # Create new colormap
    my_cmap = colors.ListedColormap(my_cmap)
    
    # Change size of markers so they get smaller over time
    Nt = len( x[:,0])
    size1 = 36
    size2 = 10
    size = np.logspace(np.log10(size1), np.log10(size2), Nt)

    if weight=='Number':
        s = Nconc # 1/cm3
        clab = 'Number concentration ($cm^{-3}$)'
    elif weight=='Area':
        s = Nconc*np.pi*(y*1e-3)**2 # um2/cm3
        clab = 'Surface Area per volume ($\mu m^2 cm^{-3}$)'
    elif weight=='Volume':
        s = Nconc*np.pi*(y*1e-3)**3/6. # um3/cm3
        clab = 'Volume concentration ($\mu m^3 cm^{-3}$)'
        
    if vmin < 0:
        vmin = 10**(np.floor( np.nanmin(np.log10(s)) ) )
    if vmax < 0:
        vmax = 10**(np.ceil( np.nanmax(np.log10(s)) ) )
    for i in range(Nt):
        indx = np.argsort( s[i,:])
        indxsrt = indx[::-1]
        plt.scatter( x[i,indxsrt],y[i,indxsrt], c=s[i,indxsrt],
                     s=size[i]*np.ones( (len(indx),) ), cmap=my_cmap, 
                    norm=LogNorm( vmin=vmin, vmax=vmax) )
    s = plt.scatter( [],[], c=[], s=size[0], cmap=cmap1, 
                    norm=LogNorm( vmin=vmin, vmax=vmax) )
    # if mmd is not None:
    plt.scatter( [],[],  s=size[0], color='None', label='SD' , 
                    edgecolors='k')
    # hist, bins = np.histogram(x, bins=bins)
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    # yout = np.log10(y)
    # yout[ np.isnan(yout) ] = -30
    # plt.hist2d(x, yout, s)
    if mmd is not None:
        plt.plot( x[:,0], mmd, '-.', color='k', label='MMD')
    plt.plot( x[:,0], yavg, '--', color='red', label='NMD')
    # ax.margins( x=0, y=0)
    # if mmd is not None:
    plt.legend(loc='best', fontsize='small')
    plt.yscale( 'log')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.ylim(0.1, 200)
    plt.xlim(0, 45)
    fig = plt.gcf()
    plt.subplots_adjust(top = .9, bottom = 0.2, right = .86, 
       			 left = 0.12)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    clb = plt.colorbar(s,format=LogFormatterMathtext(), cax = cbaxes)
    clb.ax.set_title(clab, fontsize=12, position=(-3,0) )
    ax.tick_params(which='both', top=True, right=True)
    # plt.grid(axis='y')
    
    
def make_fig_TS( ax, ts, T1, T2, S1, S2, labx, labT, labS, log=False ):
    ax.plot( ts*np.ones( (100,)), np.linspace(T2[1].min(),T1[1].max(),100), 'k--')
    ax.plot( T1[0], T1[1], 'd-', color='red')
    ax.plot( T2[0], T2[1], 'd-', color='red')  
    ax.set_xlabel(labx, fontsize=14)
    ax.set_ylabel(labT, color='red', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot( S1[0], S1[1], 'o-', color='blue')
    ax2.plot( S2[0], S2[1], 'o-', color='blue')
    plt.yscale('log')
    if log:
        plt.xscale('log')
    ax2.set_ylabel(labS, color='blue', fontsize=16)
    plt.xlim(0,45)
    
def make_fig_TV( ax, ts, T2, S2, labx, labT, labS, log=False ):
    # ax.plot( ts*np.ones( (100,)), np.linspace(T2[1].min(),T1[1].max(),100), 'k--')
    ax.plot( T2[0], T2[1], 'd-', color='red')  
    ax.set_xlabel(labx, fontsize=14)
    ax.set_ylabel(labT, color='red', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot( S2[0], S2[1], 'o-', color='blue')
    plt.yscale('log')
    if log:
        plt.xscale('log')
    ax2.set_ylabel(labS, color='blue', fontsize=16)
    ax.set_ylim(0,2400.)
    #ax2.set_ylim(10**7, 10**9)
    plt.xlim(ts)
    
def make_fig_NtotDpm( ax, t, N, d, xlab, ylabN, ylabD, log=True ):
    ax.plot( t, N, '-', color='red')
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylabN, color='red', fontsize=16)
    if log:
        plt.yscale('log')
    ax2 = ax.twinx()
    ax2.plot( t, d, '--', color='blue')
    # plt.yscale('log')
    ax2.set_ylabel(ylabD, color='blue', fontsize=16)
    
def make_fig_SanityCheck(t, vap, part, mass0, dNuc, dCond, dCoag,  MassBal=False,
                         Particles=False, Vapor=False):
    if MassBal:
        plt.subplots()
        N = len( t)
        plt.plot( t, vap, linestyle='--', label='vapor')
        plt.plot( t, part, linestyle='--', label='particles')
        plt.plot( t, part+vap-mass0*np.ones(N, ), 
                 label='mass balance')
        plt.xlabel('seconds')
        plt.ylabel('Mass FeO (g)')
        plt.legend()
        plt.title( 'Overall Mass Balance = %3.2e g' %np.sum( abs( part+
                                    vap-mass0*np.ones(N, )) )  )
    if Particles:
        plt.figure()
        plt.plot( t, dNuc[0], label='nucleation')
        plt.plot( t, dCond[0],'--', label='condensation')
        plt.plot( t, dCoag, label='coagulation')
        plt.legend()
        plt.ylabel('$\Delta$ particle mass (g)')
    if Vapor:
        plt.figure()
        plt.plot( t, dNuc[1], label='nucleation')
        plt.plot( t, dCond[1], '--', label='condensation')
        plt.legend()
        plt.ylabel('$\Delta$ vapor mass (g)')
def make_fig_diag( ax, x1, y1, y2, x3, y3, Nsd, text1, text2, text3, logy, ylab ):
	ax.set_ylabel(ylab)
	ax.set_xlabel('Time (sec)')
	ax.plot(x1, y1, 'd-', alpha=0.4)
	ax.plot([],[], 'd-', alpha=0.4, label=text1)
	ax.plot(x1, y2)
	ax.plot([],[], label=text2)
	ax.plot(x3, y3,'k-', label=text3)
	ax.margins(x=0,y=0)
	plt.grid(which='minor', axis='both')
	plt.legend(loc='best', fontsize='small')
	if logy:
		plt.yscale('log')
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