#! /bin/env python
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
Routines to run MAIN SDM, either once from command line arguments or several times in parallel
getlhspoints - get Latin Hypercube Sampling points from range
sdm - Run an instance of SDM Main code
run - Run either a single instance or several instances in parallel

"""
import main as main_code
import numpy as np
import pandas as pd
from datetime import date
import sys

def sdm(i,inputs):
	(Ys,Ms, tsim, species, save_data, condnuc_fact, coag_fact, path)=inputs
	Y = Ys[i]
	mass = Ms[i]
	name = '%05.0fkt_%05.0fkg' %(Y,mass)
	print( 'Hello from RUN! start scenario %i' %i)
	out = main_code.do_main(name,species, Y, mass, path, save_data, tsim, condnuc_fact, coag_fact)
	print( 'Finished scenario %i' %i )
	return out 
def run(ttime,species, Yin, Min, condnuc_fact,coag_fact,save_path):
	print('Hello from Main wrapper!')
	## How many second to simulate microphysics?
	dt = 0.1 #seconds
	Nt = int( ttime/dt + 2 )

	## Set up scenario
	Y1 = np.asarray([Yin])
	M1 = np.asarray([Min])

	save_data = True
	# Model inputs: Yield (kt), Mass (kg), max time simulated, name of species,
	#               logical if data saved for each simulation, scaling factor
	#               applied to Condensation & Nucleation rates, scaling factor
	#               applied to coagulation kernels, path to save results
	second_arg = (Y1,M1,ttime,species,save_data, condnuc_fact,
                         coag_fact, save_path)
	# run SDM scenarios in parallel
	result = sdm(0, inputs=second_arg)
	
	# return pd.concat( result, axis=1 )
if __name__ == "__main__":
	print('Hello!')
	## Parse inputs #########################################
	# Maximum time simulated (sec)
	tot_sec = float( sys.argv[1] )
	# Yield, kt 
	Yval = float( sys.argv[2] )
	#  Mass, kg 
	Mval = float( sys.argv[3] )
	# scaling factor for condensation & nucleation
	condnuc_fact = float( sys.argv[4] )
	# scaling factor for coagulation kernel
	coag_fact = float( sys.argv[5] )
	# Name of species (must be in thermo_properties, i.e. 'FeO', 'SrO' )
	species = sys.argv[6]
	# path to save model output
	save_path = sys.argv[7]

	## Run code and save all output in dataframe ###################
	run(tot_sec, species, Yval, Mval, condnuc_fact, coag_fact, save_path)  

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
