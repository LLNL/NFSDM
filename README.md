---
"Nuclear Fallout Super-droplet Method (NFSDM)"
---

This is a python code to simulate the formation and evolution of particles in an airburst as a homogeneous fireball expands and cools. The
numerical model applied is a Lagrangian particle super-droplet method. This work has been published: "Super-droplet Method to Simulate Lagrangian Microphysics of Nuclear Fallout in a Homogeneous Cloud" D. L. McGuffin, D. D. Lucas, J. P. Morris, G. D. Spriggs, K. B. Knight. Journal of Geophysical Research: Atmospheres **Vol**, Iss (2022)[doi:10.1016/xxx]().

Running the model
----------------

To run the model, call wrapper.py from the command line with inputs: ( maximum time (sec), Yield (kt), Mass (kg), scaling factor applied to condensation and nucleation rates (-), scaling factor applied to coagulation kernels (-), string of species name (FeO, SrO), path to directory to save data )
Example:
$ python -u wrapper.py 60 10 1500 1 1 FeO /path_to_save_directory/

Description of routines
----------------

The wrapper.py calls main.py, which starts the master time loop. Inside a master time loop condensation, nucleation, and coagulation are simulated. The respective py files contain the code to perfom each process, and do_microphysics.py calls the condensation.py, nucleation.py, and coagulation.py.
Additionally, adapt_superdroplets.py is used to merge super-droplets. Sizedist.py and figs.py are used to convert between super-droplets and real particle size distribution, and to create figures, respectively. Cooling.py includes the boundary conditions for the cloud expansion and cooling. Thermo_properties.py includes dictionaries for the thermodynamic properties of SrO and FeO.

License
----------------

The code is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE](https://github.com/LLNL/NFSDM/blob/main/LICENSE) and [NOTICE](https://github.com/LLNL/NFSDM/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE-839524
