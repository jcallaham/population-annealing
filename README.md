### Population Annealing Simulations of a Binary Hard Sphere Mixture

Data and code from Callaham and Machta, Phys. Rev. E (2017)
https://arxiv.org/abs/1701.00263

#### code\
Contains source code to run the actual PA simulations and save observables.
Requires OpenMP to compile and run in parallel

prng.hpp is the Sitmo parallel random number generator (https://www.sitmo.com/?p=1206)


#### data\
Copy of the data we reported in the paper, including the results of bootstrapping the weighted averages

#### analysis.py
Python 3 script that will perform bootstrapping (unless this is commented out), weighted averaging, and other data analysis
This will generate most of the figures published in the paper
