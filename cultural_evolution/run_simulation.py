import itertools
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from evolve_culture import run_simulation

# combinations from 0.1 to 1 of rho1, rho2, rho3, rho4
rho1_range=rho2_range=rho3_range=rho4_range=np.arange(0.1, 1.1, 0.1)
        
# create all combinations of rho1, rho2, rho3, rho4
par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, rho4_range)))       
 
# get rows that are equal to 1, so that one event happens in every iteration
par_combs_equal_1 = par_combs[par_combs[:,0] + par_combs[:,1] + \
                              par_combs[:,2] + par_combs[:,3] == 1]

# replicate each culture_group simulation 10 times
all_pars = np.repeat(par_combs_equal_1, 10, axis=0)
all_pars = all_pars.round(1)

# run simulation for each combination to get cultural complexities
cult_complex = np.zeros(shape = (len(all_pars), 10))
for i in tqdm(range(len(all_pars))):
    cult_complex[i, :] = run_simulation(rho1 = all_pars[i, 0],
                                        rho2 = all_pars[i, 1],
                                        rho3 = all_pars[i, 2],
                                        num_iter=1000)

# save simulation output
np.savetxt('cultural_evolution/output/cult_complex.txt', cult_complex)




    

