import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from AI_reasoning.evolve_AI import run_simulation    

# combinations from 0.1 to 1 of rho1, rho2, rho3
rho1_range=rho2_range=rho3_range=judge_range=np.arange(0.1, 1.1, 0.1)   

# create all combinations of rho1, rho2, rho3, rho4
par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, judge_range)))  

# get rows where the three rhos are equal to 1, so that one event happens in every iteration
#par_combs_equal_1 = par_combs[par_combs[:,0] + par_combs[:,1] + \
#                              par_combs[:,2] == 1]

# get subset where rho1 is 0.2, rho2 is 0.4 and rho3 is 0.4
par_combs_sub = par_combs[(par_combs[:,0] == 0.2) & \
                                (par_combs[:,1] == 0.4) & \
                                (par_combs[:,2] == 0.4)]

# test
#par_combs_equal_1 = par_combs_equal_1[107:109, :]

# replicate each culture_group simulation 10 times
all_pars = np.repeat(par_combs_sub, 10, axis=0)
all_pars = all_pars.round(1)

# run simulation for each combination to get cultural complexities
# 10 complexity measures are calculated
num_iter = 1000
ai_complex = np.zeros(shape = (num_iter, 10, len(all_pars)))

for i in tqdm(range(len(all_pars))):
    ai_complex[:, :, i] = run_simulation(rho1 = all_pars[i, 0],
                                        rho2 = all_pars[i, 1],
                                        rho3 = all_pars[i, 2],
                                        judge = all_pars[i, 3],
                                        num_iter=num_iter)


# reshape ai_complex into dataframe with 3rd dimension as grouping variable
ai_complex2 = ai_complex.reshape(num_iter * len(all_pars), 10)

# plot last column as lineplot
plt.plot(ai1[:, 1])
plt.show()
ai_complex_df = pd.DataFrame(ai_complex.reshape(num_iter * 10, 10), 10)

# combine all_pars, cult_complex into a dataframe
df = pd.DataFrame(np.concatenate((all_pars, cult_complex), axis=1),
                    columns=['rho1', 'rho2', 'rho3', 'rho4', 'c1', 'c2', 'c3', 'c4', 'c5',
                             'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14'])

# 10 iterations for culture groups per parameter set -> add grouping variable
df['culture_group'] = np.repeat(np.arange(0, len(all_pars), 10), 10)

# save dataframe
df.to_csv('cultural_evolution/output/cult_complex.txt', sep='\t', index=False)
#np.savetxt('cultural_evolution/output/cult_complex.txt', df)
