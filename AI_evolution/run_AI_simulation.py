# cultural evolution inspired AI simulation 

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from tqdm import tqdm
from AI_evolution.evolve_AI import run_simulation    

# suppress scientific notation in array
np.set_printoptions(suppress=True)

# combinations from 0.2 to 1 of rho1, rho2, rho3, judge, reason
rho1_range=rho2_range=rho3_range=np.arange(2, 7, 2)/10  

judge_range=reason_range=np.arange(2, 11, 2)/10  

# create all combinations of rho1, rho2, rho3, rho4
par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, judge_range,
                                            reason_range)))  

# get subset where rho1-rho3 sum up to 1 to that one action happens in 
# every iteration
par_combs_sub = par_combs[(par_combs[:,0] + par_combs[:,1] + par_combs[:,2] == 1)]

# run 10 replicates for each parameter combination
all_pars = np.repeat(par_combs_sub, 10, axis=0)

# add identifier from 1 to len(all_pars) to each row 
all_pars = np.hstack((np.arange(1, len(all_pars)+1).reshape(-1,1), all_pars))

# run simulation and calculate complexity for each parameter combination    
num_iter = 500
ai_complex = np.zeros(shape = (num_iter * len(all_pars), 10 + all_pars.shape[1] + 1))

for i in tqdm(range(len(all_pars))):
    sim, reason = run_simulation(rho1 = all_pars[i, 1],
                                        rho2 = all_pars[i, 2],
                                        rho3 = all_pars[i, 3],
                                        judge = all_pars[i, 4],
                                        reason = all_pars[i, 5],
                                        reinvest=True,
                                        num_iter=num_iter)
    # replicate all_pars[i] num_iter times
    pars = np.tile(all_pars[i], num_iter).reshape(num_iter, all_pars.shape[1])
    # add column for iteration number as integer
    pars = np.concatenate((pars, np.arange(0, num_iter, 1).reshape(-1, 1)), axis=1)
    # add pars to sim columnwise
    sim = np.concatenate((pars, sim), axis=1)
    # fill ai_complex with sim
    ai_complex[i*num_iter:(i+1)*num_iter, :] = sim

# ai_complex to dataframe
ai_complex_df = pd.DataFrame(ai_complex, columns=['sim_id', 'rho1', 'rho2', 'rho3', 'judge', 'reason',
                                                  'iter', 'c1', 'c2', 'c3', 'c4', 'c5',
                                                  'c6', 'c7', 'c8', 'c9', 'c10'])

# savee ai_complex_df to csv
ai_complex_df.to_csv('AI_evolution/output/ai_complex_df_rec.csv', index=False)


