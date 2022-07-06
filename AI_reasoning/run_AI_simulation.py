import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from AI_reasoning.evolve_AI import run_simulation    

# suppress scientific notation in array
np.set_printoptions(suppress=True)

# combinations from 0.2 to 1 of rho1, rho2, rho3, judge, reason
rho1_range=rho2_range=rho3_range=judge_range=reason_range=np.arange(2, 11, 2)/10  

# create all combinations of rho1, rho2, rho3, rho4
par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, judge_range,
                                            reason_range)))  

# get subset where rho1 is 0.4, rho2 is 0.3 and rho3 is 0.3
# par_combs_sub = par_combs[(par_combs[:,0] == 0.4) & \
#                                 (par_combs[:,1] == 0.3) & \
#                                 (par_combs[:,2] == 0.3)]


# get subset where rho1-rho3 sum up to 1
par_combs_sub = par_combs[(par_combs[:,0] + par_combs[:,1] + par_combs[:,2] == 1)]

# replicate each ai simulation 10 times
all_pars = np.repeat(par_combs_sub, 10, axis=0)

# add identifier from 1 to len(all_pars) to each row 
all_pars = np.hstack((np.arange(1, len(all_pars)+1).reshape(-1,1), all_pars))


# run simulation for each combination to get cultural complexities
# 10 complexity measures are calculated
num_iter = 1000
ai_complex = np.zeros(shape = (num_iter * len(all_pars), 10 + all_pars.shape[1] + 1))

for i in tqdm(range(len(all_pars))):
    
    sim = run_simulation(rho1 = all_pars[i, 1],
                                        rho2 = all_pars[i, 2],
                                        rho3 = all_pars[i, 3],
                                        judge = all_pars[i, 4],
                                        reason = all_pars[i, 5],
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
ai_complex_df.to_csv('AI_reasoning/output/ai_complex_df.csv', index=False)


