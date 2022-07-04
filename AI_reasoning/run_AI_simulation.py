import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from AI_reasoning.evolve_AI import run_simulation    

# combinations from 0.1 to 1 of rho1, rho2, rho3
rho1_range=rho2_range=rho3_range=judge_range=np.arange(1, 11, 1)/10  

# create all combinations of rho1, rho2, rho3, rho4
par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, judge_range)))  

# get rows where the three rhos are equal to 1, so that one event happens in every iteration
#par_combs_equal_1 = par_combs[par_combs[:,0] + par_combs[:,1] + \
#                              par_combs[:,2] == 1]

# get subset where rho1 is 0.4, rho2 is 0.3 and rho3 is 0.3
par_combs_sub = par_combs[(par_combs[:,0] == 0.4) & \
                                (par_combs[:,1] == 0.3) & \
                                (par_combs[:,2] == 0.3)]

# replicate each ai simulation 10 times
all_pars = np.repeat(par_combs_sub, 10, axis=0)

# add identifier for each simulation, ranging from 0 to 9
all_pars = np.concatenate((all_pars, np.tile(np.arange(0, 10, 1), 10).reshape(-1, 1)), axis=1)

# run simulation for each combination to get cultural complexities
# 10 complexity measures are calculated
num_iter = 100
ai_complex = np.zeros(shape = (num_iter * len(all_pars), 10 + all_pars.shape[1] + 1))

for i in tqdm(range(len(all_pars))):
    
    sim = run_simulation(rho1 = all_pars[i, 0],
                                        rho2 = all_pars[i, 1],
                                        rho3 = all_pars[i, 2],
                                        judge = all_pars[i, 3],
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
ai_complex_df = pd.DataFrame(ai_complex, columns=['rho1', 'rho2', 'rho3', 'judge', 
                                                  'sim', 'iter', 'c1', 'c2', 'c3', 'c4', 'c5',
                                                  'c6', 'c7', 'c8', 'c9', 'c10'])
# filter judge = 0.8
#ai_complex_df = ai_complex_df[ai_complex_df['judge'] == 0.8]

# lineplot for iter vs. c10, hue = sim and facetgrid by judge
g = sns.FacetGrid(ai_complex_df, col="judge", hue="sim", col_wrap=4,
                    sharex=False, sharey=True, size=5, aspect=1.5)
g.map(plt.plot, "iter", "c10")
g.add_legend()
plt.show()



sns.lineplot(x='iter', y='c10', hue='sim', data=ai_complex_df, hue_order=range(10),
                palette=sns.color_palette("hls", 10),
                facet_kws=dict(sharey=False, sharex=False, size=5),
                legend_out=False)


sns.lineplot(x='iter', y='c10', hue='sim', data=ai_complex_df)
plt.show()


# reshape ai_complex into 2d array 
a = np.arange(30).reshape(3, 5, 2)
a = np.zeros((2,2,4))
ai_complex2 = ai_complex.reshape(num_iter * len(all_pars), 10)

# plot first column
plt.plot(ai_complex2[:, 0])
plt.show()
# broadcast all_pars to the shape of ai_complex2
all_pars2 = np.repeat(all_pars, num_iter, axis=0)

# create array with number 1-10 with each number repeated 1000 times in a row and then repeat it 10 times
sim = np.arange(1, 11)    
sim = np.repeat(sim, num_iter)
# repeat full num_array 10 times
sim = np.tile(sim, 10)
# reshape to 2 dimensions
sim = sim.reshape(num_iter * len(all_pars), 1)
# create array with numbers 1-1000 repeated 100 times (num_iter)
iter = np.arange(1, num_iter+1)
# make it integer
iter = np.tile(iter, all_pars.shape[0])
# reshape to 2 dimensions
iter = iter.reshape(num_iter * len(all_pars), 1)

# combine ai_complex2 and all_pars2 into a dataframe
df = pd.DataFrame(np.concatenate((all_pars2, ai_complex2, sim, iter), axis=1), 
                  columns=['rho1', 'rho2', 'rho3', 'judge',
                           'c1', 'c2', 'c3', 'c4', 'c5', 
                            'c6', 'c7', 'c8', 'c9', 'c10', 'sim', 'iter'])

# filter judge = 0.5
df_judge = df[df['judge'] == 0.9]
df_judge

g = sns.FacetGrid(df_judge, col = 'sim', col_wrap = 5, sharey = False, sharex = False).map(plt.scatter, 'iter', 'c1')
g.map(sns.lineplot, 'iter', 'c1')
plt.show()
# plot c10 vs. iter as line plot, with facet grid for each sim
sns.lineplot(x='iter', y='c10', data=df_judge, hue='sim', facet_kws=dict(sharey=False))
plt.show()
sns.FacetGrid(df_judge, hue='sim', size=5).map(plt.scatter, 'iter', 'c10').add_legend()
plt.show()


sns.lmplot('iter', 'c10', data=df_judge, fit_reg=False, hue='judge',    
              palette='Set1', scatter_kws={"s": 100})
plt.show()

# group by iter and plot c10 per iter
sns.lineplot(x='iter', y='c10', data=df_judge)
plt.show()

# long format of dataframe with c1-c9 
pd.wide_to_long(df, stubnames=['c'], i=['rho1', 'rho2', 'rho3', 'judge'], j='c')

# create dataframe with both all_pars and ai_complex2 columns
ai_complex_df = pd.DataFrame(data = ai_complex2, columns = ['rho1', 'rho2', 'rho3', 'judge', 'complexity'])
ai_complex_df['rho1'] = all_pars[:, 0]
df = pd.DataFrame([all_pars, ai_complex2], columns = ['rho1', 'rho2', 'rho3', 'judge', 
                                                    'c1', 'c2', 'c3', 'c4', 'c5', 
                                                    'c6', 'c7', 'c8', 'c9', 'c10'])

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
