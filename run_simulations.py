import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from evolve_culture import run_simulation

# combinations from 0.1 to 1 of rho1, rho2, rho3, rho4
rho1_range=rho2_range=rho3_range=rho4_range=np.arange(0.1, 1.1, 0.1)
        
# create all combinations of rho1, rho2, rho3, rho4
# make sure it's maximally one decimal place

par_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, rho4_range)))        
# get rows that are equal to 1
par_combs_equal_1 = par_combs[par_combs[:,0] + par_combs[:,1] + \
                              par_combs[:,2] + par_combs[:,3] == 1]

# replicate each culture_group simulation 10 times
all_pars = np.repeat(par_combs_equal_1, 10, axis=0)
all_pars = all_pars.round(1)

# run simulation for each combination to get cultural complexities
cult_complex = np.zeros(shape = (len(all_pars), 5))
for i in range(len(all_pars)):
    cult_complex[i, :] = run_simulation(rho1 = all_pars[i, 0],
                                        rho2 = all_pars[i, 1],
                                        rho3 = all_pars[i, 2])

# run a PCA on all_combs to get a composite measure of cultural complexity
pca = PCA(n_components=3)
pca.fit(cult_complex)
all_combs_pca = pca.transform(cult_complex)
# explained variance per pc in percent
print(pca.explained_variance_ratio_)

# combine all_pars, cult_complex and all_combs_pca into a dataframe
df = pd.DataFrame(np.concatenate((all_pars, cult_complex, all_combs_pca), axis=1),
                    columns=['rho1', 'rho2', 'rho3', 'rho4', 'c1', 'c2', 'c3', 'c4', 'c5', 'pc1', 'pc2', 'pc3'])
# add grouping cult_complex to all_pars, every 10 rows is a new culture_group simulation
df['culture_group'] = np.repeat(np.arange(0, len(all_pars), 10), 10)


# multi-plot grid with boxplot for rho1, rho2, rho3, rho4 against pc1
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.boxplot(x = 'rho1', y='pc1', data=df, ax=ax[0, 0]).set(xlabel='novel invention rate')
sns.boxplot(x='rho2', y='pc1', data=df, ax=ax[0, 1]).set(xlabel='combination rate')
sns.boxplot(x='rho3', y='pc1', data=df, ax=ax[1, 0]).set(xlabel='modification rate')
sns.boxplot(x='rho4', y='pc1', data=df, ax=ax[1, 1]).set(xlabel='loss rate')
plt.show()


    

