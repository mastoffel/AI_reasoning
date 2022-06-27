import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

# load simulation output
cult_complex = np.loadtxt('cultural_evolution/output/cult_complex.txt')

# standardize column-wise for PCA
cult_complex_std = (cult_complex - cult_complex.mean(axis=0)) / cult_complex.std(axis=0)

# run a PCA on all_combs to get a composite measure of cultural complexity
pca = PCA(n_components=3)
pca.fit(cult_complex_std)
all_combs_pca = pca.transform(cult_complex_std)

# explained variance per pc in percent
print(pca.explained_variance_ratio_)

# combine all_pars, cult_complex and all_combs_pca into a dataframe
df = pd.DataFrame(np.concatenate((all_pars, cult_complex_std, all_combs_pca), axis=1),
                    columns=['rho1', 'rho2', 'rho3', 'rho4', 'c1', 'c2', 'c3', 'c4', 'c5',
                             'c6', 'c7', 'c8', 'c9', 'c10', 'pc1', 'pc2', 'pc3'])

# 10 iterations for culture groups per parameter set -> add grouping variable
df['culture_group'] = np.repeat(np.arange(0, len(all_pars), 10), 10)

# reverse pc1
df['pc1'] = -df['pc1']

# multi-plot grid with boxplot for rho1, rho2, rho3, rho4 against pc1
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.boxplot(x = 'rho1', y='pc1', data=df, ax=ax[0, 0]).set(xlabel='novel invention rate')
sns.boxplot(x='rho2', y='pc1', data=df, ax=ax[0, 1]).set(xlabel='combination rate')
sns.boxplot(x='rho3', y='pc1', data=df, ax=ax[1, 0]).set(xlabel='modification rate')
sns.boxplot(x='rho4', y='pc1', data=df, ax=ax[1, 1]).set(xlabel='loss rate')
plt.show()

# save plot
fig.savefig('cultural_evolution/figs/cult_evo2.png')