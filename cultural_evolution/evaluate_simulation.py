import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

# load simulation output as dataframe
cult_complex = pd.read_csv('cultural_evolution/output/cult_complex.txt', sep='\t')

# standardize c1 to c10 column-wise for PCA
cult_complex_std = cult_complex.copy()
cult_complex_std.loc[:, 'c1':'c14'] = (cult_complex_std.loc[:, 'c1':'c14'] - \
    cult_complex_std.loc[:, 'c1':'c14'].mean()) / cult_complex_std.loc[:, 'c1':'c14'].std()

# run a PCA on c1 to c10 in cult_complex_std to get a composite measure of cultural complexity
pca = PCA(n_components=2)
pca.fit(cult_complex_std.loc[:, 'c1':'c14'])

# explained variance per pc in percent
print(pca.explained_variance_ratio_)

# get the PCs and add to cult_complex_std dataframe 
df = pd.DataFrame(np.concatenate((cult_complex_std, pca.transform(cult_complex_std.loc[:, 'c1':'c14'])), axis=1),
                                columns=['rho1', 'rho2', 'rho3', 'rho4', 'c1', 'c2', 'c3', 'c4', 'c5',
                                            'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13',
                                            'c14','group', 'pc1', 'pc2'])
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