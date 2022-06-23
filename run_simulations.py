

# create pandas dataframe with combinations from 0.1 to 1 of rho1, rho2, rho3, rho4
# and run simulation for each combination
rho1_range=rho2_range=rho3_range=rho4_range=np.arange(0.1, 1.1, 0.1)
        
# create all combinations of rho1, rho2, rho3, rho4
all_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, rho4_range)))        
# get rows that are equal to 1
all_combs_equal_1 = all_combs[all_combs[:,0] + all_combs[:,1] + \
                              all_combs[:,2] + all_combs[:,3] == 1]

# run simulation for each combination
all_combs_mean = np.zeros(shape = (len(all_combs_equal_1), 5))
for i in range(len(all_combs_equal_1)):
    all_combs[i, :] = run_simulation(rho1 = all_combs_equal_1[i, 0],
                                        rho2 = all_combs_equal_1[i, 1],
                                        rho3 = all_combs_equal_1[i, 2],
                                        rho4 = all_combs_equal_1[i, 3])


# run simulations for all combinations
culture_group_list = []
for i in range(len(all_combs_equal_1)):
    # get rhos from first row, functional programming
    rho1 = all_combs_equal_1[i,0]
    rho2 = all_combs_equal_1[i,1]
    rho3 = all_combs_equal_1[i,2]
    rho4 = all_combs_equal_1[i,3]
    
    sim = run_simulation(rho1, rho2, rho3, rho4, num_iter=500)
    culture_group_list.append(sim)
    
# make pandas DataFrame with and rhos
sim_df = pd.DataFrame(all_combs_equal_1, columns=['rho1', 'rho2', 'rho3', 'rho4'])

# calculate cultural complexity from culture_group_list
# measure 1: the number of traits per element
sim_df['trait_number'] = [len(sim) for sim in culture_group_list] 

# measure 2: trait complexity: the mean length of traits in each culture_group
def get_trait_complexity(culture_group):
    if culture_group.size == 0: return 0
    traits = [len(trait) for trait in culture_group]
    return np.mean(traits)
sim_df['trait_complexity'] = list(map(get_trait_complexity, culture_group_list))

# measure 3: number of lineages in each culture group
# defined as starting with the same seed trait
def get_lineage_number(culture_group):
    if culture_group.size == 0: return 0
    lineages = {trait[0] for trait in culture_group}
    return len(lineages)
sim_df['lineage_number'] = list(map(get_lineage_number, culture_group_list))

# measure 4: mean lineage complexity
def get_lineage_complexity(culture_group):
    if culture_group.size == 0: return 0
    lineage_complexity = [len(trait.replace('m', '')) for trait in culture_group]
    return np.mean(lineage_complexity)
sim_df['lineage_complexity'] = list(map(get_lineage_complexity, culture_group_list))

# Calculate PCA from trait_number, trait_complexity, lineage_number and linear_complexity
# PCA is calculated using sklearn.decomposition.PCA
# PCA is used to reduce the number of features to 1



# get mean traits and lineages for rho4_range in sim_df
sim_df.groupby(['rho4']).agg('mean')
