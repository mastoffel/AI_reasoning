# trying to replicate model from Lewis&Laland 2012 for the evolution
# of cumulative culture in humans

# primary factors for building cumulative culture:
# 1) transmission fidelity (loss rate)
# 2) novel invention rate
# 3) modification rate
# 4) combination rate

from cmath import e
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from operator import itemgetter

from pyparsing import line

# next event can be one of four options:
# 1) new seed trait is introduced through novel invention with probability rho1
# 2) two of the cultural traits present are combined to produce a new cultural trait with probability rho2
# 3) one of the cultural traits is modified to produce new variant of the trait with probability rho3
# 4) one of the cultural traits is lost with probability rho4

# all traits can be refined or combined with all other traits.
# with the exception that composite traits cannot have the same seed trait.

# simulation one: the four events are constrained so that rho1 + rho2 + rho3 + rho4 = 1
# one of the four events must happen (assumed variable time between events)
# running time is 5000 events
# if number of traits in the group falls to zero, the next event
# is a new seed trait introduced through novel invention

# check intersection
def has_intersection(a, b):
        return not set(a).isdisjoint(b)
    
# simulation
def run_simulation(rho1, rho2, rho3, rho4, num_iter=1000):
    
    # make np array with 10 seed traits with names a-j
    seed_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    
    # give each seed_trait a utility value drawn from a random uniform distribution
    # betwee 0.75 and 1. 
    seed_utilities = np.random.uniform(size=10, low=0.75, high=1.)
    
    # combine seed names and their utilities. 
    trait_utilities = {}
    for i in range(0, len(seed_names)):
        trait_utilities[seed_names[i]] = seed_utilities[i]
        
    # initialise culture group with two seed traits drawn at random
    culture_group = np.random.choice(seed_names, 2, replace=False)
    
    for i in range(num_iter):
        
        # draw random number between 0 and 1
        r = np.random.random()
        # 1) new seed trait is introduced (necessary if there are no traits)
        if r < rho1 or len(culture_group) == 0:
            # check which seed traits are not in group 
            seed_traits_not_in_group = np.setdiff1d(seed_names, culture_group)
            
            # if all seed traits present in group repeat iteration
            if len(seed_traits_not_in_group) == 0:
                continue
            
            # otherwise take a random seed trait to add to culture group
            culture_group = np.append(culture_group, 
                                      np.random.choice(seed_traits_not_in_group, 
                                                       1, replace=False))
            
        # 2) two of the cultural traits present are combined
        elif r < rho1 + rho2:
            # draw two random traits from culture group
            trait_1 = np.random.choice(culture_group, 1, replace=False)
            
            # remove letter 'm' from trait_1
            trait_1_seed = trait_1[0].replace('m', '')
            
            # check for traits in culture_group with no overlap in seed traits
            remaining_traits = []
            for trait in culture_group:
                if not has_intersection(trait_1_seed, trait):
                    remaining_traits.append(trait)
                    
            # if there is nothing to combine with, repeat iteration
            if remaining_traits == []:
                i = i-1
                continue
            
            trait_2 = np.random.choice(remaining_traits, 1, replace=False)
            
            # combine traits and add to culture group
            culture_group = np.append(culture_group, trait_1[0] + trait_2[0])
            
            # calculate utility of new trait by taking maximum among seed traits
            # and adding value from N(0, 0.1)
            utils = itemgetter(trait_1[0], trait_2[0])(trait_utilities)
            new_util = np.max(utils) + np.random.normal(0, 0.1)
            
            # add to trait_utilities
            trait_utilities[culture_group[-1]] = new_util
            
        # 3) one of the cultural traits is modified
        elif r < rho1 + rho2 + rho3:
            # draw random trait from culture group
            trait = np.random.choice(culture_group, 1, replace=False)
            
            # modify trait
            culture_group = np.append(culture_group, trait[0] + 'm')
            
            # utility is modified by adding value from N(0, 0.1) to
            trait_utilities[culture_group[-1]] = trait_utilities[trait[0]] +\
                                                    np.random.normal(0, 0.1)
            
        # 4) one of the cultural traits is lost
        
        else:
            # utilities for each trait in culture group
            trait_utils = [trait_utilities[trait] for trait in culture_group]
            
            # translate into relative probabilities
            trait_probs = [trait_util/np.sum(trait_utils) for trait_util in trait_utils]
            
            # draw random trait from culture group
            trait = np.random.choice(culture_group, 1, replace=False, 
                                     p = trait_probs)
            
            # remove trait
            culture_group = np.setdiff1d(culture_group, trait)
       
    return culture_group

# create pandas dataframe with combinations from 0.1 to 1 of rho1, rho2, rho3, rho4
# and run simulation for each combination
rho1_range=rho2_range=rho3_range=rho4_range=np.arange(0.1, 1.1, 0.1)
        
# create all combinations of rho1, rho2, rho3, rho4
all_combs = np.array(list(itertools.product(rho1_range, rho2_range, 
                                            rho3_range, rho4_range)))        
# get rows that are equal to 1
all_combs_equal_1 = all_combs[all_combs[:,0] + all_combs[:,1] + \
                              all_combs[:,2] + all_combs[:,3] == 1]

# run simulations for all combinations
culture_group_list = []
for i in range(len(all_combs_equal_1)):
    # get rhos from first row, functional programming
    rho1 = all_combs_equal_1[i,0]
    rho2 = all_combs_equal_1[i,1]
    rho3 = all_combs_equal_1[i,2]
    rho4 = all_combs_equal_1[i,3]
    
    sim = run_simulation(rho1, rho2, rho3, rho4, num_iter=5)
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
