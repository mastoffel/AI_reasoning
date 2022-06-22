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

rho1 = 0.5
rho2 = 0.1
rho3 = 0.2
rho4 = 1 - rho1 - rho2 - rho3

# check intersection
def has_intersection(a, b):
        return not set(a).isdisjoint(b)
    
# run simulation


def run_simulation(rho1, rho2, rho3, rho4, num_iter=100):
    
    # make np array with 10 seed traits with names a-j
    seed_traits = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    # initialise culture group with two seed traits drawn at random
    culture_group = np.random.choice(seed_traits, 2, replace=False)
    culture_group_size = 2

    for i in range(num_iter):
        # draw random number between 0 and 1
        r = np.random.random()
        # 1) new seed trait is introduced (necessary if there are no traits)
        if r < rho1 or culture_group_size == 0:
            # add new seed trait to culture group that doesn't exist in the group yet
            # seed traits not in group
            seed_traits_not_in_group = np.setdiff1d(seed_traits, culture_group)
            # if all seed traits present in group repeat iteration
            if len(seed_traits_not_in_group) == 0:
                continue
            culture_group = np.append(culture_group, np.random.choice(seed_traits_not_in_group, 1, replace=False))
            culture_group_size += 1
        # two of the cultural traits present are combined
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
            # combine traits
            culture_group = np.append(culture_group, trait_1[0] + trait_2[0])
            culture_group_size += 1
        # one of the cultural traits is modified
        elif r < rho1 + rho2 + rho3:
            # draw random trait from culture group
            trait = np.random.choice(culture_group, 1, replace=False)
            # modify trait
            culture_group = np.append(culture_group, trait[0] + 'm')
            culture_group_size += 1
        # one of the cultural traits is lost
        else:
            # draw random trait from culture group
            trait = np.random.choice(culture_group, 1, replace=False)
            # remove trait
            culture_group = np.setdiff1d(culture_group, trait)
            culture_group_size -= 1
    return culture_group
        
culture_group

# create pandas dataframe with combinations from 0.1 to 1 of rho1, rho2, rho3, rho4
# and run simulation for each combination
    
rho1_range = np.arange(0.1, 1.1, 0.1)
rho2_range = np.arange(0.1, 1.1, 0.1)
rho3_range = np.arange(0.1, 1.1, 0.1)
rho4_range = np.arange(0.1, 1.1, 0.1)
        
# create all combinations of rho1, rho2, rho3, rho4
all_combinations = np.array(list(itertools.product(rho1_range, rho2_range, rho3_range, rho4_range)))        
# get rows that are equal to 1
all_combinations_equal_1 = all_combinations[all_combinations[:,0] + all_combinations[:,1] + all_combinations[:,2] + all_combinations[:,3] == 1]

# run simulations for all combinations
culture_group_list = []
for i in range(len(all_combinations_equal_1)):
    # get rhos from first row, functional programming
    rho1 = all_combinations_equal_1[i,0]
    rho2 = all_combinations_equal_1[i,1]
    rho3 = all_combinations_equal_1[i,2]
    rho4 = all_combinations_equal_1[i,3]
    
    sim = run_simulation(rho1, rho2, rho3, rho4, num_iter=5)
    culture_group_list.append(sim)
    
# make pandas DataFrame with and rhos
simulation_df = pd.DataFrame(all_combinations_equal_1, columns=['rho1', 'rho2', 'rho3', 'rho4'])
simulation_df

# calculate cultural complexity from culture_group_list
# measure 1: the number of traits per element
number_of_traits = [len(sim) for sim in culture_group_list] 

# measure 2: trait complexity: the mean length of traits in each culture_group
def trait_complexity(culture_group):
    if culture_group.size == 0: return 0
    traits = [len(trait) for trait in culture_group]
    return np.mean(traits)
trait_complexity = list(map(trait_complexity, culture_group_list))

# measure 3:

