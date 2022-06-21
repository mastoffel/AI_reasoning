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
from pandas import cut

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

# make np array with 10 seed traits with names a-j
seed_traits = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

# initialise culture group with two seed traits drawn at random
culture_group = np.random.choice(seed_traits, 2, replace=False)

# check intersection
def has_intersection(a, b):
        return not set(a).isdisjoint(b)
    
# run simulation
culture_group_size = 2
for i in range(100):
    print(culture_group, culture_group_size, i, sep='\t')
    # draw random number between 0 and 1
    r = np.random.random()
    # if r < rho1 or if there are no traits, new seed trait is introduced
    if r < rho1 or culture_group_size == 0:
        # add new seed trait to culture group that doesn't exist in the group yet
        # seed traits not in group
        seed_traits_not_in_group = np.setdiff1d(seed_traits, culture_group)
        # if all seed traits present repeat iteration
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
    # if r < rho1 + rho2 + rho3, one of the cultural traits is modified
    elif r < rho1 + rho2 + rho3:
        # draw random trait from culture group
        trait = np.random.choice(culture_group, 1, replace=False)
        # modify trait
        culture_group = np.append(culture_group, trait[0] + 'm')
        culture_group_size += 1
    # if r < rho1 + rho2 + rho3 + rho4, one of the cultural traits is lost
    else:
        # draw random trait from culture group
        trait = np.random.choice(culture_group, 1, replace=False)
        # remove trait
        culture_group = np.setdiff1d(culture_group, trait)
        culture_group_size -= 1
        

culture_group
        
        
        


