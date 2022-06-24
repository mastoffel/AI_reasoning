# trying to replicate model from Lewis&Laland 2012 for the evolution
# of cumulative culture in humans

# primary factors for building cumulative culture:
# 1) transmission fidelity (loss rate)
# 2) novel invention rate
# 3) modification rate
# 4) combination rate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math

from operator import itemgetter
from complexity_measures import get_complexity

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
    
def sigmoid(x):
      return 1 / (1 + math.exp(-x))
    
# simulation 
def run_simulation(rho1, rho2, rho3, num_iter=500):
    """Cultural evolution simulation. Starts with two (out of ten) seed traits, allows
    new traits to be introduced through novel invention (rho1), combination 
    of existing traits (rho2), and modification of existing traits (rho3). The
    loss rate or transmission fidelity (rho4) is assumed to be 1 - (rho1 + rho2 + rho3).
    

    Args:
        rho1 (_type_): probability of introducing a new seed trait through novel invention
        rho2 (_type_): probability of combining two existing traits to produce a new trait
        rho3 (_type_): probability of modifying an existing trait to produce a new variant
        num_iter (int, optional): Iterations. The last 20% are averaged over
        to calculate complexity. Defaults to 500.

    Returns:
        np.array: mean of 5 complexity measures over the last 20% of iterations.
    """
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
    
    # inititalise array of complexities over time
    cultural_complexity = np.zeros(shape = (round(0.2*num_iter), 10))
    
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
            if len(culture_group) > 1:
                
                # utilities for each trait in culture group
                trait_utils = [trait_utilities[trait] for trait in culture_group]
                # set negative utilities to 0
                trait_utils = [trait if trait > 0 else 0 for trait in trait_utils]
                # set probabilities summing to 1
                trait_probs = [1-(trait/sum(trait_utils)) for trait in trait_utils]
                trait_probs /= sum(trait_probs)
            
                # try random choice and catch error
                try:
                    trait_to_remove = np.random.choice(culture_group, 1, p=trait_probs)[0]
                except ValueError:
                    print(culture_group, trait_probs, trait_utils)
                    # if error, repeat iteration
                    i = i-1
                    continue
                
                # remove trait
                culture_group = np.setdiff1d(culture_group, trait_to_remove)
            else:
                culture_group = np.setdiff1d(culture_group, culture_group[0])

        if i >= round(0.8*num_iter):
            cultural_complexity[i-round(0.8*num_iter), :] = get_complexity(culture_group, trait_utilities)

    # get mean for each column of cultural_complexity
    cultural_complexity_mean = np.mean(cultural_complexity, axis=0)
    return  cultural_complexity_mean
