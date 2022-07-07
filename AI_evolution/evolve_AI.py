# AI evolution based on cultural evolution model by Lewis & Laland 2012

# primary factors for AI improvement:
# 1) transmission fidelity (loss rate) over reasoning steps
# 2) novel invention rate
# 3) combination rate
# 4) modification rate
# 5) idea value judgement 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from operator import itemgetter
from sklearn.metrics import mean_absolute_error
from AI_evolution.AI_complexity_measures import get_complexity

# what happens in every iteration
# 1) 
# 2) 

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
def run_simulation(rho1, rho2, rho3, judge, reason, num_iter=500):
    """Cultural evolution simulation. Starts with two (out of ten) seed traits, allows
    new traits to be introduced through novel invention (rho1), combination 
    of existing traits (rho2), and modification of existing traits (rho3). The
    loss rate or transmission fidelity (rho4) is assumed to be 1 - (rho1 + rho2 + rho3).
    

    Args:
        rho1 (_type_): probability of introducing a new seed trait through novel invention
        rho2 (_type_): probability of combining two existing traits to produce a new trait
        rho3 (_type_): probability of modifying an existing trait to produce a new variant
        judge (_type_): quality of evaluation from 0 to 1 (1 is best).
        reason (_type_): quality (fidelity?) of reasoning from 0 to 1 (1 is best).
        num_iter (int, optional): Number of iterations. 
        to calculate complexity. Defaults to 500.

    Returns:
        np.array: 10 complexity measures per iteration.
    """
    # make np array containing the 20 first letters of the alphabet
    # these are the seed traits
    num_seeds = 20
    seed_names = np.array(list(map(chr, range(97, 97 + num_seeds))))
    
    # give each seed_trait a utility value drawn from a random uniform distribution
    # betwee 0.75 and 1. 
    seed_utilities = np.random.uniform(size=num_seeds, low=0.75, high=1.)
    
    # combine seed names and their utilities. 
    trait_utilities = {}
    for i in range(0, len(seed_names)):
        trait_utilities[seed_names[i]] = seed_utilities[i]
        
    # initialise AI with five seed traits drawn at random
    ai = np.random.choice(seed_names, 2, replace=False)
    
    # inititalise array of complexities over time
    ai_complexity = np.zeros(shape = (num_iter, 10))
    
    i = 0
    while i < num_iter:
        
        # draw random number between 0 and 1
        r = np.random.random()
        
        # 1) new seed trait is introduced (necessary if there are no traits)
        if r < rho1 or len(ai) == 0:
            
            # check which seed traits are not in group 
            seed_traits_not_in_group = np.setdiff1d(seed_names, ai)
            
            # if all seed traits present in group repeat iteration
            if len(seed_traits_not_in_group) == 0:
                continue
            
            # otherwise take a random seed trait to add to culture group
            new_trait = np.random.choice(seed_traits_not_in_group, 
                                                       1, replace=False)[0]
            
        # 2) two of the cultural traits present are combined
        elif r < rho1 + rho2:
            
            # draw two random traits from culture group
            trait_1 = np.random.choice(ai, 1, replace=False)
            
            # remove letter 'm' from trait_1
            trait_1_seed = trait_1[0].replace('m', '')
            
            # check for traits in ai with no overlap in seed traits
            remaining_traits = []
            for trait in ai:
                if not has_intersection(trait_1_seed, trait):
                    remaining_traits.append(trait)
                    
            # if there is nothing to combine with, repeat iteration
            if remaining_traits == []:
                continue
            
            trait_2 = np.random.choice(remaining_traits, 1, replace=False)
            
            # combine traits 
            new_trait = trait_1[0] + trait_2[0]
            
            # calculate utility of new trait by taking maximum among seed traits
            # and adding value based on reasoning capability
            utils = itemgetter(trait_1[0], trait_2[0])(trait_utilities)
            added_mean = (reason - 0.5)
            added_sd = 0.1 + np.abs(added_mean/2)
            new_util = np.max(utils) + np.random.normal(added_mean, added_sd)
            
            # add to trait_utilities
            trait_utilities[new_trait] = new_util
            
        # 3) one of the cultural traits is modified
        elif r < rho1 + rho2 + rho3:
            
            # check whether trait exists in ai that can be modified with 'm'
            # and don't exist yet
            remaining_traits = []
            for trait in ai:
                new_trait = trait + 'm'
                if new_trait not in ai:
                    remaining_traits.append(trait)
            
            if remaining_traits == []:
                continue
            
            trait = np.random.choice(remaining_traits, 1, replace=False)[0]
            new_trait = trait + 'm'
            
             # utility is modified by adding value from N(0, 0.1) to
            added_mean = (reason - 0.5)
            added_sd = 0.1 + np.abs(added_mean/2)
            new_util = trait_utilities[trait] + np.random.normal(added_mean, added_sd)
            
             # add to trait_utilities
            trait_utilities[new_trait] = new_util
            
        # 4) evaluation step: is utility of trait greater than the mean
        # utility of the traits in ai?
        
        # average utility of existing traits
        mean_utility = np.mean(itemgetter(*ai)(trait_utilities)) 
        
        if not 'new_trait' in locals():
            print(ai, trait_utilities, r)
            
        # add trait if better than mean utility
        if (trait_utilities[new_trait]) >= mean_utility:
            # chance that new trait is added to ai is judge (0 = bad, 1 = good)
            if np.random.random() < judge:
                ai = np.append(ai, new_trait)
    
         # if trait is worse than mean utility
        else:
            # chance that new trait is added to ai when utility low
            if np.random.random() > judge:
                ai = np.append(ai, new_trait)
            
        # add to cultural_complexity over time
        ai_complexity[i, :] = get_complexity(ai, trait_utilities)
        
        i += 1
            
    return ai_complexity
