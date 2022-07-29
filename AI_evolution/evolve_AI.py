# AI evolution based on cultural evolution model by Lewis & Laland 2012

# primary factors for AI improvement:
# 1) novel invention rate
# 2) combination rate
# 3) modification rate
# 4) reasoning quality (vaguely related to transmission fidelity)
# 5) idea value judgement 

# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from operator import itemgetter
from sklearn.metrics import mean_absolute_error
from AI_evolution.AI_complexity_measures import get_complexity

# -----------------------------------------------------------------------------

# checks whether two traits overlap in any of the seed traits
def has_intersection(a, b):
        return not set(a).isdisjoint(b)
    
# simulation 
def run_simulation(rho1, rho2, rho3, judge, reason, reinvest=False, num_iter=500):
    """AI evolution simulation. Starts with two (out of 20) seed traits, allows
    new traits to be introduced through novel invention (rho1), combination 
    of existing traits (rho2), and modification of existing traits (rho3). Reason
    determines whether new traits are more useful  than old ones, and judge determines
    the capability of the AI to judge this and incorporate or not incorporate the
    new trait.
    

    Args:
        rho1 (_type_): probability of introducing a new seed trait through novel invention
        rho2 (_type_): probability of combining two existing traits to produce a new trait
        rho3 (_type_): probability of modifying an existing trait to produce a new variant
        judge (_type_): quality of evaluation from 0 to 1 (1 is best).
        reason (_type_): quality (fidelity?) of reasoning from 0 to 1 (1 is best).
        reinvest (_type_): the AI reinvests compute into better (or worse) reasoning
        num_iter (int, optional): Number of iterations. 
        to calculate complexity. Defaults to 500.

    Returns:
        np.array: 10 complexity measures per iteration.
    """
    
    # seed traits 
    num_seeds = 20
    seed_names = np.array(list(map(chr, range(97, 97 + num_seeds))))
    
    # utility values for seed traits
    seed_utilities = np.random.uniform(size=num_seeds, low=0.75, high=1.)
    
    # combine
    trait_utilities = {}
    for i in range(0, len(seed_names)):
        trait_utilities[seed_names[i]] = seed_utilities[i]
        
    # initialise AI with some of seed traits
    ai = np.random.choice(seed_names, 2, replace=False)
    
    # inititalise array of complexity measures 
    ai_complexity = np.zeros(shape = (num_iter, 10))
    
    # reasoning iterations
    i = 0
    while i < num_iter:
        # what should happen
        r = np.random.random()
        
        # 1) new seed trait is introduced (necessary if there are no traits)
        if r < rho1 or len(ai) == 0:
            # only introduce seed trait if not part of AI
            seed_traits_not_in_group = np.setdiff1d(seed_names, ai)
            if len(seed_traits_not_in_group) == 0:
                continue
            # add seed trait to AI
            new_trait = np.random.choice(
                            seed_traits_not_in_group, 
                            size = 1, 
                            replace=False)[0]
            
        # 2) two of the AI traits are combined
        elif r < rho1 + rho2:
            # combine if two traits don't share seed trait
            trait_1 = np.random.choice(ai, 1, replace=False)
            trait_1_seed = trait_1[0].replace('m', '')
            remaining_traits = []
            for trait in ai:
                if not has_intersection(trait_1_seed, trait):
                    remaining_traits.append(trait)
            # if there is nothing to combine with, repeat iteration
            if remaining_traits == []:
                continue
            trait_2 = np.random.choice(remaining_traits, 1, replace=False)
            # combine 
            new_trait = trait_1[0] + trait_2[0]
            # calculate utility of new trait by taking maximum among seed traits
            # and adding value based on reasoning capability
            utils = itemgetter(trait_1[0], trait_2[0])(trait_utilities)
            added_mean = (reason - 0.5)
            added_sd = 0.1 + np.abs(added_mean/2)
            new_util = np.max(utils) + np.random.normal(added_mean, added_sd)
            # add to trait_utilities
            trait_utilities[new_trait] = new_util
            
        # 3) one of the AI traits is modified
        elif r < rho1 + rho2 + rho3:
            # check whether trait exists that can be modified with 'm'
            remaining_traits = []
            for trait in ai:
                new_trait = trait + 'm'
                if new_trait not in ai:
                    remaining_traits.append(trait)
            # go on if no traits can be modified
            if remaining_traits == []: 
                continue
            # modify trait
            trait = np.random.choice(remaining_traits, 1, replace=False)[0]
            new_trait = trait + 'm'
             # utility for new trait 
            added_mean = (reason - 0.5)
            added_sd = 0.1 + np.abs(added_mean/2)
            new_util = trait_utilities[trait] +\
                np.random.normal(added_mean, added_sd)
            trait_utilities[new_trait] = new_util
        
        # 4) evaluation step: is utility of trait greater than the mean
        # utility of the traits in ai?
        mean_utility = np.mean(itemgetter(*ai)(trait_utilities)) 
        if (trait_utilities[new_trait]) >= mean_utility:
            # quality of judgement determines whether to add new trait
            if np.random.random() < judge:
                ai = np.append(ai, new_trait)
                if reinvest:
                    # reasoning improves by fraction of gained utility
                    x = np.random.randint(1, 10)/100
                    reason += (trait_utilities[new_trait] - mean_utility) * x
    
        else: # if trait is worse than mean utility
            # new trait might still be added to AI
            if np.random.random() > judge:
                ai = np.append(ai, new_trait)
                if reinvest:
                    # reasoning gets worse by fraction of decreased utility
                    x = np.random.randint(1, 10)/100
                    reason -= (mean_utility - trait_utilities[new_trait]) * x
                    
        # add to cultural_complexity over time
        ai_complexity[i, :] = get_complexity(ai, trait_utilities)
        i += 1
            
    return ai_complexity, reason
