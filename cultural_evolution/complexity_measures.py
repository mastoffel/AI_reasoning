# measures of cultural complexity

import numpy as np
import pandas as pd

# calculate cultural complexity from culture_group_list
# measure 1: the number of traits per element
def get_trait_number(culture_group):
    """Counts all traits in a culture group.
   
    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        int: number of traits in culture group
    """
    return len(culture_group)

# measure 2: trait complexity: the mean length of traits in each culture_group
def get_trait_complexity(culture_group):
    """Calculates the mean length of traits in a culture group.

    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        float: Mean trait length in culture group
    """
    if culture_group.size == 0: return 0
    traits = [len(trait) for trait in culture_group]
    return np.mean(traits)

# measure 3: number of lineages in each culture group
# defined as starting with the same seed trait
def get_lineage_number(culture_group):
    """Counts all lineages in a culture group.

    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        float: number of lineages in culture group
    """
    if culture_group.size == 0: return 0
    lineages = {trait[0] for trait in culture_group}
    return len(lineages)

# measure 4: mean lineage complexity
def get_lineage_complexity(culture_group):
    """Calculates the mean length of lineages in a culture group.

    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        tuple: Mean and maximum lineage complexity in culture group.
    """
    if culture_group.size == 0: return 0, 0
    lineage_complexity = [len(trait.replace('m', '')) for trait in culture_group]
    return np.mean(lineage_complexity), np.max(lineage_complexity)

def get_number_of_modifications(culture_group):
    """Calculates the number of modifications in a culture group.

    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        int: number of modifications per trait in culture group
    """
    if culture_group.size == 0: return 0
    # count m's and divide by number of traits
    return np.sum([trait.count('m') for trait in culture_group]) / len(culture_group)

def get_number_of_seed_traits(culture_group):
    """Calculates the mean number of seed traits in a culture group.

    Args:
        culture_group (np.array): array with string for each trait.

    Returns:
        float: Number of seed traits in culture group.
    """
    if culture_group.size == 0: return 0
    seed_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    # check how many seed traits in culture group, including partial matching
    # combine culture_group into one string and check how many unique letters
    # without m
    culture_group_string = ''.join(culture_group).replace('m', '')
    # string to set
    culture_group_set = set(culture_group_string)
    # count size of set
    return len(culture_group_set)

def get_utility(culture_group, trait_utilities):
    """Calculates summary stats of the utility of traits a culture group.

    Args:
       trait_utilities (dict): Mapping of trait to utility.

    Returns:
        tuple: Maximum utility of culture group, minimum utility of culture group,
        and mean utility of culture group.
    """
    if culture_group.size == 0: return 0, 0, 0
    utilities = [trait_utilities[trait] for trait in culture_group]
    return max(utilities), min(utilities), np.mean(utilities)

def get_complexity(culture_group, trait_utilities):
    """Calculates the cultural complexity of a culture group.

    Args:
        culture_group (np.array): array with string for each trait.
        trait_utilities (dict): Mapping of trait to utility.
    Returns:
        numpy array: trait_number, trait_complexity, lineage_number,    
        lineage_complexity_mean, lineage_complexity_max, seed_trait_number,
        modifications, maximum_utility, minimum_utility, mean_utility.            
    """
    trait_number = get_trait_number(culture_group)
    trait_complexity = get_trait_complexity(culture_group)
    lineage_number = get_lineage_number(culture_group)
    lineage_complexity_mean, lineage_complexity_max = get_lineage_complexity(culture_group)
    seed_trait_number = get_number_of_seed_traits(culture_group)
    modifications = get_number_of_modifications(culture_group)
    maximum_utility, minimum_utility, mean_utility = get_utility(culture_group, trait_utilities)
    
    # combine into numpy array
    return np.array([trait_number, trait_complexity, lineage_number, 
                     lineage_complexity_mean, lineage_complexity_max, 
                     seed_trait_number, modifications,
                     maximum_utility, minimum_utility, mean_utility])