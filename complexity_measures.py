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
        float: Mean lineage length in culture group.
    """
    if culture_group.size == 0: return 0
    lineage_complexity = [len(trait.replace('m', '')) for trait in culture_group]
    return np.mean(lineage_complexity)

def get_maximum_utility(culture_group, trait_utilities):
    """Calculates the maximum utility of a culture group.

    Args:
       trait_utilities (dict): Mapping of trait to utility.

    Returns:
        float: Maximum utility of culture group.
    """
    if culture_group.size == 0: return 0
    utilities = [trait_utilities[trait] for trait in culture_group]
    return max(trait_utilities.values())

def get_complexity(culture_group, trait_utilities):
    """Calculates the cultural complexity of a culture group.

    Args:
        culture_group (np.array): array with string for each trait.
        trait_utilities (dict): Mapping of trait to utility.
    Returns:
        numpy array: five measures of cultural complexity of culture group:
        trait_number, trait_complexity, lineage_number, 
        lineage_complexity, maximum_utility.
    """
    trait_number = get_trait_number(culture_group)
    trait_complexity = get_trait_complexity(culture_group)
    lineage_number = get_lineage_number(culture_group)
    lineage_complexity = get_lineage_complexity(culture_group)
    maximum_utility = get_maximum_utility(culture_group, trait_utilities)
    # combine into numpy array
    return np.array([trait_number, trait_complexity, lineage_number, 
                     lineage_complexity, maximum_utility])