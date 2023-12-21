import numpy as np
import networkx as nx

def get_species_mass(num_species, alpha, beta):
    """
        Draws masses for num_species species using
        a Beta distribution with shape parameters
        alpha and beta

        Parameters
        ----------
        num_species (int): Number of species
        alpha, beta (float): Shape parameter

        Returns
        ----------
        masses (np.array): Array of length num_species
                            containing the sampled masses

    """
    return np.random.beta(alpha, beta, num_species)


def generate_network(species_mass, C, max_attempts=100):
    """
        Constructs interactions between species
        based on their biomass/niche and temporal overlap
        and the methods in

        Williams & Martinez, Simple Rules Yield Complex Food Webs, Nature (2000).

        Parameters
        ----------
        species_masses (np.array): Mass of each species
        intervals (np.array): 2d array of intervals
        C (float): Desired aggregated network density
        max_attempts (int, default=100): Maximum number of tries to get a
                    network where all species have non-zero degree.

        Returns
        ----------
        interactions (nx.DiGraph): networkx graph with interactions
    """
    # Setup
    rng = np.random.default_rng()
    num_species = species_mass.shape[0]

    # This is the beta probability we will use, which
    # is based on the desired network density C (input)
    # Note: alpha is always 1
    beta = 1.0 / (2.0*C) - 1.0
    alpha = 1.0

    # Loop until every species has at least one interaction,
    # meaning the number of species with non-zero degree is
    # the same as the number of species we want
    interacting_species = 0
    attempts = 0
    while interacting_species < num_species:
        # Choose niche ranges using beta (computed above)
        r = rng.beta(alpha, beta, num_species) * species_mass

        # Choose the centroid uniformly from the range
        centroids = rng.uniform(r/2.0, species_mass, num_species)

        # The minimum mass species will eat nothing
        r[species_mass == species_mass.min()] = 0

        # Compute low/high of the niche ranges
        low = centroids - r
        high = centroids + r

        # We will store the interactions as network digraph
        G = nx.DiGraph()

        for pred in range(species_mass.shape[0]):
            for prey in range(species_mass.shape[0]):
                if pred == prey or G.has_edge(prey, pred):
                    continue

                # Check if the mass of the prey is in the range
                # we have assigned for the predator
                if low[pred] < species_mass[prey] < high[pred]:
                    # Add the edge to the networkx graph
                    G.add_edge(prey, pred)

        # Check whether all of the species have non-zero degree
        interacting_species = len(set([node for node in G if G.degree(node, weight="weight") > 0]))
        attempts += 1
        if attempts == max_attempts and interacting_species < num_species:
            print(f"Failed to find an interaction network with all species " + \
                f"included in interactions after {max_attempts} tries. Consider larger intervals.")
            break

    return G
