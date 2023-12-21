import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from trophic_network import *

def simulate_chaos(G, T, output_dir):
    rng = np.random.default_rng()
    # Choose arrival times uniformly at random on 1,T
    arrival_times = rng.integers(low=1, high=T, size=G.number_of_nodes())
    # Choose departure times uniformly at random from a_i, T
    departure_times = rng.integers(arrival_times+1, T+1)

    # Simulate and save adjacency matrices
    for t in range(T):
        # figure out which nodes are active
        active_nodes = set()
        for node in G.nodes():
            if arrival_times[node] <= t <= departure_times[node]:
                active_nodes.add(node)

        cascade_subgraph = nx.DiGraph(G.subgraph(active_nodes))
        # In the output we want all of the nodes present, 
        # so make sure they are all there
        for node in G.nodes():
            if node not in cascade_subgraph:
                cascade_subgraph.add_node(node)

        adjacency_matrix = nx.to_pandas_adjacency(cascade_subgraph)
        file_path = os.path.join(output_dir, f'adj_matrix_{t}.csv')
        adjacency_matrix.to_csv(file_path)


def simulate_exponential(G, T, decay_factor, output_dir, arrival_type="uniform"):
    """
    arrival (bool): If true, use exponential arrival. Otherwise use exponential
    departure.
    """
    rng = np.random.default_rng()

    if arrival_type == "uniform":
        # Exponential Departure case
        # Choose arrival times uniformly at random on 1,T
        arrival_times = rng.integers(low=1, high=T, size=G.number_of_nodes())
    else:
        # Exponential arrival case
        duration = rng.integers(low=1, high=int(T/2.0), size=G.number_of_nodes())

    # Simulate and save adjacency matrices
    node_age = np.zeros(G.number_of_nodes())
    active_nodes = set()
    inactive_nodes = list(G.nodes())
    departed_nodes = set()
    for t in range(T):
        probabilities = rng.uniform(size=len(inactive_nodes))
        if arrival_type == "uniform":
            # Compute decay probabilites for each node based on their age
            decay = np.array([np.exp(-decay_factor * (node_age[node]+1)) for node in range(node_age.shape[0])])

            # Check whether nodes need to be added to the network
            for node in inactive_nodes:
                if t > arrival_times[node]:
                    active_nodes.add(node)
                    node_age[node] = 1

            # The nodes with probability less than decay will be removed
            departed_nodes = set(np.where(probabilities < decay)[0])
            for node in active_nodes:
                if node not in departed_nodes:
                    # if not removed, increment age
                    node_age[node]+=1
            # Actually remove departing nodes
            active_nodes -= departed_nodes
        else:
            # Same decay for every still-inactive node
            decay = np.exp(-decay_factor * (t+1))
            new_nodes = np.where(probabilities < decay)[0]

            # Add the new nodes
            for node in new_nodes:
                active_nodes.add(node)
                node_age[node] = 1

            for node in active_nodes:
                # Remove a node if its age is larger than the duration we
                # assigned to it
                if node_age[node] > duration[node]:
                    departed_nodes.add(node)
                else:
                    node_age[node] += 1

            active_nodes -= departed_nodes

        subgraph = nx.DiGraph(G.subgraph(active_nodes))
        # In the output we want all of the nodes present, 
        # so make sure they are all there
        for node in G.nodes():
            if node not in subgraph:
                subgraph.add_node(node)

        adjacency_matrix = nx.to_pandas_adjacency(subgraph)
        file_path = os.path.join(output_dir, f'adj_matrix_{t}.csv')
        adjacency_matrix.to_csv(file_path)


if __name__ == "__main__":
    # Parameters
    num_species = 25
    alpha = 0.25
    beta = 1.0
    C = 0.15
    species_mass = get_species_mass(num_species, alpha, beta)
    interaction_graph = generate_network(species_mass, C)
    T = 5  # Number of timesteps to simulate

    output_dir = f'../results/S-{num_species}_alpha-{alpha}_beta-{beta}_C-{C}_T-{T}/'  # Directory to save the CSV files
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the adjacency matrix of the whole network
    adjacency_matrix = nx.to_pandas_adjacency(interaction_graph)
    file_path = os.path.join(output_dir, f'adj_matrix_full.csv')
    adjacency_matrix.to_csv(file_path)

    # Run the simulation and save the outputs
    #simulate_chaos(interaction_graph, T, output_dir)
    decay_factor = 0.8
    simulate_exponential(interaction_graph, T, decay_factor, output_dir,
                         arrival_type="exp")
