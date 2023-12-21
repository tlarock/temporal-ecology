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
    simulate_chaos(interaction_graph, T, output_dir)
