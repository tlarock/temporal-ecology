import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from trophic_network import *

def sigmoid(x, k):
    """
    Normalized tunable sigmoid function.
    """
    return (x - k*x) / (k - 2*k*abs(x) + 1)

def simulate_temporal_network(G, T, decay_factor, output_dir,
                         arrival_type="uniform", departure_type="uniform",
                        sigmoid_thresh=0.5):
    """


    arrival (bool): If true, use exponential arrival. Otherwise use exponential
    departure.
    """
    assert arrival_type in ["uniform", "exponential", "sigmoid"], f"arrival_type '{arrival_type}' not valid."
    assert departure_type in ["uniform", "exponential", "sigmoid"], f"departure_type '{departure_type}' not valid."
    rng = np.random.default_rng()

    adjacency_matrices = []
    node_activities = []

    if arrival_type == "uniform":
        # Choose arrival times uniformly at random on 1,T
        arrival_times = rng.integers(low=1, high=T, size=G.number_of_nodes())

    if departure_type == "uniform":
        duration = rng.integers(low=1, high=int(T/2.0), size=G.number_of_nodes())

    # Simulate and save adjacency matrices
    node_age = np.zeros(G.number_of_nodes())
    active_nodes = set()
    inactive_nodes = list(G.nodes())
    departed_nodes = set()
    for t in range(1, T+1):
        node_activity = np.zeros(G.number_of_nodes())

        # Manage node arrivals
        if len(inactive_nodes) > 0:
            probabilities = rng.uniform(size=len(inactive_nodes))
            if arrival_type == "uniform":
                # Check whether nodes need to be added to the network
                for node in inactive_nodes:
                    if t == arrival_times[node]:
                        active_nodes.add(node)
                        node_age[node] = 1
                        node_activity[node] = 1
            elif arrival_type == "exponential":
                # Same decay for every still-inactive node
                decay = np.exp(-decay_factor * (t+1))
                new_nodes = np.where(probabilities < decay)[0]

                # Add the new nodes
                for node in new_nodes:
                    active_nodes.add(node)
                    node_age[node] = 1
                    node_activity[node] = 1
            elif arrival_type == "sigmoid":
                entrance_probs = []
                # Update node probabilities with sigmoid function
                for node in inactive_nodes:
                    predecessors = [n for n in G.predecessors(node) if n in active_nodes]
                    total_predecessors = len(list(G.predecessors(node)))
                    proportion = len(predecessors) / total_predecessors if total_predecessors > 0 else 0
                    if proportion >= sigmoid_thresh:
                        k = (proportion-0.5)*2
                    else:
                        k = -(proportion-0.5)*2

                    if k == 1.0:
                        k -= 0.01

                    entrance_probs.append(sigmoid(t+1, k))

                entrance_probs = np.array(entrance_probs)
                entrance_probs /= entrance_probs.sum()
                new_nodes = np.where(entrance_probs < probabilities)[0]
                for node in new_nodes:
                    active_nodes.add(node)
                    node_age[node] = 1
                    node_activity[node] = 1

        # Manage node departures
        if len(active_nodes) > 0:
            probabilities = rng.uniform(size=len(active_nodes))
            if departure_type == "uniform":
                for node in active_nodes:
                    # Remove a node if its age is larger than the duration we
                    # assigned to it
                    if node_age[node] > duration[node]:
                        departed_nodes.add(node)
                    else:
                        node_age[node] += 1

            elif departure_type == "exponential":
                # Compute decay probabilites for each node based on their age
                decay = np.array([np.exp(-decay_factor * (node_age[node]+1)) for
                                  node in range(node_age.shape[0]) if node in
                                  active_nodes])
                # The nodes with probability less than decay will be removed
                departed_nodes = set(np.where(probabilities < decay)[0])
                for node in active_nodes:
                    if node not in departed_nodes:
                        # if not removed, increment age
                        node_age[node]+=1
            elif departure_type == "sigmoid":
                departure_probs = []
                # Update node probabilities with sigmoid function
                for node in active_nodes:
                    predecessors = [n for n in G.predecessors(node) if n in active_nodes]
                    total_predecessors = len(list(G.predecessors(node)))
                    proportion = len(predecessors) / total_predecessors if total_predecessors > 0 else 0
                    if proportion >= sigmoid_thresh:
                        k = (proportion-0.5)*2
                    else:
                        k = -(proportion-0.5)*2

                    if k == 1.0:
                        k -= 0.01

                    departure_probs.append(sigmoid(node_age[node], k))

                departure_probs = np.array(departure_probs)
                departure_probs /= departure_probs.sum()
                departed_nodes = set(np.where(probabilities < departure_probs)[0])
                for node in active_nodes:
                    if node not in departed_nodes:
                        # if not removed, increment age
                        node_age[node]+=1

        # Actually remove departing nodes
        active_nodes -= departed_nodes
        inactive_nodes = [node for node in inactive_nodes if node not in active_nodes]
        # Construct the subgraph from active nodes
        subgraph = nx.DiGraph(G.subgraph(active_nodes))
        # In the output we want all of the nodes present, 
        # so make sure they are all there
        for node in G.nodes():
            if node not in subgraph:
                subgraph.add_node(node)

        adjacency_matrices.append(nx.to_numpy_array(subgraph))
        for node in active_nodes:
            node_activity[node] = 1
        node_activities.append(np.array(node_activity))

    np.save(output_dir + "_temporal_adjacencies", adjacency_matrices)
    np.save(output_dir + "_node_activities", node_activities)

if __name__ == "__main__":
    # Parameters
    num_species = 25
    alpha = 0.25
    beta = 1.0
    C = 0.15
    species_mass = get_species_mass(num_species, alpha, beta)
    interaction_graph = generate_network(species_mass, C)
    T = 100  # Number of timesteps to simulate

    # Write the adjacency matrix of the whole network
    adjacency_matrix = nx.to_numpy_array(interaction_graph)

    # Run the simulation and save the outputs
    #simulate_chaos(interaction_graph, T, output_dir)
    decay_factor = 0.8
    for arrival_type in ["uniform", "exponential", "sigmoid"]:
        for departure_type in ["uniform", "exponential", "sigmoid"]:
            output_dir = f'../results/S-{num_species}_alpha-{alpha}_beta-{beta}_C-{C}_T-{T}_{arrival_type}_{departure_type}'  # Directory to save the CSV files
            # Ensure the output directory exists
            #if not os.path.exists(output_dir):
            #    os.makedirs(output_dir)
            np.save(output_dir + f'_adj_matrix_full', adjacency_matrix)
            np.save(output_dir + f'_species_mass', species_mass)
            print(arrival_type, departure_type)
            simulate_temporal_network(interaction_graph, T, decay_factor, output_dir,
                                 arrival_type=arrival_type, departure_type=departure_type)
