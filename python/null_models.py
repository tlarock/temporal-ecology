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

def uniform_arrival(t, inactive_nodes, arrival_times):
    """
    Returns the set of species that should arrive at
    timestep t.
    """
    newly_active_nodes = set()
    for node in inactive_nodes:
        if t == arrival_times[node]:
            newly_active_nodes.add(node)
    return newly_active_nodes

def uniform_departure(active_nodes, node_age, duration):
    """
    Returns the set of species that should depart at
    timestep t.
    """
    departed_nodes = set()
    for node in active_nodes:
        # Remove a node if its age is larger than the duration we
        # assigned to it
        if node_age[node] > duration[node]:
            departed_nodes.add(node)
    return departed_nodes

def exponential_arrival(rng, t, decay_factor, inactive_nodes):
    """
    Returns a set of nodes randomly selected to arrive
    at timestep t based on an exponentially increasing probability.
    """
    decay = np.exp(-decay_factor * (t+1))
    probabilities = rng.uniform(size=len(inactive_nodes))
    return set(np.where(probabilities < decay)[0])

def exponential_departure(rng, active_nodes, node_age, decay_factor):
    """
    Returns a set of currently active nodes randomly selected to
    depart at timestep t based on an exponential decay probability.
    """
    # Compute decay probabilites for each node based on their age
    decay = np.array([np.exp(-decay_factor * (node_age[node]+1)) for
                      node in range(node_age.shape[0]) if node in
                      active_nodes])
    # The nodes with probability less than decay will be removed
    probabilities = rng.uniform(size=len(active_nodes))
    departed_nodes = set(np.where(probabilities < decay)[0])
    return departed_nodes

def sigmoid_arrival(rng, t, sigmoid_thresh, G, active_nodes, inactive_nodes):
    """
    Returns a set of nodes chosen to arrive at timestep t based
    on a sigmoid arrival probability function.
    """
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
    probabilities = rng.uniform(size=len(inactive_nodes))
    new_nodes = set(np.where(entrance_probs < probabilities)[0])
    return new_nodes

def sigmoid_departure(rng, active_nodes, node_age, sigmoid_thresh, G):
    """
    Returns a set of currently active nodes chosen to depart
    at timestep t based on a sigmoid probability function.
    """
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
    probabilities = rng.uniform(size=len(active_nodes))
    departed_nodes = set(np.where(probabilities < departure_probs)[0])
    return departed_nodes

def get_subgraph(G, active_nodes):
    subg = nx.DiGraph()
    subg.add_nodes_from(G.nodes())
    for u in active_nodes:
        for ne in G.successors(u):
            if ne in active_nodes:
                subg.add_edge(u, ne)
    return subg

def simulate_temporal_network(G, T, decay_factor, output_dir,
                         arrival_type="uniform", departure_type="uniform",
                        sigmoid_thresh=0.5):
    """


    Parameters
    ---------
    G (nx.DiGraph): The underlying trophic network
    T (int): The total number of time steps
    decay_factor (float): Parameter for exponential decay
    arrival_type (str): String representing which mechanism to use for species
                    arrival. Choices are 'uniform', 'exponential', 'sigmoid'.
    departure_type (str): Same as arrival_type but for departure.
    sigmoid_thresh (float): Threshold for change between positive and negative
                            sigmoid function. Ignored for all others.

    """
    assert arrival_type in ["uniform", "exponential", "sigmoid"], f"arrival_type '{arrival_type}' not valid."
    assert departure_type in ["uniform", "exponential", "sigmoid"], f"departure_type '{departure_type}' not valid."
    assert set(G.nodes()) == set(list(range(0, G.number_of_nodes()))), f"Expected nodes to be 0,1,...,N"
    rng = np.random.default_rng()

    adjacency_matrices = []
    node_activities = []

    if arrival_type == "uniform":
        # Choose arrival times uniformly at random on 1,T
        arrival_times = rng.integers(low=1, high=T-1, size=G.number_of_nodes())

    if departure_type == "uniform":
        duration = rng.integers(low=2, high=int(T/2.0), size=G.number_of_nodes())

    # Simulate and save adjacency matrices
    node_age = np.zeros(G.number_of_nodes())
    active_nodes = set()
    # ToDo: For debugging
    prev_active_nodes = set()
    prev_edges = 0
    inactive_nodes = list(G.nodes())
    departed_nodes = set()
    for t in range(1, T+1):
        # Manage node arrivals
        if len(inactive_nodes) > 0:
            if arrival_type == "uniform":
                new_nodes = uniform_arrival(t, inactive_nodes, arrival_times)
            elif arrival_type == "exponential":
                # Same decay for every still-inactive node
                new_nodes = exponential_arrival(rng, t, decay_factor, inactive_nodes)
            elif arrival_type == "sigmoid":
                new_nodes = sigmoid_arrival(rng, t, sigmoid_thresh, G, active_nodes, inactive_nodes)

            for node in new_nodes:
                active_nodes.add(node)
                node_age[node] = 1

        # Manage node departures
        if len(active_nodes) > 0:
            probabilities = rng.uniform(size=len(active_nodes))
            if departure_type == "uniform":
                departed_nodes = uniform_departure(active_nodes, node_age, duration)
            elif departure_type == "exponential":
                departed_nodes = exponential_departure(rng, active_nodes, node_age,
                                                       decay_factor)
            elif departure_type == "sigmoid":
                departed_nodes = sigmoid_departure(rng, active_nodes, node_age,
                                                   sigmoid_thresh, G)
            # Increment age for remaining nodes
            for node in active_nodes:
                if node not in departed_nodes:
                    node_age[node] += 1

        # Actually remove departing nodes
        active_nodes -= departed_nodes
        inactive_nodes = [node for node in G if node not in active_nodes]

        # Construct the subgraph from active nodes
        subgraph = get_subgraph(G, active_nodes)

        # In the output we want all of the nodes present, 
        # so make sure they are all there
        for node in G.nodes():
            if node not in subgraph:
                subgraph.add_node(node)

        if active_nodes == prev_active_nodes:
            assert subgraph.number_of_edges() == prev_edges

        print(f"t: {t}, active nodes: {len(active_nodes)}, inactive: {len(inactive_nodes)}, subgraph edges: {subgraph.number_of_edges()}")

        adjacency_matrices.append(nx.to_numpy_array(subgraph,
                                                    nodelist=sorted(subgraph.nodes())))

        # Update node activites
        node_activity = np.zeros(G.number_of_nodes())
        for node in active_nodes:
            node_activity[node] = 1
        node_activities.append(np.array(node_activity))

        # ToDo: For debugging
        prev_active_nodes = set(active_nodes)
        prev_edges = subgraph.number_of_edges()

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
    adjacency_matrix = nx.to_numpy_array(interaction_graph,
                                         nodelist=sorted(interaction_graph.nodes()))

    # Run the simulation and save the outputs
    decay_factor = 0.8
    #for arrival_type in ["uniform", "exponential", "sigmoid"]:
    #    for departure_type in ["uniform", "exponential", "sigmoid"]:
    for arrival_type in ["sigmoid"]:
        for departure_type in ["sigmoid"]:
            output_dir = f'../results/S-{num_species}_alpha-{alpha}_beta-{beta}_C-{C}_T-{T}_{arrival_type}_{departure_type}'  # Directory to save the CSV files
            np.save(output_dir + f'_adj_matrix_full', adjacency_matrix)
            np.save(output_dir + f'_species_mass', species_mass)
            print(arrival_type, departure_type)
            simulate_temporal_network(interaction_graph, T, decay_factor, output_dir,
                                 arrival_type=arrival_type, departure_type=departure_type)
