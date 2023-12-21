import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def read_and_simulate_graph(gml_path, num_cascades, output_dir):
    # Read in directed graph G
    G = nx.read_gml(gml_path)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Simulate cascades and save adjacency matrices
    for i in range(num_cascades):
        time_interval = random.randint(5, 10)  # Select a time interval between 5 and 10
        cascade_subgraph = simulate_cascade(G, time_interval)
        adjacency_matrix = nx.to_pandas_adjacency(cascade_subgraph)
        file_path = os.path.join(output_dir, f'cascade_{i}_adj_matrix.csv')
        adjacency_matrix.to_csv(file_path)

def simulate_cascade(G, time_interval):
    """
    Simulates a random cascade model on a graph G within a given time interval.
    """
    # Select a random node
    start_node = random.choice(list(G.nodes))

    # Initialize cascade model
    cascade_nodes = set([start_node])
    for _ in range(time_interval):
        new_nodes = set()
        for node in cascade_nodes:
            new_nodes.update(G.successors(node))
        cascade_nodes.update(new_nodes)

    # Create subgraph of cascade
    cascade_subgraph = G.subgraph(cascade_nodes)
    return cascade_subgraph

# Parameters
gml_path = '../data/interaction_graph_v2.gml'  # Path to your GML file
num_cascades = 5  # Number of cascades to simulate
output_dir = '../results/cascades_output'  # Directory to save the CSV files

# Run the simulation and save the outputs
read_and_simulate_graph(gml_path, num_cascades, output_dir)

