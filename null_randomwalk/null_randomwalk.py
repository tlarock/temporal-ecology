import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import constants

def normalize_rows(matrix):
    """ Normalize each row of the matrix to sum to 1 """
    row_sums = matrix.sum(axis=1)
    
    return matrix / row_sums[:, np.newaxis]

def random_walk_with_grouped_resets(interactome, start_node, steps, reset_interval):
    """
    Perform a random walk on the interactome matrix and group the visited nodes by reset intervals.

    :param interactome: 2D numpy array representing the adjacency matrix of the interactome
    :param start_node: index of the starting node for the walk
    :param steps: total number of steps in the random walk
    :param reset_interval: number of steps after which the walk resets to the start node
    :return: A list of sets, each set containing the visited nodes in a given interval
    """
    # Normalize the interactome matrix
    normalized_interactome = normalize_rows(interactome)

    current_node = start_node
    interval_visited_nodes = set([current_node])
    grouped_visited_nodes = []

    for step in range(1, steps + 1):
        if step % reset_interval == 0:
            grouped_visited_nodes.append(interval_visited_nodes)
            interval_visited_nodes = set()
            current_node = start_node  # Reset to start node

        # Choose the next node based on the interactome probabilities
        next_node = np.random.choice(range(len(interactome)), p=normalized_interactome[current_node])
        interval_visited_nodes.add(next_node)
        current_node = next_node

    # Adding the last set if it's not empty
    if interval_visited_nodes:
        grouped_visited_nodes.append(interval_visited_nodes)

    return grouped_visited_nodes

def create_barcode_plot(grouped_walk_path, num_species):
    # Initialize a DataFrame to store presence/absence data
    barcode_data = pd.DataFrame(0, index=range(num_species), columns=range(len(grouped_walk_path)))

    # Fill in the DataFrame based on the grouped_walk_path
    for season, nodes in enumerate(grouped_walk_path):
        for node in nodes:
            barcode_data.loc[node, season] = 1

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.imshow(barcode_data, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.colorbar(label='Presence (1) / Absence (0)')
    plt.xlabel('Season')
    plt.ylabel('Species (Node)')
    plt.title('Barcode Plot of Species Presence Over Seasons')
    plt.savefig(constants.save_file_path + 'barcode_plot.png')  # Save the plot
    plt.show()

def plot_grouped_random_walk(interactome, grouped_walk_path):
    G = nx.from_numpy_matrix(interactome, create_using=nx.DiGraph)

    pos = nx.spring_layout(G)  # Position the nodes using a layout algorithm
    plt.figure(figsize=(12, 8))

    # Draw the overall network
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', alpha=0.3)

    # Highlight the path of the random walk for each season
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for i, interval_nodes in enumerate(grouped_walk_path):
        color = colors[i % len(colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=interval_nodes, node_color=color, node_size=100, label=f'Season {i+1}')

    plt.title('Grouped Random Walk on Interactome Network')
    plt.legend()
    plt.savefig(constants.save_file_path + 'network_plot.png')  # Save the plot
    plt.show()


if __name__ == "__main__":
    # Define your interactome matrix and other parameters here
    num_species = 25  # Example number of species
    interactome_matrix = np.random.rand(num_species, num_species)  # Example interactome matrix or load your own!
    start_species = 0  # Starting species index
    total_steps = 75  # Total steps in the random walk
    season_length = 25  # Reset interval to mimic seasons

    # Remove self-loops and normalize the matrix
    np.fill_diagonal(interactome_matrix, 0)
    interactome_matrix = normalize_rows(interactome_matrix)

    # Perform the random walk with grouped resets
    grouped_walk_path = random_walk_with_grouped_resets(interactome_matrix, start_species, total_steps, season_length)

    # Create and save the plots
    create_barcode_plot(grouped_walk_path, num_species)
    plot_grouped_random_walk(interactome_matrix, grouped_walk_path)
