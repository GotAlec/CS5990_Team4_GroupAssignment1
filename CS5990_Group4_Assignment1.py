# from networkx.algorithms import community
# from operator import itemgetter
# import graphlib
# from venv import create
import matplotlib

matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt
import networkx as nx
import csv
import random
import sys
import scipy
import multiprocessing
import numpy as np
# import cuda
# import cugraph
# import cudf


# Define a CUDA kernel for degree calculation
# @cuda.jit
# def calculate_degree_kernel(src_nodes, dst_nodes, degrees, num_iterations):
#     tid = cuda.grid(1)
#     stride = cuda.gridsize(1)
#
#     for i in range(tid, num_iterations, stride):
#         degrees[tid] = degrees[tid] + src_nodes[i]
#
# def get_avg_degree_gpu(G, i):
#     degree_sum = 0
#     num_nodes = G.number_of_nodes()
#
#     src_nodes = np.random.randint(0, num_nodes, i)
#     dst_nodes = np.random.randint(0, num_nodes, i)
#
#     # Move data to the GPU
#     src_nodes_gpu = cuda.to_device(src_nodes)
#     dst_nodes_gpu = cuda.to_device(dst_nodes)
#     degrees_gpu = cuda.device_array(i)
#
#     # Define CUDA kernel configuration
#     threadsperblock = 128
#     blockspergrid = (i + (threadsperblock - 1)) // threadsperblock
#
#     # Launch the CUDA kernel
#     calculate_degree_kernel[blockspergrid, threadsperblock](src_nodes_gpu, dst_nodes_gpu, degrees_gpu, i)
#
#     # Copy the result back from GPU to CPU
#     degrees_gpu.copy_to_host(degrees_gpu)
#
#     degree_sum = degrees_gpu.sum()
#     return degree_sum / i

def Watts_Strogatz(V, c, B):
    '''
    Require: Number of nodes |V|, mean degree c, parameter B
    1: return A small-world graph G(V,E)
    2: G = A regular ring lattice with |V| nodes and degree c
    3: for node vi (starting from v1), and all edges e(vi,vj), i < j do
    4:  vk = Select a node from V uniformly at random.
    5:  if rewiring e(vi; vj) to e(vi; vk) does not create loops in the graph or
        multiple edges between vi and vk then
    6:  rewire e(vi; vj) with probability B: E = E-{e(vi,vj)}, E = E U {e(vi; vk)};
    7:  end if
    8: end for
    9: Return G(V,E)
    '''

    # Create a regular ring lattice with V nodes and degree c
    G = nx.Graph()
    for vi in range(V):
        for d in range(1, c // 2 + 1):  # Connecting each node to c/2 neighbors on both sides
            vj = (vi + d) % V
            G.add_edge(vi, vj)

    # Loop over each node and its edges to possibly rewire
    for vi in G.nodes():
        for vj in list(G.neighbors(vi)):
            if vi < vj:
                vk = random.choice(list(G.nodes()))

                # Check if the edge rewiring would create loops or multiple edges
                if vk != vi and vk != vj and not G.has_edge(vi, vk):
                    if random.random() < B:
                        G.remove_edge(vi, vj)
                        G.add_edge(vi, vk)
    return G


def Barabasi_Albert(m0, m, t):
    '''
        Require: Graph G(V0; E0), where jV0j = m0 and dv  1 8 v 2 V0, number of expected connections m  m0, time to run the algorithm t
            1: return A scale-free network
            2: //Initial graph with m0 nodes with degrees at least 1
            3: G(V,E) = G(V0,E0);
            4: for 1 to t do
            5:  V = V [ fvig; // add new node vi
            6:  while di , m do
            7:      Connect vi to a random node vj 2 V, i , j ( i.e., E = E [ {e(vi,vj)} )
                    with probability P(vj) = dj / E dk
            8:  end while
            9: end for
            10: Return G(V; E)
    '''

    #  Initial graph with m0 nodes with degrees at least 1
    G = nx.complete_graph(m0)

    # Main loop where nodes are added to the network
    for _ in range(t):
        #  Add new node vi
        new_node = len(G)
        G.add_node(new_node)

        #  Connect the new node with m existing nodes
        degrees = nx.degree(G)
        node_probabilities = [degree / sum(dict(degrees).values()) for _, degree in degrees]

        targets = set()
        while len(targets) < m:
            # Connect to existing node with preferential attachment
            target = random.choices(population=list(G.nodes), weights=node_probabilities, k=1)[0]
            if target not in targets:
                targets.add(target)

        # Add edges to new node
        G.add_edges_from([(new_node, target) for target in targets])

    # Return the final graph
    return G


def load_data(filename, print=False):
    if filename == 'com-amazon.ungraph.txt':
        with open('com-amazon.ungraph.txt') as fin, open('com-amazon.ungraph(fixed).txt', 'w') as fout:
            for line in fin:
                fout.write(line.replace('\t', ','))
        fin.close()
        fout.close()
        filename = 'com-amazon.ungraph(fixed).txt'

    with open(filename, 'r') as nodecsv:
        nodereader = csv.reader(nodecsv)
        nodes = [n for n in nodereader][1:]

    node_names = [n[0] for n in nodes]
    # node_names = list(set(node_names))

    with open(filename, 'r') as edgecsv:
        edgereader = csv.reader(edgecsv)
        edges = [tuple(e) for e in edgereader][1:]

    if print:
        print('Nodes:' + str(len(node_names)))
        print('Edges:' + str(len(edges)))

    return node_names, edges


def make_graph(node_names, edges):
    G = nx.Graph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)
    return G


def disp_graph(G):
    nx.draw(G)
    plt.show()
    return


def get_avg_degree(G, i):
    x = 0

    for count in range(i):
        percent_complete = (count + 1) / i
        progress_bar(percent_complete)
        cur = random.choice(list(G.nodes()))
        x += G.degree[cur]
    progress_bar(1, complete=True)
    return x / i


def worker(G, count, i, results):
    x = 0
    for _ in range(count):
        cur = random.choice(list(G.nodes()))
        x += G.degree[cur]
    results.append(x)


def get_avg_degree_parallel(G, i, num_processes=4):
    pool = multiprocessing.Pool(processes=num_processes)
    results = multiprocessing.Manager().list()
    chunk_size = i // num_processes
    count_per_process = [chunk_size] * num_processes

    # Distribute the work among processes
    pool.starmap(worker, [(G, count, chunk_size, results) for count in count_per_process])
    pool.close()
    pool.join()

    total = sum(results)
    return total / i


def get_avg_clust(G, i):
    x = 0

    for count in range(i):
        percent_complete = (count + 1) / i
        progress_bar(percent_complete)
        cur = random.choice(list(G.nodes()))
        x += nx.clustering(G, cur)
    progress_bar(1, complete=True)
    return x / i


def get_avg_path(G, i):
    x = 0

    # Create a list of connected components in the graph
    components = list(nx.connected_components(G))

    if len(components) == 0:
        return 0  # The graph is empty

    for count in range(i):
        # Update the loading bar
        percent_complete = (count + 1) / i
        progress_bar(percent_complete)
        # Randomly select a connected component
        component = random.choice(components)

        # If the component has only one node, its shortest path length is 0
        if len(component) == 1:
            x += 0
        else:
            # Select two random nodes from the component
            start, end = random.sample(component, 2)
            try:
                shortest_path_length = nx.shortest_path_length(G, source=start, target=end)
                x += shortest_path_length
            except nx.NetworkXNoPath:
                # If there's no path between the selected nodes, NetworkX raises an exception
                # In this case, you might want to handle it appropriately (e.g., skip or set to a special value)
                pass
    # Finish the loading bar
    progress_bar(1, complete=True)
    return x / i


def progress_bar(percent, complete=False):
    bar_length = 50
    block = int(round(bar_length * percent))
    progress = "=" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\r[{progress}] {int(percent * 100)}%")

    if complete:
        sys.stdout.write("\n")

    sys.stdout.flush()


def print_table(ON_amazon, ON_twitch, WS, BA):
    '''
    | ------------------------------------------------------------------------------------------------------------------------------------------- |
    |                         Original Network                                |          Watts-Strogatz         |         Barabasi-Albert         |
    | ----------------------------------------------------------------------- | ------------------------------- | ------------------------------- |
    | Network         |   Size  |  Avg Deg  |   Avg Path Len  |  Clust Coeff  |   Avg Path Len  |  Clust Coeff  |   Avg Path Len  |  Clust Coeff  |
    | Com-Amazon      |         |           |                 |               |                 |               |                 |               |
    | Twitch Gamers   |         |           |                 |               |                 |               |                 |               |
    | ------------------------------------------------------------------------------------------------------------------------------------------- |
    '''
    i = 1

    t = [['', 'Original Network', 'Watts-Strogatz', 'Barabasi-Albert'],
         ['Network', 'Size', 'Average Degree', 'Average Path Length', 'Clustering Coefficient', 'Average Path Length',
          'Clustering Coefficient', 'Average Path Length', 'Clustering Coefficient'],
         ['Com-Amazon', ON_amazon.size(), get_avg_degree(ON_amazon, i), get_avg_path(ON_amazon, i),
          get_avg_clust(ON_amazon, i), get_avg_path(WS, i), get_avg_clust(WS, i)],
         ['Twitch Gamers', ON_twitch.size(), get_avg_degree(ON_twitch, i), get_avg_path(ON_twitch, i),
          get_avg_clust(ON_twitch, i), get_avg_path(BA, i), get_avg_clust(BA, i)]]

    print(t)
    return


def main(print_status=True):
    '''
    Important: The network in each of these data sets may contain multiple components.
    You should extract the largest connected component and simulate this component using the network model implementation.
    '''
    print('CS5990 - Team 4 - Group Assignment 1\nGabriel Alfredo Siguenza, Alec Gotts, Jun Ho Ha, Ardavan Sherafat\n')

    if print_status: print('Loading "large_twitch_edges.csv" data')
    node_names, edges = load_data('large_twitch_edges.csv')
    if print_status: print('Making "large_twitch_edges.csv" graph')
    Graph_twitch = make_graph(node_names, edges)
    Graph_twitch = Graph_twitch.subgraph(max(nx.connected_components(Graph_twitch), key=len))

    if print_status: print('Loading "com-amazon.ungraph.txt" data')
    # node_names, edges = load_data('com-amazon.ungraph.txt')
    if print_status: print('Making "com-amazon.ungraph.txt" graph')
    # Graph_amazon = make_graph(node_names, edges)
    # Graph_amazon = Graph_amazon.subgraph(max(nx.connected_components(Graph_amazon), key=len))

    if print_status: print('Making Watts-Strogatz (4.1) graph')
    Graph_WS = Watts_Strogatz(100, 4, 0.2)

    if print_status: print('Making Barabasi-Albert (4.2) graph')
    Graph_BA = Barabasi_Albert(5, 2, 50)

    # Convert the NetworkX graph to a cuGraph object
    # G_cugraph = cugraph.Graph()
    # G_cugraph.from_cudf_edgelist(nx.to_pandas_edgelist(G))

    # Display graphs
    # disp_graph(Graph_WS)
    # disp_graph(Graph_BA)

    # disp_graph(Graph_amazon)
    # disp_graph(Graph_twitch)

    # # Define the size of the subgraph (number of edges)
    # subgraph_size = 5000
    #
    # # Randomly select a subset of edges
    # subgraph_edges = random.sample(Graph_amazon.edges(), subgraph_size)
    #
    # # Create a subgraph containing only the selected edges and their connected nodes
    # subgraph_a = Graph_amazon.edge_subgraph(subgraph_edges)
    #
    # # Randomly select a subset of edges
    # subgraph_edges = random.sample(Graph_amazon.edges(), subgraph_size)
    #
    # # Create a subgraph containing only the selected edges and their connected nodes
    # subgraph_t = Graph_amazon.edge_subgraph(subgraph_edges)
    #
    # disp_graph(subgraph_a)
    # disp_graph(subgraph_t)

    input('Press any key to continue...')

    # set the total nodes to calculate averages on
    # num_nodes = 100000

    # get average path length, degree, and clustering coefficient for twitch and amazon graphs
    # avg_path_twitch = get_avg_path(Graph_twitch, num_nodes)
    # avg_degree_twitch = get_avg_degree(Graph_twitch, num_nodes)
    # avg_degree_twitch = get_avg_degree_parallel(Graph_twitch, 15, num_processes=4)
    # avg_clust_twitch = get_avg_clust(Graph_twitch, num_nodes)

    # avg_path_amazon = get_avg_path(Graph_amazon, num_nodes)
    # avg_degree_amazon = get_avg_degree(Graph_amazon, num_nodes)
    # avg_clust_amazon = get_avg_clust(Graph_amazon, num_nodes)

    # display results
    # print("Average Shortest Path Length for Twitch Graph:", avg_path_twitch)
    # print("Average Degree for Twitch Graph:", avg_degree_twitch)
    # print("Average Clustering Coefficient for Twitch Graph:", avg_clust_twitch)

    # print("Average Shortest Path Length for Amazon Graph:", avg_path_amazon)
    # print("Average Degree for Amazon Graph:", avg_degree_amazon)
    # print("Average Shortest Path Length for Amazon Graph:", avg_clust_amazon)

    # print_table(Graph_amazon, Graph_twitch, Graph_WS, Graph_BA)
    return


if __name__ == "__main__":
    main()
