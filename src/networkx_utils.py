### IMPORTS AND STATIC STUFF ###
from copy import deepcopy
import networkx as nx
from pprint import pprint

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from collections import OrderedDict
from src.metrics import custom_silhouette_score


def create_mst_from_distance_matrix(distance_matrix, label_col='peptide', index_col='raw_index',
                                    weight_col=None, algorithm='kruskal'):
    """
    Given a labelled distance matrix, create the Graph and minimum spanning tree associated
    Args:
        distance_matrix:
        label_col:
        index_col:
        algorithm:

    Returns:

    """
    distance_matrix = distance_matrix.copy(deep=True)
    indexing = [str(x) for x in distance_matrix.index] if type(
        distance_matrix.columns[0]) == str else distance_matrix.index
    values = distance_matrix[indexing].values
    labels = distance_matrix[label_col].values
    if index_col is not None:
        # print(f'here in create_mst_from_distance_matrix, index_col is not None, using {index_col}')
        raw_indices = distance_matrix[index_col].values
    else:
        print(f'Here in create_mst_from_distance_matrix, index_col is None. Will create and use {"raw_index"} instead')
        index_col = 'raw_index'
        distance_matrix[index_col] = [f'seqid_{i}' for i in range(len(distance_matrix))]
        raw_indices = distance_matrix[index_col].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    if weight_col is not None and weight_col in distance_matrix.columns:
        weights = distance_matrix[weight_col].values

    # Creating the graph and storing data in nodes
    G = nx.Graph(values)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['label'] = labels[i]
        G.nodes[node]['index'] = raw_indices[i]
        if weight_col is not None and weight_col in distance_matrix.columns:
            G.nodes[node]['weight'] = weights[i]

    tree = nx.minimum_spanning_tree(G, algorithm=algorithm)
    # For now, return all values as they may be useful later for analysis
    return G, tree, distance_matrix, values, labels, encoded_labels, label_encoder, raw_indices


def prune_by_distance(tree, distance_threshold, prune_node=True, labels=None, label_key='peptide', verbose=False):
    """
    Given a minimum spanning tree, prune it based on a distance threshold
    Args:
        tree:
        distance_threshold:
        prune_node: Bool, remove nodes that have become isolated after edge-pruning
        labels:
        verbose:

    Returns:

    """
    tree_pruned = tree.copy()
    removed_edges = []
    removed_nodes = []
    for u, v in tree_pruned.edges():
        weight = tree_pruned.get_edge_data(u, v)['weight']
        if weight >= distance_threshold:  # assuming weight is a distance and not a similarity
            # If providing label, use it to get l1, l2 for the class check
            if labels is not None:
                l1 = labels[u]
                l2 = labels[v]
            # else, if label-key exists and the key is in the tree nodes attributes, use those
            elif label_key is not None and label_key in tree.nodes[0]:
                l1 = tree_pruned.nodes[u][label_key]
                l2 = tree_pruned.nodes[v][label_key]
            else:
                l1, l2 = np.nan, np.nan
            removed_edges.append((u, v, round(weight, 4), int(l1 == l2)))
            tree_pruned.remove_edge(u, v)
    if prune_node:
        removed_nodes = list(nx.isolates(tree_pruned))
        tree_pruned.remove_nodes_from(removed_nodes)

    if verbose:
        print(f'***\nbefore pruning : {len(tree.edges())} edges, {len(tree.nodes())} nodes\n***')
        print(f'***\nafter pruning : {len(tree_pruned.edges())} edges, {len(tree_pruned.nodes())} nodes\n***')
        print(len(removed_edges), 'removed edges',
              f'\n{np.array([x[3] for x in removed_edges]).mean():.2%} of the removed edges were actually same-class')

    return tree_pruned, removed_edges, removed_nodes


def get_color_map(distance_matrix, label_col='peptide', palette='tab10'):
    labels = distance_matrix[label_col].unique()
    color_map = {k: v for k, v in zip(sorted(np.unique(labels)), sns.color_palette(palette, len(np.unique(labels))))}
    return color_map


def plot_mst_spring(tree, color_map=None, title=None, figsize=(8, 8),
                    iterations=300, threshold=1e-5, scale=0.9, k=0.05, dim=2, seed=131):
    sns.set_style('darkgrid')
    # Visualize the graph and the minimum spanning tree
    f, a = plt.subplots(1, 1, figsize=figsize)
    pos = nx.spring_layout(tree, iterations=iterations, threshold=threshold, seed=seed, scale=scale, k=k, dim=dim)
    if color_map is None:
        unique_labels = list({tree.nodes[node]['label'] for node in tree.nodes()})
        palette = sns.color_palette("tab10", len(unique_labels))  # Use HSV for distinct colors
        color_map = {label: color for label, color in zip(unique_labels, palette)}

    node_colors = [color_map[tree.nodes[node]['label']] for node in tree.nodes()]

    nx.draw_networkx_nodes(tree, pos, node_color=node_colors, node_size=52.5,
                           ax=a)  # nx.draw_networkx_edges(G, pos, edge_color="grey")
    nx.draw_networkx_labels(tree, pos, font_size=4, font_color='w', font_weight='semibold', font_family="monospace")
    nx.draw_networkx_edges(tree, pos, edge_color="k", width=0.75)
    # Create a legend
    legend_labels = {k: color_map[k] for k in sorted(color_map.keys())}  # Sorted keys for consistent order
    patches = [Patch(color=color, label=f'{label}') for label, color in legend_labels.items()]
    plt.legend(handles=patches, title='Node Classes', loc='best', borderaxespad=0.)
    # Find bottleneck score ; with centrality AND connectivity
    if title is not None:
        a.set_title(title)
    plt.axis("off")
    plt.show()


def get_cluster_stats_from_graph(graph, nodes_list):
    """
    Here this is a potentially bad behaviour because of how graphs can be created
    and saved with a "weight" node data even if we don't intend to use it
    240902 I made it like this because I didn't want to re-do too much code
    Args:
        graph:
        nodes_list:

    Returns:

    """
    cluster = [graph.nodes(data=True)[x] for x in nodes_list]
    weighted = 'weight' in cluster[0].keys()
    labels = [x['label'] for x in cluster]
    if weighted:
        # assumes we have a "weight" parameter in the node data
        weights = [x['weight'] for x in cluster]  # ex here could be the normalized count
        counts = {label: sum(weight for l, weight in zip(labels, weights) if l == label) for label in set(labels)}

    else:
        counts = {k: labels.count(k) for k in np.unique(labels)}
    majority_label = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]
    purity = counts[majority_label] / sum(counts.values())
    return {'cluster_size': len(nodes_list),
            'majority_label': majority_label,
            'purity': purity,
            'counts': counts,
            'members': nodes_list}


def get_nodes_stats(G, node):
    neighbours = list(nx.neighbors(G, node))
    edges = [tuple(sorted((node, target_node))) for target_node in nx.neighbors(G, node)]
    edge_weights = {edge: G.edges[edge]['weight'] for edge in edges}
    centralities = nx.edge_betweenness_centrality(G)
    filtered_centralities = {k: v for k, v in centralities.items() if k in edges}
    return {'n_neighbours': len(neighbours), 'neighbours': neighbours,
            'max_dist': max(edge_weights.values()), 'sum_dist': sum(edge_weights.values()),
            'mean_dist': np.mean(list(edge_weights.values())),
            'max_edge_centrality': max(filtered_centralities.values()),
            'sum_edge_centrality': sum(filtered_centralities.values()),
            'mean_edge_centrality': np.mean(list(filtered_centralities.values()))}


def silhouette_rank_cut(tree, dist_array, which='edge',
                        cut_threshold=1, cut_method='top', scale_aggregation='mean'):
    """
    Args:
        tree:
        cut_threshold:
        cut_method:
        distance_weighted: Here it means "weighted" for distance weighted edge betweenness

    Returns:

    """
    assert cut_method in ['threshold',
                          'top'], f'`cut_method` must be either "threshold" or "top". Got {cut_method} instead'
    assert (cut_method == 'threshold' and type(cut_threshold) == float) or (
            cut_method == 'top' and type(cut_threshold) == int), \
        '`cut_threshold` should of type either int or float (for cut_method=="threshold" or cut_method=="top")'
    assert which in ['betweenness', 'edge'], f'got {which} ?? instead of betweenness or rank'
    # deep copy the tree to preserve it in case we need the original tree for other things
    tree_cut = tree.copy()
    # Get the tree and its stats
    initial_clusters = sorted([get_cluster_stats_from_graph(tree_cut, x) for x in nx.connected_components(tree_cut)],
                              key=lambda x: x['cluster_size'], reverse=True)
    current_silhouette_samples = get_score_at_cut(dist_array, initial_clusters, aggregation='none')
    silhouette_scale = 1 - current_silhouette_samples
    if which == 'edge':
        sorted_edges = rank_edges(tree_cut, silhouette_scale, agg=scale_aggregation)
    elif which == 'betweenness':
        sorted_edges = rank_betweenness(tree_cut, silhouette_scale, agg=scale_aggregation)
    else:
        print('wtf')
    if cut_method == "threshold":
        edges_to_remove = [x for x in sorted_edges if x[1] > cut_threshold]
    elif cut_method == "top":
        edges_to_remove = sorted_edges[:cut_threshold]
    # Remove edges (x[1] is the centrality), computes the disconnected nodes and remove them (discard singletons)
    # OR should the singletons be kept in the graph and put in new "clusters" ?
    tree_cut.remove_edges_from([x[0] for x in edges_to_remove])
    nodes_to_remove = list(nx.isolates(tree_cut))
    tree_cut.remove_nodes_from(nodes_to_remove)
    clusters = sorted([get_cluster_stats_from_graph(tree_cut, x) for x in nx.connected_components(tree_cut)],
                      key=lambda x: x['cluster_size'], reverse=True)

    return tree_cut, clusters, edges_to_remove, nodes_to_remove


def get_per_cluster_score_labels_update(dist_array, clusters):
    """
    TODO : 241028
           This is not yet used ; For now let's look at a global SI-weighted edge cut instead of a more
           complicated cluster-ranked intra-cluster cuts
    Args:
        dist_array:
        clusters:

    Returns:

    """
    pred_labels = get_pred_labels(dist_array, clusters)
    current_score = get_score_at_cut(dist_array, clusters, aggregation='none')
    # this is a per-cluster avg for each cluster in pred_labels_5
    per_cluster_score = [current_score[np.where(pred_labels == label)[0]].mean() for label in np.unique(pred_labels)]
    updated_clusters = [
        dict(OrderedDict([('score', s), *d.items()])) for d, s in zip(clusters, per_cluster_score)]
    list_of_metrics = [
        {
            'max_dist': dist_array[idx][:, idx].max(),
            'med_dist': np.median(dist_array[idx][:, idx][dist_array[idx][:, idx] != 0]),
            'avg_dist': dist_array[idx][:, idx][dist_array[idx][:, idx] != 0].mean()
        }
        for label in np.unique(pred_labels)
        for idx in [np.where(pred_labels == label)[0]]
    ]
    updated_clusters = [
        dict(OrderedDict([(k, v) for k, v in metrics.items()] + list(cluster.items())))
        for cluster, metrics in zip(updated_clusters, list_of_metrics)]
    silhouette_scale = 1 - current_score
    return updated_clusters, current_score, pred_labels, silhouette_scale


def rank_edges(tree, silhouette_scale, agg='mean'):
    # Here, silhouette_scale is 1-silhouette_sample
    # for now, define either sum or mean for the aggregation of the silhouette_scale
    # That is the same thing lol (use mean to keep values close to 0 and 1)
    assert agg in ['mean', 'max'], f'agg must be "mean" or "max"'
    edge_list = list(tree.edges(data=True))
    fc = {'mean': np.mean, 'max': np.max}
    return sorted(
        {(x[0], x[1]): x[2]['weight'] * fc[agg](silhouette_scale[[x[0], x[1]]]) for x in edge_list}.items(),
        key=lambda item: item[1], reverse=True)


def rank_betweenness(tree, silhouette_scale, agg='mean'):
    # Here, silhouette_scale is 1-silhouette_sample
    # for now, define either sum or mean for the aggregation of the silhouette_scale
    # That is the same thing lol (use mean to keep values close to 0 and 1)
    assert agg in ['mean', 'max'], f'agg must be "mean" or "max"'
    fc = {'mean': np.mean, 'max': np.max}
    edge_betweenness = nx.edge_betweenness_centrality(tree)
    # Silhouette + distance weighted betweenness?
    edge_betweenness = {k: tree.edges[k]['weight'] * fc[agg](silhouette_scale[[k[0], k[1]]]) * v
                        for k, v in edge_betweenness.items()
                        }

    return sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True)


def edge_betweenness_cut(tree, cut_threshold, cut_method='threshold', distance_weighted=False):
    """

    Args:
        tree:
        cut_threshold:
        cut_method:
        distance_weighted: Here it means "weighted" for distance weighted edge betweenness

    Returns:

    """
    assert cut_method in ['threshold',
                          'top'], f'`cut_method` must be either "threshold" or "top". Got {cut_method} instead'
    assert (cut_method == 'threshold' and type(cut_threshold) == float) or (
            cut_method == 'top' and type(cut_threshold) == int), \
        '`cut_threshold` should of type either int or float (for cut_method=="threshold" or cut_method=="top")'
    # deep copy the tree to preserve it in case we need the original tree for other things
    tree_cut = tree.copy()
    edge_betweenness = nx.edge_betweenness_centrality(tree_cut)

    # Weight of an edge here is the distance between two nodes
    if distance_weighted:
        edge_betweenness = {k: tree_cut.edges[k]['weight'] * v for k, v in edge_betweenness.items()}
    # sorted for printing purposes
    sorted_edges = sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True)

    if cut_method == "threshold":
        edges_to_remove = [x for x in sorted_edges if x[1] > cut_threshold]
    elif cut_method == "top":
        edges_to_remove = sorted_edges[:cut_threshold]
    # Remove edges (x[1] is the centrality), computes the disconnected nodes and remove them (discard singletons)
    # OR should the singletons be kept in the graph and put in new "clusters" ?
    tree_cut.remove_edges_from([x[0] for x in edges_to_remove])
    nodes_to_remove = list(nx.isolates(tree_cut))
    tree_cut.remove_nodes_from(nodes_to_remove)
    clusters = sorted([get_cluster_stats_from_graph(tree_cut, x) for x in nx.connected_components(tree_cut)],
                      key=lambda x: x['cluster_size'], reverse=True)

    return tree_cut, clusters, edges_to_remove, nodes_to_remove


def node_betweenness_cut(tree, cut_threshold, cut_method='threshold', distance_weighted=False):
    assert cut_method in ['threshold',
                          'top'], f'`cut_method` must be either "threshold" or "top". Got {cut_method} instead'
    assert (cut_method == 'threshold' and type(cut_threshold) == float) or (
            cut_method == 'top' and type(cut_threshold) == int), \
        '`cut_threshold` should of type either int or float (for cut_method=="threshold" or cut_method=="top")'
    # deep copy the tree to preserve it in case we need the original tree for other things
    tree_cut = tree.copy()
    node_betweenness = nx.betweenness_centrality(tree_cut)

    # Weight of an edge here is the distance between two nodes
    if distance_weighted:
        pass  # For now, don't do this part because it can get extremely slow
        # node_betweenness = {k:get_nodes_stats(tree_cut, k)['mean_dist']*v for k,v in node_betweenness.items()}
    sorted_nodes = sorted(node_betweenness.items(), key=lambda item: item[1], reverse=True)
    if cut_method == "threshold":
        nodes_to_remove = [x for x in sorted_nodes if x[1] > cut_threshold]
    elif cut_method == "top":
        nodes_to_remove = sorted_nodes[:cut_threshold]

    tree_cut.remove_nodes_from([x[0] for x in nodes_to_remove])
    # Further identify nodes that have become singletons and must be removed
    isolated_nodes = list(nx.isolates(tree_cut))
    tree_cut.remove_nodes_from(isolated_nodes)
    nodes_to_remove.extend([x for x in sorted_nodes if x[0] in isolated_nodes])
    # Identify the edges that have been cut
    list_nodes = [x[0] for x in nodes_to_remove]
    # Those edges don't exist anymore in tree_cut but we take them for return and logging purposes here
    edges_to_remove = [x for x in tree.edges() if any([x[0] in list_nodes or x[1] in list_nodes])]
    clusters = sorted([get_cluster_stats_from_graph(tree_cut, x) for x in nx.connected_components(tree_cut)],
                      key=lambda x: x['cluster_size'], reverse=True)
    return tree_cut, clusters, edges_to_remove, nodes_to_remove


def betweenness_cut(tree, cut_threshold, cut_method='threshold', which='edge', distance_weighted=False, verbose=1):
    """
    Wrap-around method to do either edge or node cut.
    Can select either threshold (centrality>cut_threshold) or top (first `cut_threshold` elements) cutting.
    Weighted works only for edge cutting for now, using the distance as weight.
    Verbosity levels of 2 will reduce the speed in order to log every centrality
    Args:
        tree: A minimum spanning tree
        cut_threshold (int or float) : A float if cut_method is "threshold", or int if cut_method is "top"
        cut_method (str): "threshold" or "top" cut mode ;
        which (str) : "edge" or "node", which way to cut the tree
        verbose (int): verbosity levels, from 0 to 2
        distance_weighted (bool) : whether to use distance to weight the centrality.

    Returns:

    """
    assert which in ['edge', 'node'], f'`which` must be either "node" or "edge". Got {which} instead'
    assert cut_method in ['threshold',
                          'top'], f'`cut_method` must be either "threshold" or "top". Got {cut_method} instead'
    assert (cut_method == 'threshold' and type(cut_threshold) == float) or (
            cut_method == 'top' and type(cut_threshold) == int), \
        '`cut_threshold` should of type either int or float (for cut_method=="threshold" or cut_method=="top")'

    # Take from the original tree to get the original centrality 
    if which == 'edge':
        tree_cut, clusters, edges_to_remove, nodes_to_remove = edge_betweenness_cut(tree, cut_threshold, cut_method,
                                                                                    distance_weighted)
        if verbose > 1:
            node_betweenness = nx.betweenness_centrality(tree)
            nodes_to_remove = [x for x in sorted(node_betweenness.items(), key=lambda item: item[1], reverse=True) if
                               x[0] in nodes_to_remove]
    elif which == 'node':
        tree_cut, clusters, edges_to_remove, nodes_to_remove = node_betweenness_cut(tree, cut_threshold, cut_method,
                                                                                    distance_weighted)
        if verbose > 1:
            edge_betweenness = nx.edge_betweenness_centrality(tree)
            edges_to_remove = [x for x in sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True) if
                               x[0] in edges_to_remove]

    if verbose > 0:
        print(f'\t\tedges, nodes')
        print(f'before cutting:\t{len(list(tree.edges()))},\t{len(list(tree.nodes()))}')
        print(f'after cutting:\t{len(list(tree_cut.edges()))},\t{len(list(tree_cut.nodes()))}')
        print('N components before cutting:\t', len(list(nx.connected_components(tree))))
        print('N components after cutting:\t', len(list(nx.connected_components(tree_cut))))
        print('\nEdges removed:')
        pprint(edges_to_remove)
        print('\nNodes removed:')
        pprint(nodes_to_remove)

    return tree_cut, clusters, edges_to_remove, nodes_to_remove


def get_pred_labels(dist_array, clusters):
    # Initialize pred_labels with -1
    pred_labels = np.full(dist_array.shape[0], -1, dtype=np.int16)
    # Assign cluster labels
    for i, c in enumerate(clusters):
        pred_labels[np.array(list(c['members']))] = i
    # Identify singleton indices
    singleton_indices = np.where(pred_labels == -1)[0]
    # Calculate the number of clusters and singletons
    n_classes = len(clusters)
    n_singletons = len(singleton_indices)
    # Assign unique labels to singletons
    pred_labels[singleton_indices] = np.arange(n_classes, n_classes + n_singletons)

    return pred_labels


def iterative_size_cut(dist_array, tree, initial_cut_threshold, initial_cut_method, top_n=1, which='edge',
                       distance_weighted=False, verbose=1, max_size=6, silhouette_aggregation='micro'):
    # From a tree take the top N ?? and continue cutting until the subgraphs all reach a certain size or ?
    # Or if the weighted mean edge distance is some threshold ??
    # Initial cut, takes the input parameters
    tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
                                                                       which, distance_weighted, verbose)
    current_silhouette_score = get_score_at_cut(dist_array, clusters, precision=4, aggregation=silhouette_aggregation)
    subgraphs = []
    # What exit condition ??
    # Silhouette score --> Report per iteration across entire graph
    # maybe do top_n across all subgraphs and not iteratively 
    # Adjusted rand index
    iter = 0
    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    n_clusters = [len(clusters)]
    while any([x['cluster_size'] >= max_size for x in clusters]):
        for i, c in enumerate(clusters):
            if c['cluster_size'] >= max_size:
                # Remove the current cluster from the list ; Whatever they get cut into will be extended to the back of the clusters list
                current_cluster = clusters.pop(i)
                # Subsequent cuts : Take the cluster subgraph, cut top_n with 'top' cut method.
                subgraph = nx.subgraph(tree_cut, current_cluster['members'])
                subgraph_cut, subgraph_clusters, subgraph_edges_removed, subgraph_nodes_removed = betweenness_cut(
                    subgraph, cut_threshold=top_n, cut_method='top', which=which, distance_weighted=distance_weighted,
                    verbose=verbose)
                clusters.extend(subgraph_clusters)
                # max_size=max([x['cluster_size'] for x in clusters if x['majority_label'] != 'healthy'])
                edges_removed.extend(subgraph_edges_removed)
                nodes_removed.extend(subgraph_nodes_removed)
                subgraphs.append(subgraph_cut)
                current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
                retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
                scores.append(current_silhouette_score)
                purities.append(np.mean([x['purity'] for x in clusters]))
                mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
                n_clusters.append(len(clusters))
                # print(iter, np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score, round(sum([x['cluster_size'] for x in clusters])/len(dist_array),4))
                # iter+=1
        else:
            continue

    # Create an empty graph to combine subgraphs into
    tree_return = nx.Graph()
    # Combine all subgraphs
    for subgraph in subgraphs:
        tree_return = nx.compose(tree_return, subgraph)
    return tree_return, subgraphs, clusters, edges_removed, nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters


def get_score_at_cut(dist_array, clusters, precision=4, aggregation='micro'):
    """
    From the various clusters and pruned nodes, reconstruct an array of predicted label with integers representing predicted class
    Uses that and true labels and dist_matrix to compute scores
    Args:
        clusters:
        pruned_nodes:
    Returns:
    """
    pred_labels = get_pred_labels(dist_array, clusters)
    return custom_silhouette_score(dist_array, pred_labels,
                                   metric='precomputed', aggregation=aggregation, precision=precision)


def iterative_topn_cut(dist_array, tree, initial_cut_threshold=1, initial_cut_method='top', top_n=1, which='edge',
                       distance_weighted=False, verbose=1, score_threshold=.75, silhouette_aggregation='micro'):
    # Set initial_cut_method to 'top' and initial_cut_threshold to top_n=1 to have fully iterative behaviour
    tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
                                                                       which, distance_weighted, verbose)
    current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
    print('Initial mean purity, silhouette score, retention')
    print(np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score,
          round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))

    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    n_clusters = [len(clusters)]
    best_silhouette_score = -1
    # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
    best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
        edges_removed), deepcopy(nodes_removed)
    # Make a copy before starting the iteration to re-use the variable
    tree_trimmed = tree_cut.copy()
    while current_silhouette_score <= score_threshold or retentions[-1] > 0:
        tree_trimmed, clusters, edges_trimmed, nodes_trimmed = betweenness_cut(tree_trimmed, cut_threshold=top_n,
                                                                               cut_method='top', which=which,
                                                                               distance_weighted=distance_weighted,
                                                                               verbose=verbose)
        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            n_clusters.append(len(clusters))
            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except ValueError:
            # Fake an exit condition when we reach the error where we only have singletons
            current_silhouette_score = score_threshold + 1
            break

        # print(iter, np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score, round(sum([x['cluster_size'] for x in clusters])/len(dist_array),4))
        # iter += 1
    subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
    return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters


def iterative_topn_cut_logsize(dist_array, tree, initial_cut_threshold=1, initial_cut_method='top', cut_threshold=1,
                               which='edge', distance_weighted=True, verbose=0, score_threshold=1,
                               min_purity=.8, min_size=6,
                               silhouette_aggregation='micro'):
    # Set initial_cut_method to 'top' and initial_cut_threshold to top_n=1 to have fully iterative behaviour
    # -> If using augmented tree, we need to iteratively pre-cut until we have at least 2 connected components
    # For now, find some percentile threshold so that this doesn't happen (let's use 0.5)
    tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
                                                                       which, distance_weighted, verbose)
    current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
    print('Initial mean purity, silhouette score, retention')
    print(np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score,
          round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4))

    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
    clusters_above = [[x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity]]
    n_clusters = [len(clusters)]
    best_silhouette_score = -1
    # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
    best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
        edges_removed), deepcopy(nodes_removed)
    # Make a copy before starting the iteration to re-use the variable
    tree_trimmed = tree_cut.copy()
    # while current_silhouette_score <= score_threshold or retentions[-1] > 0:
    for _ in tqdm(range(len(best_tree.edges()) - 1), desc='cuts', leave=True, position=0):
        tree_trimmed, clusters, edges_trimmed, nodes_trimmed = betweenness_cut(tree_trimmed,
                                                                               cut_threshold=cut_threshold,
                                                                               cut_method='top', which=which,
                                                                               distance_weighted=distance_weighted,
                                                                               verbose=verbose)
        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            all_cluster_sizes.append([x['cluster_size'] for x in clusters])
            n_clusters.append(len(clusters))
            clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])

            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except ValueError:
            # Fake an exit condition when we reach the error where we only have singletons
            current_silhouette_score = score_threshold + 1
            break

        # print(iter, np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score, round(sum([x['cluster_size'] for x in clusters])/len(dist_array),4))
        # iter += 1
    subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
    return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, all_cluster_sizes, clusters_above


def iterative_SIrank_cut_logsize(dist_array, tree, initial_cut_threshold=1, initial_cut_method='top',
                                 which='edge', cut_threshold=1, cut_method='top', scale_aggregation='max',
                                 min_purity=.8, min_size=6,
                                 silhouette_aggregation='micro'):
    # Start with a betweenness cut in order to have some initial cluster silhouette score
    tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
                                                                       which='edge', distance_weighted=True, verbose=0)
    # TODO At some point if I keep doing these experiments I should probably do both micro and macro avg at once
    current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)

    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
    n_clusters = [len(clusters)]
    clusters_above = [[x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity]]

    best_silhouette_score = -1
    # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
    best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
        edges_removed), deepcopy(nodes_removed)
    # Make a copy before starting the iteration to re-use the variable
    tree_trimmed = tree_cut.copy()
    # TODO: For now, do range(size(array)) to cut everything down to the last edge
    for _ in tqdm(range(len(best_tree.edges()) - 1), desc='cuts', leave=True, position=0):
        tree_trimmed, clusters, edges_trimmed, nodes_trimmed = silhouette_rank_cut(tree_trimmed, dist_array, which,
                                                                                   cut_threshold,
                                                                                   cut_method,
                                                                                   scale_aggregation=scale_aggregation)

        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            all_cluster_sizes.append([x['cluster_size'] for x in clusters])
            clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])

            n_clusters.append(len(clusters))
            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except:
            break
    subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
    return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, all_cluster_sizes, clusters_above


def twophase_cuts(tree, dist_array, phase1_proportion=0.1, initial_cut_threshold=1, initial_cut_method='top',
                  which_top='edge', which_rank='edge', scale_aggregation='max', verbose=0, distance_weighted=True,
                  min_purity=.8, min_size=6,
                  silhouette_aggregation='micro'):
    """

    Args:
        tree:
        dist_array:
        phase1_proportion:
        initial_cut_threshold:
        initial_cut_method:
        which_top: 'edge' or 'node'
        which_rank: 'edge' or 'betweenness'
        scale_aggregation:
        verbose:
        distance_weighted:
        min_purity:
        min_size:
        silhouette_aggregation:

    Returns:

    """
    assert phase1_proportion <= 0.95, 'wtf man'
    # Initialise with a first cut (if nx connected_components == 1)
    tree_cut = tree.copy()
    if len(list(nx.connected_components(tree))) == 1:
        while len(list(nx.connected_components(tree_cut))) == 1:
            tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree_cut, initial_cut_threshold,
                                                                               initial_cut_method,
                                                                               which='edge', distance_weighted=True,
                                                                               verbose=0)
    else:
        tree_cut = tree.copy()
        clusters = sorted([get_cluster_stats_from_graph(tree_cut, x) for x in nx.connected_components(tree_cut)],
                          key=lambda x: x['cluster_size'], reverse=True)
        edges_removed = [None]
        nodes_removed = [None]

    # Initialise variables
    current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
    n_clusters = [len(clusters)]
    clusters_above = [[x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity]]
    best_silhouette_score = -1
    # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
    best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
        edges_removed), deepcopy(nodes_removed)
    # Make a copy before starting the iteration to re-use the variable
    tree_trimmed = tree_cut.copy()

    n_edges = len(tree_trimmed.edges())
    # Phase 1 is betweenness cuts, Phase 2 is silhouette-rank cuts;
    # Assumption : Betweenness cuts splits the graph, Silhouette-rank "trims" the extremities (? TBC ?)
    n_iter_p1 = int(n_edges * phase1_proportion)
    # Phase 1
    for _ in tqdm(range(n_iter_p1), desc='Phase 1 : Betweenness cuts', position=0):
        tree_trimmed, clusters, edges_trimmed, nodes_trimmed = betweenness_cut(tree_trimmed,
                                                                               cut_threshold=1,
                                                                               cut_method='top', which=which_top,
                                                                               distance_weighted=distance_weighted,
                                                                               verbose=verbose)
        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            all_cluster_sizes.append([x['cluster_size'] for x in clusters])
            n_clusters.append(len(clusters))
            clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])

            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except ValueError:
            # Fake an exit condition when in case we reach an error with the silhouette score (i.e. only singletons)
            break

    # Phase 2
    # n_iter_p2 = n_edges - n_iter_p1 - 1
    n_iter_p2 = len(tree_trimmed.edges()) - 1
    for _ in tqdm(range(n_iter_p2), desc='Phase 2 : Silhouette trims', position=1):
        tree_trimmed, clusters, edges_trimmed, nodes_trimmed = silhouette_rank_cut(tree_trimmed, dist_array, which_rank,
                                                                                   1, 'top', scale_aggregation)
        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            all_cluster_sizes.append([x['cluster_size'] for x in clusters])
            clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])

            n_clusters.append(len(clusters))
            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except:
            break

    subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
    results_df = pd.DataFrame(
        np.array([scores, purities, retentions, mean_cluster_sizes, n_clusters, [len(x) for x in clusters_above]]).T,
        columns=['silhouette', 'mean_purity', 'retention', 'mean_cluster_size', 'n_cluster', 'n_above']).assign(
        silhouette_aggregation=silhouette_aggregation)
    best_si = results_df.iloc[int(0.1 * len(results_df)):]['silhouette'].idxmax()
    results_df['best'] = False
    results_df.loc[best_si, 'best'] = True
    return results_df, best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, all_cluster_sizes, clusters_above


def plot_sprm(df, title=None, fn=None, burn_in=0.05, vline=None, random_label=False,
              vline_label=None, hline=None, hline_label=None, f=None, a=None):
    sns.set_palette('tab10', 4)
    if f is None and a is None:
        f, a = plt.subplots(1, 1, figsize=(9, 5))
    x = range(len(df))
    s = df['silhouette']
    a.plot(x, s, label='silhouette', ls='--', lw=1)
    if not random_label:
        p = df['mean_purity']
        a.plot(x, p, label='purity', ls='--', lw=1)
    r = df['retention']
    a.plot(x, r, label='retention', ls='--', lw=1)
    if any([x in df.columns for x in ['mean_cluster_size', 'n_above']]):
        twa = a.twinx()
        twa.set_yscale('log', base=2)
        twa.grid(False)
    if 'mean_cluster_size' in df.columns:
        m = df['mean_cluster_size']
        twa.plot(x, m, label='mean size', ls='--', lw=1, c='m')
    if 'n_above' in df.columns and not random_label:
        n = df['n_above']
        twa.plot(x, n, label='n_clust above', ls='--', lw=1, c='c')
    if vline is not None:
        a.axvline(vline, label=vline_label, ls='-.', lw=1, c='r')
    if hline is not None:
        a.axhline(hline, label=hline_label, ls='-.', lw=1, c='b')
    # Take after the 5% of cuts to avoid getting the best silhouette at the very start ?
    a.axvline(df.loc[int(burn_in * len(df)):]['silhouette'].idxmax(), label='max SI', ls=':', lw=1, c='k')
    a.legend()
    twa.legend()
    if 'n_above' in df.columns:
        twa.set_ylim([1, max(df['n_above'].max()+2, df['mean_cluster_size'].max()+2)])
    else:
        twa.set_ylim([1, df['mean_cluster_size'].max()+2])
    best = df.loc[[df.loc[int(burn_in * len(df)):]['silhouette'].idxmax()]]
    title = f'{title}'+f'\n Mean Pur:{best["mean_purity"].item():.3f}, Retention:{best["retention"].item():.3f}, Silhouette:{best["silhouette"].item():.3f}, mean_cluster_size:{best["mean_cluster_size"].item():.2f}'
    if 'n_above' in df.columns:
        title = title+f' N_above:{best["n_above"].item():.1f}'
    if title is not None:
        a.set_title(title)
    if fn is not None:
        f.savefig(f'{fn}.png', dpi=300, bbox_inches='tight')
    return best


def alternating_cuts(dist_array, tree, initial_cut_threshold=1, initial_cut_method='top', cut_threshold=1,
                     which='edge', distance_weighted=True, verbose=0, score_threshold=1,
                     min_purity=.8, min_size=6,
                     silhouette_aggregation='micro', scale_aggregation='mean'):
    # Set initial_cut_method to 'top' and initial_cut_threshold to top_n=1 to have fully iterative behaviour
    # -> If using augmented tree, we need to iteratively pre-cut until we have at least 2 connected components
    # For now, find some percentile threshold so that this doesn't happen (let's use 0.5)
    tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold,
                                                                       initial_cut_method,
                                                                       which, distance_weighted, verbose)
    current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
    print('Initial mean purity, silhouette score, retention')
    print(np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score,
          round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4))

    scores = [current_silhouette_score]
    purities = [np.mean([x['purity'] for x in clusters])]
    retentions = [round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4)]
    mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
    all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
    clusters_above = [[x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity]]
    n_clusters = [len(clusters)]
    best_silhouette_score = -1
    # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
    best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
        edges_removed), deepcopy(nodes_removed)
    # Make a copy before starting the iteration to re-use the variable
    tree_trimmed = tree_cut.copy()
    # while current_silhouette_score <= score_threshold or retentions[-1] > 0:
    for i in tqdm(range(len(best_tree.edges()) - 1), desc='cuts', leave=True, position=0):
        if i % 2 == 0:
            tree_trimmed, clusters, edges_trimmed, nodes_trimmed = betweenness_cut(tree_trimmed,
                                                                                   cut_threshold=cut_threshold,
                                                                                   cut_method='top', which=which,
                                                                                   distance_weighted=distance_weighted,
                                                                                   verbose=verbose)
        else:
            tree_trimmed, clusters, edges_trimmed, nodes_trimmed = silhouette_rank_cut(tree_trimmed, dist_array, 'edge',
                                                                                       1, 'top', scale_aggregation)

        try:
            current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
            scores.append(current_silhouette_score)
            purities.append(np.mean([x['purity'] for x in clusters]))
            retentions.append(round(sum([x['cluster_size'] for x in clusters]) / dist_array.shape[0], 4))
            mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
            all_cluster_sizes.append([x['cluster_size'] for x in clusters])
            n_clusters.append(len(clusters))
            clusters_above.append(
                [x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])

            edges_removed.extend(edges_trimmed)
            nodes_removed.extend(nodes_trimmed)
            # print(iter, current_silhouette_score, best_silhouette_score)
            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
                    edges_removed), deepcopy(nodes_removed)
        except ValueError:
            # Fake an exit condition when we reach the error where we only have singletons
            current_silhouette_score = score_threshold + 1
            break

        # print(iter, np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score, round(sum([x['cluster_size'] for x in clusters])/len(dist_array),4))
        # iter += 1
    subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
    return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, all_cluster_sizes, clusters_above

#
#
# def iterative_topn_cut_niter_logsize(dist_array, tree, n_iter=None, initial_cut_threshold=1, initial_cut_method='top', cut_threshold=1,
#                                which='edge', distance_weighted=True, verbose=0, score_threshold=1,
#                                min_purity=.8, min_size=6,
#                                silhouette_aggregation='micro'):
#     # Set initial_cut_method to 'top' and initial_cut_threshold to top_n=1 to have fully iterative behaviour
#     # -> If using augmented tree, we need to iteratively pre-cut until we have at least 2 connected components
#     # For now, find some percentile threshold so that this doesn't happen (let's use 0.5)
#     tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
#                                                                        which, distance_weighted, verbose)
#     current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
#     print('Initial mean purity, silhouette score, retention')
#     print(np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score,
#           round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
#     scores = [current_silhouette_score]
#     purities = [np.mean([x['purity'] for x in clusters])]
#     retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
#     mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
#     all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
#     clusters_above = [[x for x in clusters if x['cluster_size']>min_size and x['purity']>min_purity]]
#     n_clusters = [len(clusters)]
#     best_silhouette_score = -1
#     # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
#     best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
#         edges_removed), deepcopy(nodes_removed)
#     # Make a copy before starting the iteration to re-use the variable
#     tree_trimmed = tree_cut.copy()
#     # while current_silhouette_score <= score_threshold or retentions[-1] > 0:
#     for _ in tqdm(range(n_iter), desc='cuts', leave=True, position=0):
#         tree_trimmed, clusters, edges_trimmed, nodes_trimmed = betweenness_cut(tree_trimmed,
#                                                                                cut_threshold=cut_threshold,
#                                                                                cut_method='top', which=which,
#                                                                                distance_weighted=distance_weighted,
#                                                                                verbose=verbose)
#         try:
#             current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
#             scores.append(current_silhouette_score)
#             purities.append(np.mean([x['purity'] for x in clusters]))
#             retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
#             mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
#             all_cluster_sizes.append([x['cluster_size'] for x in clusters])
#             n_clusters.append(len(clusters))
#             clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])
#
#             edges_removed.extend(edges_trimmed)
#             nodes_removed.extend(nodes_trimmed)
#             # print(iter, current_silhouette_score, best_silhouette_score)
#             if current_silhouette_score > best_silhouette_score:
#                 best_silhouette_score = current_silhouette_score
#                 best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
#                     edges_removed), deepcopy(nodes_removed)
#         except ValueError:
#             # Fake an exit condition when we reach the error where we only have singletons
#             current_silhouette_score = score_threshold + 1
#             break
#
#         # print(iter, np.mean([x['purity'] for x in clusters]).round(4), current_silhouette_score, round(sum([x['cluster_size'] for x in clusters])/len(dist_array),4))
#         # iter += 1
#     subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
#     return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, all_cluster_sizes, clusters_above
#
#
# def iterative_SIrank_cut_niter_logsize(dist_array, tree, n_iter=None, initial_cut_threshold=1, initial_cut_method='top',
#                                  which='edge', cut_threshold=1, cut_method='top', scale_aggregation='max',
#                                  min_purity=.8, min_size=6,
#                                  silhouette_aggregation='micro'):
#
#     # Start with a betweenness cut in order to have some initial cluster silhouette score
#     tree_cut, clusters, edges_removed, nodes_removed = betweenness_cut(tree, initial_cut_threshold, initial_cut_method,
#                                                                        which='edge', distance_weighted=True, verbose=0)
#     # TODO At some point if I keep doing these experiments I should probably do both micro and macro avg at once
#     current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
#     if n_iter is None:
#         n_iter=len(tree_cut.edges()) - 1
#     scores = [current_silhouette_score]
#     purities = [np.mean([x['purity'] for x in clusters])]
#     retentions = [round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4)]
#     mean_cluster_sizes = [np.mean([x['cluster_size'] for x in clusters])]
#     all_cluster_sizes = [[x['cluster_size'] for x in clusters]]
#     n_clusters = [len(clusters)]
#     clusters_above = [[x for x in clusters if x['cluster_size']>min_size and x['purity']>min_purity]]
#
#     best_silhouette_score = -1
#     # deepcopy because lists are mutable: otherwise the update in `if best_silhouette` doesn't work as intended
#     best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_cut, clusters, deepcopy(
#         edges_removed), deepcopy(nodes_removed)
#     # Make a copy before starting the iteration to re-use the variable
#     tree_trimmed = tree_cut.copy()
#     # TODO: For now, do range(size(array)) to cut everything down to the last edge
#     for _ in tqdm(range(n_iter), desc='cuts', leave=True, position=0):
#         tree_trimmed, clusters, edges_trimmed, nodes_trimmed = silhouette_rank_cut(tree_trimmed, dist_array, which,
#                                                                                    cut_threshold,
#                                                                                    cut_method,
#                                                                                    scale_aggregation=scale_aggregation)
#
#         try:
#             current_silhouette_score = get_score_at_cut(dist_array, clusters, aggregation=silhouette_aggregation)
#             scores.append(current_silhouette_score)
#             purities.append(np.mean([x['purity'] for x in clusters]))
#             retentions.append(round(sum([x['cluster_size'] for x in clusters]) / len(dist_array), 4))
#             mean_cluster_sizes.append(np.mean([x['cluster_size'] for x in clusters]))
#             all_cluster_sizes.append([x['cluster_size'] for x in clusters])
#             clusters_above.append([x for x in clusters if x['cluster_size'] > min_size and x['purity'] > min_purity])
#
#             n_clusters.append(len(clusters))
#             edges_removed.extend(edges_trimmed)
#             nodes_removed.extend(nodes_trimmed)
#             # print(iter, current_silhouette_score, best_silhouette_score)
#             if current_silhouette_score > best_silhouette_score:
#                 best_silhouette_score = current_silhouette_score
#                 best_tree, best_clusters, best_edges_removed, best_nodes_removed = tree_trimmed, clusters, deepcopy(
#                     edges_removed), deepcopy(nodes_removed)
#         except:
#             break
#     subgraphs = [nx.subgraph(best_tree, c['members']) for c in best_clusters]
#     return best_tree, subgraphs, best_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, all_cluster_sizes, clusters_above
