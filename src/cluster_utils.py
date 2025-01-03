import glob
import os
import re
from copy import deepcopy
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import auc, adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score as ch_score, davies_bouldin_score as db_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm

from src.conv_models import CNNVAE
from src.datasets import FullTCRDataset
from src.metrics import custom_silhouette_score
from src.models import TwoStageVAECLF, FullTCRVAE
from src.networkx_utils import create_mst_from_distance_matrix, iterative_size_cut, iterative_topn_cut, \
    iterative_topn_cut_logsize
from src.sim_utils import make_dist_matrix
from src.torch_utils import load_model_full
from src.train_eval import predict_model
from src.utils import get_palette, mkdirs


##################################
#           PIPELINES            #
##################################
def do_4vae_clustering_pipeline(dm_vae_os_notrp, dm_vae_ts_notrp, dm_vae_os_cstrp, dm_vae_ts_cstrp, dm_tbcr, dm_tcrdist,
                                label_col='peptide', index_col=None, weight_col=None, initial_cut_threshold=1,
                                initial_cut_method='top', silhouette_aggregation='micro', outdir='../output/',
                                filename='output_', title='', n_jobs=8):
    print('Running clustering pipeline for VAE_os_notrp')
    vae_os_notrp_size_results, vae_os_notrp_topn_results, vae_os_notrp_agglo_results = do_three_clustering(
        dm_vae_os_notrp, label_col, index_col, weight_col, dm_name='VAE_OS_NoTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs,
        silhouette_aggregation=silhouette_aggregation)
    print('Running clustering pipeline for VAE_ts_notrp')
    vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results = do_three_clustering(
        dm_vae_ts_notrp, label_col, index_col, weight_col, dm_name='VAE_TS_NoTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs,
        silhouette_aggregation=silhouette_aggregation)
    print('Running clustering pipeline for VAE_os_cstrp')
    vae_os_cstrp_size_results, vae_os_cstrp_topn_results, vae_os_cstrp_agglo_results = do_three_clustering(
        dm_vae_os_cstrp, label_col, index_col, weight_col, dm_name='VAE_OS_CsTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs,
        silhouette_aggregation=silhouette_aggregation)
    print('Running clustering pipeline for VAE_ts_cstrp')
    vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results = do_three_clustering(
        dm_vae_ts_cstrp, label_col, index_col, weight_col, dm_name='VAE_TS_CsTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs,
        silhouette_aggregation=silhouette_aggregation)

    print('Running clustering pipeline for TBCRalign')
    tbcr_size_results, tbcr_topn_results, tbcr_agglo_results = do_three_clustering(dm_tbcr, label_col, index_col,
                                                                                   weight_col, dm_name='TBCRalign',
                                                                                   initial_cut_threshold=initial_cut_threshold,
                                                                                   initial_cut_method=initial_cut_method,
                                                                                   n_jobs=n_jobs,
                                                                                   silhouette_aggregation=silhouette_aggregation)
    print('Running clustering pipeline for tcrdist3')
    tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results = do_three_clustering(dm_tcrdist, label_col,
                                                                                            index_col, weight_col,
                                                                                            dm_name='tcrdist3',
                                                                                            initial_cut_threshold=initial_cut_threshold,
                                                                                            initial_cut_method=initial_cut_method,
                                                                                            n_jobs=n_jobs,
                                                                                            silhouette_aggregation=silhouette_aggregation)

    # TODO: CHANGE VARIABLE NAMES FOR VAE AND INCREASE NUMBER OF PLOTS // PALETTE
    plot_silhouette_scores(vae_os_notrp_size_results['purities'], vae_os_notrp_size_results['retentions'],
                           vae_os_notrp_size_results['scores'],
                           vae_os_notrp_topn_results['purities'], vae_os_notrp_topn_results['retentions'],
                           vae_os_notrp_topn_results['scores'],
                           vae_os_notrp_agglo_results['purities'], vae_os_notrp_agglo_results['retentions'],
                           vae_os_notrp_agglo_results['scores'],
                           dm_name='VAE_OS_NoTRP', outdir=outdir, filename=filename)
    plot_silhouette_scores(vae_ts_notrp_size_results['purities'], vae_ts_notrp_size_results['retentions'],
                           vae_ts_notrp_size_results['scores'],
                           vae_ts_notrp_topn_results['purities'], vae_ts_notrp_topn_results['retentions'],
                           vae_ts_notrp_topn_results['scores'],
                           vae_ts_notrp_agglo_results['purities'], vae_ts_notrp_agglo_results['retentions'],
                           vae_ts_notrp_agglo_results['scores'],
                           dm_name='VAE_TS_NoTRP', outdir=outdir, filename=filename)
    plot_silhouette_scores(vae_os_cstrp_size_results['purities'], vae_os_cstrp_size_results['retentions'],
                           vae_os_cstrp_size_results['scores'],
                           vae_os_cstrp_topn_results['purities'], vae_os_cstrp_topn_results['retentions'],
                           vae_os_cstrp_topn_results['scores'],
                           vae_os_cstrp_agglo_results['purities'], vae_os_cstrp_agglo_results['retentions'],
                           vae_os_cstrp_agglo_results['scores'],
                           dm_name='VAE_OS_CsTRP', outdir=outdir, filename=filename)
    plot_silhouette_scores(vae_ts_cstrp_size_results['purities'], vae_ts_cstrp_size_results['retentions'],
                           vae_ts_cstrp_size_results['scores'],
                           vae_ts_cstrp_topn_results['purities'], vae_ts_cstrp_topn_results['retentions'],
                           vae_ts_cstrp_topn_results['scores'],
                           vae_ts_cstrp_agglo_results['purities'], vae_ts_cstrp_agglo_results['retentions'],
                           vae_ts_cstrp_agglo_results['scores'],
                           dm_name='VAE_TS_CsTRP', outdir=outdir, filename=filename)
    plot_silhouette_scores(tbcr_size_results['purities'], tbcr_size_results['retentions'], tbcr_size_results['scores'],
                           tbcr_topn_results['purities'], tbcr_topn_results['retentions'], tbcr_topn_results['scores'],
                           tbcr_agglo_results['purities'], tbcr_agglo_results['retentions'],
                           tbcr_agglo_results['scores'], dm_name='TBCRalign', outdir=outdir, filename=filename)
    plot_silhouette_scores(tcrdist_size_results['purities'], tcrdist_size_results['retentions'],
                           tcrdist_size_results['scores'], tcrdist_topn_results['purities'],
                           tcrdist_topn_results['retentions'], tcrdist_topn_results['scores'],
                           tcrdist_agglo_results['purities'], tcrdist_agglo_results['retentions'],
                           tcrdist_agglo_results['scores'], dm_name='tcrdist3', outdir=outdir, filename=filename)

    plot_4vae_retpur_curves(vae_os_notrp_topn_results, vae_os_notrp_agglo_results,
                            vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results,
                            vae_os_cstrp_topn_results, vae_os_cstrp_agglo_results,
                            vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results,
                            tbcr_topn_results, tbcr_agglo_results,
                            tcrdist_topn_results, tcrdist_agglo_results,
                            title, outdir, filename)

    # WRITE FUNCTION HERE TO RECOVER "PRED" LABEL BY TOPN/SIZE/ETC CLUSTERS FOR EACH DISTMATRIX
    # AND MERGE WITH INITIAL INPUT_DF AND RE-SAVE AS THE OUTPUT.
    return vae_os_notrp_size_results, vae_os_notrp_topn_results, vae_os_notrp_agglo_results, \
           vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results, \
           vae_os_cstrp_size_results, vae_os_cstrp_topn_results, vae_os_cstrp_agglo_results, \
           vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results, \
           tbcr_size_results, tbcr_topn_results, tbcr_agglo_results, \
           tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results


def do_twostage_2vae_clustering_pipeline(dm_vae_ts_notrp, dm_vae_ts_cstrp,
                                         dm_tbcr, dm_tcrdist, label_col='peptide',
                                         index_col=None,
                                         weight_col=None,
                                         initial_cut_threshold=1, initial_cut_method='top',
                                         outdir='../output/',
                                         filename='output_', title='',
                                         n_jobs=8) -> object:
    print('Running clustering pipeline for VAE_ts_notrp')
    vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results = do_three_clustering(
        dm_vae_ts_notrp, label_col, index_col, weight_col, dm_name='VAE_TS_NoTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs)

    print('Running clustering pipeline for VAE_ts_cstrp')
    vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results = do_three_clustering(
        dm_vae_ts_cstrp, label_col, index_col, weight_col, dm_name='VAE_TS_CsTRP',
        initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method, n_jobs=n_jobs)

    print('Running clustering pipeline for TBCRalign')
    tbcr_size_results, tbcr_topn_results, tbcr_agglo_results = do_three_clustering(dm_tbcr, label_col, index_col,
                                                                                   weight_col, dm_name='TBCRalign',
                                                                                   initial_cut_threshold=initial_cut_threshold,
                                                                                   initial_cut_method=initial_cut_method,
                                                                                   n_jobs=n_jobs)
    print('Running clustering pipeline for tcrdist3')
    tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results = do_three_clustering(dm_tcrdist, label_col,
                                                                                            index_col, weight_col,
                                                                                            dm_name='tcrdist3',
                                                                                            initial_cut_threshold=initial_cut_threshold,
                                                                                            initial_cut_method=initial_cut_method,
                                                                                            n_jobs=n_jobs)

    # TODO: CHANGE VARIABLE NAMES FOR VAE AND INCREASE NUMBER OF PLOTS // PALETTE

    plot_silhouette_scores(vae_ts_notrp_size_results['purities'], vae_ts_notrp_size_results['retentions'],
                           vae_ts_notrp_size_results['scores'],
                           vae_ts_notrp_topn_results['purities'], vae_ts_notrp_topn_results['retentions'],
                           vae_ts_notrp_topn_results['scores'],
                           vae_ts_notrp_agglo_results['purities'], vae_ts_notrp_agglo_results['retentions'],
                           vae_ts_notrp_agglo_results['scores'],
                           dm_name='VAE_TS_NoTRP', outdir=outdir, filename=filename)

    plot_silhouette_scores(vae_ts_cstrp_size_results['purities'], vae_ts_cstrp_size_results['retentions'],
                           vae_ts_cstrp_size_results['scores'],
                           vae_ts_cstrp_topn_results['purities'], vae_ts_cstrp_topn_results['retentions'],
                           vae_ts_cstrp_topn_results['scores'],
                           vae_ts_cstrp_agglo_results['purities'], vae_ts_cstrp_agglo_results['retentions'],
                           vae_ts_cstrp_agglo_results['scores'],
                           dm_name='VAE_TS_CsTRP', outdir=outdir, filename=filename)
    plot_silhouette_scores(tbcr_size_results['purities'], tbcr_size_results['retentions'], tbcr_size_results['scores'],
                           tbcr_topn_results['purities'], tbcr_topn_results['retentions'], tbcr_topn_results['scores'],
                           tbcr_agglo_results['purities'], tbcr_agglo_results['retentions'],
                           tbcr_agglo_results['scores'], dm_name='TBCRalign', outdir=outdir, filename=filename)
    plot_silhouette_scores(tcrdist_size_results['purities'], tcrdist_size_results['retentions'],
                           tcrdist_size_results['scores'], tcrdist_topn_results['purities'],
                           tcrdist_topn_results['retentions'], tcrdist_topn_results['scores'],
                           tcrdist_agglo_results['purities'], tcrdist_agglo_results['retentions'],
                           tcrdist_agglo_results['scores'], dm_name='tcrdist3', outdir=outdir, filename=filename)

    # SUPER BAD ON THE GO REUSING LINES
    plot_4vae_retpur_curves(vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results,
                            vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results,
                            vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results,
                            vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results,
                            tbcr_topn_results, tbcr_agglo_results,
                            tcrdist_topn_results, tcrdist_agglo_results,
                            title, outdir, filename)

    # WRITE FUNCTION HERE TO RECOVER "PRED" LABEL BY TOPN/SIZE/ETC CLUSTERS FOR EACH DISTMATRIX
    # AND MERGE WITH INITIAL INPUT_DF AND RE-SAVE AS THE OUTPUT.
    return vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results, \
           vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results, \
           tbcr_size_results, tbcr_topn_results, tbcr_agglo_results, \
           tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results


def do_full_clustering_pipeline(dm_vae, dm_tbcr, dm_tcrdist, label_col='peptide', index_col=None, weight_col=None,
                                initial_cut_threshold=1, initial_cut_method='top', outdir='../output/',
                                filename='output_', title='', n_jobs=8):
    print('Running clustering pipeline for VAE')
    vae_size_results, vae_topn_results, vae_agglo_results = do_three_clustering(dm_vae, label_col, index_col,
                                                                                weight_col, dm_name='VAE',
                                                                                initial_cut_threshold=initial_cut_threshold,
                                                                                initial_cut_method=initial_cut_method,
                                                                                n_jobs=n_jobs)
    print('Running clustering pipeline for TBCRalign')
    tbcr_size_results, tbcr_topn_results, tbcr_agglo_results = do_three_clustering(dm_tbcr, label_col, index_col,
                                                                                   weight_col, dm_name='TBCRalign',
                                                                                   initial_cut_threshold=initial_cut_threshold,
                                                                                   initial_cut_method=initial_cut_method,
                                                                                   n_jobs=n_jobs)
    print('Running clustering pipeline for tcrdist3')
    tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results = do_three_clustering(dm_tcrdist, label_col,
                                                                                            index_col, weight_col,
                                                                                            dm_name='tcrdist3',
                                                                                            initial_cut_threshold=initial_cut_threshold,
                                                                                            initial_cut_method=initial_cut_method,
                                                                                            n_jobs=n_jobs)
    plot_silhouette_scores(vae_size_results['purities'], vae_size_results['retentions'], vae_size_results['scores'],
                           vae_topn_results['purities'], vae_topn_results['retentions'], vae_topn_results['scores'],
                           vae_agglo_results['purities'], vae_agglo_results['retentions'], vae_agglo_results['scores'],
                           dm_name='VAE', outdir=outdir, filename=filename)
    plot_silhouette_scores(tbcr_size_results['purities'], tbcr_size_results['retentions'], tbcr_size_results['scores'],
                           tbcr_topn_results['purities'], tbcr_topn_results['retentions'], tbcr_topn_results['scores'],
                           tbcr_agglo_results['purities'], tbcr_agglo_results['retentions'],
                           tbcr_agglo_results['scores'], dm_name='TBCRalign', outdir=outdir, filename=filename)
    plot_silhouette_scores(tcrdist_size_results['purities'], tcrdist_size_results['retentions'],
                           tcrdist_size_results['scores'], tcrdist_topn_results['purities'],
                           tcrdist_topn_results['retentions'], tcrdist_topn_results['scores'],
                           tcrdist_agglo_results['purities'], tcrdist_agglo_results['retentions'],
                           tcrdist_agglo_results['scores'], dm_name='tcrdist3', outdir=outdir, filename=filename)

    plot_retpur_curves(vae_topn_results, vae_agglo_results,
                       tbcr_topn_results, tbcr_agglo_results,
                       tcrdist_topn_results, tcrdist_agglo_results,
                       title, outdir, filename)

    # WRITE FUNCTION HERE TO RECOVER "PRED" LABEL BY TOPN/SIZE/ETC CLUSTERS FOR EACH DISTMATRIX
    # AND MERGE WITH INITIAL INPUT_DF AND RE-SAVE AS THE OUTPUT.
    return vae_size_results, vae_topn_results, vae_agglo_results, \
           tbcr_size_results, tbcr_topn_results, tbcr_agglo_results, \
           tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results


def get_optimal_point(results):
    best_idx = np.argmax(np.where(np.isnan(results['scores']), -999, results['scores']))
    best_silhouette = results['scores'][best_idx]
    best_purity = results['purities'][best_idx]
    best_retention = results['retentions'][best_idx]
    return {'idx': best_idx, 'purity': best_purity, 'retention': best_retention, 'silhouette': best_silhouette}


def do_three_clustering(dist_matrix, label_col, index_col=None, weight_col=None, dm_name='', initial_cut_threshold=1,
                        initial_cut_method='top', n_jobs=8, silhouette_aggregation='micro'):
    if index_col not in dist_matrix.columns:
        if index_col is None:
            index_col = 'index_col'
        dist_matrix[index_col] = [f'seq_{i:05}' for i in range(len(dist_matrix))]
    # Get MST
    G, original_tree, dist_matrix, values_array, labels, encoded_labels, label_encoder, raw_indices = create_mst_from_distance_matrix(
        dist_matrix, label_col=label_col, index_col=index_col, weight_col=weight_col, algorithm='kruskal')
    # Size-cut
    size_tree, size_subgraphs, size_clusters, edges_cut, nodes_cut, size_scores, size_purities, size_retentions, size_mean_cluster_sizes, size_n_clusters = iterative_size_cut(
        values_array, original_tree, initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method,
        top_n=1, which='edge', distance_weighted=True, verbose=0, max_size=4,
        silhouette_aggregation=silhouette_aggregation)
    # TopN-cut
    topn_tree, topn_subgraphs, topn_clusters, edges_removed, nodes_removed, topn_scores, topn_purities, topn_retentions, topn_mean_cluster_sizes, topn_n_clusters = iterative_topn_cut(
        values_array, original_tree, initial_cut_threshold=initial_cut_threshold, initial_cut_method=initial_cut_method,
        top_n=1, which='edge', distance_weighted=True, verbose=0, score_threshold=.75,
        silhouette_aggregation=silhouette_aggregation)
    # Agglo and get the best threshold and redo the clustering to get the best results
    agglo_output = cluster_all_thresholds(values_array, values_array, labels, encoded_labels, label_encoder,
                                          n_points=300, silhouette_aggregation=silhouette_aggregation, n_jobs=n_jobs, )
    best_agglo = agglo_output.loc[agglo_output['silhouette'].idxmax()]
    agglo_single_summary, agglo_single_df, agglo_single_c = cluster_single_threshold(values_array, values_array, labels,
                                                                                     encoded_labels, label_encoder,
                                                                                     threshold=best_agglo['threshold'],
                                                                                     silhouette_aggregation=silhouette_aggregation,
                                                                                     return_df_and_c=True)

    size_results = {'tree': size_tree, 'clusters': size_clusters,
                    'n_clusters': size_n_clusters,
                    'mean_cluster_size': size_mean_cluster_sizes,
                    'scores': size_scores, 'purities': size_purities, 'retentions': size_retentions,
                    'df': get_cut_cluster_df(original_tree, size_clusters).assign(dm_name=dm_name,
                                                                                  method='size_cut').rename(
                        columns={'index': index_col,
                                 'label': label_col})}
    topn_results = {'tree': topn_tree, 'clusters': topn_clusters,
                    'n_clusters': topn_n_clusters,
                    'mean_cluster_size': topn_mean_cluster_sizes,
                    'scores': topn_scores, 'purities': topn_purities, 'retentions': topn_retentions,
                    'df': get_cut_cluster_df(original_tree, topn_clusters).assign(dm_name=dm_name,
                                                                                  method='topn_cut').rename(
                        columns={'index': index_col,
                                 'label': label_col})}
    agglo_results = {'scores': agglo_output['silhouette'].values,
                     'purities': agglo_output['mean_purity'].values,
                     'retentions': agglo_output['retention'].values,
                     'mean_cluster_size': agglo_output['mean_cluster_size'].values,
                     'n_clusters': agglo_output['n_cluster'].values,
                     'df': get_agglo_cluster_df(agglo_single_c, dist_matrix, agglo_single_df, index_col, label_col,
                                                rest_cols=[index_col, label_col]).assign(dm_name=dm_name,
                                                                                         method='agglo'),
                     'best': best_agglo}

    return size_results, topn_results, agglo_results


def get_cut_cluster_df(original_tree: nx.Graph, clusters):
    """

    Args:
        original_tree: UNCUT, UNPRUNED original tree as created from `create_mst_from_dist_matrix`
        clusters: the list of clusters returned by the cut algos

    Returns:

    """

    series = []
    original_tree = deepcopy(original_tree)
    nodes = dict(original_tree.nodes(data=True))
    for i, c in enumerate(clusters):
        common_nodes = {k: v for k, v in nodes.items() if k in c['members']}
        zz = [{'node': k, 'label': v['label'], 'index': v['index'], 'pred_label': i, 'cluster_size': c['cluster_size'],
               'majority_label': c['majority_label'], 'purity': c['purity']} for k, v in common_nodes.items()]
        series.append(pd.DataFrame(zz))
    results = pd.concat(series)
    non_singletons = results.dropna().node.unique()
    singletons = pd.DataFrame([{'node': n, **v} for n, v in nodes.items() if n not in non_singletons])
    if len(singletons) > 0:
        singletons['pred_label'] = range(int(results.pred_label.max()) + 1,
                                         int(results.pred_label.max()) + len(singletons) + 1)
        singletons['cluster_size'] = 1
        singletons['majority_label'] = singletons['label']
        singletons['purity'] = np.nan
    return pd.concat([results, singletons]).reset_index(drop=True)


def plot_silhouette_scores(size_purities, size_retentions, size_scores,
                           topn_purities, topn_retentions, topn_scores,
                           agglo_purities, agglo_retentions, agglo_scores, dm_name='',
                           outdir='../output/', filename='silhouette'
                           ):
    f, a = plt.subplots(1, 1, figsize=(7, 4))
    a2 = a.twinx()
    # "title" here should be the name of the distmatrix ; dm_name should be VAE (or TS128/OS128, TBCR, tcrdist)
    a.set_title(
        f'{dm_name}; No initial pruning ; Using Top-5 as initial cut ; distance-weighted betweenness\nTBCR comparison of size-cut and top-1 (silhouette) cut')

    a.plot(range(len(size_purities)), size_purities, lw=.75, ls=':', c='r',
           label='size(4)-cut avg purity')
    a.plot(range(len(size_retentions)), size_retentions, lw=.75, ls='-', c='r',
           label='size(4)-cut avg retention')
    a2.plot(range(len(size_scores)), size_scores, lw=.75, ls='--', c='r',
            label='size(4)-cut silhouette score')

    a.plot(range(len(topn_purities)), topn_purities, lw=1, ls=':', c='g',
           label='Top1-cut avg purity')
    a.plot(range(len(topn_retentions)), topn_retentions, lw=.75, ls='-', c='g',
           label='Top1-cut avg retention')
    a2.plot(range(len(topn_scores)), topn_scores, lw=.8, ls='--', c='g',
            label='Top1-cut silhouette score')

    a.plot(range(len(agglo_purities)), agglo_purities, lw=1, ls=':', c='b',
           label='Agglomerative avg purity')
    a.plot(range(len(agglo_retentions)), agglo_retentions, lw=.75, ls='-', c='b',
           label='Agglomerative avg retention')
    a2.plot(range(len(agglo_scores)), agglo_scores, lw=.8, ls='--', c='b',
            label='Agglomerative silhouette score')

    # Align the tick marks
    a.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks from primary y-axis
    a2.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks from secondary y-axis
    # Align the gridlines
    a.grid(True)
    a2.grid(False)  # Disable secondary y-axis gridlines
    a.set_ylabel('Retention // Purity (%)')
    a2.set_ylabel('Silhouette score')
    ymin = min([np.nanmin(size_scores), np.nanmin(topn_scores), np.nanmin(agglo_scores)])
    ymin = ymin - 0.1 * ymin
    ymax = max([np.nanmax(size_scores), np.nanmax(topn_scores), np.nanmax(agglo_scores)])
    ymax = ymax + 0.1 * ymax
    a2.set_ylim([ymin, ymax])
    a2.legend(bbox_to_anchor=(1.62, .25))
    a.set_xlabel('iteration')
    a.legend(bbox_to_anchor=(1.62, .88))
    f.savefig(f'{outdir}{dm_name}{filename}_silhouette.png', dpi=150, bbox_inches='tight')


def plot_retpur_curves(  # vae_size_results,
        vae_topn_results, vae_agglo_results,
        # tbcr_size_results,
        tbcr_topn_results, tbcr_agglo_results,
        # tcrdist_size_results,
        tcrdist_topn_results, tcrdist_agglo_results, title, outdir, filename):
    palette = get_palette('cool', 3)
    # So apparently I don't use the size
    # Get marker positions
    # best_vae_size = get_optimal_point(vae_size_results)
    best_vae_topn = get_optimal_point(vae_topn_results)
    best_vae_agglo = get_optimal_point(vae_agglo_results)
    print('best_vae_topn', best_vae_topn)
    print('best_vae_agglo', best_vae_agglo)
    # best_tbcr_size = get_optimal_point(tbcr_size_results)
    best_tbcr_topn = get_optimal_point(tbcr_topn_results)
    best_tbcr_agglo = get_optimal_point(tbcr_agglo_results)
    print('best_tbcr_topn', best_tbcr_topn)
    print('best_tbcr_agglo', best_tbcr_agglo)
    # best_tcrdist_size = get_optimal_point(tcrdist_size_results)
    best_tcrdist_topn = get_optimal_point(tcrdist_topn_results)
    best_tcrdist_agglo = get_optimal_point(tcrdist_agglo_results)
    print('best_tcrdist_topn', best_tcrdist_topn)
    print('best_tcrdist_agglo', best_tcrdist_agglo)
    # Plotting options
    f, ax = plt.subplots(1, 1, figsize=(16, 16))
    marker_size = 22
    agglo_lw = 0.8
    agglo_marker = '*'
    agglo_ls = '-'
    topn_lw = 1.1
    topn_ls = '--'
    topn_marker = 'x'
    c_vae = palette[1]
    c_tbcr = 'g'
    c_tcrdist = 'y'

    # VAE agglo
    ax.plot(vae_agglo_results['retentions'][1:-1], vae_agglo_results['purities'][1:-1],
            label='VAE + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_vae)
    ax.plot(vae_topn_results['retentions'][1:-1], vae_topn_results['purities'][1:-1],
            label='VAE + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_vae)
    ax.scatter(best_vae_agglo['retention'], best_vae_agglo['purity'], c=c_vae, marker=agglo_marker,
               label='Best VAE+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_vae_topn['retention'], best_vae_topn['purity'], c=c_vae, marker=topn_marker,
               label='Best VAE+Top-1 Cut', lw=1.2, s=marker_size)
    # TBCR
    ax.plot(tbcr_agglo_results['retentions'][1:-1], tbcr_agglo_results['purities'][1:-1],
            label='TBCR + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_tbcr)
    ax.plot(tbcr_topn_results['retentions'][1:-1], tbcr_topn_results['purities'][1:-1],
            label='TBCR + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_tbcr)
    ax.scatter(best_tbcr_agglo['retention'], best_tbcr_agglo['purity'], c=c_tbcr, marker=agglo_marker,
               label='Best TBCR+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_tbcr_topn['retention'], best_tbcr_topn['purity'], c=c_tbcr, marker=topn_marker,
               label='Best TBCR+Top-1 Cut', lw=1.2, s=marker_size)
    # TCRDIST
    ax.plot(tcrdist_agglo_results['retentions'][1:-1], tcrdist_agglo_results['purities'][1:-1],
            label='tcrdist3 + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_tcrdist)
    ax.plot(tcrdist_topn_results['retentions'][1:-1], tcrdist_topn_results['purities'][1:-1],
            label='tcrdist3 + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_tcrdist)
    ax.scatter(best_tcrdist_agglo['retention'], best_tcrdist_agglo['purity'], c=c_tcrdist, marker=agglo_marker,
               label='Best tcrdist3+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_tcrdist_topn['retention'], best_tcrdist_topn['purity'], c=c_tcrdist, marker=topn_marker,
               label='Best tcrdist3+Top-1 Cut', lw=1.2, s=marker_size)

    ax.set_ylim([-0.015, 1.015])
    ax.set_xlim([-0.015, 1.015])
    ax.set_xlabel('Retention', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Mean purity', fontsize=12, fontweight='semibold')
    # Enable grids
    ax.grid(True, which='major', linestyle='-')
    ax.minorticks_on()  # This enables the minor ticks
    ax.grid(True, which='minor', linestyle='--')

    # Customizing the legend
    ax.legend(title='Method', prop={'weight': 'semibold', 'size': 13},
              title_fontproperties={'weight': 'semibold', 'size': 15})
    ax.set_title(
        f'Purity Retention curves for {title}\n Agglomerative vs MST cutting ; Retention/Purity range : (0.5-1.0)',
        fontweight='semibold', fontsize=14)
    f.savefig(f'{outdir}{filename}_retpur_curves.png', dpi=150,
              bbox_inches='tight')


def plot_4vae_retpur_curves(  # vae_size_results,
        vae_os_notrp_topn_results, vae_os_notrp_agglo_results,
        vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results,
        vae_os_cstrp_topn_results, vae_os_cstrp_agglo_results,
        vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results,
        # tbcr_size_results,
        tbcr_topn_results, tbcr_agglo_results,
        # tcrdist_size_results,
        tcrdist_topn_results, tcrdist_agglo_results, title, outdir, filename):
    # So apparently I don't use the size
    # Get marker positions
    # best_vae_size = get_optimal_point(vae_size_results)
    best_vae_os_notrp_topn = get_optimal_point(vae_os_notrp_topn_results)
    best_vae_os_notrp_agglo = get_optimal_point(vae_os_notrp_agglo_results)
    best_vae_ts_notrp_topn = get_optimal_point(vae_ts_notrp_topn_results)
    best_vae_ts_notrp_agglo = get_optimal_point(vae_ts_notrp_agglo_results)
    best_vae_os_cstrp_topn = get_optimal_point(vae_os_cstrp_topn_results)
    best_vae_os_cstrp_agglo = get_optimal_point(vae_os_cstrp_agglo_results)
    best_vae_ts_cstrp_topn = get_optimal_point(vae_ts_cstrp_topn_results)
    best_vae_ts_cstrp_agglo = get_optimal_point(vae_ts_cstrp_agglo_results)

    # best_tbcr_size = get_optimal_point(tbcr_size_results)
    best_tbcr_topn = get_optimal_point(tbcr_topn_results)
    best_tbcr_agglo = get_optimal_point(tbcr_agglo_results)
    print('best_tbcr_topn', best_tbcr_topn)
    print('best_tbcr_agglo', best_tbcr_agglo)
    # best_tcrdist_size = get_optimal_point(tcrdist_size_results)
    best_tcrdist_topn = get_optimal_point(tcrdist_topn_results)
    best_tcrdist_agglo = get_optimal_point(tcrdist_agglo_results)
    print('best_tcrdist_topn', best_tcrdist_topn)
    print('best_tcrdist_agglo', best_tcrdist_agglo)
    # Plotting options
    palette = get_palette('gnuplot2', 4)[:-1]
    palette.extend(['r', 'g', 'y'])
    marker_size = 22
    agglo_lw = 0.8
    agglo_marker = '*'
    agglo_ls = '-'
    topn_lw = 1.1
    topn_ls = '--'
    topn_marker = 'x'
    c_vae_os_notrp = palette[0]
    c_vae_ts_notrp = palette[1]
    c_vae_os_cstrp = palette[2]
    c_vae_ts_cstrp = palette[3]
    c_tbcr = palette[4]
    c_tcrdist = palette[5]
    f, ax = plt.subplots(1, 1, figsize=(16, 16))

    # OS NO TRP
    ax.plot(vae_os_notrp_agglo_results['retentions'][1:-1], vae_os_notrp_agglo_results['purities'][1:-1],
            label='VAE_os_notrp + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_vae_os_notrp)
    ax.plot(vae_os_notrp_topn_results['retentions'][1:-1], vae_os_notrp_topn_results['purities'][1:-1],
            label='VAE_os_notrp + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_vae_os_notrp)
    ax.scatter(best_vae_os_notrp_agglo['retention'], best_vae_os_notrp_agglo['purity'], c=c_vae_os_notrp,
               marker=agglo_marker,
               label='Best VAE_os_notrp+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_vae_os_notrp_topn['retention'], best_vae_os_notrp_topn['purity'], c=c_vae_os_notrp,
               marker=topn_marker,
               label='Best VAE_os_notrp+Top-1 Cut', lw=1.2, s=marker_size)
    # TS NO TRP
    ax.plot(vae_ts_notrp_agglo_results['retentions'][1:-1], vae_ts_notrp_agglo_results['purities'][1:-1],
            label='VAE_ts_notrp + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_vae_ts_notrp)
    ax.plot(vae_ts_notrp_topn_results['retentions'][1:-1], vae_ts_notrp_topn_results['purities'][1:-1],
            label='VAE_ts_notrp + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_vae_ts_notrp)
    ax.scatter(best_vae_ts_notrp_agglo['retention'], best_vae_ts_notrp_agglo['purity'], c=c_vae_ts_notrp,
               marker=agglo_marker,
               label='Best VAE_ts_notrp+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_vae_ts_notrp_topn['retention'], best_vae_ts_notrp_topn['purity'], c=c_vae_ts_notrp,
               marker=topn_marker,
               label='Best VAE_ts_notrp+Top-1 Cut', lw=1.2, s=marker_size)
    # OS CS TRP
    ax.plot(vae_os_cstrp_agglo_results['retentions'][1:-1], vae_os_cstrp_agglo_results['purities'][1:-1],
            label='VAE_os_cstrp + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_vae_os_cstrp)
    ax.plot(vae_os_cstrp_topn_results['retentions'][1:-1], vae_os_cstrp_topn_results['purities'][1:-1],
            label='VAE_os_cstrp + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_vae_os_cstrp)
    ax.scatter(best_vae_os_cstrp_agglo['retention'], best_vae_os_cstrp_agglo['purity'], c=c_vae_os_cstrp,
               marker=agglo_marker,
               label='Best VAE_os_cstrp+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_vae_os_cstrp_topn['retention'], best_vae_os_cstrp_topn['purity'], c=c_vae_os_cstrp,
               marker=topn_marker,
               label='Best VAE_os_cstrp+Top-1 Cut', lw=1.2, s=marker_size)
    # TS CS TRP
    ax.plot(vae_ts_cstrp_agglo_results['retentions'][1:-1], vae_ts_cstrp_agglo_results['purities'][1:-1],
            label='VAE_ts_cstrp + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_vae_ts_cstrp)
    ax.plot(vae_ts_cstrp_topn_results['retentions'][1:-1], vae_ts_cstrp_topn_results['purities'][1:-1],
            label='VAE_ts_cstrp + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_vae_ts_cstrp)
    ax.scatter(best_vae_ts_cstrp_agglo['retention'], best_vae_ts_cstrp_agglo['purity'], c=c_vae_ts_cstrp,
               marker=agglo_marker,
               label='Best VAE_ts_cstrp+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_vae_ts_cstrp_topn['retention'], best_vae_ts_cstrp_topn['purity'], c=c_vae_ts_cstrp,
               marker=topn_marker,
               label='Best VAE_ts_cstrp+Top-1 Cut', lw=1.2, s=marker_size)
    # TBCR
    ax.plot(tbcr_agglo_results['retentions'][1:-1], tbcr_agglo_results['purities'][1:-1],
            label='TBCR + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_tbcr)
    ax.plot(tbcr_topn_results['retentions'][1:-1], tbcr_topn_results['purities'][1:-1],
            label='TBCR + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_tbcr)
    ax.scatter(best_tbcr_agglo['retention'], best_tbcr_agglo['purity'], c=c_tbcr, marker=agglo_marker,
               label='Best TBCR+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_tbcr_topn['retention'], best_tbcr_topn['purity'], c=c_tbcr, marker=topn_marker,
               label='Best TBCR+Top-1 Cut', lw=1.2, s=marker_size)
    # TCRDIST
    ax.plot(tcrdist_agglo_results['retentions'][1:-1], tcrdist_agglo_results['purities'][1:-1],
            label='tcrdist3 + AggClustering', lw=agglo_lw, ls=agglo_ls, c=c_tcrdist)
    ax.plot(tcrdist_topn_results['retentions'][1:-1], tcrdist_topn_results['purities'][1:-1],
            label='tcrdist3 + Top-1 Cut', lw=topn_lw, ls=topn_ls, c=c_tcrdist)
    ax.scatter(best_tcrdist_agglo['retention'], best_tcrdist_agglo['purity'], c=c_tcrdist, marker=agglo_marker,
               label='Best tcrdist3+AggClustering', lw=1.2, s=marker_size)
    ax.scatter(best_tcrdist_topn['retention'], best_tcrdist_topn['purity'], c=c_tcrdist, marker=topn_marker,
               label='Best tcrdist3+Top-1 Cut', lw=1.2, s=marker_size)

    ax.set_ylim([-0.015, 1.015])
    ax.set_xlim([-0.015, 1.015])
    ax.set_xlabel('Retention', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Mean purity', fontsize=12, fontweight='semibold')
    # Enable grids
    ax.grid(True, which='major', linestyle='-')
    ax.minorticks_on()  # This enables the minor ticks
    ax.grid(True, which='minor', linestyle='--')

    # Customizing the legend
    ax.legend(title='Method', prop={'weight': 'semibold', 'size': 13},
              title_fontproperties={'weight': 'semibold', 'size': 15})
    ax.set_title(
        f'Purity Retention curves for {title}\n Agglomerative vs MST cutting ; Retention/Purity range : (0.5-1.0)',
        fontweight='semibold', fontsize=14)
    print(f'HERE FUCK OFF {filename}')
    f.savefig(f'{outdir}{filename}_retpur_curves.png', dpi=150,
              bbox_inches='tight')


def do_baseline(baseline_distmatrix, identifier='baseline', label_col='peptide', n_points=500):
    cols = [str(x) for x in baseline_distmatrix.index.to_list()] if type(
        baseline_distmatrix.columns[0] == str) else baseline_distmatrix.index.to_list()
    dist_array = baseline_distmatrix[cols].values
    encoder = LabelEncoder()
    labels = baseline_distmatrix[label_col].values
    encoded_labels = encoder.fit_transform(labels)
    results = cluster_all_thresholds(dist_array, torch.rand([len(dist_array), 1]), labels, encoded_labels, encoder,
                                     n_points=n_points)
    results['input_type'] = identifier
    return results


def sort_key(item):
    if item == 'last_epoch':
        return float('inf'), 1  # last_epoch will be the second last
    elif item == 'checkpoint_best':
        return float('inf'), 2  # checkpoint_best will be last
    else:
        # Extract the number from strings like 'epoch_6000' and convert to integer
        return int(re.search(r'\d+', item).group()), 0


def run_interval_clustering(model_folder, input_df, index_col, identifier='VAEmodel', n_points=1500, n_jobs=1):
    # Assuming we are using VAE based models
    if not model_folder.endswith("/"):
        model_folder = model_folder + '/'
    files = glob.glob(f'{model_folder}*.pt')
    js = glob.glob(f'{model_folder}*JSON*.json')[0]
    # Get the epochs and sort them
    epochs = ['_'.join(os.path.basename(x.replace(model_folder, '')).split('_')[:2]) for x in files]
    sorted_epochs = sorted(epochs, key=sort_key)
    # Get the mapping to load the files in the correct order
    map_epochs_files = {k: v for k, v in zip(epochs, files)}

    # Run the analysis for each of these VAE models
    cat_results = []
    best_dm = None
    best_auc = -1
    for name in sorted_epochs:
        if 'last_epoch' in name:
            continue
        checkpoint = map_epochs_files[name]
        model, model_json = load_model_full(checkpoint, js, map_location='cpu', return_json=True)
        if hasattr(model, 'vae'):
            model = model.vae
        latent_df = get_latent_df(model, input_df)
        dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
            latent_df, index_col=index_col)

        results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points,
                                         n_jobs=n_jobs)
        results['input_type'] = f'{identifier}_{name}'
        retentions = results['retention'].values[1:-1]
        purities = results['mean_purity'].values[1:-1]
        # Small fix to set a max threshold for retention at 0.98 to avoid cases where 
        # we have 100% retention with one large garbage cluster very unpure and 2-3 small clusters 100% purity
        # that would give an overall mean_purity of 70% or somethings
        p70_r35_auc = get_retpur_auc(retentions, purities, min_retention=0.35, min_purity=0.70, max_retention=0.975)

        # Saving just the best in case nothing beats the best AUC and no best_dm has been found
        if 'checkpoint_best' in checkpoint and best_dm is None and 'interval' not in checkpoint:
            best_dm = dist_matrix
            best_auc = p70_r35_auc
            best_name = 'checkpoint_best'
        # Then do the actual auc check
        if p70_r35_auc > best_auc:
            best_dm = dist_matrix
            best_auc = p70_r35_auc
            best_name = name
        cat_results.append(results)
    cat_results = pd.concat(cat_results)
    return cat_results, best_dm, best_name


def plot_retention_purities(runs, title=None, fn=None, palette='tab10', add_clustersize=False,
                            add_best_silhouette=False, hue='input_type', figsize=(9, 9), legend=True, outdir=None):
    # plotting options
    sns.set_palette(palette, n_colors=len(runs[hue].unique()))  # - 2)
    f, a = plt.subplots(1, 1, figsize=figsize)
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Mean Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    if add_clustersize:
        ax2 = a.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_yscale('log', base=2)
        ax2.set_ylabel('Mean Cluster Size (Log2)', fontweight='semibold', fontsize=14)
    for input_type in runs[hue].unique():
        query = runs.query(f'{hue}==@input_type').reset_index()
        retentions = query['retention'][1:-1].values
        purities = query['mean_purity'][1:-1].values
        if add_clustersize:
            cluster_sizes = query['mean_cluster_size'].values[1:-1]
        marker = '*' if 'agglo' in input_type.lower() else 'x' if 'cut' in input_type.lower() else 'o'
        ls = ':' if 'agglo' in input_type.lower() else '-' if 'cut' in input_type.lower() else '-.'
        lw = .7 if 'cut' in input_type.lower() else 1
        if "tbcralign" in input_type.lower():
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls=ls, c='m', lw=lw)
            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='x', lw=0.25, s=6, c='m')
            if add_best_silhouette:
                best_silhouette = query.loc[find_true_maximum(query['silhouette'].values, patience=50)[1]]
            a.scatter(x=best_silhouette['retention'], y=best_silhouette['mean_purity'], marker=marker, s=11, lw=1.2,
                      c='m',
                      label=f"Best SI: {input_type.lstrip('_')}")

        elif "tcrdist3" in input_type.lower():
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls=ls, c='y', lw=lw)
            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='.', lw=0.25, s=6, c='y')
            if add_best_silhouette:
                best_silhouette = query.loc[find_true_maximum(query['silhouette'].values, patience=50)[1]]
                a.scatter(x=best_silhouette['retention'], y=best_silhouette['mean_purity'], marker=marker, s=11, lw=1.2,
                          c='y',
                          label=f"Best SI: {input_type.lstrip('_')}")

        else:
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls=ls, lw=lw)
            if add_best_silhouette:
                best_silhouette = query.loc[find_true_maximum(query['silhouette'].values, patience=50)[1]]
                a.scatter(x=best_silhouette['retention'], y=best_silhouette['mean_purity'], marker=marker, s=11, lw=1.2,
                          label=f"Best SI: {input_type.lstrip('_')}")

            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='*', lw=0.25, s=6)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')
    if legend:
        a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
                 prop={'weight': 'semibold', 'size': 12}, loc='lower left')
    f.suptitle(f'{title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fn is not None:
        if outdir is not None:
            mkdirs(outdir)
            fn = f'{outdir}{fn}'
        f.savefig(f'{fn}.png', dpi=200)
    return f, a


def run_interval_plot_pipeline(model_folder, input_df, index_col, label_col, tbcr_dm, identifier='', n_points=250,
                               baselines=None, plot_title='None', fig_fn=None, n_jobs=1):
    try:
        interval_runs, best_dm, best_name = run_interval_clustering(model_folder, input_df, index_col, identifier,
                                                                    n_points, n_jobs=n_jobs)
    except:
        print('\n\n\n\n', model_folder, identifier, '\n\n\n\n')
        raise ValueError(model_folder, identifier)
    # Only do TBCRalign and tcrdist3 because we know we are better than KernelSim by a good margin
    tbcr_dm, tbcr_da = resort_baseline(tbcr_dm, best_dm, index_col)
    agg_results = do_agg_clustering_best(best_dm, tbcr_dm, index_col, label_col, n_points)
    interval_runs = pd.concat([baselines.query('input_type=="TBCRalign"'),
                               baselines.query('input_type == "tcrdist3"'),
                               interval_runs,
                               agg_results.assign(input_type=f'agg_{best_name}')])
    interval_runs['input_type'] = interval_runs['input_type'].apply(
        lambda x: x.replace(identifier, '').lstrip('_').rstrip('_'))
    # plotting options
    sns.set_palette('gnuplot2', n_colors=len(interval_runs.input_type.unique()) - 2)
    f, a = plt.subplots(1, 1, figsize=(15, 15))
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Avg Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    for input_type in interval_runs.input_type.unique():
        query = interval_runs.query('input_type==@input_type')
        retentions = query['retention'][1:-1].values
        purities = query['mean_purity'][1:-1].values
        if input_type == "TBCRalign":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='g', lw=1.)
        elif input_type == "tcrdist3":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='y', lw=1.)
        else:
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='--', lw=1.)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')

    a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
             prop={'weight': 'semibold', 'size': 12})
    f.suptitle(f'{plot_title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fig_fn is not None:
        f.savefig(f'{fig_fn}.png', dpi=200)
    return interval_runs


def do_agg_clustering_best(input_dm, other_dm_labelled, index_col, label_col='peptide', n_points=500):
    # provide either a model path or an actual loaded model
    baseline_dm, baseline_da = resort_baseline(other_dm_labelled, input_dm, index_col)
    input_da = input_dm.iloc[:len(baseline_da), :len(baseline_da)].values
    labels = input_dm[label_col].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit(labels)
    agg_array = 1 - np.multiply(1 - input_da, 1 - baseline_da)
    results = cluster_all_thresholds(agg_array, torch.rand([len(agg_array), 1]), labels, encoded_labels, label_encoder,
                                     n_points=n_points)
    return results


def do_agg_clustering(model, input_df, other_dm_labelled, n_points=500):
    # provide either a model path or an actual loaded model
    if type(model) == str:
        model = get_model(model)
    # Assuming both input_df and other_dm_labelled have the raw_index...
    latent_df = get_latent_df(model, input_df)
    # Sort it accordingly
    latent_df = latent_df.set_index('raw_index').loc[other_dm_labelled['raw_index'].values].reset_index()
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    other_array = other_dm_labelled.iloc[:other_dm_labelled, :other_dm_labelled].values
    agg_array = 1 - np.multiply((1 - dist_array, 1 - other_array))
    results = cluster_all_thresholds(agg_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    return results


def pipeline_best_model_clustering(model_folder, input_df, name, n_points=500):
    # Assuming we are using VAE based models
    model = get_model(model_folder,
                      map_location='cpu')
    latent_df = get_latent_df(model, input_df)
    print(len(latent_df))
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    results['input_type'] = name
    return results


def cluster_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, threshold,
                             silhouette_aggregation='micro', return_df_and_c=False):
    try:
        c = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=threshold, linkage='complete')
    except:
        c = AgglomerativeClustering(n_clusters=None, affinity='precomputed', distance_threshold=threshold,
                                    linkage='complete')
    c.fit(dist_array)
    if return_df_and_c:
        return *get_all_metrics(threshold, features, c, dist_array, labels, encoded_labels, label_encoder,
                                silhouette_aggregation,
                                return_df=return_df_and_c), c
    else:
        return get_all_metrics(threshold, features, c, dist_array, labels, encoded_labels, label_encoder,
                               silhouette_aggregation)


# Here, do a run ith only the best
def cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, decimals=5, n_points=500,
                           silhouette_aggregation='micro', n_jobs=1):
    # Getting clustering at all thresholds
    limits = get_linspace(dist_array, decimals, n_points)
    if n_jobs > 1:
        wrapper = partial(cluster_single_threshold, dist_array=dist_array, features=features, labels=labels,
                          encoded_labels=encoded_labels, label_encoder=label_encoder,
                          silhouette_aggregation=silhouette_aggregation)
        results = Parallel(n_jobs=n_jobs)(delayed(wrapper)(threshold=t) for t in tqdm(limits))
    else:
        results = []
        for t in tqdm(limits):
            results.append(cluster_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, t,
                                                    silhouette_aggregation))
    results = pd.DataFrame(results).sort_values('threshold')
    results['retention'] = (dist_array.shape[0] - results['n_singletons']) / dist_array.shape[0]
    return results


def vae_clustering_pipeline(model_folder, input_df, name, dataset_params=None, n_points=500):
    model = get_model(model_folder, map_location='cpu')
    latent_df = get_latent_df(model, input_df, dataset_params)
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    results['input_type'] = name
    return results


# Here, do a run ith only the best
def plot_pipeline(results, b, plot_title='None', fig_fn=None, filter=None, palette=None, more=False,
                  add_cluster_size=False):
    runs = pd.concat([b, results])
    # plotting options
    if filter is None:
        filter = ['TBCRalign', 'KernelSim', 'tcrdist3'] + list(results.input_type.unique())

    if palette is None:
        palette = 'gnuplot2'
    if more:
        palette = get_palette(palette, n_colors=len(filter) - 3)
    else:
        palette = sns.color_palette(palette, n_colors=len(filter) - 3)

    sns.set_palette(palette)
    f, a = plt.subplots(1, 1, figsize=(9, 9))
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Avg Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    print(runs.duplicated().any())
    if add_cluster_size:
        ax2 = a.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_yscale('log', base=2)

    for i, input_type in enumerate(filter):
        query = runs.query('input_type==@input_type')
        retentions = query['retention'].values[1:-1]
        purities = query['mean_purity'].values[1:-1]
        print(input_type, '\t', round(get_retpur_auc(retentions, purities), 4))
        ls = '-' if i % 2 == 0 else '--'
        if add_cluster_size:
            cluster_sizes = query['mean_cluster_size'].values[1:-1]
        # Plotting baselines with fixed styles colors etc
        if input_type == "TBCRalign":
            a.plot(retentions, purities, label='TBCRalign', ls=':', c='k', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='x', lw=0.5, s=8, c='k')
        elif input_type == "KernelSim":
            a.plot(retentions, purities, label='KernelSim', ls=':', c='m', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='v', lw=0.1, s=8, c='m')
        elif input_type == "tcrdist3":
            a.plot(retentions, purities, label='tcrdist3', ls=':', c='y', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='*', lw=0.1, s=8, c='y')
        # Plotting the actual things
        else:
            a.plot(retentions, purities, label=input_type.lstrip('_').replace('_', ' ').replace('checkpoint best', ''),
                   ls=ls, lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='+', lw=1.15, s=12)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')

    a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
             prop={'weight': 'semibold', 'size': 12})

    f.suptitle(f'{plot_title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fig_fn is not None:
        f.savefig(f'../output/240411_ClusteringTests/{fig_fn}.png', dpi=200)
    return runs


##################################
#      LOAD MODEL / LATENT       #
##################################

def resort_baseline(baseline_dm, input_dm, index_col,
                    cols=('peptide', 'original_peptide', 'binder', 'partition')):
    """
    Resorts the baseline_dm to match input_dm in order to do agg_clustering
    Args:
        baseline_dm:
        input_dm:
        index_col:

    Returns:

    """
    cols = list(cols)
    if index_col not in cols:
        cols.append(index_col)
    baseline_copy = baseline_dm.copy()
    baseline_copy = baseline_copy.drop_duplicates(index_col)
    original_index_col = baseline_dm.index.name
    if original_index_col is None:
        original_index_col = 'index'
    reindex = input_dm[index_col].values
    baseline_copy = baseline_copy.reset_index().set_index(index_col).loc[reindex].reset_index().set_index(
        original_index_col)
    idxcols = [str(x) for x in baseline_copy.index.to_list()] if type(
        [x for x in baseline_copy.columns if x not in cols][0]) == str else baseline_copy.index.to_list()
    # print(idxcols+cols)
    baseline_copy = baseline_copy[idxcols + cols]
    # baseline_copy = baseline_copy.reset_index().set_index(original_index_col)
    baseline_array = baseline_copy.iloc[:len(baseline_copy), :len(baseline_copy)].values
    return baseline_copy, baseline_array


def get_retpur_auc(retentions, purities, min_retention=0.0, max_retention=1.0, min_purity=0.0, max_purity=1.0):
    ridx = np.where((min_retention <= retentions) & (retentions <= max_retention))
    pidx = np.where((min_purity <= purities) & (purities <= max_purity))
    idx = np.intersect1d(ridx, pidx)
    if len(idx) <= 1:
        return -1
    retentions = retentions[idx]
    purities = purities[idx]
    return auc(retentions, purities)


def get_all_rpauc(retentions, purities, input_type=None, name=None):
    total_auc = get_retpur_auc(retentions, purities)
    p60_auc = get_retpur_auc(retentions, purities, min_purity=0.6)
    p70_auc = get_retpur_auc(retentions, purities, min_purity=0.7)

    p60_r40_auc = get_retpur_auc(retentions, purities, min_retention=0.4, min_purity=0.6, max_retention=0.975)
    p70_r35_auc = get_retpur_auc(retentions, purities, min_retention=0.35, min_purity=0.70, max_retention=0.975)
    p70_r50_auc = get_retpur_auc(retentions, purities, min_retention=0.5, min_purity=0.70, max_retention=0.975)
    out = {'input_type': input_type,
           'name': name,
           'total_auc': total_auc,
           'p70_auc': p70_auc,
           'p60_auc': p60_auc,
           'p60_r40_auc': p60_r40_auc,
           'p70_r35_auc': p70_r35_auc,
           'p70_r50_auc': p70_r50_auc}

    return out


def get_all_inputs_rpauc(df, input_col='input_type'):
    # Utility function to return the rpauc df for all groups and sort by best
    return pd.json_normalize(df.groupby('input_type').apply(lambda group: get_all_rpauc(group['retention'].values[1:-1],
                                                                                        group['mean_purity'].values[
                                                                                        1:-1],
                                                                                        group[input_col].iloc[
                                                                                            0]))).drop(
        columns=['name']).set_index(input_col).sort_values('p70_r35_auc', ascending=False)


def get_model(folder, map_location='cpu'):
    pt = glob.glob(folder + '/*checkpoint_best*.pt')
    pt = [x for x in pt if 'interval' not in x][0]
    js = glob.glob(folder + '/*checkpoint*.json')[0]
    model = load_model_full(pt, js, map_location=map_location)
    # Extract the vae part if the model comes from a two stage VAE
    if type(model) == TwoStageVAECLF:
        model = model.vae
    model.eval()
    return model


def get_latent_df(model, df, dataset_params: dict = None, batch_size=512):
    # Init dataset and pred fct depending on model type
    dataset_params = dict(max_len_a1=7, max_len_a2=8, max_len_a3=22,
                          max_len_b1=6, max_len_b2=7, max_len_b3=23, max_len_pep=0,
                          encoding='BL50LO', pad_scale=-20,
                          a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                          pep_col='peptide') if dataset_params is None else dataset_params

    if hasattr(model, 'vae'):
        model = model.vae

    if model.max_len > 7 + 8 + 22 + 6 + 7 + 23:
        dataset_params['max_len_pep'] = 12

    elif model.max_len <= 22 + 23:
        dataset_params['max_len_a1'] = 0
        dataset_params['max_len_a2'] = 0
        dataset_params['max_len_b1'] = 0
        dataset_params['max_len_b2'] = 0
        dataset_params['max_len_pep'] = 0

    dataset_params['add_positional_encoding'] = model.add_positional_encoding

    if type(model) == FullTCRVAE:
        print(dataset_params)
        dataset = FullTCRDataset(df, **dataset_params)
        dataloader = dataset.get_dataloader(batch_size, SequentialSampler)
        latent_df = predict_model(model, dataset, dataloader)

    elif type(model) == CNNVAE:
        dataset_params['conv'] = True
        dataset = FullTCRDataset(df, **dataset_params)
        dataloader = dataset.get_dataloader(batch_size, SequentialSampler)
        latent_df = predict_model(model, dataset, dataloader)

    return latent_df


def get_distances_labels_from_latent(latent_df, label_col='peptide', seq_cols=('A1', 'A2', 'A3', 'B1', 'B2', 'B3'),
                                     index_col='raw_index', rest_cols=None, low_memory=False):
    # Columns for making distmatrix
    if rest_cols is None:
        rest_cols = list(
            x for x in latent_df.columns if
            x in ['peptide', 'original_peptide', 'partition', 'origin', 'binder', index_col])
    if index_col not in rest_cols:
        rest_cols.append(index_col)
    if label_col not in rest_cols:
        rest_cols.append(label_col)
    # Getting distmatrix and arrays
    dist_matrix = make_dist_matrix(latent_df, label_col, seq_cols, cols=rest_cols, low_memory=low_memory)
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values
    # Getting label encoder and features for computing metrics
    features = latent_df[[z for z in latent_df.columns if z.startswith('z_')]].values
    label_encoder = LabelEncoder()
    labels = dist_matrix[label_col].values
    encoded_labels = label_encoder.fit_transform(labels)
    return dist_matrix, dist_array, features, labels, encoded_labels, label_encoder


##################################
#        tbcralign fcts          #
##################################
def get_merged_distances_labels(dist_matrix, original_df, index_tcr_df, label_col='peptide', query_subset=None):
    # Assumes a square matrix with no other columns, and that the original_df and index_tcr_df match
    merged = pd.merge(index_tcr_df, original_df[
        [x for x in original_df.columns if x in ['seq_id', 'peptide', 'partition', 'binder', 'origin', 'fulltcr']]],
                      left_on=['q_index', 'tcr'], right_on=['seq_id', 'fulltcr'])

    assert ((merged['seq_id'] == merged['q_index']).all() and (merged['tcr'] == merged['fulltcr']).all()), 'fuck'
    merged = merged.set_index('q_index')[
        [x for x in merged.columns if x in ['peptide', 'partition', 'binder', 'origin']]]
    merged_dist_matrix = pd.merge(dist_matrix, merged, left_index=True, right_index=True)
    extra_cols = merged_dist_matrix.columns.difference(dist_matrix.columns)

    if query_subset is not None:
        query = merged_dist_matrix.query(query_subset)
        merged_dist_matrix = query[list(str(x) for x in query.index) + list(extra_cols)]

    return merged_dist_matrix, extra_cols


def get_distances_labels_from_distmatrix(dist_matrix, original_df, index_tcr_df, label_col='peptide',
                                         query_subset=None):
    merged_dist_matrix, extra_cols = get_merged_distances_labels(dist_matrix, original_df, index_tcr_df, label_col,
                                                                 query_subset)
    dist_array = merged_dist_matrix.iloc[:, :-len(extra_cols)].values
    features = torch.randn([dist_array.shape[0], 3])
    label_encoder = LabelEncoder()
    labels = merged_dist_matrix[label_col].values
    encoded_labels = label_encoder.fit_transform(labels)
    return merged_dist_matrix, dist_array, features, labels, encoded_labels, label_encoder


##################################
#            METRICS             #
##################################

def get_purity(counts):
    # Purity in absolute % of a cluster, taking the majority label
    # high = better
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_counts[list(sorted_counts.keys())[0]] / sum(sorted_counts.values())


def get_mixity(counts):
    # how many different labels are inside a cluster, weighted by the number of members
    # low = better
    return len(counts.keys()) / sum(counts.values())


def get_coherence(dist_array):
    # Assumes dist_array is the subset of the distance array for a given cluster label
    # mean distance within a cluster
    # low = better

    # get upper triangle mask without the diagonale
    mask = np.triu(np.ones(dist_array.shape), k=0) - np.eye(dist_array.shape[0])
    flat_array = dist_array[mask == 1]
    return np.mean(flat_array)


def get_purity_mixity_coherence(cluster_label: int,
                                true_labels: list,
                                pred_labels: list,
                                dist_array: np.array,
                                label_encoder):
    """
        For a given cluster label (int) returned by clustering.labels_,
        Return the purity, mixity, coherence, cluster_size, and silhouette_scale (==cluster_size/total_size)
        silhouette_scale should be used to get a weighted average metric at the end
    """
    indices = np.where(pred_labels == cluster_label)[0]
    cluster_size = len(indices)
    if cluster_size <= 1:
        # return np.nan, np.nan, np.nan, 1, 1/len(true_labels)
        return {'purity': np.nan, 'coherence': np.nan, 'cluster_size': 1}

    # Query the subset of the true labels belonging to this cluster using indices
    # Convert to int label encodings in order to use np.bincount to get purity and mixity

    subset = true_labels[indices]
    encoded_subset = label_encoder.transform(subset)
    counts = {i: k for i, k in enumerate(np.bincount(encoded_subset)) if k > 0}
    majority_label = label_encoder.inverse_transform([sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]])[
        0]
    purity = get_purity(counts)
    # mixity = get_mixity(counts)
    # index the distance matrix and return the mean distance within this cluster (i.e. coherence)
    coherence = get_coherence(dist_array[indices][:, indices])

    # return purity, mixity, coherence, cluster_size, cluster_size / len(true_labels)
    return {'purity': purity, 'coherence': coherence, 'cluster_size': cluster_size, 'majority_label': majority_label}


# Here put get_agglo_df_results():
def get_agglo_cluster_df(c, dist_matrix, res_df, index_col, label_col, rest_cols):
    """

    Args:
        c: The estimator object
        dist_matrix: original distance matrix (labelled)
        res_df: the "df" that we get from cluster_single_threshold(return_df_and_c=True)
        index_col: ...
        label_col: ...
        rest_cols: ...

    Returns:

    """
    columns = [index_col, label_col] + [x for x in list(rest_cols) if x != index_col and x != label_col]
    subset = dist_matrix[columns]
    subset['pred_label'] = c.labels_
    subset = subset.merge(res_df.drop(columns=['coherence']), left_on=['pred_label'], right_on=['pred_label'],
                          how='left')
    subset['cluster_size'].fillna(1, inplace=True)
    subset['majority_label'] = subset.apply(
        lambda x: x[label_col] if x['majority_label'] != np.nan else x['majority_label'], axis=1)
    return subset.sort_values('pred_label')


def get_all_metrics(t, features, c, array, true_labels, encoded_labels, label_encoder, silhouette_aggregation='micro',
                    min_purity=0.8, min_size=6, return_df=False):
    n_cluster = np.sum((np.bincount(c.labels_) > 1))
    n_singletons = (np.bincount(c.labels_) == 1).sum()
    try:
        # Here assumes features is the distance array...
        # So using pre-computed !!
        # metric='precomputed' if (features.shape==array.shape and (features.values.diagonal()==0).all())\
        #         else 'cosine'
        # s_score = silhouette_score(array, c.labels_, metric='precomputed')
        s_score = custom_silhouette_score(array, c.labels_, metric='precomputed', aggregation=silhouette_aggregation)
    except:
        s_score = np.nan
    try:
        c_score = ch_score(features, c.labels_)
    except:
        c_score = np.nan
    try:
        d_score = db_score(features, c.labels_)
    except:
        d_score = np.nan
    try:
        ari_score = adjusted_rand_score(encoded_labels, c.labels_)
    except:
        ari_score = np.nan

    df_out = pd.concat(
        [pd.DataFrame(get_purity_mixity_coherence(k, true_labels, c.labels_, array, label_encoder), index=[0])
         for k in set(c.labels_)]).dropna().reset_index(drop=True).reset_index().rename(columns={'index': 'pred_label'})
    nc_07 = len(df_out.query('purity>=0.7'))

    # Here do the n_above thing :

    n_above = len(df_out.query('purity>@min_purity and cluster_size>@min_size'))
    if return_df:
        return {'threshold': t,
                'n_cluster': n_cluster, 'n_singletons': n_singletons,
                'n_cluster_over_70p': nc_07,
                'mean_purity': df_out['purity'].mean(),
                'min_purity': df_out['purity'].min(),
                'max_purity': df_out['purity'].max(),
                'mean_coherence': df_out['coherence'].mean(),
                'min_coherence': df_out['coherence'].min(),
                'max_coherence': df_out['coherence'].max(),
                'mean_cluster_size': df_out['cluster_size'].mean(),
                'min_cluster_size': df_out['cluster_size'].min(),
                'max_cluster_size': df_out['cluster_size'].max(),
                'silhouette': s_score,
                'n_above': n_above,
                'ch_index': c_score, 'db_index': d_score, 'ARI': ari_score}, df_out
    else:
        return {'threshold': t,
                'n_cluster': n_cluster, 'n_singletons': n_singletons,
                'n_cluster_over_70p': nc_07,
                'mean_purity': df_out['purity'].mean(),
                'min_purity': df_out['purity'].min(),
                'max_purity': df_out['purity'].max(),
                'mean_coherence': df_out['coherence'].mean(),
                'min_coherence': df_out['coherence'].min(),
                'max_coherence': df_out['coherence'].max(),
                'mean_cluster_size': df_out['cluster_size'].mean(),
                'min_cluster_size': df_out['cluster_size'].min(),
                'max_cluster_size': df_out['cluster_size'].max(),
                'silhouette': s_score,
                'n_above': n_above,
                'ch_index': c_score, 'db_index': d_score, 'ARI': ari_score}


def get_bounds(array, decimals=5):
    lower_bound = array[array > 0].min()
    upper_bound = array.max()
    factor = 10 ** decimals
    return np.floor(lower_bound * factor) / factor, np.ceil(upper_bound * factor) / factor


def get_linspace(array, decimals=5, n_points=500):
    # Here maybe take something else instead of min, max but min, X quantile (excluding 0)
    return np.round(np.linspace(*get_bounds(array, decimals), n_points), decimals)


def find_true_maximum(scores, patience, idx_only=False):
    """
    Efficiently finds the correct maximum score based on a patience-based early stopping criterion,
    handling NaN values in the scores array.

    Parameters:
    - scores: list or array-like, the sequence of scores for each iteration.
    - patience: int, the number of iterations to wait for an improvement before stopping.

    Returns:
    - max_score: float, the maximum score ignoring artificial spikes due to malfunctions.
    - max_index: int, the index of the detected maximum score.
    """
    max_score = float('-inf')  # Initialize to a very low value
    max_index = -1
    patience_counter = 0

    for i, score in enumerate(scores):
        # Skip NaN values
        if np.isnan(score):
            continue

        # If the current score improves the maximum, update max_score and reset patience
        if score > max_score:
            max_score = score
            max_index = i
            patience_counter = 0  # Reset patience because we found a new max

        # If the score does not improve, increase the patience counter
        else:
            patience_counter += 1

        # If patience limit is reached, break and return the best score seen so far
        if patience_counter >= patience:
            break

    # If no valid maximum was found (e.g., all NaNs), handle gracefully
    if max_index == -1:
        return None, None

    if idx_only:
        return max_index
    else:
        return max_score, max_index


### KMeans stuff
def kmeans_all_thresholds(features, labels, encoded_labels, label_encoder, silhouette_aggregate='micro', n_jobs=1,
                          min_purity=0.8, min_size=6, min_k=None, max_k=None):
    # Getting clustering at all thresholds, from 2 clusters to every clusters being duplets
    min_k = 2 if min_k is None else min_k
    max_k = max_k if max_k is not None else len(features)
    limits = np.arange(min_k, max_k)
    if n_jobs > 1 or n_jobs == -1:
        wrapper = partial(kmeans_single_threshold, features=features, labels=labels,
                          silhouette_aggregate=silhouette_aggregate,
                          encoded_labels=encoded_labels, label_encoder=label_encoder,
                          min_purity=min_purity, min_size=min_size)
        results = Parallel(n_jobs=n_jobs)(delayed(wrapper)(k=k) for k in tqdm(limits))
    else:
        results = []
        for t in tqdm(limits):
            results.append(
                kmeans_single_threshold(features, labels, encoded_labels, label_encoder, t, silhouette_aggregate,
                                        min_purity=min_purity, min_size=min_size))
    results = pd.DataFrame(results).sort_values('threshold')
    results['retention'] = (len(labels) - results['n_singletons']) / len(labels)
    return results


def kmeans_single_threshold(features, labels, encoded_labels, label_encoder, k, silhouette_aggregate,
                            return_df_and_c=False,
                            min_purity=0.8, min_size=6):
    max_iter = 75 if k >= len(features) // 2 else 300
    c = KMeans(n_clusters=k, max_iter=max_iter)
    c.fit(features)
    if return_df_and_c:
        return *get_kmeans_metrics(k, features, c, labels, encoded_labels, label_encoder, silhouette_aggregate,
                                   min_purity=min_purity, min_size=min_size,
                                   return_df=return_df_and_c), c
    else:
        return get_kmeans_metrics(k, features, c, labels, encoded_labels, label_encoder, silhouette_aggregate,
                                  min_purity=min_purity, min_size=min_size)


def get_km_purity_mixity(cluster_label: int,
                         true_labels: list,
                         pred_labels: list,
                         label_encoder):
    """
        For a given cluster label (int) returned by clustering.labels_,
        Return the purity, mixity, coherence, cluster_size, and silhouette_scale (==cluster_size/total_size)
        silhouette_scale should be used to get a weighted average metric at the end
    """
    indices = np.where(pred_labels == cluster_label)[0]
    cluster_size = len(indices)
    if cluster_size <= 1:
        # return np.nan, np.nan, np.nan, 1, 1/len(true_labels)
        return {'purity': np.nan, 'cluster_size': 1}

    # Query the subset of the true labels belonging to this cluster using indices
    # Convert to int label encodings in order to use np.bincount to get purity and mixity

    subset = true_labels[indices]
    encoded_subset = label_encoder.transform(subset)
    counts = {i: k for i, k in enumerate(np.bincount(encoded_subset)) if k > 0}
    majority_label = label_encoder.inverse_transform([sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]])[
        0]
    purity = get_purity(counts)
    # mixity = get_mixity(counts)
    # index the distance matrix and return the mean distance within this cluster (i.e. coherence)

    # return purity, mixity, coherence, cluster_size, cluster_size / len(true_labels)
    return {'purity': purity, 'cluster_size': cluster_size, 'majority_label': majority_label}


def kmeans_pipeline(model, df, model_name=None, dataset_name=None, partition=None, silhouette_aggregate='micro',
                    n_jobs=8):
    latent = get_latent_df(model, df)
    features = latent[[z for z in latent.columns if z.startswith('z_')]]
    labels = latent['peptide'].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    out = kmeans_all_thresholds(features, labels, encoded_labels, label_encoder, silhouette_aggregate, n_jobs)

    out['input_type'] = model_name
    out['dataset'] = dataset_name
    out['partition'] = partition
    return out


def agglo_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, threshold,
                           min_purity=0.8, min_size=6,
                           silhouette_aggregation='micro', return_df_and_c=False):
    try:
        c = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=threshold,
                                    linkage='complete')
    except:
        c = AgglomerativeClustering(n_clusters=None, affinity='precomputed', distance_threshold=threshold,
                                    linkage='complete')
    c.fit(dist_array)
    if return_df_and_c:
        return *get_all_metrics(threshold, dist_array, c, dist_array, labels, encoded_labels, label_encoder,
                                silhouette_aggregation, min_purity=min_purity, min_size=min_size, return_df=return_df_and_c), c
    else:
        return get_all_metrics(threshold, dist_array, c, dist_array, labels, encoded_labels, label_encoder,
                               silhouette_aggregation,
                               min_purity=min_purity, min_size=min_size)


# Here, do a run ith only the best
def agglo_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, decimals=5, n_points=500,
                         min_purity=0.8, min_size=6,
                         silhouette_aggregation='micro', n_jobs=1):
    assert type(dist_array)==np.ndarray, f'{type(dist_array)}'
    assert dist_array.shape[0] == dist_array.shape[1], f'{dist_array.shape}'
    print(label_encoder, decimals, n_points, min_size, min_purity, silhouette_aggregation, n_jobs)
    # Getting clustering at all thresholds
    limits = get_linspace(dist_array, decimals, n_points)
    if n_jobs > 1 or n_jobs == -1:
        wrapper = partial(agglo_single_threshold, dist_array=dist_array, features=features, labels=labels,
                          min_purity=min_purity, min_size=min_size,
                          encoded_labels=encoded_labels, label_encoder=label_encoder,
                          silhouette_aggregation=silhouette_aggregation)
        results = Parallel(n_jobs=n_jobs)(delayed(wrapper)(threshold=t) for t in tqdm(limits))
    else:
        results = []
        for t in tqdm(limits):
            results.append(agglo_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, t,
                                                  silhouette_aggregation=silhouette_aggregation, min_purity=min_purity,
                                                  min_size=min_size))
    results = pd.DataFrame(results).sort_values('threshold')
    results['retention'] = (dist_array.shape[0] - results['n_singletons']) / dist_array.shape[0]
    return results


def topn_pipe(matrix, initial_cut_threshold=1, distance_weighted=True, silhouette_aggregation='micro',
              mst_algo='kruskal', min_purity=0.8, min_size=6):
    G, tree, matrix, values, labels, encoded_labels, label_encoder, raw_indices = create_mst_from_distance_matrix(
        matrix, label_col='peptide', index_col='raw_index', algorithm=mst_algo)

    best_tree, subgraphs, micro_clusters, best_edges_removed, best_nodes_removed, scores, purities, retentions, mean_cluster_sizes, n_clusters, cluster_sizes_micro, c_above \
        = iterative_topn_cut_logsize(values, tree, initial_cut_threshold=initial_cut_threshold,
                                     initial_cut_method='top',
                                     cut_threshold=1, which='edge', distance_weighted=distance_weighted, verbose=0,
                                     score_threshold=1, silhouette_aggregation='micro',
                                     min_purity=min_purity, min_size=min_size)
    # df results
    n_above = [len(x) for x in c_above]
    micro_cut_df = pd.DataFrame(np.array([scores, purities, retentions, mean_cluster_sizes, n_clusters, n_above]).T,
                                columns=['silhouette', 'mean_purity', 'retention', 'mean_cluster_size', 'n_cluster',
                                         'n_above'])  # .assign(silhouette_aggregate='micro')
    micro_cut_df['best'] = False
    micro_cut_df.loc[micro_cut_df['silhouette'].idxmax(), 'best'] = True
    return micro_cut_df


def agglo_pipe(matrix, min_purity=0.8, min_size=6, silhouette_aggregation='micro',
               decimals=5, n_points=500, n_jobs=-1):
    da = matrix.iloc[:len(matrix), :len(matrix)].values
    labels = matrix['peptide'].values
    labenc = LabelEncoder()
    encoded_labels = labenc.fit_transform(labels)
    results = agglo_all_thresholds(da, da, labels, encoded_labels, labenc, decimals, n_points, min_purity, min_size,
                                   silhouette_aggregation, n_jobs)
    results['best'] = False
    results.loc[results['silhouette'].idxmax(), 'best'] = True
    return results[
        ['threshold', 'silhouette', 'best', 'n_cluster', 'n_singletons', 'mean_purity', 'min_purity', 'max_purity',
         'mean_cluster_size', 'min_cluster_size', 'max_cluster_size', 'n_above', 'retention']]


def kmeans_pipe(latent, min_purity=0.8, min_size=6, silhouette_aggregation='micro', n_jobs=-1):
    features = latent[[z for z in latent.columns if z.startswith('z_')]].values
    labels = latent['peptide'].values
    labenc = LabelEncoder()
    encoded_labels = labenc.fit_transform(labels)
    results = kmeans_all_thresholds(features, labels, encoded_labels, labenc, silhouette_aggregation, n_jobs,
                                    min_purity=min_purity, min_size=min_size, min_k=2,
                                    max_k=int(2 * len(features) // 3) + 1)
    results['best'] = False
    results.loc[results['silhouette'].idxmax(), 'best'] = True
    return results[
        ['threshold', 'silhouette', 'best', 'n_cluster', 'n_singletons', 'mean_purity', 'min_purity', 'max_purity',
         'mean_cluster_size', 'min_cluster_size', 'max_cluster_size', 'n_above', 'retention']]


def get_kmeans_metrics(t, features, c, true_labels, encoded_labels, label_encoder, aggregate='micro', return_df=False,
                       min_purity=0.8, min_size=6):
    n_cluster = np.sum((np.bincount(c.labels_) > 1))
    n_singletons = (np.bincount(c.labels_) == 1).sum()
    s_score = custom_silhouette_score(features, c.labels_, metric='cosine', aggregation=aggregate)
    try:
        c_score = ch_score(features, c.labels_)
    except:
        c_score = np.nan
    try:
        d_score = db_score(features, c.labels_)
    except:
        d_score = np.nan
    try:
        ari_score = adjusted_rand_score(encoded_labels, c.labels_)
    except:
        ari_score = np.nan

    df_out = pd.concat(
        [pd.DataFrame(get_km_purity_mixity(label_i, true_labels, c.labels_, label_encoder), index=[0])
         for label_i in set(c.labels_)]).reset_index(drop=True).reset_index().rename(columns={'index': 'pred_label'})
    nc_07 = len(df_out.query('purity>=0.7'))
    n_above = len(df_out.query('purity>@min_purity and cluster_size>@min_size'))
    if return_df:
        return {'threshold': t,
                'n_cluster': n_cluster, 'n_singletons': n_singletons,
                'n_cluster_over_70p': nc_07,
                'mean_purity': df_out['purity'].mean(),
                'min_purity': df_out['purity'].min(),
                'max_purity': df_out['purity'].max(),
                'mean_cluster_size': df_out['cluster_size'].mean(),
                'min_cluster_size': df_out['cluster_size'].min(),
                'max_cluster_size': df_out['cluster_size'].max(),
                'silhouette': s_score,
                'n_above': n_above,
                'ch_index': c_score, 'db_index': d_score, 'ARI': ari_score}, df_out
    else:
        return {'threshold': t,
                'n_cluster': n_cluster, 'n_singletons': n_singletons,
                'n_cluster_over_70p': nc_07,
                'mean_purity': df_out['purity'].mean(),
                'min_purity': df_out['purity'].min(),
                'max_purity': df_out['purity'].max(),
                'mean_cluster_size': df_out['cluster_size'].mean(),
                'min_cluster_size': df_out['cluster_size'].min(),
                'max_cluster_size': df_out['cluster_size'].max(),
                'silhouette': s_score,
                'n_above': n_above,
                'ch_index': c_score, 'db_index': d_score, 'ARI': ari_score}
