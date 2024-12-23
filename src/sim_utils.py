import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from src.metrics import compute_cosine_distance

def make_self_db_distmatrix(preds, peptide, label_col='peptide', cols=('peptide','original_peptide')):
    # Set self as db using the positives and query as the whole set
    db = preds.query('peptide==@peptide and binder==1').assign(set='db')
    query = preds.copy().assign(set='query')
    # Use the set columns to remove dupes and rearrange matrix
    dist_matrix = make_dist_matrix(pd.concat([db, query]), label_col, seq_cols=('A1','A2','A3','B1','B2','B3','set'), cols=cols)
    no_query = [x for x in dist_matrix.columns if 'query' not in x]
    only_query = [x for x in dist_matrix.index if 'query' in x]
    dist_matrix = dist_matrix.loc[only_query][no_query]
    dist_matrix.drop_duplicates(inplace=True)
    dist_matrix.columns = dist_matrix.columns.str.replace('db','')
    dist_matrix.index = dist_matrix.index.str.replace('query','')
    # Puts nan to ignore during sorting
    dist_matrix.replace(0., np.nan, inplace=True)
    dist_matrix['label'] = (dist_matrix[label_col]==peptide).astype(int)
    return dist_matrix, db, query

def make_dist_matrix(df, label_col='peptide',
                     seq_cols=('A1', 'A2', 'A3', 'B1', 'B2', 'B3'), 
                     cols=('peptide', 'original_peptide'), low_memory=False):
    """
    From a latent vector dataframe, compute a square distance matrix with cosine distance
    Args:
        df:
        label_col:
        seq_cols:
        cols:

    Returns:

    """
    df['seq'] = df.apply(lambda x: ''.join([x[c] for c in seq_cols]), axis=1)
    seqs = df.seq.values
    # Getting dist matrix
    zcols = [z for z in df.columns if z.startswith("z_")]
    zs = torch.from_numpy(df[zcols].values)
    dist_matrix = pd.DataFrame(compute_cosine_distance(zs),
                               columns=seqs, index=seqs)
    # dist_matrix = pd.merge(dist_matrix, df.set_index('seq')[list(cols)],
    #                        left_index=True, right_index=True).drop_duplicates()  # .rename(columns={label_col: 'label'})
    if low_memory:
        dist_matrix[list(cols)] = df[list(cols)]
    else:
        dist_matrix = pd.concat([dist_matrix, df.set_index('seq')[list(cols)]], axis=1)
        # dm.drop_duplicates()
        dist_matrix = dist_matrix[dist_matrix.index.to_list() + list(cols)]
    return dist_matrix



def get_tcrbase_method(tcr, ref):
    # here take the top 1 (shorted distance = highest sim)
    best = ref[tcr].sort_values().head(1)
    best_name = best.index[0]
    best_dist = best.item()
    label = ref.loc[best_name]['label']
    return label, best_name, best_dist


def do_tcrbase(query_distmatrix, db_distmatrix, label='GILGFVFTL'):
    """
    dist_matrix is the filtered query matrix
    ref is the filtered database matrix
    Args:
        query_distmatrix:
        db_distmatrix:
        label:

    Returns:

    """
    output = query_distmatrix.drop(
        columns=[x for x in query_distmatrix.columns if x != 'label']).copy().reset_index().rename(
        columns={'index': 'tcr'})

    output[['similar_label', 'best_match', 'best_dist']] = output.apply(
        lambda x: get_tcrbase_method(x['tcr'], ref=db_distmatrix),
        axis=1, result_type='expand')
    output['y_true'] = (output['label'] == label).astype(int)
    output['score'] = 1 - output['best_dist']
    return output.sort_values(['y_true', 'score'], ascending=False)


def do_histplot_distribution(dist_matrix, peptide, f=None, ax=None, xlim=None, label_col='label',
                             unique_filename=None, outdir=None, bins=100, title=None):
    """
    Plots the distribution of distances in a score all against all kind of method,
    Grouping the datapoints to be plotted by same label or not
    Args:
        dist_matrix:
        peptide:
        f:
        ax:
        label_col:
        unique_filename:
        outdir:
        bins:

    Returns:

    """
    # splitting the matrix by label
    dist_matrix = dist_matrix[dist_matrix.index.tolist() + [label_col, 'original_peptide']]
    same = dist_matrix.query(f'{label_col}==@peptide')
    same_tcrs = same.index.tolist()
    diff = dist_matrix.query(f'{label_col}!=@peptide')
    diff_tcrs = diff.index.tolist()
    same_matrix = same[same_tcrs].values
    diff_matrix = same[diff_tcrs].values
    # Getting the flattened distributions (upper triangle and making df for plot)
    trimask_same = np.triu(np.ones(same_matrix.shape), k=1)
    flattened_same = same_matrix[trimask_same == 1]
    trimask_diff = np.triu(np.ones(diff_matrix.shape), k=1)
    flattened_diff = diff_matrix[trimask_diff == 1]
    cat = np.concatenate([flattened_same, flattened_diff])
    labels = np.concatenate([np.array(['same'] * len(flattened_same) + ['diff'] * len(flattened_diff))])
    ntr = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', label_col])
    ntr['distance'] = ntr['distance'].astype(float)
    # plotting
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=ntr, x='distance', hue=label_col, ax=ax, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    xlim = [0, 1.1] if xlim is None else xlim
    ax.set_xlim(xlim)
    title = peptide if title is None else title
    ax.set_title(title, fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f.savefig(f'{outdir}{peptide}_AllvAll_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')


def do_histplot_best(dist_matrix, peptide, f=None, ax=None, label_col='original_peptide',
                     unique_filename=None, outdir=None, bins=100):
    """
    Takes the distance matrix, and takes the "best" (which is the minimum distance)
    and plots that instead of the full distribution for each data point

    Args:
        dist_matrix:
        peptide:
        f:
        ax:
        label_col:
        unique_filename:
        outdir:
        bins:

    Returns:

    """
    # splitting the matrix by label
    dist_matrix = dist_matrix[dist_matrix.index.tolist() + [label_col, 'original_peptide']]
    same = dist_matrix.query(f'{label_col}==@peptide')
    diff = dist_matrix.query(f'{label_col}!=@peptide')
    same_tcrs = same.index.tolist()
    diff_tcrs = diff.index.tolist()
    same_matrix = same[same_tcrs].values
    diff_matrix = same[diff_tcrs].values
    # Add ones to diagonal before taking the "minimum" on axis 1
    same_matrix = same_matrix + np.diag(np.ones(same_matrix.shape[0]))

    flattened_diff = diff_matrix.min(axis=1).flatten()
    flattened_same = same_matrix.min(axis=1).flatten()

    cat = np.concatenate([flattened_same, flattened_diff])
    labels = np.concatenate([np.array(['same'] * len(flattened_same) + ['diff'] * len(flattened_diff))])
    ntr = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', label_col])
    ntr['distance'] = ntr['distance'].astype(float)
    # plotting
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=ntr, x='distance', hue=label_col, ax=ax, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    # ax.set_xlim([0])
    ax.set_title(peptide, fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f.savefig(f'{outdir}{peptide}_BEST_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')


def do_tcrbase_and_histplots(preds, peptide, partition, f=None, ax=None,
                             unique_filename=None, outdir=None, bins=100):
    query = preds.query('partition==@partition and peptide==@peptide').assign(set='query')
    database = preds.query('partition!=@partition and peptide==@peptide and original_peptide==@peptide').assign(
        set='database')
    concat = pd.concat([query, database])
    dist_matrix = make_dist_matrix(concat, cols=('set', 'peptide', 'original_peptide', 'binder'))

    query = dist_matrix.query('set=="query"')
    database = dist_matrix.query('set=="database"')
    db_tcrs = database.index.tolist()
    # Scoring query against database & splitting by label
    pos = query[db_tcrs + ['binder']].query('binder==1')
    neg = query[db_tcrs + ['binder']].query('binder==0')
    tcrbase_output = pd.concat([pos, neg])
    pos = pos.drop(columns=['binder']).values
    neg = neg.drop(columns=['binder']).values
    # Plot both the distribution of scores and the "best" score as done above
    #   Plotting distribution of all scores
    pos_flat = pos.flatten()
    neg_flat = neg.flatten()
    cat = np.concatenate([pos_flat, neg_flat])
    labels = np.concatenate([np.array(['pos'] * len(pos_flat) + ['neg'] * len(neg_flat))])
    df_plot_allvsall = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', 'label'])
    df_plot_allvsall['distance'] = df_plot_allvsall['distance'].astype(float)
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=df_plot_allvsall, x='distance', hue='label', ax=ax, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    # ax.set_xlim([0,1.1])
    ax.set_title(f'TCRBase: All vs All {peptide}', fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f.savefig(f'{outdir}TCRBase_AllvAll_{peptide}_distances_histplot_{unique_filename}', dpi=150,
                  bbox_inches='tight')

    #   Plotting "Best" score
    pos_best = pos.min(axis=1).flatten()
    neg_best = neg.min(axis=1).flatten()
    cat_best = np.concatenate([pos_best, neg_best])
    labels_best = np.concatenate([np.array(['pos'] * len(pos_best) + ['neg'] * len(neg_best))])
    df_plot_best = pd.DataFrame(data=np.stack([cat_best, labels_best]).T, columns=['distance', 'label'])
    df_plot_best['distance'] = df_plot_best['distance'].astype(float)
    f2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=df_plot_best, x='distance', hue='label', ax=ax2, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    # ax.set_xlim([0,1.1])
    ax2.set_title(f'TCRBase: Best score {peptide}', fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f2.savefig(f'{outdir}TCRBase_BEST_{peptide}_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')

    return tcrbase_output

# def wrapper(dist_matrix, peptide, unique_filename, outdir):
#     query = dist_matrix.query('set=="query"').copy()
#     database = dist_matrix.query('set=="database" and label==@peptide').copy()
#     output = do_tcrbase(query, database, label=peptide)
#     output.to_csv(f'{outdir}tcrbase_{peptide}_{unique_filename}.csv')
#     auc = roc_auc_score(output['y_true'], output['score'])
#     auc01 = roc_auc_score(output['y_true'], output['score'], max_fpr=0.1)
#     text = f'\n{peptide}:\tauc={auc:.3f}\tauc01={auc01:.3f}'
#     do_histplot(dist_matrix, peptide, unique_filename, outdir)
#     tqdm.write(text)
#     with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
#         file.write(f'{text}')
