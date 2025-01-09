import argparse
from tqdm.auto import tqdm
import os, sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Get the absolute path of the "src" directory and HARDCODEDDLY ADDING IT
src_path = '/home/local/tools/src/TCRcluster-1.0/src/'
# Add the "src" directory to the Python module search path
sys.path.append(src_path)
import pandas as pd
import seaborn as sns
from cluster_utils import *
from networkx_utils import *
from torch_utils import load_model_full
from utils import str2bool, make_jobid_filename, get_linkage_sorted_dm
from datetime import datetime as dt


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input file')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default='../tmp/', help='Output directory, should be ${TMP} from the bashscript')
    """
    Models args 
    """
    parser.add_argument('-m', '--model', type=str, default='TSCSTRP',
                        help='which model to use ; can be "OSNOTRP", "OSCSTRP", "TSNOTRP", "TSCSTRP"')
    parser.add_argument('-index_col', type=str, required=False, default=None,
                        help='index col to sort both baselines and latent df')
    parser.add_argument('-label_col', type=str, required=False, default='label',
                        help='column containing the labels (eg peptide)')
    parser.add_argument('-weight_col', type=str, required=False, default=None,
                        help='Column that contains the weight for a count (ex: norm_count); Leave empty to not use it')
    parser.add_argument('-rest_cols', type=str, required=False, default=None,
                        nargs='*', help='Other columns to be added; ex : -rest_cols peptide partition binder')
    parser.add_argument('-low_memory', type=str2bool, default=False,
                        help='whether to use "low memory merge mode. Might get wrong results...')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=750,
                        help='How many points to do for the bounded limits')
    parser.add_argument('-t', '--threshold', dest='threshold', default=None,
                        help='If provided, will skip the n_points iteration thing and run a single clustering at the given threshold.')
    parser.add_argument('-mp', '--min_purity', dest='min_purity', type=float, default=.8,
                        help='minimum purity for n_above')
    parser.add_argument('-ms', '--min_size', dest='min_size', type=int, default=6, help='minimum sizefor n_above')
    """
    TODO: Misc. 
    """
    parser.add_argument('-j', '--job_id', dest='job_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-n_jobs', dest='n_jobs', default=20, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    # print('\nStarting run_pipeline.py\n')
    # ev = dict(os.environ)
    # for k,v in ev.items():
    #     print(f'{k}: {v}')
    start = dt.now()
    sns.set_style('darkgrid')
    args = vars(args_parser())

    # TODO : Make output filepath work. Here, need args['out'] as None,
    #        Then define the outdir as the ${TMP} given by the bashscript
    #        For debugging purpouses, here use '../tmp/' so that uniquefilename doesn't use it
    # print(args)
    unique_filename, jobid, connector = make_jobid_filename(args)

    args['device'] = 'cpu'
    outdir = os.path.join(args['outdir'], unique_filename) + '/'
    # print(outdir)
    mkdirs(outdir)
    # dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
    # TODO HERE make sure columns etc are correct (ex A/B3 doesn't contain starting C/F etc.
    try:
        df = pd.read_csv(args['file'])
    except:
        print('Couldn\'t read file')
        sys.exit(1)
    # TODO : Hardcoded path or something server specific but the main directory would be in engine/src/tools/etc/models/
    # --> create directory structure to have each model saved in a separate folder and make loading easy
    # srcpath = '/tools'
    model_paths = {'OSNOTRP': {'pt': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni.pt',
                               'json': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni_JSON_kwargs.json'},
                   'OSCSTRP': {'pt': '../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ.pt',
                               'json': '../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ_JSON_kwargs.json'},
                   'TSNOTRP': {
                       'pt': '../models/TwoStage_NoTriplet_N1jMC/epoch_4500_interval_checkpoint_TwoStage_NoTriplet_N1jMC.pt',
                       'json': '../models/TwoStage_NoTriplet_N1jMC/checkpoint_best_TwoStage_NoTriplet_N1jMC_JSON_kwargs.json'},
                   'TSCSTRP': {
                       'pt': '../models/TwoStage_CosTriplet_jyGpd/epoch_4500_interval_checkpoint_TwoStage_CosTriplet_jyGpd.pt',
                       'json': '../models/TwoStage_CosTriplet_jyGpd/checkpoint_best_TwoStage_CosTriplet_jyGpd_JSON_kwargs.json'}}
    assert args['model'] in model_paths.keys(), f"model provided is {args['model']} and is not in the keys of the dict!"
    model_paths = model_paths[args['model']]
    model = load_model_full(model_paths['pt'], model_paths['json'], map_location=args['device'], verbose=False);
    # print('Loaded model')
    # TODO: Handle input with or without header ?_?
    index_col = args['index_col']
    label_col = args['label_col']
    # rest_cols = args['rest_cols']
    rest_cols = [x for x in df.columns if x not in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'label']]

    # # TODO : phase this out ;
    # weight_col = args['weight_col']
    # if weight_col is not None:
    #     if weight_col not in rest_cols and weight_col in df.columns:
    #         rest_cols.append(weight_col)

    # TODO : Merge latent df back to predicted cluster df
    latent_df = get_latent_df(model, df)

    # Here, if indices are not provided, we give it a random index column to not have to change all the code
    if index_col is None or index_col == '' or index_col not in latent_df.columns:
        latent_df.reset_index(inplace=True)
        latent_df.rename(columns={'index':'original_input_order'}, inplace=True)
        index_col = 'index_col'
        rest_cols.extend([index_col, 'original_input_order'])
        latent_df[index_col] = [f'seq_{i:04}' for i in range(len(latent_df))]

    seq_cols = ('A1', 'A2', 'A3', 'B1', 'B2', 'B3')

    # Here, if labels are not provided, we give it a random label column to not have to change all the code
    random_label = label_col is None or label_col == '' or label_col not in latent_df.columns
    if random_label:
        label_col = 'placeholder_label'
        random_classes = np.random.randint(2, 7, 1)[0]
        labels = [f'class_{i}' for i in range(random_classes)]
        latent_df[label_col] = np.random.choice(labels, (len(latent_df)), replace=True)
        rest_cols.append(label_col)

    # Trying something here...
    if 300 > len(latent_df)//3 and 4*len(latent_df) < 300:
        args['n_points'] = max(50, max(4*len(latent_df), len(latent_df)//3))
    else:
        args['n_points'] = len(latent_df)//3

    dist_matrix, dist_array, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(latent_df,
                                                                                                         label_col,
                                                                                                         seq_cols,
                                                                                                         index_col,
                                                                                                         rest_cols,
                                                                                                         args[
                                                                                                             'low_memory'])
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values

    # TODO : Nice to have but not required
    # TODO : MAKE A SORTED DIST MATRIX SOMEHOW
    #        MAKE A HEATMAP PLOT FOR VIZ
    if args['threshold'] is None or args['threshold'] == "None":
        # print('\nOptim\n')
        optimisation_results = agglo_all_thresholds(dist_array, dist_array, labels, encoded_labels, label_encoder, 5,
                                                    args['n_points'], args['min_purity'], args['min_size'], 'micro',
                                                    args['n_jobs'])
        # print('Got optim')
        optimisation_results['best'] = False
        optimisation_results.loc[
            optimisation_results.iloc[:int(0.8 * len(optimisation_results))]['silhouette'].idxmax(), 'best'] = True
        plot_sprm(optimisation_results, fn=f'{outdir}optimisation_curves', random_label=random_label)
        threshold = optimisation_results.query('best')['threshold'].item()
        optimisation_results[['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']] = optimisation_results[['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']].round(3)
        optimisation_results['max_cluster_size'] = optimisation_results['max_cluster_size'].round(0)
        optimisation_results.to_csv(f'{outdir}optimisation_results_df.csv')
        # print('saved optim')
    else:
        threshold = float(args['threshold'])
        optimisation_results = None
    # print('\nSingle threshold\n')
    metrics, clusters_df, c = agglo_single_threshold(dist_array, dist_array, labels, encoded_labels,
                                                     label_encoder, threshold,
                                                     min_purity=args['min_purity'], min_size=args['min_size'],
                                                     silhouette_aggregation='micro',
                                                     return_df_and_c=True)
    # print('Done single threshold')
    # Assigning labels and saving
    dist_matrix['cluster_label'] = c.labels_
    keep_columns = ['index_col', 'cluster_label']
    results_df = pd.merge(latent_df, dist_matrix[keep_columns], left_on=index_col, right_on=index_col)
    # print('Merged dfs')
    clusters_df.to_csv(f'{outdir}clusters_summary.csv', index=False)
    # Here now sort DF / results + plot heatmap
    sorted_dm, sorted_da = get_linkage_sorted_dm(dist_matrix, 'complete', 'cosine', True)
    sorted_dm.to_csv(f'{outdir}sorted_cosine_distance_matrix.csv')
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    sns.heatmap(sorted_da, ax=ax, square=True, cmap='viridis', xticklabels=False, yticklabels=False)
    results_df = results_df.set_index(index_col).loc[sorted_dm[index_col]].reset_index()
    fig.savefig(f'{outdir}complete_cosine_sorted_heatmap.png', dpi=150)
    results_df.to_csv(f'{outdir}TCRcluster_results.csv', index=False)

    return results_df, clusters_df, optimisation_results, unique_filename, jobid, args


if __name__ == '__main__':
    # TODO : Check the tmp output path and make this downloadable
    results_df, clusters_df, optimisation_results, unique_filename, jobid, args = main()
    print('\n\n')
    print('Click ' + '<a href="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/' \
          + f'{jobid}/{unique_filename}/' \
            'TCRcluster_results.csv" target="_blank">here</a>' + ' to download the latent vector and predicted clusters in .csv format.')

    print('Click ' + '<a href="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/' \
          + f'{jobid}/{unique_filename}/' \
            'clusters_summary.csv" target="_blank">here</a>' + ' to download the clusters summary in .csv format.')

    print('Click ' + '<a href="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/' \
          + f'{jobid}/{unique_filename}/' \
            'cosine_distance_matrix.csv" target="_blank">here</a>' + ' to download the cosine distance matrix in .csv format.')

    print(f'<p>Below is a complete-linkage sorted cosine distance heatmap:</p>')
    print(
        f'<img src="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/{jobid}/{unique_filename}/complete_cosine_sorted_heatmap.png" alt="Cosine sorted heatmap" style="max-width:100%; height:auto;">')

    if optimisation_results is not None:
        pd.set_option('display.max_columns', 30)
        pd.set_option('display.max_rows', 101)
        print('Click ' + '<a href="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/' \
              + f'{jobid}/{unique_filename}/' \
                'TCRcluster_results.csv" target="_blank">here</a>' + ' to download the optimisation results in .csv format.')
        print('Click ' + '<a href="https://services.healthtech.dtu.dk/services/TCRcluster-1.0/tmp/' \
              + f'{jobid}/{unique_filename}/' \
                'optimisation_curves.png" target="_blank">here</a>' + ' to download the optimisation curve plot in .png format.')
        print("\n \nBelow is a table preview of clustering metrics at each threshold tested.\n"
              f"A total of {args['n_points']} points are tested, showing only 10 points centered around the best solution."
              "\nthe 'best' column denotes the best silhouette solution.\n")
        best_index = optimisation_results.query('best').index
        min_index = max(0, (best_index - 10).item())
        max_index = min((best_index + 10).item(), 500)
        optimisation_results[['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']] = optimisation_results[
            ['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']].round(3)
        optimisation_results['max_cluster_size'] = optimisation_results['max_cluster_size'].round(0)

        print(optimisation_results.loc[min_index:max_index][['threshold', 'best', 'n_cluster', 'n_singletons',
                                                             'silhouette', 'mean_purity', 'retention',
                                                             'mean_cluster_size', 'max_cluster_size']]\
              .rename(columns={'mean_cluster_size':'mean_size', 'max_cluster_size':'max_size'}))

