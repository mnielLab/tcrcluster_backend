import argparse
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import seaborn as sns
from cluster_utils import *
from networkx_utils import *
from torch_utils import load_model_full
from utils import str2bool, make_jobid_filename, pkl_dump
from datetime import datetime as dt


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default='cpu', type=str,
                        help='device to use for cuda')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input file')
    # parser.add_argument('-tcrdist', '--tcrdist_file', dest='tcrdist_file', type=str,
    #                     default=None, help='External labelled tcrdist baseline')
    # parser.add_argument('-tbcralign', '--tbcralign_file', dest='tbcralign_file', type=str,
    #                     default=None, help='External labelled tbcralign baseline')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default=None, help='Additional output directory')
    parser.add_argument('-a1', '--a1_col', dest='a1_col', default='A1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a2', '--a2_col', dest='a2_col', default='A2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a3', '--a3_col', dest='a3_col', default='A3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b1', '--b1_col', dest='b1_col', default='B1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b2', '--b2_col', dest='b2_col', default='B2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b3', '--b3_col', dest='b3_col', default='B3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
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
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=None,
                        help='If provided, will skip the n_points iteration thing and run a single clustering at the given threshold.')
    parser.add_argument('-mp', '--min_purity', dest='min_purity', type=float, default=.8,
                        help='minimum purity for n_above')
    parser.add_argument('-ms', '--min_size', dest='min_size', type=int, default=6, help='minimum sizefor n_above')
    """
    TODO: Misc. 
    """
    parser.add_argument('-nh', '--no_header', dest='no_header', type=str2bool, default=False,
                        help='Workaround for inputs with no headers with A LOT OF ASSUMPTIONS')
    parser.add_argument('-j', '--job_id', dest='job_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-n_jobs', dest='n_jobs', default=-1, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    start = dt.now()
    sns.set_style('darkgrid')
    args = vars(args_parser())
    unique_filename, rid, connector = make_jobid_filename(args)

    # TODO: 
    # HERE NEED TO CHANGE OUTDIR
    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    # Here this is commented because we handle the uniquefilename creation already
    # in the overall bash script
    # TODO : These things here need to change for a Webserver

    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    # TODO HERE make sure columns etc are correct (ex A/B3 doesn't contain starting C/F etc.
    df = pd.read_csv(args['file'])
    # TODO : Hardcoded path or something server specific but the main directory would be in engine/src/tools/etc/models/
    # --> create directory structure to have each model saved in a separate folder and make loading easy
    model_paths = {'OSNOTRP': {'pt': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni.pt',
                               'json': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni_JSON_kwargs.json'},
                   'OSCSTRP': {'pt': '../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ.pt',
                               'json':'../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ_JSON_kwargs.json'},
                   'TSNOTRP': {'pt': '../models/TwoStage_NoTriplet_N1jMC/epoch_4500_interval_checkpoint_TwoStage_NoTriplet_N1jMC.pt',
                               'json': '../models/TwoStage_NoTriplet_N1jMC/checkpoint_best_TwoStage_NoTriplet_N1jMC_JSON_kwargs.json'},
                   'TSCSTRP': {'pt': '../models/TwoStage_CosTriplet_jyGpd/epoch_4500_interval_checkpoint_TwoStage_CosTriplet_jyGpd.pt',
                               'json': '../models/TwoStage_CosTriplet_jyGpd/checkpoint_best_TwoStage_CosTriplet_jyGpd_JSON_kwargs.json'}}
    assert args['model'] in model_paths.keys(), f"model provided is {args['model']} and is not in the keys of the dict!"
    model_paths = model_paths[args['model']]
    model = load_model_full(model_paths['pt'], model_paths['json'], map_location=args['device'], verbose=True)

    # TODO: Handle input with or without header ?_?
    index_col = args['index_col']
    label_col = args['label_col']
    # rest_cols = args['rest_cols']
    rest_cols = [x for x in df.columns if x not in ['A1','A2','A3','B1','B2','B3','label']]

    # # TODO : phase this out ;
    # weight_col = args['weight_col']
    # if weight_col is not None:
    #     if weight_col not in rest_cols and weight_col in df.columns:
    #         rest_cols.append(weight_col)

    # TODO : Merge latent df back to predicted cluster df
    latent_df = get_latent_df(model, df)

    # Here, if indices are not provided, we give it a random index column to not have to change all the code
    if index_col is None or index_col == '' or index_col not in latent_df.columns:
        index_col = 'index_col'
        latent_df[index_col] = [f'seq_{i:04}' for i in range(len(latent_df))]

    seq_cols = tuple(args[f'{x.lower()}_col'] for x in ('A1', 'A2', 'A3', 'B1', 'B2', 'B3'))

    # Here, if labels are not provided, we give it a random label column to not have to change all the code
    random_label = label_col is None or label_col == '' or label_col not in latent_df.columns
    if random_label:
        label_col = 'placeholder_label'
        random_classes = np.random.randint(2, 7, 1)[0]
        labels = [f'class_{i}' for i in range(random_classes)]
        latent_df[label_col] = np.random.choice(labels, (len(latent_df)), replace=True)
        rest_cols.append(label_col)

    dist_matrix, dist_array, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(latent_df,
                                                                                                         label_col,
                                                                                                         seq_cols,
                                                                                                         index_col,
                                                                                                         rest_cols,
                                                                                                         args[
                                                                                                             'low_memory'])

    if args['threshold'] is None or args['threshold'] == "None":
        optimisation_results = agglo_all_thresholds(dist_array, dist_array, labels, encoded_labels, label_encoder, 5,
                                                    args['n_points'], args['min_purity'], args['min_size'], 'micro',
                                                    args['n_jobs'])
        optimisation_results['best'] = False
        optimisation_results.loc[
            optimisation_results.iloc[:int(0.8 * len(optimisation_results))]['silhouette'].idxmax(), 'best'] = True
        plot_sprm(optimisation_results, fn=f'{outdir}{unique_filename}optimisation_curves', random_label=random_label)
        threshold = optimisation_results.query('best')['threshold'].item()
        optimisation_results.to_csv(f'{outdir}{unique_filename}optimisation_results_df.csv')
    else:
        threshold = float(args['threshold'])

    metrics, clusters_df, c = agglo_single_threshold(dist_array, dist_array, labels, encoded_labels,
                                                     label_encoder, threshold,
                                                     min_purity = args['min_purity'], min_size=args['min_size'],
                                                     silhouette_aggregation='micro',
                                                     return_df_and_c=True)
    # Assigning labels and saving
    dist_matrix['cluster_label'] = c.labels_
    if random_label:
        clusters_df.drop(columns=[label_col], inplace=True)
    dist_matrix['cluster_label'] = c.labels_
    keep_columns = ['index_col', 'cluster_label']
    results_df = pd.merge(latent_df, dist_matrix[keep_columns], left_on=index_col, right_on=index_col)
    clusters_df.to_csv(f'{outdir}{unique_filename}clusters_summary.csv', index=False)
    results_df.to_csv(f'{outdir}{unique_filename}TCRcluster_results.csv', index=False)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')


if __name__ == '__main__':
    main()
