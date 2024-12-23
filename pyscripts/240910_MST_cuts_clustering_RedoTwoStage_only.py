import argparse
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from src.cluster_utils import *
from src.networkx_utils import *
from src.torch_utils import load_model_full
from src.utils import str2bool, make_filename, pkl_dump
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
    parser.add_argument('-tcrdist', '--tcrdist_file', dest='tcrdist_file', type=str,
                        default=None, help='External labelled tcrdist baseline')
    parser.add_argument('-tbcralign', '--tbcralign_file', dest='tbcralign_file', type=str,
                        default=None, help='External labelled tbcralign baseline')
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

    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=8,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=6,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb3', '--max_len_b3', dest='max_len_b3', type=int, default=23,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlpep', '--max_len_pep', dest='max_len_pep', type=int, default=0,
                        help='Max seq length admitted for peptide. Set to 0 to disable adding peptide to the input')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool,
                        default=False,
                        help='Adding positional encoding to the sequence vector. False by default')
    """
    Models args 
    """
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory')

    parser.add_argument('-pt_file_ts_notrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_ts_notrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')

    parser.add_argument('-pt_file_ts_cstrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_ts_cstrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-index_col', type=str, required=False, default=None,
                        help='index col to sort both baselines and latent df')
    parser.add_argument('-label_col', type=str, required=False, default='peptide',
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

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=500,
                        help='How many points to do for the bounds limits')
    parser.add_argument('-link', '--linkage', dest='linkage', type=str, default='complete',
                        help='Which linkage to use for AgglomerativeClustering')
    """
    TODO: Misc. 
    """
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    parser.add_argument('-reset', dest='reset', type=str2bool, default=False,
                        help='Whether to reset the encoder\'s weight for a blank run')
    parser.add_argument('-random_latent', dest='random_latent', type=str2bool, default=False,
                        help='Whether to set RANDOM latent vectors')
    parser.add_argument('-newmodel', dest='newmodel', type=str2bool, default=False,
                        help='re instanciate a new model from scratch')
    parser.add_argument('-tcr_enc', dest='tcr_enc', type=str, default=None,
                        help='Whether to do "alternative" TCR encoding')
    parser.add_argument('-n_jobs', dest='n_jobs', default=8, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    start = dt.now()
    print('Starting MST-cut pyscript')
    sns.set_style('darkgrid')
    args = vars(args_parser())
    unique_filename, kf, rid, connector = make_filename(args)
    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    # Here this is commented because we handle the uniquefilename creation already
    # in the overall bash script
    # outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    df = pd.read_csv(args['file'])

    model_ts_notrp = load_model_full(args['pt_file_ts_notrp'], args['json_file_ts_notrp'],
                                     map_location=args['device'], verbose=False)

    model_ts_cstrp = load_model_full(args['pt_file_ts_cstrp'], args['json_file_ts_cstrp'],
                                     map_location=args['device'], verbose=False)
    index_col = args['index_col']
    label_col = args['label_col']
    rest_cols = args['rest_cols']
    weight_col = args['weight_col']
    if weight_col is not None:
        if weight_col not in rest_cols and weight_col in df.columns:
            rest_cols.append(weight_col)

    latent_df_ts_notrp = get_latent_df(model_ts_notrp, df)
    latent_df_ts_cstrp = get_latent_df(model_ts_cstrp, df)

    if index_col is None or index_col == '' or index_col not in latent_df_ts_cstrp.columns:
        index_col = 'index_col'
        latent_df_ts_cstrp[index_col] = [f'seq_{i:04}' for i in range(len(latent_df_ts_cstrp))]
    seq_cols = tuple(args[f'{x.lower()}_col'] for x in ('A1', 'A2', 'A3', 'B1', 'B2', 'B3'))

    dm_vae_ts_notrp, vals_vae, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df_ts_notrp,
        label_col, seq_cols,
        index_col,
        rest_cols,
        args['low_memory'])

    dm_vae_ts_cstrp, vals_vae, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df_ts_cstrp,
        label_col, seq_cols,
        index_col,
        rest_cols,
        args['low_memory'])
    # This part assumes that the script will read a pre-formatted distance matrix (with labels etc) from a source.
    # Given the full pipeline (MSTcut_all_pipeline.sh) we should run --> do_tbcralign.sh, do_tcrdist.py, then the current script
    dm_tbcr, _ = resort_baseline(pd.read_csv(args['tbcralign_file'], index_col=0), dm_vae_ts_notrp, index_col,
                                 rest_cols)
    dm_tcrdist, _ = resort_baseline(pd.read_csv(args['tcrdist_file'], index_col=0), dm_vae_ts_notrp, index_col,
                                    rest_cols)
    vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results, \
    vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results, \
    tbcr_size_results, tbcr_topn_results, tbcr_agglo_results, \
    tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results = do_twostage_2vae_clustering_pipeline(
        dm_vae_ts_notrp,
        dm_vae_ts_cstrp,
        dm_tbcr, dm_tcrdist,
        label_col=label_col,
        index_col=index_col,
        weight_col=weight_col,
        outdir=outdir,
        filename=args['out'],
        title=args['out'])
    for result in [vae_ts_notrp_size_results, vae_ts_notrp_topn_results, vae_ts_notrp_agglo_results,
                   vae_ts_cstrp_size_results, vae_ts_cstrp_topn_results, vae_ts_cstrp_agglo_results,
                   tbcr_size_results, tbcr_topn_results, tbcr_agglo_results,
                   tcrdist_size_results, tcrdist_topn_results, tcrdist_agglo_results]:
        # Get the cluster df and save it to a txt file
        resdf = result.pop('df')
        dm_name = resdf['dm_name'].unique()[0]
        method = resdf['method'].unique()[0]
        # Merge to inputdf first!
        resdf.to_csv(f'{outdir}cluster_results_{dm_name}_{method}.csv')
        # Save the remaining results (curves etc) to a pickle file
        pkl_dump(result, f'result_{dm_name}_{method}.pkl', outdir)

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'MST-cut finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')


if __name__ == '__main__':
    main()
