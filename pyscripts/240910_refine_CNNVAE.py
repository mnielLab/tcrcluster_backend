import glob
from typing import Union, Dict
import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# import wandb
import math
import torch
from torch import optim
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists, make_filename, get_class_initcode_keys, pkl_load
from src.torch_utils import load_checkpoint, save_model_full, get_available_device, save_json, load_json, \
    load_model_full
from src.conv_models import CNNVAE, TwoStageCNNVAECLF
from src.train_eval import predict_model, train_eval_loops
from src.datasets import TCRSpecificDataset
from src.samplers import GroupClassBatchSampler, MinorityClassSampler
from src.metrics import CombinedVAELoss, get_metrics
from sklearn.metrics import roc_auc_score, precision_score
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Script to resume training and evaluate a multimodal VAE model' \
                                                 'Most parameters / arguments are set to None and will only be used if explicitely set, ' \
                                                 'otherwise it will use the ones loaded from the run_parameters.' \
                                                 'Dataset parameters and models parameters are removed ; ' \
                                                 'KLD Warm-up will resume from the epoch counter')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
                        help='Specify a device (cpu, cuda:0, cuda:1)')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default=None,
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default=None, help='Additional output ¡name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default=None, help='Additional output directory')
    """
    Models args 
    """
    parser.add_argument('-model_folder', dest='model_folder', type=str, required=True, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory and load the "best" checkpoint' \
                             'Unless another -pt_file is provided')
    parser.add_argument('-model_pt', dest='model_pt', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-model_json', dest='model_json', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    """
    Training hyperparameters & args ; DEFINE MOST AS NONE TO USE THOSE FROM RUN-PARAMS, OVERRIDE IF PROVIDED
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=None, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=None, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=None, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=None, required=True,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=1,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld', '--weight_kld', dest='weight_kld', type=float, default=1e-2,
                        help='Which weight to use for the KLD term in the loss')
    parser.add_argument('-wu', '--warm_up', dest='warm_up', type=int, default=150,
                        help='Whether to do a warm-up period for the loss (without the KLD term). ' \
                             'Default = 10. Set to 0 if you want this disabled')
    parser.add_argument('-kldts', '--kld_tahn_scale', dest='kld_tahn_scale', type=float, default=0.075,
                        help='Scale for the TanH annealing in the KLD_n term')
    parser.add_argument('-fp', '--flat_phase', dest='flat_phase', default=50, type=int,
                        help='If used, the duration (in epochs) of the "flat phase" in the KLD annealing')
    parser.add_argument('-kld_dec', dest='kld_decrease', type=float, default=1e-2,
                        help="KLD_N linear decrease rate per epoch")
    parser.add_argument('-lwvae', '--weight_vae', dest='weight_vae', default=1,
                        help='Weight for the VAE term (reconstruction+KLD)')
    parser.add_argument('-lwtrp', '--weight_triplet',
                        dest='weight_triplet', type=float, default=0, help='Weight for the triplet loss term')
    parser.add_argument('-dist_type', '--dist_type', dest='dist_type', default='cosine', type=str,
                        help='Which distance metric to use ')
    parser.add_argument('-margin', dest='margin', default=0.2, type=float,
                        help='Margin for the triplet loss (Default is None and will have the default behaviour depending on the distance type)')
    parser.add_argument('-dwpos', dest='pos_dist_weight', default=1, type=float,
                        help='Weight for the positive distance part of triplet loss')
    parser.add_argument('-dwneg', dest='neg_dist_weight', default=1, type=float,
                        help='Weight for the positive distance part of triplet loss')
    parser.add_argument('-wutrp', dest='triplet_warm_up', default=None, type=int,
                        help='Warm-up period for the triplet loss')
    parser.add_argument('-cdtrp', dest='triplet_cool_down', default=None, type=int,
                        help='Cooldown period for the triplet loss')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-pepweight', dest='pep_weighted', type=str2bool, default=False,
                        help='Using per-sample (by peptide label) weighted loss')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool,
                        default=True,
                        help='Adding positional encoding to the sequence vector. False by default')
    parser.add_argument('-posweight', '--positional_weighting', dest='positional_weighting', type=str2bool,
                        default=True,
                        help='Whether to use positional weighting in reconstruction loss to prioritize CDR3 chains.')
    parser.add_argument('-minority_sampler', dest='minority_sampler', default=True, type=str2bool,
                        help='Whether to use a custom batch sampler to handle minority classes')
    parser.add_argument('-minority_count', dest='minority_count', default=50, type=int,
                        help='Counts to consider a class a minority classes')
    # TODO: TBD what to do with these!
    """
    TODO: Misc. 
    """
    # These two arguments are to be phased out or re-used, in the case of fold.
    # For now, for the exercise, I will do KCV and try to see if there is any robustsness across folds in the VAE
    # later on, it makes no sense to concatenate the latent dimensions so we need to figure something else out.
    # parser.add_argument('-s', '--split', dest='split', required=False, type=int,
    #                     default=5, help='How to split the train/test data (test size=1/X)')
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()


def load_previous_run(args, device) -> (Union[CNNVAE, TwoStageCNNVAECLF], Dict, Dict):
    if args['model_folder'] is not None:
        # model_folder
        # model_pt
        # model_json
        curves = list(filter(lambda x: 'loss' in x or 'metric' in x,
                             glob.glob(f"{args['model_folder']}/*.pkl")))
        # Reading old dict curves
        dict_curves = {}
        for k in ['train_losses', 'train_metrics', 'valid_losses', 'valid_metrics']:
            dict_curves[k] = pkl_load(next(filter(lambda x: k in x, curves)))
        run_params = load_json(glob.glob(f"{args['model_folder']}/*run_param*.json")[0])
        checkpoint_file = next(
            filter(lambda x: 'best' in x.lower() and 'checkpoint' in x and 'interval' not in x and 'last' not in x,
                   glob.glob(f'{args["model_folder"]}/*.pt')))
        checkpoint_file = checkpoint_file if args['model_pt'] is None else args['model_pt']
        model_json = next(filter(lambda x: 'checkpoint' in x,
                                 glob.glob(f'{args["model_folder"]}/*.json'))) if args['model_json'] is None else args[
            'model_json']
        try:
            model, model_params = load_model_full(checkpoint_file, model_json,
                                                  return_json=True, verbose=True,
                                                  return_best_dict=True, map_location=device)
        except:
            print(args['model_folder'], '\n', os.listdir(args['model_folder']))
            raise ValueError(f'\n\n\nCouldn\'t load your files!! at {args["model_folder"]}\n\n\n')
    else:
        model, model_params = load_model_full(args['pt_file'], args['json_file'],
                                              return_json=True, verbose=True,
                                              return_best_dict=True, map_location=device)
        dict_curves = None
    # update params to disable warmup KLD
    for k in run_params:
        if 'warm_up' in k:
            run_params[k] = 0
    best_dict = model_params.pop('best')
    filter_key = next(filter(lambda x: 'epoch' in x, best_dict.keys()))
    restart_epoch = best_dict[filter_key]
    # override device setting
    run_params['device'] = device
    run_params['restart_epoch'] = restart_epoch if 'last_epoch' not in checkpoint_file else run_params['n_epochs']
    # Extract the VAE part only if it's a two-stage:
    if hasattr(model, 'vae'):
        model = model.vae
    model.train()

    return model, model_params, run_params, dict_curves


def update_loss_metric_curves(dict_curves, train_losses_dict, train_metrics_dict, valid_losses_dict,
                              valid_metrics_dict, restart_epoch):
    # Do a try/except here because I really couldn't care less if this broke and I need the full script to run
    try:
        if dict_curves is not None:
            for new_dict, (key_old, old_dict) in zip([train_losses_dict, train_metrics_dict,
                                                      valid_losses_dict, valid_metrics_dict],
                                                     dict_curves.items()):
                # Here, dict_curves contains the previous curves (from the old training)
                # Each of dict_curves' items is a dict (i.e. train_losses_dict, train_metrics_dict, etc.)
                # Meaning dict_old should have the same keys as curve
                # So I want to concatenate the lists (add the old list first then extend the new list
                for key_lm in new_dict:
                    filter_cdt = not (
                                'precision' in key_lm or 'auc_01' in key_lm or key_lm == 'train_accuracy' or key_lm == 'valid_accuracy')
                    if key_lm in old_dict and filter_cdt:
                        concat = old_dict[key_lm][:restart_epoch] + new_dict[key_lm]
                        new_dict[key_lm] = concat

    except:
        print('Couldn\'t append old curves')
        return train_losses_dict, train_metrics_dict, valid_losses_dict, valid_metrics_dict

    return train_losses_dict, train_metrics_dict, valid_losses_dict, valid_metrics_dict


def main():
    start = dt.now()
    args = vars(args_parser())
    if torch.cuda.is_available() and args['cuda']:
        device = get_available_device()
    else:
        device = torch.device('cpu')

    # load stuff from the folder
    model, model_params, run_params, dict_curves = load_previous_run(args, device)
    model.to(device)
    # Update args so I can just copy-paste my old code
    for k, v in run_params.items():
        if k in args and args[k] is not None:
            continue
        else:
            args[k] = v

    if args['device'] is not None:
        device = args['device']
    print("Using : {}".format(device))
    seed = args['seed'] if args['fold'] is None else args['fold']
    torch.manual_seed(seed)
    df = pd.read_csv(args['file'])
    if args['debug']:
        df = df.sample(frac=0.15)

    fold = args["fold"]
    train_df = df.query('partition!=@fold')
    valid_df = df.query('partition==@fold')
    # TODO: get rid of this bad hardcoded behaviour for AA_dim ; Let's see if we end up using Xs
    args['aa_dim'] = 20
    # if 'log_wandb' in args and args['log_wandb']:
    #     wandb.login()
    # File-saving stuff
    unique_filename, kf, rid, connector = make_filename(args)

    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    outdir = '../output/'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # Do the same as normal script but without model_params because it's already loaded

    model_class = model.__class__.__name__
    dataset_keys = get_class_initcode_keys(TCRSpecificDataset, args)
    loss_keys = get_class_initcode_keys(CombinedVAELoss, args)
    # Get params from args
    dataset_params = {k: args[k] for k in dataset_keys}
    dataset_params['conv']=True
    loss_params = {k: args[k] for k in loss_keys}
    # TODO IMPORTANT REFACTOR;
    #   FOR NOW : Manually set triplet weight to 0 to avoid refining on covid/healthy triplet
    loss_params['weight_triplet'] = 0
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    train_dataset = TCRSpecificDataset(train_df, **dataset_params)
    valid_dataset = TCRSpecificDataset(valid_df, **dataset_params)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory

    fold_filename = f'kcv_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_{fold_filename}.pt'
    # instantiate objects
    torch.manual_seed(args["fold"])
    # TODO: This behaviour might bite me in the ass later
    criterion = CombinedVAELoss(**loss_params)
    criterion.to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), **optim_params)

    model.set_counter(run_params['restart_epoch'])
    criterion.set_counter(run_params['restart_epoch'])
    train_dataset.set_counter(run_params['restart_epoch'])
    valid_dataset.set_counter(run_params['restart_epoch'])
    save_json(args, f'run_parameters_{unique_filename}.json', outdir)

    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)
    # Running a test here on the valid loader to see how bad reconstruction is :
    model.eval()
    predict_model(model, valid_dataset, valid_loader).to_csv(f'{outdir}/valid_pre_retrain_results.csv')
    model.train()

    print(f'Restarting training model from epoch {args["restart_epoch"]}.')

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_metrics = train_eval_loops(args['n_epochs'], args['tolerance'], model,
                                                                       criterion, optimizer, train_loader, valid_loader,
                                                                       checkpoint_filename, outdir)
    best_epoch = best_epoch + run_params['restart_epoch']
    # Convert list of dicts to dicts of lists
    train_losses_dict = get_dict_of_lists(train_losses,
                                          'train')
    train_metrics_dict = get_dict_of_lists(train_metrics,
                                           'train')
    valid_losses_dict = get_dict_of_lists(valid_losses,
                                          'valid')
    valid_metrics_dict = get_dict_of_lists(valid_metrics,
                                           'valid')

    train_losses_dict, train_metrics_dict, valid_losses_dict, valid_metrics_dict = update_loss_metric_curves(
        dict_curves, train_losses_dict, train_metrics_dict, valid_losses_dict, valid_metrics_dict, run_params['restart_epoch'])
    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}
    keys_to_remove = []

    for k in accs_dict:
        if 'precision' in k or 'auc_01' in k or k == 'train_accuracy' or k == 'valid_accuracy':
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del accs_dict[k]
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Fold: {args["fold"]}\n')
        file.write(f"Best valid epoch: {best_epoch}\n")
        for k, v in best_val_loss.items():
            file.write(f'{k}:\t{v}\n')
        for k, v in best_val_metrics.items():
            file.write(f'{k}:\t{v}\n')

    pkl_dump(train_losses_dict, f'{outdir}/train_losses_{fold_filename}.pkl')
    pkl_dump(valid_losses_dict, f'{outdir}/valid_losses_{fold_filename}.pkl')
    pkl_dump(train_metrics_dict, f'{outdir}/train_metrics_{fold_filename}.pkl')
    pkl_dump(valid_metrics_dict, f'{outdir}/valid_metrics_{fold_filename}.pkl')
    # Here plot the whole thing disable warm-up printing; include the whole thing
    plot_vae_loss_accs(losses_dict, accs_dict, unique_filename, outdir,
                       dpi=120, palette='gnuplot2_r', warm_up=0, restart=run_params['restart_epoch'])

    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=checkpoint_filename,
                            dir_path=outdir)
    valid_preds = predict_model(model, valid_dataset, valid_loader)
    valid_preds['fold'] = args["fold"]
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{fold_filename}.csv', index=False)
    valid_seq_acc = valid_preds['seq_acc'].mean()

    print(f'Final valid reconstruction accuracy: \t{valid_seq_acc:.3%}')

    if args['test_file'] is not None:
        test_df = pd.read_csv(args['test_file'])
        test_basename = os.path.basename(args['test_file']).split(".")[0]
        test_dataset = TCRSpecificDataset(test_df, **dataset_params)
        test_loader = test_dataset.get_dataloader(batch_size=3 * args['batch_size'],
                                                  sampler=SequentialSampler)

        test_preds = predict_model(model, test_dataset, test_loader)
        test_preds['fold'] = args["fold"]
        test_preds.to_csv(f'{outdir}test_predictions_{test_basename}_{fold_filename}.csv', index=False)
        test_seq_acc = test_preds['seq_acc'].mean()

        print(f'Final test reconstruction accuracy: \t{test_seq_acc:.3%}')
    else:
        test_seq_acc = None

    save_json(args, f'run_parameters_{unique_filename}.json', outdir)
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:

        file.write(f'Fold: {kf}')
        file.write(f"Best valid seq acc: {valid_seq_acc}\n")
        if args['test_file'] is not None:
            file.write(f"Best test seq acc: {test_seq_acc}\n")
    best_dict = {'Best epoch': best_epoch}
    best_dict.update(best_val_loss)
    best_dict.update(best_val_metrics)
    # reshape dict for saving
    if 'vae_kwargs' in model_params:
        model_params = model_params['vae_kwargs']
    save_model_full(model, checkpoint_filename, outdir,
                    best_dict=best_dict, dict_kwargs=model_params)
    load_model_full(checkpoint_filename, f'{checkpoint_filename.split(".pt")[-2]}_JSON_kwargs.json',
                    outdir)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')


if __name__ == '__main__':
    main()
