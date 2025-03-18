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
    get_dict_of_lists, make_filename, get_class_initcode_keys
from src.torch_utils import load_checkpoint, save_model_full, get_available_device, save_json
from src.models import FullTCRVAE
from src.train_eval import predict_model, train_eval_loops
from src.datasets import TCRSpecificDataset
from src.samplers import GroupClassBatchSampler, MinorityClassSampler
from src.metrics import CombinedVAELoss, get_metrics
from sklearn.metrics import roc_auc_score, precision_score
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
                        help='Specify a device (cpu, cuda:0, cuda:1)')

    parser.add_argument('-logwb', '--log_wandb', dest='log_wandb', required=False, default=False,
                        type=str2bool, help='Whether to log a run using WandB. False by default')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/230921_nettcr_immrepnegs_noswap.csv',
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
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
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool, default=False,
                        help='Adding positional encoding to the sequence vector. False by default')
    parser.add_argument('-posweight', '--positional_weighting', dest='positional_weighting', type=str2bool, default=False,
                        help='Whether to use positional weighting in reconstruction loss to prioritize CDR3 chains.')
    parser.add_argument('-minority_sampler', dest='minority_sampler', default=False, type=str2bool,
                        help='Whether to use a custom batch sampler to handle minority classes')
    parser.add_argument('-minority_count', dest='minority_count', default=50, type=int,
                        help='Counts to consider a class a minority classes')

    """
    Models args 
    """
    parser.add_argument('-nh', '--hidden_dim', dest='hidden_dim', type=int, default=256,
                        help='Number of hidden units in the VAE. Default = 256')
    parser.add_argument('-nl', '--latent_dim', dest='latent_dim', type=int, default=128,
                        help='Size of the latent dimension. Default = 128')
    parser.add_argument('-act', '--activation', dest='activation', type=str, default='selu',
                        help='Which activation to use. Will map the correct nn.Module for the following keys:' \
                             '[selu, relu, leakyrelu, elu]')
    parser.add_argument('-ale', dest='add_layer_encoder', default=False, type=str2bool,
                        help='Add an extra encoder layer')
    parser.add_argument('-ald', dest='add_layer_decoder', default=False, type=str2bool,
                        help='Add an extra decoder layer')
    parser.add_argument('-bn' ,dest='batchnorm', default=False, type=str2bool,
                        help='Add Batchnorm to the layers')
    parser.add_argument('-ob', dest='old_behaviour', default=True, type=str2bool,
                        help='switchcase to have old behaviour to ahve models we can load/save')


    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=5e-4, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=1024, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=20000, required=False,
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
                        dest='weight_triplet', type=float, default=3, help='Weight for the triplet loss term')
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
    parser.add_argument('-leave_pep_out', dest='leave_pep_out', type=str, default=None,
                        help='Adding a leave-pep-out behaviour for triplet loss')
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


def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    seed = args['seed'] if args['fold'] is None else args['fold']
    if torch.cuda.is_available() and args['cuda']:
        device = get_available_device()
    else:
        device = torch.device('cpu')

    if args['device'] is not None:
        device = args['device']

    print("Using : {}".format(device))
    torch.manual_seed(seed)
    # Convert the activation string codes to their nn counterparts
    args['activation'] = {'selu': nn.SELU(), 'relu': nn.ReLU(),
                          'leakyrelu': nn.LeakyReLU(), 'elu': nn.ELU()}[args['activation']]
    # Loading data and getting train/valid
    # TODO: Restore valid kcv behaviour // or not
    df = pd.read_csv(args['file'])
    dfname = args['file'].split('/')[-1].split('.')[0]
    train_df = df.query(f'partition!={args["fold"]}')
    valid_df = df.query(f'partition=={args["fold"]}')
    # TODO: get rid of this bad hardcoded behaviour for AA_dim ; Let's see if we end up using Xs
    args['aa_dim'] = 20
    # if args['log_wandb']:
    #     wandb.login()

    # File-saving stuff
    unique_filename, kf, rid, connector = make_filename(args)
    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # Def params so it's tidy

    # Maybe this is better? Defining the various keys using the constructor's init arguments
    # model_init_code = FullTCRVAE.__init__.__code__
    # model_init_code = FullTCRVAE.__init__.__code__.co_varnames[1:model_init_code.co_argcount]
    # model_keys = [x for x in args.keys() if x in model_init_code]
    # dataset_init_code = TCRSpecificDataset.__init__.__code__
    # dataset_init_code = TCRSpecificDataset.__init__.__code__.co_varnames[1:dataset_init_code.co_argcount]
    # dataset_keys = [x for x in args.keys() if x in dataset_init_code]
    # loss_init_code = CombinedVAELoss.__init__.__code__
    # loss_init_code = CombinedVAELoss.__init__.__code__.co_varnames[1:loss_init_code.co_argcount]
    # loss_keys = [x for x in args.keys() if x in loss_init_code]

    # Def params so it's tidy
    model_keys = get_class_initcode_keys(FullTCRVAE, args)
    dataset_keys = get_class_initcode_keys(TCRSpecificDataset, args)
    loss_keys = get_class_initcode_keys(CombinedVAELoss, args)

    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    loss_params = {k: args[k] for k in loss_keys}
    # loss_params['max_len'] = sum([v for k, v in model_params.items() if 'max_len' in k])
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
    # Dump args to json for potential resume training.
    save_json(args, f'run_parameters_{unique_filename}.json', outdir)
    # Here, don't specify V and J map to use the default V/J maps loaded from src.data_processing
    train_dataset = TCRSpecificDataset(train_df, **dataset_params)
    valid_dataset = TCRSpecificDataset(valid_df, **dataset_params)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory
    train_sampler = GroupClassBatchSampler if args['minority_sampler'] else RandomSampler
    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=train_sampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)

    fold_filename = f'kcv_{dfname}_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_fold{args["fold"]:02}_{fold_filename}.pt'

    # instantiate objects
    torch.manual_seed(args["fold"])
    model = FullTCRVAE(**model_params)
    model.to(device)
    criterion = CombinedVAELoss(**loss_params)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), **optim_params)
    # Adding the wandb watch statement ; Only add them in the script so that it never interferes anywhere in train_eval
    # if args['log_wandb']:
    #     # wandb stuff
    #     wandb.init(project=unique_filename, name=f'fold_{args["fold"]:02}', config=args)
    #     wandb.watch(model, criterion=criterion, log_freq=len(train_loader))
    print(f'Main script device check : model={model.device}; crit={criterion.device}')
    model, train_metrics, valid_metrics, train_losses, valid_losses, \
        best_epoch, best_val_loss, best_val_metrics = train_eval_loops(args['n_epochs'], args['tolerance'], model,
                                                                       criterion, optimizer, train_loader, valid_loader,
                                                                       checkpoint_filename, outdir)


    # Convert list of dicts to dicts of lists
    train_losses_dict = get_dict_of_lists(train_losses,
                                          'train')
    train_metrics_dict = get_dict_of_lists(train_metrics,
                                           'train')
    valid_losses_dict = get_dict_of_lists(valid_losses,
                                          'valid')
    valid_metrics_dict = get_dict_of_lists(valid_metrics,
                                           'valid')

    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}

    # Saving text file for the run:
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
    plot_vae_loss_accs(losses_dict, accs_dict, unique_filename, outdir,
                       dpi=300, palette='gnuplot2_r', warm_up=args['warm_up'])

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

    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:

        file.write(f'Fold: {kf}')
        file.write(f"Best valid seq acc: {valid_seq_acc}\n")
        if args['test_file'] is not None:
            file.write(f"Best test seq acc: {test_seq_acc}\n")
    best_dict = {'Best epoch': best_epoch}
    best_dict.update(best_val_loss)
    best_dict.update(best_val_metrics)
    save_model_full(model, checkpoint_filename, outdir,
                    best_dict=best_dict, dict_kwargs=model_params)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
