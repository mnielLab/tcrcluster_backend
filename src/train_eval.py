import torch
import math
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.torch_utils import save_checkpoint, load_checkpoint
from src.utils import epoch_counter, get_loss_metric_text
from src.metrics import get_metrics, model_reconstruction_stats
from torch.nn import functional as F


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-6, name='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_best_score = np.Inf
        self.delta = delta
        self.path = f'{name}.pt'

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            # This condition works for AUC ; checks that the AUC is below the best AUC +/- some delta
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Prev best score: ({self.prev_best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.prev_best_score = score


def invoke(early_stopping, loss, model, implement=False):
    if implement:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            return True
    else:
        return False


# noinspection PyUnboundLocalVariable
def train_model_step(model, criterion, optimizer, train_loader):
    """
    Args:
        model:
        criterion:
        optimizer:
        train_loader:

    Returns:

    """
    model.train()
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss = 0, 0, 0, 0
    x_reconstructed, x_true = [], []
    for batch in train_loader:
        if (criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion, 'triplet_loss')) \
                and train_loader.dataset.__class__.__name__ == 'TCRSpecificDataset':
            x, labels = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
            pep_weights = batch[0] if train_loader.dataset.pep_weighted else None
            x_hat, mu, logvar = model(x)
            recon_loss, kld_loss, triplet_loss = criterion(x_hat, x, mu, logvar, z=mu, labels=labels,
                                                           pep_weights=pep_weights)
            loss = recon_loss + kld_loss + triplet_loss
            acum_triplet_loss += triplet_loss.item() * x.shape[0]
        else:
            x = batch.to(model.device)
            x_hat, mu, logvar = model(x)
            recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
            loss = recon_loss + kld_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x_reconstructed.append(x_hat.detach().cpu())
        x_true.append(x.detach().cpu())
        acum_total_loss += loss.item() * x.shape[0]
        acum_recon_loss += recon_loss.item() * x.shape[0]
        acum_kld_loss += kld_loss.item() * x.shape[0]

    # Normalizes to loss per batch
    acum_total_loss /= len(train_loader.dataset)
    acum_recon_loss /= len(train_loader.dataset)
    acum_kld_loss /= len(train_loader.dataset)

    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)

    train_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}
    if criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion, 'triplet_loss'):
        acum_triplet_loss /= len(train_loader.dataset)
        train_loss['triplet'] = acum_triplet_loss

    train_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    return train_loss, train_metrics


# noinspection PyUnboundLocalVariable
def eval_model_step(model, criterion, valid_loader):
    model.eval()
    # disables gradient logging
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss = 0, 0, 0, 0
    x_reconstructed, x_true = [], []
    with torch.no_grad():
        # Same workaround as above
        for batch in valid_loader:
            if (criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion,
                                                                             'triplet_loss')) and valid_loader.dataset.__class__.__name__ == 'TCRSpecificDataset':
                x, labels = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
                pep_weights = batch[0] if valid_loader.dataset.pep_weighted else None
                x_hat, mu, logvar = model(x)
                # Model is already in eval mode here
                recon_loss, kld_loss, triplet_loss = criterion(x_hat, x, mu, logvar, z=mu, labels=labels,
                                                               pep_weights=pep_weights)
                loss = recon_loss + kld_loss + triplet_loss
                acum_triplet_loss += triplet_loss.item() * x.shape[0]
            else:
                x = batch.to(model.device)
                x_hat, mu, logvar = model(x)
                recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
                loss = recon_loss + kld_loss
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            acum_total_loss += loss.item() * x.shape[0]
            acum_recon_loss += recon_loss.item() * x.shape[0]
            acum_kld_loss += kld_loss.item() * x.shape[0]
    # Normalizes to loss per batch
    acum_total_loss /= len(valid_loader.dataset)
    acum_recon_loss /= len(valid_loader.dataset)
    acum_kld_loss /= len(valid_loader.dataset)
    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)
    valid_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}

    # Only save and normalize the triplet loss if the criterion has that attribute
    if criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion, 'triplet_loss'):
        acum_triplet_loss /= len(valid_loader.dataset)
        valid_loss['triplet'] = acum_triplet_loss
    valid_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    return valid_loss, valid_metrics


def predict_model(model, dataset,
                  dataloader: torch.utils.data.DataLoader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'
    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'
    model.eval()
    df = dataset.df.reset_index(drop=True).copy()
    x_reconstructed, x_true, z_latent = [], [], []
    with torch.no_grad():
        # Same workaround as above
        for batch in dataloader:
            if dataloader.dataset.__class__.__name__ == 'TCRSpecificDataset':
                # pop(-1) works here because we don't care about potential pep weights (only used for loss)
                x, labels = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
                x_hat, mu, logvar = model(x)
                # Model is already in eval mode here ; Here, z shuold just be mu
                z = model.embed(x)
            else:
                x = batch.to(model.device)
                x_hat, _, _ = model(x)
                z = model.embed(x)
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            z_latent.append(z.detach().cpu())

    x_reconstructed = torch.cat(x_reconstructed)
    x_true = torch.cat(x_true)
    metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=True, modality_mask=None)

    # In theory we don't care about the positional encoding reconstruction, I assume it's very good
    x_seq_recon, _ = model.slice_x(x_reconstructed)
    x_seq_true, _ = model.slice_x(x_true)

    df['seq_acc'] = metrics['seq_accuracy']
    seq_hat_reconstructed = model.recover_sequences_blosum(x_seq_recon)
    seq_true_reconstructed = model.recover_sequences_blosum(x_seq_true)
    df['hat_reconstructed'] = seq_hat_reconstructed
    df['true_reconstructed'] = seq_true_reconstructed
    df['n_errors_seq'] = df.apply(
        lambda x: sum([c1 != c2 for c1, c2 in zip(x['hat_reconstructed'], x['true_reconstructed'])]), axis=1)

    z_latent = torch.cat(z_latent).detach()
    df[[f'z_{i}' for i in range(z_latent.shape[1])]] = z_latent
    return df


def train_eval_loops(n_epochs, tolerance, model, criterion, optimizer,
                     train_loader, valid_loader, checkpoint_filename, outdir):
    """ Trains and validates a model over n_epochs, then reloads the best checkpoint

    Args:
        n_epochs:
        tolerance:
        model:
        criterion:
        optimizer:
        train_loader:
        valid_loader:
        checkpoint_filename:
        outdir:

    Returns:
        model
        train_metrics
        valid_metrics
        train_losses
        valid_losses
        best_epoch
        best_val_loss
        best_val_auc
    """
    if not checkpoint_filename.endswith('.pt'):
        checkpoint_filename = checkpoint_filename + '.pt'
    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_seq_reconstruction, best_pos_reconstruction, best_epoch = 1000, 0., 0., 1
    best_val_losses, best_val_metrics = {}, {}
    # To normalize the mean accuracy thing depending on the amount of different metrics we are using
    # Only consider the sequence reconstruction, but report the positional enc reconstruction in the prints and saves
    # Because positional encoding reconstruction _should_ be trivial
    early_intervals = np.arange(0, 4500, 500)
    best_dict = {}
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_model_step(model, criterion, valid_loader)
        epoch_counter(model, criterion)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs:
            text = get_loss_metric_text(e, train_loss, valid_loss, train_metric, valid_metric)
            tqdm.write(text)

        # Maybe here we can separate the reconstruction accuracy because sequence is more important than positional for ex
        seq_accuracy = valid_metric['seq_accuracy']
        pos_accuracy = valid_metric.get('pos_accuracy', -1)

        if e > 1 and ((valid_loss[
                           "total"] <= best_val_loss + tolerance and seq_accuracy > best_seq_reconstruction) or seq_accuracy > best_seq_reconstruction):
            # Getting the individual components for asserts
            best_epoch = e
            best_val_loss = valid_loss['total']
            best_seq_reconstruction = seq_accuracy
            best_pos_reconstruction = pos_accuracy
            # Saving the actual dictionaries for logging purposes
            best_val_losses = valid_loss
            best_val_metrics = valid_metric

            best_dict = {'Best epoch': best_epoch}
            best_dict.update(valid_loss)
            best_dict.update(valid_metric)
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)

        # Adding a new thing where we log the model every 10% of the epochs, could make it easier to re-train to a certain point ??
        if e % math.ceil(0.1 * n_epochs) == 0 or e == n_epochs or e in early_intervals:
            fn = f'epoch_{e}_interval_' + checkpoint_filename.replace('best', '')
            savedict = {'epoch': e}
            savedict.update(valid_loss)
            savedict.update(valid_metric)
            save_checkpoint(model, filename=fn, dir_path=outdir, best_dict=savedict)

    last_filename = 'last_epoch_' + checkpoint_filename.replace('best', '')
    save_dict = {'epoch': e}
    save_dict.update(valid_loss)
    save_dict.update(valid_metric)
    save_checkpoint(model, filename=last_filename, dir_path=outdir, best_dict=save_dict)

    print(f'End of training cycles')
    print(best_dict)
    best_train_reconstruction = max([np.sum([x for x in z.values()]) for z in train_metrics])

    print(f'Best train loss:\t{min([x["total"] for x in train_losses]):.3e}'
          # f'Best train reconstruction Acc:\t{max([x["seq_accuracy"] for x in train_metrics])}')
          f'Best train reconstruction Acc:\t{best_train_reconstruction:.3%}')
    print(f'Best valid epoch: {best_epoch}')
    print(
        f'Best valid loss :\t{best_val_loss:.3e}, best valid mean reconstruction acc:\t{best_seq_reconstruction:.3%}, pos_encode reconstruction acc:\t{best_pos_reconstruction:.3%}')
    # print(f'Reloaded best model at {os.path.abspath(os.path.join(outdir, checkpoint_filename))}')
    model = load_checkpoint(model, checkpoint_filename, outdir)
    # Here for now it's not the best implementation but save the full model with a undefined json filename
    model.eval()
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics


def train_classifier_step(model, criterion, optimizer, train_loader):
    """

    Args:
        model: Should be a standard MLP (like PeptideClassifier)
        criterion: nn.BCEWithLogitsLoss(reduction='none') since we are doing the mean manually
        optimizer:
        train_loader: LatentTCRpMHCDataset's loader

    Returns:

    """
    model.train()
    acum_loss = 0
    y_scores, y_true = [], []
    for batch in train_loader:
        x, y = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
        # Here, don't set weight as None because criterion is not a custom loss class with custom behaviour
        # After popping, if dataset has the pep_weighted flag, then there should be a single element left in batch
        pep_weights = batch[0].to(model.device) if (
                train_loader.dataset.pep_weighted and len(batch) == 1) else torch.ones([len(y)]).to(model.device)
        # output here should be logits to use BCEWithLogitLoss
        output = model(x)
        # Here criterion should be with reduction='none' and then manually do the mean() because `weight` is not used in forward but init
        loss = (criterion(output, y) * pep_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Saving scores and true for aucs etc
        y_scores.append(output.detach().cpu())
        y_true.append(y.detach().cpu())
        acum_loss += loss.item() * x.shape[0]

    acum_loss /= len(train_loader.dataset)
    # Saving the scores as logits but getting the metrics using sigmoid, shouldn't change much but makes the y_pred and thresholding easier
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    train_metrics = get_metrics(y_true, F.sigmoid(y_scores), threshold=0.5, reduced=True, round_digit=5)
    return acum_loss, train_metrics


def eval_classifier_step(model, criterion, valid_loader):
    """

    Args:
        model: Should be a standard MLP (like PeptideClassifier)
        criterion: nn.BCEWithLogitsLoss(reduction='none') since we are doing the mean manually
        valid_loader: LatentTCRpMHCDataset's loader

    Returns:

    """
    model.eval()
    acum_loss = 0
    y_scores, y_true = [], []
    with torch.no_grad():
        for batch in valid_loader:
            x, y = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
            # Here, don't set weight as None because criterion is not a custom loss class with custom behaviour
            # After popping, if dataset has the pep_weighted flag, then there should be a single element left in batch
            pep_weights = batch[0].to(model.device) if (
                    valid_loader.dataset.pep_weighted and len(batch) == 1) else torch.ones([len(y)]).to(
                model.device)
            # output here should be logits to use BCEWithLogitLoss
            output = model(x)
            # Here criterion should be with reduction='none' and then manually do the mean() because `weight` is not used in forward but init
            loss = (criterion(output, y) * pep_weights).mean()
            # Saving scores and true for aucs etc
            y_scores.append(output.detach().cpu())
            y_true.append(y.detach().cpu())
            acum_loss += loss.item() * x.shape[0]

    acum_loss /= len(valid_loader.dataset)
    # Saving the scores as logits but getting the metrics using sigmoid, shouldn't change much but makes the y_pred and thresholding easier
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    valid_metrics = get_metrics(y_true, F.sigmoid(y_scores), threshold=0.5, reduced=True, round_digit=5)
    return acum_loss, valid_metrics


def predict_classifier(model, dataset, dataloader):
    assert type(
        dataloader.sampler) == torch.utils.data.SequentialSampler, 'Test/Valid loader MUST use SequentialSampler!'
    df = dataset.df.reset_index(drop=True).copy()
    y_true, y_logit = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Here, again, no need to check for pep_weights because there is no loss in `predict`
            x, y = batch.pop(0).to(model.device), batch.pop(-1).to(model.device)
            output = model(x)
            y_true.append(y.detach().cpu())
            y_logit.append(output.detach().cpu())

    y_logit, y_true = torch.cat(y_logit), torch.cat(y_true)
    y_probs = F.sigmoid(y_logit)

    df['pred_logit'] = y_logit
    df['pred_prob'] = y_probs

    return df


def classifier_train_eval_loops(n_epochs, tolerance, model, criterion, optimizer, train_loader, valid_loader,
                                checkpoint_filename, outdir):
    if not checkpoint_filename.endswith('.pt'):
        checkpoint_filename = checkpoint_filename + '.pt'
    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_auc, best_val_auc01, best_epoch = 1000, 0.5, 0.5, 1
    best_val_metrics = {}
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_classifier_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_classifier_step(model, criterion, valid_loader)
        epoch_counter(model, criterion)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs:
            text = get_loss_metric_text(e, train_loss, valid_loss, train_metric, valid_metric)
            tqdm.write(text)

        if e % math.ceil(0.1 * n_epochs)==0 or e == 1:
            interval_fn = f'epoch_{e}_interval' + checkpoint_filename.replace('best', '')
            interval_save_dict = {'epoch': e, 'valid_loss':valid_loss}
            interval_save_dict.update(valid_metric)

            save_checkpoint(model, filename=interval_fn, dir_path=outdir, best_dict=interval_save_dict)

        loss_cdt = valid_loss <= best_val_loss + tolerance
        auc_cdt = valid_metric['auc'] > best_val_auc
        auc01_cdt = valid_metric['auc_01'] > best_val_auc01
        if loss_cdt and (auc_cdt or auc01_cdt):
            best_dict = {'epoch':e, 'loss':valid_loss}
            best_epoch = e
            best_val_loss = valid_loss
            best_val_auc = valid_metric['auc']
            best_val_auc01 = valid_metric['auc_01']
            best_val_metrics = valid_metric
            # Saving best dict for logging purposes
            best_dict['epoch'] = best_epoch
            best_dict['loss'] = valid_loss
            best_dict.update(valid_metric)
            # Saving model
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)
        elif auc_cdt or auc01_cdt :
            auc_save_dict = {'epoch': e, 'loss': valid_loss}
            auc_save_dict.update(valid_metric)
            if auc_cdt and auc01_cdt:
                alternate_fn = checkpoint_filename.replace('best','bestOverall')
            else:
                if auc_cdt:
                    alternate_fn = checkpoint_filename.replace('best', 'bestAUC')
                elif auc01_cdt:
                    alternate_fn = checkpoint_filename.replace('best','bestAUC01')
            save_checkpoint(model, filename=alternate_fn, dir_path=outdir, best_dict=auc_save_dict)


    last_filename = 'last_epoch_' + checkpoint_filename.replace('best','')
    save_dict={'epoch':e, 'valid_loss':valid_loss}
    save_dict.update(valid_metric)
    save_checkpoint(model, filename=last_filename, dir_path=outdir, best_dict=save_dict)

    print(f'End of training cycles')
    print(best_dict)

    print(f'Best train loss:\t{min(train_losses):.3e}, at epoch = {train_losses.index(min(train_losses))}'
          f'\tTrain AUC, AUC01:\t{train_metrics[train_losses.index(min(train_losses))]["auc"]:.3%},\t{train_metrics[train_losses.index(min(train_losses))]["auc_01"]}')
    print(f'Best valid epoch: {best_epoch}')
    print(
        f'Best valid loss :\t{best_val_loss:.3e}, best valid AUC, AUC01:\t{best_val_auc:.3%},\t{best_val_auc01:.3%}')
    model = load_checkpoint(model, checkpoint_filename, outdir)
    model.eval()

    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_loss, best_val_metrics


def train_twostage_step(model, criterion, optimizer, train_loader):
    model.train()
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss, acum_clf_loss = 0, 0, 0, 0, 0
    x_reconstructed, x_true, y_score, y_true = [], [], [], []
    # Here, didn't check for class of dataset whether it is TCRSpecificDataset or issubclass of TCRSpecificDataset,
    # Assume it returns the x, x_pep, label, binder in a batch by default
    # TODO: Maybe a bad way to unpack a batch to get the weights??
    for batch in train_loader:
        if train_loader.dataset.pep_weighted:
            x, x_pep, label, binder, pep_weights = batch
        else:
            x, x_pep, label, binder = batch
            pep_weights = torch.ones([len(x)])

        x, x_pep, label, binder, pep_weights = x.to(model.device), x_pep.to(model.device), label.to(
            model.device), binder.to(
            model.device), pep_weights.to(model.device)
        x_hat, mu, logvar, x_out = model(x, x_pep)

        recon_loss, kld_loss, triplet_loss, clf_loss = criterion(x_hat, x, mu, logvar, mu, label, x_out, binder,
                                                                 pep_weights=pep_weights)
        loss = recon_loss + kld_loss + triplet_loss + clf_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accumulate losses and save preds for metrics
        acum_total_loss += loss.item() * x.shape[0]
        acum_recon_loss += recon_loss.item() * x.shape[0]
        acum_kld_loss += kld_loss.item() * x.shape[0]
        acum_triplet_loss += triplet_loss.item() * x.shape[0]
        acum_clf_loss += clf_loss.item() * x.shape[0]
        x_reconstructed.append(x_hat.detach().cpu())
        x_true.append(x.detach().cpu())
        y_score.append(x_out.detach().cpu())
        y_true.append(binder.detach().cpu())

    y_score, y_true = [x for x in y_score if x is not None], [x for x in y_true if x is not None]
    # Normalize loss per batch
    acum_total_loss /= len(train_loader.dataset)
    acum_recon_loss /= len(train_loader.dataset)
    acum_kld_loss /= len(train_loader.dataset)
    acum_triplet_loss /= len(train_loader.dataset)
    acum_clf_loss /= len(train_loader.dataset)

    # Cat outputs to compute metrics
    x_reconstructed, x_true, y_score, y_true = torch.cat(x_reconstructed), torch.cat(x_true), torch.cat(
        y_score), torch.cat(y_true)
    recon_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    pred_metrics = get_metrics(y_true, F.sigmoid(y_score), threshold=0.5, reduced=True, round_digit=5)
    train_metrics = {**recon_metrics, **pred_metrics}
    train_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss,
                  'triplet': acum_triplet_loss, 'BCE': acum_clf_loss}
    return train_loss, train_metrics


def eval_twostage_step(model, criterion, valid_loader):
    model.eval()
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss, acum_clf_loss = 0, 0, 0, 0, 0
    x_reconstructed, x_true, y_score, y_true = [], [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            if valid_loader.dataset.pep_weighted:
                x, x_pep, label, binder, pep_weights = batch
            else:
                x, x_pep, label, binder = batch
                pep_weights = torch.ones([len(x)])

            x, x_pep, label, binder, pep_weights = x.to(model.device), x_pep.to(model.device), label.to(
                model.device), binder.to(model.device), pep_weights.to(model.device)

            # TODO : same as for train with z / mu
            x_hat, mu, logvar, x_out = model(x, x_pep)
            # TODO: here, give "mu" as Z
            recon_loss, kld_loss, triplet_loss, clf_loss = criterion(x_hat, x, mu, logvar, mu, label, x_out, binder,
                                                                     pep_weights=pep_weights)
            loss = recon_loss + kld_loss + triplet_loss + clf_loss

            # accumulate losses and save preds for metrics
            acum_total_loss += loss.item() * x.shape[0]
            acum_recon_loss += recon_loss.item() * x.shape[0]
            acum_kld_loss += kld_loss.item() * x.shape[0]
            acum_triplet_loss += triplet_loss.item() * x.shape[0]
            acum_clf_loss += clf_loss.item() * x.shape[0]
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            y_score.append(x_out.detach().cpu())
            y_true.append(binder.detach().cpu())

    # Drop the returned None values from clf
    y_score, y_true = [x for x in y_score if x is not None], [x for x in y_true if x is not None]
    # Normalize loss per batch
    acum_total_loss /= len(valid_loader.dataset)
    acum_recon_loss /= len(valid_loader.dataset)
    acum_kld_loss /= len(valid_loader.dataset)
    acum_triplet_loss /= len(valid_loader.dataset)
    acum_clf_loss /= len(valid_loader.dataset)

    # Cat outputs to compute metrics
    x_reconstructed, x_true, y_score, y_true = torch.cat(x_reconstructed), torch.cat(x_true), torch.cat(
        y_score), torch.cat(y_true)
    recon_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    pred_metrics = get_metrics(y_true, F.sigmoid(y_score), threshold=0.5, reduced=True, round_digit=5)
    valid_metrics = {**recon_metrics, **pred_metrics}
    valid_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss,
                  'triplet': acum_triplet_loss, 'BCE': acum_clf_loss}
    return valid_loss, valid_metrics


def predict_twostage(model, dataset, dataloader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'
    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'

    df = dataset.df.reset_index(drop=True).copy()
    x_reconstructed, x_true, z_latent, y_score, y_true = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, x_pep, label, binder = batch[0], batch[1], batch[2], batch[3]
            x, x_pep, label, binder = x.to(model.device), x_pep.to(model.device), label.to(model.device), binder.to(
                model.device)
            # TODO : same as for train with z / mu
            x_hat, mu, logvar, x_out = model(x, x_pep)
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            z_latent.append(mu.detach().cpu())
            y_score.append(x_out.detach().cpu())
            y_true.append(binder.detach().cpu())

    # Cat outputs to compute metrics
    x_reconstructed, x_true, y_score, y_true, z_latent = torch.cat(x_reconstructed), torch.cat(x_true), torch.cat(
        y_score), torch.cat(y_true), torch.cat(z_latent)

    # Reconstruction metrics
    metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=True)
    df['seq_acc'] = metrics['seq_accuracy']

    # TODO : Here, add something regarding handling the positional encoding vector
    x_seq_recon, _ = model.slice_x(x_reconstructed)
    x_seq_true, _ = model.slice_x(x_true)
    # Reconstructed sequences
    seq_hat_reconstructed = model.recover_sequences_blosum(x_seq_recon)
    seq_true_reconstructed = model.recover_sequences_blosum(x_seq_true)
    df['hat_reconstructed'] = seq_hat_reconstructed
    df['true_reconstructed'] = seq_true_reconstructed
    df['n_errors_seq'] = df.apply(
        lambda x: sum([c1 != c2 for c1, c2 in zip(x['hat_reconstructed'], x['true_reconstructed'])]), axis=1)

    # Saving latent representations
    df[[f'z_{i}' for i in range(z_latent.shape[1])]] = z_latent

    # Saving MLP prediction scores
    df['pred_logit'] = y_score
    df['pred_prob'] = F.sigmoid(y_score)
    pred_metrics = get_metrics(y_true, F.sigmoid(y_score))
    # print(f'Mean reconstruction accuracy: {metrics["seq_accuracy"].mean()}')
    print('MLP metrics:', '\t'.join(f'{k}: {v:.3f}' for k, v in pred_metrics.items()))
    return df


def twostage_train_eval_loops(n_epochs, tolerance, model, criterion, optimizer,
                              train_loader, valid_loader, checkpoint_filename, outdir):
    """
    Trains a bi-modal VAE-MLP model,
    then reloads the best checkpoint based on aggregate metrics (losses, reconstruction, prediction aucs?)
    Args:
        n_epochs:
        tolerance:
        model:
        criterion:
        optimizer:
        train_loader:
        valid_loader:
        checkpoint_filename:
        outdir:

    Returns:

    """
    print(f'Staring {n_epochs} training cycles')
    # Pre-save model at epoch 0
    save_checkpoint(model, checkpoint_filename, dir_path=outdir, verbose=False, best_dict=None)
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_reconstruction, best_epoch, best_val_auc, best_agg_metric = 1000, 0., 1, 0.5, 0.5
    # "best_val_losses" is a dictionary of all the various split losses
    best_val_losses, best_val_metrics, best_dict = {}, {}, {}
    intervals = (np.arange(0.2, 1, 0.2) * n_epochs).astype(int)

    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_twostage_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_twostage_step(model, criterion, valid_loader)
        epoch_counter(model, criterion)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # Periodic prints for tracking
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs:
            text = get_loss_metric_text(e, train_loss, valid_loss, train_metric, valid_metric)
            tqdm.write(text)
        # Saving every 10%
        if e % math.ceil(0.1 * n_epochs) == 0 or e == n_epochs:
            fn = f'epoch_{e}_interval_' + checkpoint_filename.replace('best', '')
            savedict = {'epoch': e}
            savedict.update(valid_loss)
            savedict.update(valid_metric)
            save_checkpoint(model, filename=fn, dir_path=outdir, best_dict=savedict)

        # Saving the best model out of 3 conditions (Total loss, VAE reconstruction, MLP AUC)
        loss_condition = valid_loss['total'] <= best_val_loss + tolerance
        recon_condition = valid_metric['seq_accuracy'] >= best_val_reconstruction - tolerance
        clf_condition = valid_metric['auc'] >= best_val_auc - tolerance
        if e > 1 and loss_condition and recon_condition and clf_condition:
            best_epoch = e
            best_val_loss = valid_loss['total']
            best_val_reconstruction = valid_metric['seq_accuracy']
            best_val_auc = valid_metric['auc']
            # Saving the actual dictionaries for logging purposes
            best_val_losses = valid_loss
            best_val_metrics = valid_metric

            best_dict = {'Best epoch': best_epoch, 'Best val loss': best_val_loss}
            best_dict.update(valid_loss)
            best_dict.update(valid_metric)
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)


    last_filename = 'last_epoch_' + checkpoint_filename.replace('best', '')
    save_dict = {'epoch': e}
    save_dict.update(valid_loss)
    save_dict.update(valid_metric)
    save_checkpoint(model, filename=last_filename, dir_path=outdir, best_dict=save_dict)

    print(f'End of training cycles')
    print(best_dict)
    model = load_checkpoint(model, checkpoint_filename, outdir)
    model.eval()
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics

