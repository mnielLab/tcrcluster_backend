import math

import numpy as np
import torch
from overrides import override

from src.torch_utils import mask_modality, filter_modality
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, \
    precision_score, precision_recall_curve, auc, average_precision_score, recall_score, silhouette_score, \
    silhouette_samples


class LossParent(nn.Module):

    def __init__(self, debug=False):
        super(LossParent, self).__init__()
        self.positional_weights = torch.empty(1)
        self.counter = 0
        self.debug = debug
        self.device = 'cpu'

    def increment_counter(self):
        "one level of children modules ; If we get too many might need to have a recursive method somewhere"
        self.counter += 1
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'increment_counter'):
                c.increment_counter()

    def set_counter(self, counter):
        self.counter = counter
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'set_counter'):
                c.set_counter(counter)

    def to(self, device):
        super(LossParent, self).to(device)
        self.device = device
        if hasattr(self, 'positional_weights') and hasattr(self, 'positional_weighting'):
            if self.positional_weighting:
                self.positional_weights = self.positional_weights.to(device)
        for c in self.children():
            if hasattr(c, 'device') and hasattr(c, 'to'):
                c.to(device)
            if hasattr(c, 'positional_weighting') and hasattr(c, 'positional_weights'):
                if c.positional_weighting:
                    c.positional_weights = c.positional_weights.to(device)


class VAELoss(LossParent):
    """
    No fucking annealing, just some basic stuff for now
    TODO: re-do the tanh behaviour for the KLD loss
    Does the VAE part of the loss, including reconstruction error (here for the sequence and possible positional encoding vectors)
         and the KLD to control the latent dimensions

    Inherits from the LossParent class and has a built-in counter
    counter increment should be called externally, e.g. 1x per epoch, in train-eval loop functions

    **17.01.24 : warm_up counter should now be an epoch counter and not batch counter**
    --> All I need to ensure : FullTCRvae works, fulltcrvae_triplet works, bimodal works
    """

    def __init__(self, sequence_criterion=nn.MSELoss(reduction='none'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 positional_weighting=False, weight_seq=3, weight_kld=1e-2, warm_up=100, tanh_scale=0.1,
                 kld_decrease=1e-3, flat_phase=100, debug=False):
        super(VAELoss, self).__init__()
        max_len = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])
        pos_dim = sum([int(mlx) > 0 for mlx in
                       [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep]]) \
            if add_positional_encoding else 0
        # Refactoring : Do sequence_criterion with reduction='none' and manually do the mean
        self.positional_weighting = positional_weighting
        if positional_weighting:
            # take 2d weights (sum_lens, 1), and broadcast using torch.mul so it takes less memory
            # Set weights for CDR3 to 3x while cdr1,2,pep weights are at 1
            alpha_weights = torch.cat([torch.full([max_len_a1 + max_len_a2, 1], 1),
                                       torch.full([max_len_a3, 1], 3)], dim=0)
            beta_weights = torch.cat([torch.full([max_len_b1 + max_len_b2, 1], 1),
                                      torch.full([max_len_b3, 1], 3)], dim=0)
            pep_weights = torch.full([max_len_pep, 1], 1)
            self.positional_weights = torch.cat([alpha_weights, beta_weights, pep_weights], dim=0).to(self.device)
            assert self.positional_weights.shape == (max_len, 1), 'wrong shape for pos weights'
            print('Here in Vae loss pos weights init', self.positional_weights.device, self.positional_weighting)

        self.max_len = max_len
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.add_positional_encoding = add_positional_encoding
        self.norm_factor = weight_seq + weight_kld
        self.sequence_criterion = sequence_criterion
        self.weight_seq = weight_seq  # / self.norm_factor
        # KLD weight things
        self.base_weight_kld = weight_kld  # / self.norm_factor
        self.weight_kld = 0
        print('here', self.base_weight_kld, self.weight_kld)
        self.kld_warm_up = warm_up
        self.kld_tanh_scale = tanh_scale
        self.flat_phase = warm_up // 3 if flat_phase is None else flat_phase
        self.kld_decrease = kld_decrease

        self.counter = 0
        self.debug = debug
        self.kld_warm_up = warm_up
        print(f'Weights: seq, kld_base: ', self.weight_seq, self.base_weight_kld)

    def to(self, device):
        super(VAELoss, self).to(device)
        self.device = device
        if self.positional_weighting:
            self.positional_weights = self.positional_weights.to(device)

    def reconstruction_loss(self, x_hat, x):
        x_hat_seq, positional_hat = self.slice_x(x_hat)
        x_true_seq, positional_true = self.slice_x(x)
        reconstruction_loss = self.sequence_criterion(x_hat_seq, x_true_seq)
        # if positional weighting, then multiply the loss to give larger/smaller gradients w.r.t. chains and positions
        if self.positional_weighting:
            reconstruction_loss = reconstruction_loss.mul(self.positional_weights)

        # Here, take the mean before checking for positional encoding because we have un-reduced loss
        # and silhouette_scale it by the "weight_seq" (versus weight_kld)
        reconstruction_loss = self.weight_seq * reconstruction_loss.mean()

        # check if we use positional encoding, if yes we add the loss
        if self.add_positional_encoding:
            # Here use 1e-3
            positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true)
            reconstruction_loss += (1e-3 * self.weight_seq * positional_loss)

        return reconstruction_loss

    def kullback_leibler_divergence(self, mu, logvar):
        # 240426 Replaced with TanH regime
        # KLD weight regime control
        # if self.counter < self.kld_warm_up:
        #     # While in the warm-up phase, weight_kld is 0
        #     self.weight_kld = 0
        # else:
        #     # Otherwise, it starts at the base_kld weight and decreases a 1% of max weight with the epoch counter
        #     # until it reaches a minimum of the base weight / 5
        #     self.weight_kld = max(
        #         self.base_weight_kld - (0.01 * self.base_weight_kld * (self.counter - self.kld_warm_up)),
        #         self.base_weight_kld / 5)

        return self.weight_kld * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

    def slice_x(self, x):
        """
        Slices and extracts // reshapes the sequence vector
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        # TODO : Here, since it's reshaped, we can maybe put a positional_weight to give more weight on reconstruction of some chains
        sequence = x[:, 0:(self.max_len * (self.aa_dim + self.pos_dim))].view(-1, self.max_len,
                                                                              (self.aa_dim + self.pos_dim))
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def forward(self, x_hat, x, mu, logvar):
        """
        x_hat and x should be the flattened vectors, where hat is the reconstructed and x the true
        Then, self.slice_x will deal with slicing, reshaping and (if appl.) extracting the positional encoding
        KLD weight follows 2 regimes:
            1: Warm-Up period, w_kld = 0 until counter==warm_up
            2: w_kld = w_base_kld, then decreases each epoch by a set amount until it reaches a min of w_base / 5
        Args:
            x_hat:
            x:
            mu:
            logvar:

        Returns:

        """
        # Refactoring : Use the newly made self methods in forward
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        kld = self.kullback_leibler_divergence(mu, logvar)

        if self.debug:
            print('seq_loss', reconstruction_loss)
            print('kld_weight', self.weight_kld)
            print('kld_loss', kld, '\n')

        # Return them separately and sum later so that I can debug each component
        return reconstruction_loss, kld

    def set_debug(self, debug):
        self.debug = debug

    def reset_parameters(self):
        self.counter = 0
        self._kld_weight_regime()

    def _kld_weight_regime(self):
        """
        if counter <= warm_up : tanh annealing warm_up procedure
        else: Starts at 1 * base_weight then decrease 5/1000 of max weight until it reaches base_weight / 5
        Should be called at each increment counter step!
        Returns:
            -
        """
        # TanH warm-up phase
        if self.counter <= self.kld_warm_up:
            self.weight_kld = self._tanh_annealing(self.counter, self.base_weight_kld,
                                                   self.kld_tanh_scale, self.kld_warm_up, shift=None)
        # "flat" phase : No need to update
        if self.kld_warm_up < self.counter <= (self.kld_warm_up + self.flat_phase):
            self.weight_kld = self.base_weight_kld
        # Start to decrease weight once counter > warm_up+flat phase for KLD
        elif self.counter > self.kld_warm_up + self.flat_phase:
            self.weight_kld = max(
                self.base_weight_kld - (self.kld_decrease * self.base_weight_kld * (
                        self.counter - (self.kld_warm_up + self.flat_phase))),
                self.base_weight_kld / 4)

    @staticmethod
    def _tanh_annealing(counter, base_weight, scale, warm_up, shift=None):
        """
        epoch_shift sets the epoch at which weight==weight/2, should be set at warm_up//2
        Only updates self.weight_kld as current weight, doesn't return anything
        Computes a 1-shifted (y-axis) tanh and 2/3 * self.warm_up shifted (x-axis) to compute the weight
        using 1+tanh(speed * (epoch - shift))
        Args:
            base_weight: base total weight
            scale: ramp-up speed ~ range [0.05, 0.5]
            epoch_shift:

        Returns:
            weight: kld_weight_n at current epoch
        """

        shift = 2 * warm_up / 3 if shift is None else shift
        return base_weight * (
                1 + math.tanh(scale * (counter - shift))) / 2
        #
        # return self.base_weight_kld_n * (
        #         1 + math.tanh(self.kld_tanh_scale * (self.counter - 2 * self.kld_warm_up / 3))) / 2

    @override
    def increment_counter(self):
        super(VAELoss, self).increment_counter()
        self._kld_weight_regime()

    @override
    def set_counter(self, counter):
        super(VAELoss, self).set_counter(counter)
        self._kld_weight_regime()


class TripletLoss(LossParent):
    def __init__(self, dist_type='cosine', margin=None, pos_dist_weight=1, neg_dist_weight=1, weight=1,
                 warm_up=None, cool_down=None):
        super(TripletLoss, self).__init__()
        assert dist_type in ['cosine', 'l1',
                             'l2'], f'Distance type must be in ["l1", "l2", "cosine"]! Got {dist_type} instead.'
        self.distance = {'cosine': compute_cosine_distance,
                         'l2': torch.cdist,
                         'l1': torch.cdist}[dist_type]
        self.p = 1 if dist_type == 'l1' else 2 if dist_type == 'l2' else None
        if margin is None:
            margin = 0.15 if dist_type == 'cosine' else 1.0
        self.margin = margin
        self.pos_dist_weight = pos_dist_weight
        self.neg_dist_weight = neg_dist_weight
        self.activated = weight > 0
        self.weight = weight
        self.warm_up = warm_up
        self.cool_down = cool_down

    def forward(self, z, labels, **kwargs):
        if self.activated:
            weights = kwargs['pep_weights'] if (
                    'pep_weights' in kwargs and kwargs['pep_weights'] is not None) else torch.ones([len(z)])
            weights = weights.to(z.device)
            pairwise_distances = self.distance(z, z, p=self.p)
            mask_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
            mask_negative = 1 - mask_positive
            # Get the distances for each of the masks
            positive_distances = mask_positive * pairwise_distances
            negative_distances = mask_negative * pairwise_distances
            # Take the relu to encourage error above margin (i.e. threshold)
            # Here, weights should be 1 if activated, 0 if not.
            loss = torch.nn.functional.relu(
                self.pos_dist_weight * positive_distances - self.neg_dist_weight * negative_distances + self.margin) * weights
            # 240424: Fixing an error where self-self distance would lead to a loss of self.margin ; The mask now zeroes out those elements
            diag_mask = torch.ones_like(mask_positive).fill_diagonal_(0)
            loss = torch.mul(loss, diag_mask)
            loss = loss.mean()
        else:
            loss = torch.tensor(0)
        return loss

    def _triplet_weight_regime(self):
        # TODO : Maybe use the KLD regime with slightly different schedules
        # warm-up means activating triplet after `triplet_warm_up` epochs. STARTING WITHOUT TRIPLET
        wu_cdt = self.counter >= self.warm_up if self.warm_up is not None else True
        cd_cdt = self.counter <= self.cool_down if self.cool_down is not None else True
        self.activated = wu_cdt & cd_cdt & (self.weight > 0)

    @override
    def increment_counter(self):
        super(TripletLoss, self).increment_counter()
        self._triplet_weight_regime()

    @override
    def set_counter(self, counter):
        super(TripletLoss, self).set_counter(counter)
        self._triplet_weight_regime()


class CombinedVAELoss(LossParent):
    """
    This is the VAE + Triplet Loss, that is used kind of everywhere now
    """

    def __init__(self, sequence_criterion=nn.MSELoss(reduction='none'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 positional_weighting=False, weight_seq=1, weight_kld=1e-2, warm_up=100, kld_tanh_scale=0.1,
                 kld_decrease=5e-3, flat_phase=None, weight_vae=1, weight_triplet=1, dist_type='cosine', margin=None,
                 debug=False, pos_dist_weight=1, neg_dist_weight=1, triplet_warm_up=None, triplet_cool_down=None):
        super(CombinedVAELoss, self).__init__()
        # TODO: PHASE OUT N BATCHES
        self.vae_loss = VAELoss(sequence_criterion, aa_dim=aa_dim, max_len_a1=max_len_a1, max_len_a2=max_len_a2,
                                max_len_a3=max_len_a3, max_len_b1=max_len_b1, max_len_b2=max_len_b2,
                                max_len_b3=max_len_b3, max_len_pep=max_len_pep,
                                add_positional_encoding=add_positional_encoding,
                                positional_weighting=positional_weighting, weight_seq=weight_seq, weight_kld=weight_kld,
                                warm_up=warm_up, tanh_scale=kld_tanh_scale, kld_decrease=kld_decrease,
                                flat_phase=flat_phase,
                                debug=debug)
        self.positional_weighting = positional_weighting
        # Very crude triplet weight regime rn ; quick fix
        # TODO: Implement a proper regime (like KLD) with warm-up and down if needed
        # self.triplet_warm_up = triplet_warm_up
        # self.triplet_cool_down = triplet_cool_down
        # self.triplet_activated = weight_triplet > 0
        self.triplet_loss = TripletLoss(dist_type, margin=margin, pos_dist_weight=pos_dist_weight,
                                        neg_dist_weight=neg_dist_weight, weight=weight_triplet,
                                        warm_up=triplet_warm_up, cool_down=triplet_cool_down)
        self.weight_triplet = weight_triplet
        self.weight_vae = weight_vae
        self.norm_factor = (self.weight_triplet + self.weight_vae)
        print(f'weights: vae, triplet: ', self.weight_vae, self.weight_triplet)

    # Maybe this normalisation factor thing is not needed
    def forward(self, x_hat, x, mu, logvar, z, labels, **kwargs):
        # Both the vaeloss and triplet loss are mean reduced already ; Just need to adjust weights for each term (?)
        recon_loss, kld_loss = self.vae_loss(x_hat, x, mu, logvar)
        recon_loss = self.weight_vae * recon_loss / self.norm_factor
        kld_loss = self.weight_vae * kld_loss / self.norm_factor
        triplet_loss = self.weight_triplet * self.triplet_loss(z, labels, **kwargs) / self.norm_factor
        return recon_loss, kld_loss, triplet_loss

    # @override
    # def increment_counter(self):
    #     super(CombinedVAELoss, self).increment_counter()


class TwoStageVAELoss(LossParent):
    # Here
    def __init__(self, sequence_criterion=nn.MSELoss(reduction='none'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 positional_weighting=False, weight_seq=3, weight_kld=1, debug=False, warm_up=0, warm_up_clf=0,
                 weight_vae=1, weight_triplet=1, weight_classification=1, dist_type='cosine', triplet_loss_margin=None):
        super(TwoStageVAELoss, self).__init__()
        # TODO: Here, maybe change the additional term from triplet loss to something else (Center Loss or Contrastive loss?)
        self.vae_loss = VAELoss(sequence_criterion, aa_dim=aa_dim, max_len_a1=max_len_a1, max_len_a2=max_len_a2,
                                max_len_a3=max_len_a3, max_len_b1=max_len_b1, max_len_b2=max_len_b2,
                                max_len_b3=max_len_b3, max_len_pep=max_len_pep,
                                add_positional_encoding=add_positional_encoding,
                                positional_weighting=positional_weighting, weight_seq=weight_seq, weight_kld=weight_kld,
                                warm_up=warm_up, debug=debug)
        self.triplet_loss = TripletLoss(dist_type, triplet_loss_margin)
        self.classification_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.positional_weighting = positional_weighting
        self.weight_triplet = float(weight_triplet)
        self.weight_vae = float(weight_vae)
        self.weight_classification = float(weight_classification)
        self.norm_factor = (self.weight_triplet + self.weight_vae + self.weight_classification)
        self.warm_up_clf = warm_up_clf
        self.counter = 0

    # Maybe this normalisation factor thing is not needed
    def forward(self, x_hat, x, mu, logvar, z, triplet_labels, x_out, binder_labels, **kwargs):
        weights = kwargs['pep_weights'] if (
                'pep_weights' in kwargs and kwargs['pep_weights'] is not None) else torch.ones([len(z)])
        weights = weights.to(x.device)
        # Both the vaeloss and triplet loss are mean reduced already ; Just need to adjust weights for each term (?)
        recon_loss, kld_loss = self.vae_loss(x_hat, x, mu, logvar)
        recon_loss = self.weight_vae * recon_loss / self.norm_factor
        kld_loss = self.weight_vae * kld_loss / self.norm_factor
        triplet_loss = self.weight_triplet * self.triplet_loss(z, triplet_labels,
                                                               pep_weights=weights) / self.norm_factor

        if self.counter >= self.warm_up_clf:
            # Here criterion should be with reduction='none' and then manually do the mean() because `weight` is not used in forward but init
            # but now I don't even use necessarily use sample weights
            # TODO : Here, removed "* weights" from the classification loss before the mean() because we are using weights to remove triplet from swapped datapoints
            #       i.e. : Triplet is only trained with positive points, whereas the rest are trained with all the losses
            classification_loss = (self.weight_classification * self.classification_loss(x_out,
                                                                                         binder_labels)).mean() / self.norm_factor
            if self.debug:
                print('clf pep weights, loss', weights[:10], classification_loss)
        else:
            classification_loss = torch.tensor([0], device=self.device)

        return recon_loss, kld_loss, triplet_loss, classification_loss

    def increment_counter(self):
        self.counter += 1


class TrimodalVAELoss(LossParent):
    """
        Disabling some elements' loss term should be done here by using the masks
        Keep the VAE model itself simple to only always return all modalities, unmasked
    """

    def __init__(self, alpha_dim, beta_dim, pep_dim, sequence_criterion=nn.MSELoss(reduction='none'),
                 aa_dim=20, add_positional_encoding=False,
                 weight_seq=1, weight_kld=1e-2, weight_triplet=0, weight_vae=1,
                 dist_type='cosine', triplet_loss_margin=0.075,
                 debug=False, warm_up=15):
        super(TrimodalVAELoss, self).__init__()
        # Here check refactored VAE Loss for positional weights thingy
        # TODO : to do the positional weights thing here we need to refactor XX_dim into mla1, etc
        # since we want to change the trimodal from a-b-pep to apep, bpep, a+b_pep, wait a bit before refactoring
        assert sequence_criterion.reduction == 'none', f'Reduction mode should be "none" for sequence criterion! Got {sequence_criterion.reduction} instead.'
        self.sequence_criterion = sequence_criterion
        self.alpha_dim = alpha_dim
        self.beta_dim = beta_dim
        self.pep_dim = pep_dim
        self.aa_dim = aa_dim
        self.pos_dim = 1 if add_positional_encoding else 0
        self.add_positional_encoding = add_positional_encoding
        self.norm_factor = weight_seq + weight_kld + weight_triplet
        self.weight_seq = weight_seq
        self.base_weight_kld = weight_kld
        self.weight_triplet = weight_triplet
        self.weight_kld = self.base_weight_kld
        self.weight_vae = weight_vae

        self.kld_warm_up = warm_up
        self.triplet_loss = TripletLoss(dist_type, triplet_loss_margin)
        self.norm_factor = (self.weight_triplet + self.weight_vae)
        self.debug = debug
        print(f'weights: vae, triplet: ', self.weight_vae, self.weight_triplet)

    def forward(self, x_hat_alpha, x_true_alpha, x_hat_beta, x_true_beta, x_hat_pep, x_true_pep,
                mu_joint, logvar_joint, mus: list, logvars: list, mask_alpha, mask_beta, mask_pep,
                triplet_labels=None, **kwargs):
        """
        Uses the masks to remove missing modalities for each loss term.
        This masking behaviour is taken care of in the various functions doing the losses instead
        of being explicitely laid out in the forward here
        Args:
            x_hat_alpha:
            x_true_alpha:
            x_hat_beta:
            x_true_beta:
            x_hat_pep:
            x_true_pep:
            mu_joint:
            logvar_joint:
            mus:
            logvars:
            mask_alpha:
            mask_beta:
            mask_pep:
            triplet_labels:
            **kwargs:

        Returns:

        """
        # Reconstruction losses ; Put more weight on alpha and beta because longer seq
        recon_loss_alpha = 3 * self.reconstruction_loss(x_hat_alpha, x_true_alpha, 'alpha', mask=mask_alpha)
        recon_loss_beta = 3 * self.reconstruction_loss(x_hat_beta, x_true_beta, 'beta', mask=mask_beta)
        recon_loss_pep = self.reconstruction_loss(x_hat_pep, x_true_pep, 'pep', mask=mask_pep)
        reconstruction_loss = (recon_loss_alpha + recon_loss_beta + recon_loss_pep) / 7

        kld_joint = self.kullback_leibler_divergence(mu_joint, logvar_joint, mask=None)
        kld_alpha = self.kullback_leibler_divergence(mus[0], logvars[0], mask=mask_alpha)
        kld_beta = self.kullback_leibler_divergence(mus[1], logvars[1], mask=mask_beta)
        kld_pep = self.kullback_leibler_divergence(mus[2], logvars[2], mask=mask_pep)
        kld_marginal = kld_alpha + kld_beta + kld_pep
        if self.debug:
            print('counter', self.counter)
            print('seq_loss', reconstruction_loss)
            print('kld_weight', self.weight_kld)
            print('kld_loss', kld_joint.round(decimals=4), kld_alpha.round(decimals=4), kld_beta.round(decimals=4),
                  kld_pep.round(decimals=4), '\n')

        # This here should probably be changed : Triplet labels wouldn't exist for all of mu_joint
        # Because of the tri-modality, technically we could have datapoints with missing peptide input (?)
        triplet_loss = self.weight_triplet * self.triplet_loss(mu_joint, triplet_labels, **kwargs)
        # Return them separately and sum later so that I can debug each component
        return reconstruction_loss / self.norm_factor, kld_joint / self.norm_factor, kld_marginal / self.norm_factor, triplet_loss / self.norm_factor

    def kullback_leibler_divergence(self, mu, logvar, mask=None):
        # KLD weight regime control ; Handles both the joint and marginal KLDs
        if self.counter < self.kld_warm_up:
            # While in the warm-up phase, weight_kld is 0
            self.weight_kld = 0
        else:
            # Otherwise, it starts at the base_kld weight and decreases a 1% of max weight with the epoch counter
            # until it reaches a minimum of the base weight / 5
            self.weight_kld = max(
                self.base_weight_kld - (0.01 * self.base_weight_kld * (self.counter - self.kld_warm_up)),
                self.base_weight_kld / 5)
        kld = 1 + logvar - mu.pow(2) - logvar.exp()
        if mask is not None:
            # kld = mask_modality(kld, mask)
            kld = filter_modality(kld, mask, -99)
        return self.weight_kld * (-0.5 * torch.mean(kld))

    def reconstruction_loss(self, x_hat, x_true, which, mask):
        x_hat_seq, positional_hat = self.slice_x(x_hat, which)
        x_true_seq, positional_true = self.slice_x(x_true, which)
        # TODO Maybe here something is wrong with the sequence criterion that creates nans ?
        # Maybe should not do torch.nanmean but use filter_modality and a custom value here (ex: 1.2345678e-8)
        # --> should also maybe try and see if anywhere there's a problem with exploding loss or params
        reconstruction_loss = self.weight_seq * self.sequence_criterion(x_hat_seq, x_true_seq)
        # reconstruction_loss = torch.nanmean(mask_modality(reconstruction_loss, mask, np.nan))
        reconstruction_loss = torch.mean(filter_modality(reconstruction_loss, mask, -99))
        if self.add_positional_encoding:
            positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true, reduction='none')
            # positional_loss = torch.nanmean(mask_modality(positional_loss, mask, np.nan))
            positional_loss = torch.mean(filter_modality(positional_loss, mask, -99))
            reconstruction_loss += (1e-4 * self.weight_seq * positional_loss)
        return reconstruction_loss

    def slice_x(self, x_flat, which: str):
        """
        Slices and extracts // reshapes the sequence vector from a flattened tensor
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        max_len = {'alpha': self.alpha_dim,
                   'beta': self.beta_dim,
                   'pep': self.pep_dim}[which]

        sequence_tensor = x_flat[:, 0:(max_len * (self.aa_dim + self.pos_dim))].view(-1, max_len,
                                                                                     (self.aa_dim + self.pos_dim))
        positional_encoding_tensor = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding_tensor = sequence_tensor[:, :, self.aa_dim:]
            sequence_tensor = sequence_tensor[:, :, :self.aa_dim]
        return sequence_tensor, positional_encoding_tensor


def compute_cosine_similarity(z_embedding: torch.Tensor, *args, **kwargs):
    """
    Computes a cosine similarity matrix (All vs All), given Z
    Cos Sim : AxB = ||A|| ||B|| * cos(theta), value ranges in [-1, 1]
    Args:
        z_embedding:
        *args:
        **kwargs:

    Returns:

    """
    dot_product = torch.mm(z_embedding, z_embedding.t())
    norms = torch.norm(z_embedding, p=2, dim=1, keepdim=True)
    return dot_product / (norms * norms.t())


def compute_cosine_distance(z_embedding: torch.Tensor, *args, **kwargs):
    """
    Computes a square cosine distance matrix (All vs ALl) for a given sample of latent Z
    Cos Sim : AxB = ||A|| ||B|| * cos(theta)
              in range [-1 ; 1]
    Args:
        z_embedding (torch.Tensor): Latent embedding, dimension N x latent_dim
        *args:
        **kwargs:

    Returns:
        cosine_distance_matrix (torch.Tensor): Square tensor of dimension NxN containing pairwise cosine distances
    """
    cosine_similarity = compute_cosine_similarity(z_embedding)
    cosine_distance_matrix = 1 - cosine_similarity
    # Clamps the low values to 0 for numerical stability
    cosine_distance_matrix[cosine_distance_matrix <= 1e-6] = 0

    return cosine_distance_matrix


def model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False, modality_mask=None):
    # Here concat the list in case we are given a list of tensors (as done in the train/valid batching system)
    x_reconstructed = torch.cat(x_reconstructed) if type(x_reconstructed) == list else x_reconstructed
    x_true = torch.cat(x_true) if type(x_true) == list else x_true
    seq_hat, positional_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, positional_true = model.reconstruct_hat(x_true)
    seq_accuracy, positional_accuracy = reconstruction_accuracy(seq_true, seq_hat, positional_true, positional_hat,
                                                                pad_index=20, return_per_element=return_per_element,
                                                                modality_mask=modality_mask)
    metrics = {'seq_accuracy': seq_accuracy, 'pos_accuracy': positional_accuracy}
    if positional_accuracy is None:
        del metrics['pos_accuracy']

    return metrics


def reconstruct_and_compute_accuracy(model, x_true, x_recon, recon_only=False):
    if type(x_true) != torch.Tensor and type(x_true) == list:
        x_true = torch.cat(x_true)
        x_recon = torch.cat(x_recon)
    # do recon_only for when we need to cat the TCR and pep

    metrics = {'seq_accuracy': [0] * len(x_true)} if recon_only else model_reconstruction_stats(model, x_recon, x_true,
                                                                                                return_per_element=True)
    x_seq_recon, pos_recon = model.slice_x(x_recon)
    x_seq_true, pos_true = model.slice_x(x_true)
    seq_hat_recon = model.recover_sequences_blosum(x_seq_recon)
    seq_hat_true = model.recover_sequences_blosum(x_seq_true)
    return seq_hat_true, seq_hat_recon, metrics


def get_acc_list_string(strings_1, strings_2):
    return [get_acc_single_string(s1, s2) for s1, s2 in zip(strings_1, strings_2)]


def get_acc_single_string(s1, s2):
    return sum([int(x1 == x2) for x1, x2 in zip(s1, s2) if x1 != 'X' and x2 != 'X']) / len(s2.replace('X', ''))


# NOTE : Fixed version with the true acc
# TODO : add thing for positional encoding
def reconstruction_accuracy(seq_true, seq_hat, positional_true, positional_hat,
                            pad_index=20, return_per_element=False, modality_mask=None):
    """
    Args:
        mask:
        seq_true: ordinal vector of true sequence (Here, the number is to map back to an amino acid with AA_KEYS, with 20 = X)
        seq_hat: Same but reconstructed
        positional_true: positional encoding (true), shape is either 2D or flattened vector
        positional_hat: positional encoding (reconstructed), same as above
        pad_index: This is the INDEX for the AA used for padding. 20 for default (which represents X)
        return_per_element: Return a per-element acc instead of mean

    Returns:
        seq_accuracy : sequence reconstruction accuracy
                       Either a list of accuracy per sample if return_per_element is True, or a mean number
        pos_accuracy : positional encoding reconstruction accuracy
                       Same as above, except it can be None if positional_hat is None (i.e. disabled)

    """
    # Use modality_mask to do something

    # Compute the mask and true lengths
    mask = (seq_true != pad_index).float()
    true_lens = mask.sum(dim=1)
    # difference here for per element is that we don't take the mean(dim=0) and
    # have to detach() from graph to do tolist() ; Here is still per element
    seq_accuracy = ((seq_true == seq_hat).float() * mask).sum(1) / true_lens
    # Assuming pos_accuracy are logits : sigmoid->threshold(0.5)->compare to true
    pos_accuracy = ((F.sigmoid(positional_hat) > 0.5).float() == positional_true).float().mean(
        dim=(1, 2)) if positional_hat is not None else None
    # fill with nans to ignore those datapoints
    if modality_mask is not None:
        seq_accuracy = mask_modality(seq_accuracy, modality_mask, torch.nan)
        pos_accuracy = mask_modality(pos_accuracy, modality_mask, torch.nan) if positional_hat is not None else None
    if return_per_element:
        # Here give a per element accuracy (so that we have individual metric for each datapoint in a df)
        seq_accuracy = seq_accuracy.detach().cpu().tolist()
        pos_accuracy = pos_accuracy.detach().cpu().tolist() if positional_hat is not None else None
    else:
        # Here gives the mean accuracy of the model for all the datapoints
        seq_accuracy = seq_accuracy.nanmean(dim=0).item()
        pos_accuracy = pos_accuracy.nanmean(dim=0).item() if positional_hat is not None else None

    return seq_accuracy, pos_accuracy


def auc01_score(y_true: np.ndarray, y_pred: np.ndarray, max_fpr=0.1) -> float:
    """Compute the partial AUC of the ROC curve for FPR up to max_fpr.
    Args:
        y_true (array-like): The true labels of the data (0 or 1).
        y_pred (array-like): The predicted probabilities or scores.
        max_fpr (float): Maximum false positive rate.
    Returns:
        float: Partial AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return auc(fpr, tpr) * 10


def get_metrics(y_true, y_score, y_pred=None, threshold=0.50, keep=False, reduced=True, round_digit=4,
                no_curves=False) -> dict:
    """
    Computes all classification metrics & returns a dictionary containing the various key/metrics
    incl. ROC curve, AUC, AUC_01, F1 score, Accuracy, Recall
    Args:
        y_true:
        y_pred:
        y_score:

    Returns:
        metrics (dict): Dictionary containing all results
    """
    metrics = {}
    # DETACH & PASS EVERYTHING TO CPU
    if threshold is not None and y_pred is None:
        # If no y_pred is provided, will threshold score (y in [0, 1])
        y_pred = (y_score > threshold)
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy()
        elif type(y_pred) == np.ndarray:
            y_pred = y_pred.astype(int)
    elif y_pred is not None and type(y_pred) == torch.Tensor:
        y_pred = y_pred.int().cpu().detach().numpy()

    if type(y_true) == torch.Tensor and type(y_score) == torch.Tensor:
        y_true, y_score = y_true.int().cpu().detach().numpy(), y_score.cpu().detach().numpy()

    metrics['auc'] = roc_auc_score(y_true, y_score)
    metrics['auc_01'] = roc_auc_score(y_true, y_score, max_fpr=0.1)
    metrics['auc_01_real'] = auc01_score(y_true, y_score, max_fpr=0.1)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AP'] = average_precision_score(y_true, y_score)
    if not reduced:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics['roc_curve'] = fpr, tpr
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['pr_curve'] = recall, precision  # So it follows the same x,y format as roc_curve
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
            metrics['prauc'] = auc(recall, precision)
            metrics['AP'] = average_precision_score(y_true, y_score)
        except:
            print('Couldn\'t get AUCs/etc because there\'s only one class in the dataset')
            print(f'Only negatives: {all(y_true == 0)}, Only positives: {all(y_true == 1)}')
            raise ValueError
        if keep:
            metrics['y_true'] = y_true
            metrics['y_score'] = y_score
    if round_digit is not None:
        for k, v in metrics.items():
            try:
                metrics[k] = round(v, round_digit)
            except:
                print(f'Couldn\'t round {k} of type ({type(v)})! continuing')
                continue
    if no_curves:
        td = []
        for k in metrics:
            if 'curve' in k:
                td.append(k)
        for k in td:
            del metrics[k]
    return metrics


def get_roc(df, score='pred', target='agg_label', binder=None, anchor_mutation=None):
    """
    Args:
        df: DF containing the prediction or scores
        score: Name of the score columns, 'pred' by default
        target: Name of the target column, 'pred' by default
        binder: None, "Improved" or "Conserved" ; None by default
        anchor_mutation: None, True, False ; None by default

    Returns:

    """
    if binder is not None and anchor_mutation is not None:
        df = df.query('binder==@binder and anchor_mutation==@anchor_mutation').copy()
    try:
        fpr, tpr, _ = roc_curve(df[target].values, df[score].values)
        auc = roc_auc_score(df[target].values, df[score].values)
        auc01 = roc_auc_score(df[target].values, df[score].values, max_fpr=0.1)
    except KeyError:
        print('here in get_roc KeyError')
        try:
            fpr, tpr, _ = roc_curve(df[target].values, df['mean_pred'].values)
            auc = roc_auc_score(df[target].values, df['mean_pred'].values)
            auc01 = roc_auc_score(df[target].values, df['mean_pred'].values, max_fpr=0.1)
        except:
            raise KeyError(f'{target} or "mean_pred" not in df\'s columns!')
    output = {"roc": (fpr, tpr),
              "auc": auc,
              "auc01": auc01,
              "npep": len(df)}
    return output


# def custom_silhouette_score(input_matrix, labels, metric='precomputed', aggregation='micro'):
#     """
#     Implements the silhouette score to do either micro (all samples) or macro (per cluster) averaging
#     Args:
#         input_matrix:
#         labels:
#         metric:
#         aggregation:
#
#     Returns:
#
#     """
#     if aggregation == 'micro':
#         score = silhouette_score(input_matrix, labels, metric=metric)
#     elif aggregation == 'macro':
#         # Per sample silhouette score
#         all_scores = silhouette_samples(input_matrix, labels, metric=metric)
#         macro_averages = []
#         # per cluster averaging
#         for label in np.unique(labels):
#             indices = np.where(labels == label)[0]
#             macro_averages.append(np.mean(all_scores[indices]))
#         score = np.mean(macro_averages)
#
#     elif aggregation == 'size_weighted':
#         all_scores = silhouette_samples(input_matrix, labels, metric=metric)
#
#         # Step 2: Identify unique clusters and their sizes
#         unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
#
#         # Step 3: Compute the silhouette score for each cluster
#         cluster_silhouette_values = np.array([
#             np.mean(all_scores[labels == label]) for label in unique_labels
#         ])
#
#         # Step 4: Multiply the average silhouette score by the cluster size (vectorized)
#         weighted_silhouette = np.dot(cluster_silhouette_values, cluster_sizes)
#
#         # Step 5: Compute the overall weighted silhouette score
#         score = weighted_silhouette / np.sum(cluster_sizes)
#
#     elif aggregation == 'size_log2':
#         all_scores = silhouette_samples(input_matrix, labels, metric=metric)
#         # Step 2: Identify unique clusters and their sizes
#         unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
#         cluster_weights = np.log2(cluster_sizes)+1
#         # normalizes the weights to have em in 0,1
#         cluster_weights = cluster_weights / np.sum(cluster_weights)
#         # Compute the silhouette score for each cluster (is this almost macro aggregation??)
#         cluster_avg_silhouettes = np.array([np.mean(all_scores[labels == label]) for label in unique_labels])
#         # Step 5: Calculate the weighted silhouette score using vectorized operations
#         score = np.sum(cluster_avg_silhouettes * cluster_weights)
#
#         # weighted_sum = 0
#         # for label, size in zip(unique_labels, cluster_sizes):
#         #     # Get the silhouette scores for the current cluster
#         #     cluster_silhouette_values = all_scores[labels == label]
#         #     # Compute the average silhouette score for the current cluster
#         #     cluster_avg_silhouette = np.mean(cluster_silhouette_values)
#         #
#         #     # Multiply the average silhouette score by the normalized weight
#         #     weighted_sum += cluster_avg_silhouette * weight
#     else:
#         raise ValueError('Aggregation must be "micro" or "macro" or "size_weighted" or "size_log2"')
#
#     return score

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples


def custom_silhouette_score(input_matrix, labels, metric='precomputed', aggregation='micro', precision=4):
    """
    Implements the silhouette score to do either micro (all samples) or macro (per cluster) averaging
    Args:
        input_matrix:
        labels:
        metric:
        aggregation:
        precision:

    Returns:
        silhouette score based on the selected aggregation method
    """
    if aggregation == 'micro':
        # Micro-average: Standard silhouette score across all samples
        score = silhouette_score(input_matrix, labels, metric=metric)

    elif aggregation == 'macro':
        # Macro-average: Average silhouette score per cluster, then average across clusters
        all_scores = silhouette_samples(input_matrix, labels, metric=metric)
        macro_averages = []

        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            macro_averages.append(np.mean(all_scores[indices]))

        score = np.mean(macro_averages)

    elif aggregation == 'size_weighted':
        # Size-weighted average: Weight each cluster's silhouette score by its size
        all_scores = silhouette_samples(input_matrix, labels, metric=metric)
        unique_labels, cluster_sizes = np.unique(labels, return_counts=True)

        # Compute the average silhouette score per cluster
        cluster_avg_silhouettes = np.array([np.mean(all_scores[labels == label]) for label in unique_labels])

        # Weight each cluster's silhouette score by its size
        weighted_silhouette = np.dot(cluster_avg_silhouettes, cluster_sizes)

        # Normalize by the total number of samples
        score = weighted_silhouette / np.sum(cluster_sizes)

    elif aggregation == 'size_log2':
        # Size-log2 weighted average: Weight each cluster's silhouette score by log2(size) + 1
        all_scores = silhouette_samples(input_matrix, labels, metric=metric)
        unique_labels, cluster_sizes = np.unique(labels, return_counts=True)

        # Compute the log2(size) + 1 weight for each cluster
        cluster_weights = np.log2(cluster_sizes) + 1

        # Compute the average silhouette score per cluster
        cluster_avg_silhouettes = np.array([np.mean(all_scores[labels == label]) for label in unique_labels])

        # Compute the weighted sum (without normalizing the weights)
        score = np.sum(cluster_avg_silhouettes * cluster_weights) / np.sum(cluster_weights)
    elif aggregation == 'none':
        return silhouette_samples(input_matrix, labels, metric=metric)
    else:
        raise ValueError('Aggregation must be "micro", "macro", "size_weighted", or "size_log2", "none"')

    return round(score, precision)