import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from src.data_processing import get_positional_encoding, encoding_matrix_dict
import math


class NetParent(nn.Module):
    """
    Mostly a QOL superclass
    Creates a parent class that has reset_parameters implemented and .device
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()
        # device is cpu by default
        self.device = 'cpu'
        self.counter = 0

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    @staticmethod
    def reset_weight(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            if hasattr(child, 'reset_parameters'):
                self.reset_weight(child)

    def to(self, device):
        # Work around, so we can get model.device for all NetParent
        #
        super(NetParent, self).to(device)
        self.device = device

    def increment_counter(self):
        self.counter += 1
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'increment_counter'):
                c.increment_counter()

    def set_counter(self, counter):
        self.counter = counter
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'set_counter'):
                c.set_counter(counter)

class CDR3bVAE(NetParent):
    """
    TODO : To deprecate this class
    """

    # Define the input dimension as some combination of sequence length, AA dim,
    def __init__(self, max_len=23, encoding='BL50LO', pad_scale=-20, aa_dim=20, use_v=True, use_j=True, v_dim=51,
                 j_dim=13, activation=nn.SELU(), hidden_dim=128, latent_dim=32, max_len_pep=0):
        super(CDR3bVAE, self).__init__()
        # Init params that will be needed at some point for reconstruction
        v_dim = v_dim if use_v else 0
        j_dim = j_dim if use_j else 0
        max_len = max_len + max_len_pep
        input_dim = (max_len * aa_dim) + v_dim + j_dim
        self.encoding = encoding
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale
        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))
        self.input_dim = input_dim
        self.max_len = max_len
        self.aa_dim = aa_dim
        self.v_dim = v_dim if use_v else 0
        self.use_v = use_v
        self.j_dim = j_dim if use_j else 0
        self.use_j = use_j
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: For now, just use a fixed set of layers.
        # Encoder : in -> in//2 -> hidden -> latent_mu, latent_logvar, where z = mu + logvar*epsilon
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                     nn.Linear(input_dim // 2, hidden_dim), activation)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                     nn.Linear(hidden_dim, hidden_dim), activation)
        self.decoder_sequence = nn.Sequential(nn.Linear(hidden_dim, input_dim // 2), activation,
                                              nn.Linear(input_dim // 2, input_dim - self.v_dim - self.j_dim))

        self.decoder_v = nn.Linear(hidden_dim, self.v_dim) if use_v else None
        self.decoder_j = nn.Linear(hidden_dim, self.j_dim) if use_j else None

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        # During evaluation, the trick is disabled and z = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        else:
            return mu

    def encode(self, x):
        mu_logvar = self.encoder(x.flatten(start_dim=1))
        mu = self.encoder_mu(mu_logvar)
        logvar = self.encoder_logvar(mu_logvar)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        x_hat = self.decoder_sequence(z)
        if self.use_v:
            v = self.decoder_v(z)
            x_hat = torch.cat([x_hat, v], dim=1)
        if self.use_j:
            j = self.decoder_j(z)
            x_hat = torch.cat([x_hat, j], dim=1)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def slice_x(self, x):
        sequence = x[:, 0:(self.max_len * self.aa_dim)].view(-1, self.max_len, self.aa_dim)
        # Reconstructs the v/j gene as one hot vectors
        v_gene = x[:, (self.max_len * self.aa_dim):(self.max_len * self.aa_dim + self.v_dim)] if self.use_v else None
        j_gene = x[:, ((self.max_len * self.aa_dim) + self.v_dim):] if self.use_j else None
        return sequence, v_gene, j_gene

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, v_gene, j_gene = self.slice_x(x_hat)
            return sequence, v_gene, j_gene

    def embed(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    def recover_indices(self, seq_tensor):
        # Sample data
        N, max_len = seq_tensor.shape[0], seq_tensor.shape[1]

        # Expand MATRIX_VALUES to have the same shape as x_seq for broadcasting
        expanded_MATRIX_VALUES = self.MATRIX_VALUES.unsqueeze(0).expand(N, -1, -1, -1)
        # Compute the absolute differences
        abs_diff = torch.abs(seq_tensor.unsqueeze(2) - expanded_MATRIX_VALUES)
        # Sum along the last dimension (20) to get the absolute differences for each character
        abs_diff_sum = abs_diff.sum(dim=-1)

        # Find the argmin along the character dimension (21)
        argmin_indices = torch.argmin(abs_diff_sum, dim=-1)
        return argmin_indices

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        return [''.join([AA_KEYS[y] for y in x]) for x in self.recover_indices(seq_tensor)]

    def reconstruct_hat(self, x_hat):
        seq, v, j = self.slice_x(x_hat)
        seq_idx = self.recover_indices(seq)
        return seq_idx, v, j


# TODO : Make number of layers more flexible and add batchnorm-dropout, try leaky relu
class FullTCRVAE(NetParent):
    # Define the input dimension as some combination of sequence length, AA dim,
    def __init__(self, max_len_a1=0, max_len_a2=0, max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23,
                 max_len_pep=0, encoding='BL50LO', pad_scale=-20, aa_dim=20, add_positional_encoding=False,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=64,
                 add_layer_encoder=True, add_layer_decoder=True, old_behaviour=True, batchnorm=True):
        super(FullTCRVAE, self).__init__()
        # Init params that will be needed at some point for reconstruction
        # Here, define aa_dim and pos_dim ; pos_dim is the dimension of a positional encoding.
        # pos_dim should be given by how many sequences are used, i.e. how many max_len_x > 0
        # But also use a flag `add_positional_encoding` to make it more explicit that it's active or not
        max_len = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])
        pos_dim = sum([int(mlx) > 0 for mlx in
                       [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep]]) \
            if add_positional_encoding else 0
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.add_positional_encoding = add_positional_encoding
        input_dim = max_len * (aa_dim + pos_dim)
        self.input_dim = input_dim
        self.max_len = max_len
        self.encoding = encoding
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale

        # TODO : Maybe should use -1 instead of -20 for pad values since that's the value for X in BL50LO ?
        # Create the encoding matrix to recover / rebuild sequences
        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))
        # Neural network params
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: For now, just use a fixed set of layers.
        #       Might need more layers because we are compressing more information

        # Encoder : in -> in//2 -> hidden -> latent_mu, latent_logvar, where z = mu + logvar*epsilon
        bn = nn.BatchNorm1d if batchnorm else nn.Identity
        #####
        # HERE DO THINGS WITH ADD LAYER AND BATCHNORM ; add an "old_behaviour" parameter so we can still read old models

        if old_behaviour:
            self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                         nn.Linear(input_dim // 2, hidden_dim), activation)
            self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
            self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                         nn.Linear(hidden_dim, hidden_dim), activation)
            self.decoder_sequence = nn.Sequential(nn.Linear(hidden_dim, input_dim // 2), activation,
                                                  nn.Linear(input_dim // 2, input_dim))
        else:
            # Create encoder layers
            encoder_layers = [nn.Linear(input_dim, input_dim // 2), activation, bn(input_dim//2),
                              nn.Linear(input_dim // 2, hidden_dim), activation, bn(hidden_dim)]
            if add_layer_encoder:
                encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), activation, bn(hidden_dim)])
            # Create decoder layers:
            decoder_layers = [nn.Linear(latent_dim, hidden_dim), activation, bn(hidden_dim)]
            if add_layer_decoder:
                decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), activation, bn(hidden_dim)])
            decoder_sequence_layers = [nn.Linear(hidden_dim, input_dim // 2), activation, bn(input_dim // 2),
                                       nn.Linear(input_dim // 2, input_dim)]

            self.encoder = nn.Sequential(*encoder_layers)
            self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
            self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.Sequential(*decoder_layers)
            self.decoder_sequence = nn.Sequential(*decoder_sequence_layers)
            


    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        # During evaluation, the trick is disabled and z = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        else:
            return mu

    def encode(self, x):
        mu_logvar = self.encoder(x.flatten(start_dim=1))
        mu = self.encoder_mu(mu_logvar)
        logvar = self.encoder_logvar(mu_logvar)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        x_hat = self.decoder_sequence(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

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
        sequence = x[:, 0:(self.max_len * (self.aa_dim + self.pos_dim))].view(-1, self.max_len,
                                                                              (self.aa_dim + self.pos_dim))
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.slice_x(x_hat)
            return sequence, positional_encoding

    def embed(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    def recover_indices(self, seq_tensor):
        # Sample data
        N, max_len = seq_tensor.shape[0], seq_tensor.shape[1]

        # Expand MATRIX_VALUES to have the same shape as x_seq for broadcasting
        expanded_MATRIX_VALUES = self.MATRIX_VALUES.unsqueeze(0).expand(N, -1, -1, -1)
        # Compute the absolute differences
        abs_diff = torch.abs(seq_tensor.unsqueeze(2) - expanded_MATRIX_VALUES)
        # Sum along the last dimension (20) to get the absolute differences for each character
        abs_diff_sum = abs_diff.sum(dim=-1)

        # Find the argmin along the character dimension (21)
        argmin_indices = torch.argmin(abs_diff_sum, dim=-1)
        return argmin_indices

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        return [''.join([AA_KEYS[y] for y in x]) for x in self.recover_indices(seq_tensor)]

    def reconstruct_hat(self, x_hat):
        sequence, positional_encoding = self.slice_x(x_hat)
        seq_idx = self.recover_indices(sequence)
        return seq_idx, positional_encoding


class PeptideClassifier(NetParent):

    def __init__(self, pep_dim=12, latent_dim=64, n_layers=0, n_hidden_clf=32, dropout=0, batchnorm=False,
                 decrease_hidden=False, add_pep=True):
        super(PeptideClassifier, self).__init__()
        # self.sigmoid = nn.Sigmoid()

        # self.softmax = nn.Softmax()

        in_dim = pep_dim + latent_dim if add_pep else latent_dim
        in_layer = [nn.Linear(in_dim, n_hidden_clf), nn.ReLU()]
        if batchnorm:
            in_layer.append(nn.BatchNorm1d(n_hidden_clf))
        in_layer.append(nn.Dropout(dropout))

        self.in_layer = nn.Sequential(*in_layer)
        # Hidden layers
        layers = []
        for _ in range(n_layers):
            if decrease_hidden:
                layers.append(nn.Linear(n_hidden_clf, n_hidden_clf // 2))
                n_hidden_clf = n_hidden_clf // 2
            else:
                layers.append(nn.Linear(n_hidden_clf, n_hidden_clf))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden_clf))
            layers.append(nn.Dropout(dropout))

        self.hidden_layers = nn.Sequential(*layers) if n_layers > 0 else nn.Identity()
        self.out_layer = nn.Linear(n_hidden_clf, 1)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x


class AttentionPeptideClassifier(NetParent):

    def __init__(self, pep_dim, latent_dim, num_heads=4, n_layers=1, n_hidden_clf=32, dropout=0.0, batchnorm=False,
                 decrease_hidden=False):
        #
        super(AttentionPeptideClassifier, self).__init__()
        self.in_dim = pep_dim + latent_dim
        self.attention = nn.MultiheadAttention(embed_dim=self.in_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        in_layer = [nn.Linear(self.in_dim, n_hidden_clf), nn.ReLU()]
        if batchnorm:
            in_layer.append(nn.BatchNorm1d(n_hidden_clf))
        in_layer.append(self.dropout)
        self.in_layer = nn.Sequential(*in_layer)
        # Hidden layers
        layers = []
        for _ in range(n_layers):
            if decrease_hidden:
                layers.append(nn.Linear(n_hidden_clf, n_hidden_clf // 2))
                n_hidden_clf = n_hidden_clf // 2
            else:
                layers.append(nn.Linear(n_hidden_clf, n_hidden_clf))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden_clf))
            layers.append(nn.Dropout(dropout))
        self.hidden_layers = nn.Sequential(*layers) if n_layers > 0 else nn.Identity()

        self.out_layer = nn.Linear(n_hidden_clf, 1)

    def forward(self, x_cat):
        # x_cat is the concatenated vector between the latent and the sequence
        x_attention, _ = self.attention(x_cat, x_cat, x_cat)
        # Residual connection adding back the input
        x_attention += x_cat
        # Classifier
        x_out = self.in_layer(x_attention)
        x_out = self.hidden_layers(x_out)
        x_out = self.out_layer(x_out)
        return x_out


class TwoStageVAECLF(NetParent):
    # This name is technically wrong, it should be TwoStageVAEClassifier
    # Refactoring now would be a bit annoying given the previously saved JSONs that have to be manually changed
    # Should be refactored at some point.
    # TODO : Refactoring to include pep : This here should be fine but CAREFUL when adding positional encoding
    def __init__(self, vae_kwargs, clf_kwargs, warm_up_clf=0):
        super(TwoStageVAECLF, self).__init__()
        self.vae = FullTCRVAE(**vae_kwargs)
        self.clf = PeptideClassifier(**clf_kwargs)
        self.warm_up_clf = warm_up_clf
        self.counter = 0

    def forward(self, x_tcr, x_pep):
        """
        Needs to back-propagate on and return :
            x_hat -> reconstruction loss
            mu/logvar -> KLD loss
            mu -> Triplet/Contrastive loss: This is just `z_embed` without reparameterisation !
            x_out -> Prediction loss (Binary Cross Entropy? MSE?)
        Args:
            x_tcr:
            x_pep:

        Returns:
            x_hat: reconstructed input
            mu: Mean latent vector (used for KLD + Triplet)
            logvar: std latent vector (used for KLD)
            x_out: pMHC-TCR prediction output logit (used for classifier loss)
        """
        x_hat, mu, logvar = self.vae(x_tcr)
        # Here, set the vae to eval before embedding in order to disable reparameterisation
        # TODO: Should the classifier be trained using z_tcr (reparameterised) or mu (not reparameterised?)
        #       Figure out which latent representation should be used for classifier. For now, use z_tcr
        z = self.vae.embed(x_tcr)
        z = torch.cat([z, x_pep.flatten(start_dim=1)], dim=1)
        # Only do the CLF part if the counter is above the warm_up threshold ; This is redundant but just in case
        if self.counter < self.warm_up_clf:
            with torch.no_grad():
                x_out = self.clf(z)
        else:
            x_out = self.clf(z)
        return x_hat, mu, logvar, x_out

    def reconstruct_hat(self, x):
        return self.vae.reconstruct_hat(x)

    def slice_x(self, x):
        return self.vae.slice_x(x)

    def reconstruct(self, z):
        return self.vae.reconstruct(z)

    def embed(self, x):
        return self.vae.embed(x)

    def sample_latent(self, n_samples):
        return self.vae.sample_latent(n_samples)

    def recover_indices(self, seq_tensor):
        return self.vae.recover_indices(seq_tensor)

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        return self.vae.recover_sequences_blosum(seq_tensor, AA_KEYS)





class StandardizerSequence(nn.Module):
    def __init__(self, n_feats=20):
        super(StandardizerSequence, self).__init__()
        # Here using 20 because 20 AA alphabet. With this implementation, it shouldn't need custom state_dict fct
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats
        self.dimensions = None

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = (torch.sum(masked_values, dim=1) / torch.sum(x_mask, dim=1))
            sigma = (torch.sqrt(torch.sum((masked_values - mu.unsqueeze(1)) ** 2, dim=1) / torch.sum(x_mask, dim=1)))
            self.mu.data.copy_(mu.mean(dim=0))
            sigma = sigma.mean(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = (self.view_3d_to_2d(x) - self.mu) / self.sigma
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def recover(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = self.view_3d_to_2d(x)
            # Return to original silhouette_scale by multiplying with sigma and adding mu
            x = x * self.sigma + self.mu
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros(self.n_feats))
            self.sigma.data.copy_(torch.ones(self.n_feats))
            self.fitted.data = torch.tensor(False)

    def view_3d_to_2d(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                self.dimensions = (x.shape[0], x.shape[1], x.shape[2])
                return x.view(-1, x.shape[2])
            else:
                return x

    def view_2d_to_3d(self, x):
        with torch.no_grad():
            if len(x.shape) == 2 and self.dimensions is not None:
                return x.view(self.dimensions[0], self.dimensions[1], self.dimensions[2])
            else:
                return x

    def to(self, device):
        super(StandardizerSequence, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StandardizerSequenceVector(nn.Module):
    def __init__(self, input_dim=20, max_len=12):
        super(StandardizerSequenceVector, self).__init__()
        self.mu = nn.Parameter(torch.zeros((max_len, input_dim)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((max_len, input_dim)), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.input_dim = input_dim
        self.max_len = max_len

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = masked_values.mean(dim=0)
            sigma = masked_values.std(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.mu.data.copy_(mu)
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'Standardizer not fitted!'
        return (x - self.mu) / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros((self.max_len, self.input_dim)))
            self.sigma.data.copy_(torch.ones((self.max_len, self.input_dim)))
            self.fitted.data = torch.tensor(False)

    def to(self, device):
        super(StandardizerSequenceVector, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StandardizerFeatures(nn.Module):
    def __init__(self, n_feats=2):
        super(StandardizerFeatures, self).__init__()
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats

    def fit(self, x_features: torch.Tensor):
        """ Will consider the mask (padded position) and ignore them before computing the mean/std
        Args:
            x_features:

        Returns:
            None
        """
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            self.mu.data.copy_(x_features.mean(dim=0))
            self.sigma.data.copy_(x_features.std(dim=0))
            # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
            self.sigma.data[torch.where(self.sigma.data == 0)] = 1e-12
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            return x - self.mu / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy(torch.zeros(self.n_feats))
            self.sigma.data.copy(torch.ones(self.n_feats))
            self.fitted.data = torch.tensor(False)

    def to(self, device):
        super(StandardizerFeatures, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StdBypass(nn.Module):
    def __init__(self, **kwargs):
        super(StdBypass, self).__init__()
        self.requires_grad = False
        self.bypass = nn.Identity(**kwargs)
        self.fitted = False
        self.mu = 0
        self.sigma = 1

    def forward(self, x_tensor, *args):
        """
        Args:
            x:
        Returns:

        """

        return x_tensor

    def fit(self, x_tensor, *args):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """
        self.fitted = True
        return x_tensor
