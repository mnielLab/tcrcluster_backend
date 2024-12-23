from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.data_processing import encoding_matrix_dict
from src.models import NetParent, PeptideClassifier


class CNNEncoder(NetParent):
    def __init__(self, kernel_size, stride, pad, max_len, features_dim,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=128, batchnorm=True):
        super(CNNEncoder, self).__init__()
        self.features_dim = features_dim
        self.max_len = max_len
        self.len_in = 1 + ((1 + ((max_len + 2 * pad - kernel_size) // stride) + 2 * pad - kernel_size) // stride)
        # Neural network params
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        bn = nn.BatchNorm1d if batchnorm else nn.Identity
        self.conv_layers = nn.Sequential(nn.Conv1d(features_dim, hidden_dim, kernel_size, stride, pad),
                                         activation, bn(hidden_dim),
                                         nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, stride, pad),
                                         activation, bn(hidden_dim*2))
        self.fc_mu = nn.Linear(self.len_in * 2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.len_in * 2 * hidden_dim, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu_logvar = self.conv_layers(x)
        mu_logvar = mu_logvar.flatten(start_dim=1)
        mu = self.fc_mu(mu_logvar)
        logvar = self.fc_logvar(mu_logvar)
        return mu, logvar


class CNNDecoder(NetParent):
    def __init__(self, kernel_size, stride, pad, len_in, features_dim, output_padding_1, output_padding_2,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=128, batchnorm=True):
        super(CNNDecoder, self).__init__()
        self.features_dim = features_dim
        self.len_in = len_in  # Should be the len_in of CNNEncoder ; See CNNVAE wrapper class
        self.len_out_trans = stride * (stride * (len_in - 1) + kernel_size - 2 * pad + output_padding_1 - 1) + kernel_size - 2 * pad + output_padding_2
        # Neural network params
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        bn = nn.BatchNorm1d if batchnorm else nn.Identity
        print(bn, hidden_dim, kernel_size, stride, pad, output_padding_1, output_padding_2, activation)
        self.conv_transpose_layers = nn.Sequential(nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size, stride, pad, output_padding_1), nn.SELU(), bn(hidden_dim),
                                                   nn.ConvTranspose1d(hidden_dim, features_dim, kernel_size, stride, pad, output_padding_2), nn.SELU(), bn(features_dim))
        self.fc_z = nn.Linear(latent_dim, len_in * 2 * hidden_dim)

    def forward(self, z):
        x_hat = self.fc_z(z)
        x_hat = x_hat.view(-1, self.hidden_dim * 2, self.len_in)
        x_hat = self.conv_transpose_layers(x_hat)
        x_hat = x_hat.permute(0, 2, 1)
        return x_hat


class CNNVAE(NetParent):
    def __init__(self, kernel_size_in=9, stride_in=4, pad_in=2,
                 kernel_size_trans=9, stride_trans=4, pad_trans=2, output_padding_trans_1=1, output_padding_trans_2=0,
                 max_len_a1=7, max_len_a2=8, max_len_a3=22, max_len_b1=6, max_len_b2=7, max_len_b3=23,
                 encoding='BL50LO', pad_scale=-20, aa_dim=20, add_positional_encoding=True,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=128, batchnorm=True):
        super(CNNVAE, self).__init__()
        # Init params that will be needed at some point for reconstruction
        # Here, define aa_dim and pos_dim ; pos_dim is the dimension of a positional encoding.
        # pos_dim should be given by how many sequences are used, i.e. how many max_len_x > 0
        # But also use a flag `add_positional_encoding` to make it more explicit that it's active or not
        max_len = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3])
        pos_dim = sum([int(mlx) > 0 for mlx in
                       [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]]) \
            if add_positional_encoding else 0
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.add_positional_encoding = add_positional_encoding
        features_dim = aa_dim + pos_dim
        input_dim = max_len * features_dim
        self.input_dim = input_dim
        self.features_dim = features_dim
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
        self.batchnorm = batchnorm
        self.len_in = 1 + ((1 + (
                (max_len + 2 * pad_in - kernel_size_in) // stride_in) + 2 * pad_in - kernel_size_in) // stride_in)
        self.len_out = stride_trans * (stride_trans * (self.len_in - 1) + kernel_size_trans - 2 * pad_trans + output_padding_trans_1 - 1) + kernel_size_trans - 2 * pad_trans + output_padding_trans_2

        self.encoder = CNNEncoder(kernel_size_in, stride_in, pad_in, max_len, features_dim, activation,
                                  hidden_dim, latent_dim, batchnorm)
        self.decoder = CNNDecoder(kernel_size_trans, stride_trans, pad_trans, self.len_in, features_dim,
                                  output_padding_trans_1, output_padding_trans_2, activation, hidden_dim, latent_dim,
                                  batchnorm)

    # VAE functions (fwd, embed, reparam, etc)
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def embed(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return z

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        # During evaluation, the trick is disabled and z = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        else:
            return mu

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    # Reshaping / reconstruction functions
    def slice_x(self, x):
        """
        Slices and extracts // reshapes the sequence vector
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # Here this function exists for compatibility reason
        # In theory the CNN should reconstruct the sequence in the right dimension and require no slicing - reshaping
        # We only slice to extract the sequence and positional encoding if it exists
        sequence = x
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def recover_indices(self, seq_tensor):
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


class TwoStageCNNVAECLF(NetParent):
    def __init__(self, vae_kwargs, clf_kwargs, warm_up_clf=0):
        super(TwoStageCNNVAECLF, self).__init__()
        self.vae = CNNVAE(**vae_kwargs)
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
        # Mu is the non-reparameterised Z_latent ; So we just take that as the embedding and concat with x_pep
        z = torch.cat([mu, x_pep.flatten(start_dim=1)], dim=1)
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
