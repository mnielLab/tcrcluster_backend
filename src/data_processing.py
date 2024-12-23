import copy

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing
import math
from torch.utils.data import TensorDataset
from src.utils import pkl_load, pkl_dump
from copy import deepcopy
import os
import warnings

warnings.filterwarnings('ignore')

DATADIR = '/Users/riwa/Documents/code/tclustr/data/' if os.path.exists(os.path.abspath(
    '/Users/riwa/Documents/code/tclustr/data')) else '/home/projects/vaccine/people/yatwan/tclustr/data/' if os.path.exists(os.path.abspath('/home/projects/vaccine/people/yatwan/tclustr/data/')) else '/home/projects2/riwa/tclustr/data/'
OUTDIR = '/Users/riwa/Documents/code/tclustr/output/' if os.path.exists(os.path.abspath(
    '/Users/riwa/Documents/code/tclustr/output')) else '/home/projects/vaccine/people/yatwan/tclustr/output/' if os.path.exists(os.path.abspath('/home/projects/vaccine/people/yatwan/tclustr/output/')) else '/home/projects2/riwa/tclustr/output/'
# Stupid hardcoded variable
CNN_FEATS = ['EL_ratio', 'anchor_mutation', 'delta_VHSE1', 'delta_VHSE3', 'delta_VHSE7', 'delta_VHSE8',
             'delta_aliphatic_index',
             'delta_boman', 'delta_hydrophobicity', 'delta_isoelectric_point', 'delta_rank']


def _init(DATADIR):
    #### ==== CONST (blosum, multiprocessing, keys, etc) ==== ####
    # TODO : remove this
    VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
    N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

    MATRIXDIR = f'{DATADIR}Matrices/'
    # ICSDIR = f'{DATADIR}ic_dicts/'
    AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']

    CHAR_TO_INT = dict((c, i) for i, c in enumerate(AA_KEYS))
    INT_TO_CHAR = dict((i, c) for i, c in enumerate(AA_KEYS))

    CHAR_TO_INT['-'] = -1
    INT_TO_CHAR[-1] = '-'

    BG = np.loadtxt(f'{MATRIXDIR}bg.freq.fmt', dtype=float)
    BG = dict((k, v) for k, v in zip(AA_KEYS, BG))

    # BLOSUMS 50
    BL50 = {}
    _blosum50 = np.loadtxt(f'{MATRIXDIR}BLOSUM50', dtype=float).T
    for i, letter_1 in enumerate(AA_KEYS):
        BL50[letter_1] = {}
        for j, letter_2 in enumerate(AA_KEYS):
            BL50[letter_1][letter_2] = _blosum50[i, j]
    BL50_VALUES = {k: np.array([v for v in BL50[k].values()]) for k in BL50}
    # BLOSUMS 62
    BL62_DF = pd.read_csv(f'{MATRIXDIR}BLOSUM62', sep='\s+', comment='#', index_col=0)
    BL62 = BL62_DF.to_dict()
    BL62_VALUES = BL62_DF.drop(columns=['B', 'Z', 'X', '*'], index=['B', 'Z', 'X', '*'])
    BL62_VALUES = dict((x, BL62_VALUES.loc[x].values) for x in BL62_VALUES.index)

    # BLOSUMS 62 FREQS
    _blosum62 = np.loadtxt(f'{MATRIXDIR}BLOSUM62.freq_rownorm', dtype=float).T
    BL62FREQ = {}
    BL62FREQ_VALUES = {}
    for i, letter_1 in enumerate(AA_KEYS):
        BL62FREQ[letter_1] = {}
        BL62FREQ_VALUES[letter_1] = _blosum62[i]
        for j, letter_2 in enumerate(AA_KEYS):
            BL62FREQ[letter_1][letter_2] = _blosum62[i, j]
    # ICS_KL = pkl_load(ICSDIR + 'ics_kl_new.pkl')
    # ICS_SHANNON = pkl_load(ICSDIR + 'ics_shannon.pkl')
    # HLAS = ICS_SHANNON[9].keys()
    ICS_KL = None
    ICS_SHANNON = None
    HLAS = None
    V_MAP = pkl_load(f'{MATRIXDIR}230927_nettcr_dataset_vmap.pkl')
    J_MAP = pkl_load(f'{MATRIXDIR}230927_nettcr_dataset_jmap.pkl')
    PEP_MAP = pkl_load(f'{MATRIXDIR}231031_nettcr_pep_map.pkl')
    PEP_MAP2 = pkl_load(f'{MATRIXDIR}240226_nettcr2-2_alpha_beta_paired_pepmap.pkl')
    return VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, \
           BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, \
           HLAS, ICS_KL, ICS_SHANNON, V_MAP, J_MAP, PEP_MAP, PEP_MAP2


VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, HLAS, ICS_KL, ICS_SHANNON, V_MAP, J_MAP, PEP_MAP, PEP_MAP2 = _init(
    DATADIR)

encoding_matrix_dict = {'onehot': None,
                        'BL62LO': BL62_VALUES,
                        'BL62FREQ': BL62FREQ_VALUES,
                        'BL50LO': BL50_VALUES}
ics_dict = {'KL': ICS_KL,
            'Shannon': ICS_SHANNON}


######################################
####      assertion / checks      ####
######################################

def verify_df(df, seq_col, hla_col, target_col):
    df = copy.deepcopy(df)
    unique_labels = sorted(df[target_col].dropna().unique())
    # Checks binary label
    assert ([int(x) for x in sorted(unique_labels)]) in [[0, 1], [0], [1]], f'Labels are not 0, 1! {unique_labels}'
    # Checks if any seq not in alphabet
    try:
        df = df.drop(df.loc[df[seq_col].apply(lambda x: any([z not in AA_KEYS and not z == '-' for z in x]))].index)
    except:
        print(len(df), df.columns, seq_col, AA_KEYS)
        raise ValueError
    # Checks if HLAs have correct format
    if all(df[hla_col].apply(lambda x: not x.startswith('HLA-'))):
        df[hla_col] = df[hla_col].apply(lambda x: 'HLA-' + x)
    df[hla_col] = df[hla_col].apply(lambda x: x.replace('*', '').replace(':', ''))
    # Check HLA only in subset
    try:
        df = df.query(f'{hla_col} in @HLAS')
    except:
        print(type(df), type(HLAS), HLAS, hla_col)
        raise ValueError(f'{type(df)}, {type(HLAS)}, {HLAS}, {hla_col}, {df[hla_col].unique()}')

    return df


def assert_encoding_kwargs(encoding_kwargs, mode_eval=False):
    """
    Assertion / checks for encoding kwargs and verify all the necessary key-values
    are in
    """
    # Making a deep copy since dicts are mutable between fct calls
    encoding_kwargs = copy.deepcopy(encoding_kwargs)
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'standardize': False}
    essential_keys = ['max_len', 'encoding', 'standardize']
    keys_check = [x in encoding_kwargs.keys() for x in essential_keys]
    keys_check_dict = {k: v for (k, v) in zip(essential_keys, keys_check) if v == False}
    assert all(keys_check), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                            f"{list(keys_check_dict.keys())} are missing!"

    if mode_eval:
        if any([(x not in encoding_kwargs.keys()) for x in ['seq_col', 'hla_col', 'target_col', 'rank_col']]):
            if 'seq_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'seq_col': 'icore_mut'})
            if 'hla_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'hla_col': 'HLA'})
            if 'target_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'target_col': 'agg_label'})
            if 'rank_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'rank_col': 'EL_rank_mut'})

        # This KWARGS not needed in eval mode since I'm using Pipeline and Pipeline
        del encoding_kwargs['standardize']
    return encoding_kwargs


######################################
####      SEQUENCES ENCODING      ####
######################################

#
# def get_aa_properties(df, seq_col='icore_mut', do_vhse=True, prefix=''):
#     """
#     Compute some AA properties that I have selected
#     keep = ['aliphatic_index', 'boman', 'hydrophobicity',
#         'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
#     THIS KEEP IS BASED ON SOME FEATURE DISTRIBUTION AND CORRELATION ANALYSIS
#     Args:
#         df (pandas.DataFrame) : input dataframe, should contain at least the peptide sequences
#         seq_col (str) : column name containing the peptide sequences
#
#     Returns:
#         out (pandas.DataFrame) : The same dataframe but + the computed AA properties
#
#     """
#     out = df.copy()
#
#     out[f'{prefix}aliphatic_index'] = out[seq_col].apply(lambda x: peptides.Peptide(x).aliphatic_index())
#     out[f'{prefix}boman'] = out[seq_col].apply(lambda x: peptides.Peptide(x).boman())
#     out[f'{prefix}hydrophobicity'] = out[seq_col].apply(lambda x: peptides.Peptide(x).hydrophobicity())
#     out[f'{prefix}isoelectric_point'] = out[seq_col].apply(lambda x: peptides.Peptide(x).isoelectric_point())
#     # out['PD2'] = out[seq_col].apply(lambda x: peptides.Peptide(x).physical_descriptors()[1])
#     # out['charge_7_4'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=7.4))
#     # out['charge_6_65'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=6.65))
#     if do_vhse:
#         vhse = out[seq_col].apply(lambda x: peptides.Peptide(x).vhse_scales())
#         # for i in range(1, 9):
#         #     out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
#         for i in [1, 3, 7, 8]:
#             out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
#
#     # Some hardcoded bs
#     return out, ['aliphatic_index', 'boman', 'hydrophobicity',
#                  'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']


def encode_cat(sequence, max_len, pad_value=-1):
    return F.pad(torch.tensor([CHAR_TO_INT[x] for x in sequence]), (0, max_len - (len(sequence))), value=pad_value)


def batch_encode_cat(sequences, max_len, pad_value=-1):
    return torch.stack([encode_cat(x, max_len, pad_value) for x in sequences])


def encode(sequence, max_len=None, encoding='onehot', pad_scale=None):
    """
    encodes a single peptide into a matrix, using 'onehot' or 'blosum'
    if 'blosum', then need to provide the blosum dictionary as argument
    """
    assert encoding in encoding_matrix_dict.keys(), f'Wrong encoding key {encoding} passed!' \
                                                    f'Should be any of {encoding_matrix_dict.keys()}'
    if pad_scale is None:
        pad_scale = 0 if encoding == 'onehot' else -20
    # One hot encode by setting 1 to positions where amino acid is present, 0 elsewhere
    size = len(sequence)
    blosum_matrix = encoding_matrix_dict[encoding]
    if encoding == 'onehot':
        int_encoded = [CHAR_TO_INT[char] for char in sequence]
        onehot_encoded = list()
        for value in int_encoded:
            letter = [0 for _ in range(len(AA_KEYS))]
            letter[value] = 1 if value != -1 else 0
            onehot_encoded.append(letter)
        tmp = np.array(onehot_encoded)

    # BLOSUM encode
    else:
        if blosum_matrix is None or not isinstance(blosum_matrix, dict):
            raise Exception('No BLOSUM matrix provided!')

        tmp = np.zeros([size, len(AA_KEYS)], dtype=np.float32)
        for idx in range(size):
            if sequence[idx] in AA_KEYS:
                tmp[idx, :] = blosum_matrix[sequence[idx]]
            # TODO : Hotfix for Xs in input ; Should probably actually take the BLOSUM50 values instead
            #        But this would mean that we need to expand the actual matrix size from 20 to 21 ...
            elif sequence[idx] == 'X':
                tmp[idx, :] = np.array([pad_scale]).repeat(20)
    # Padding if max_len is provided
    if max_len is not None and max_len > size:
        diff = int(max_len) - int(size)
        try:
            tmp = np.concatenate([tmp, pad_scale * np.ones([diff, len(AA_KEYS)], dtype=np.float32)],
                                 axis=0)
        except:
            print('Here in encode', type(tmp), tmp.shape, len(AA_KEYS), type(diff), type(max_len), type(size), diff, sequence)
            #     return tmp, diff, len(AA_KEYS)
            raise Exception
    return torch.from_numpy(tmp).float()


def encode_batch(sequences, max_len=None, encoding='onehot', pad_scale=None):
    """
    Encode multiple sequences at once.

    encoding should take value : 'onehot', ''BL62LO', 'BL62FREQ', 'BL50LO'.
    """
    if max_len is None:
        max_len = max([len(x) for x in sequences])
    # Contiguous to allow for .view operation
    return torch.stack([encode(seq, max_len, encoding, pad_scale) for seq in sequences]).contiguous()


def onehot_decode(onehot_sequence):
    if type(onehot_sequence) == np.ndarray:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[1]])
    elif type(onehot_sequence) == torch.Tensor:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[:, 1]])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def positional_encode(seq, pad=(0, 0)):
    return F.pad(torch.ones([len(seq)]), pad=(pad[0], pad[1] - len(seq)))


def batch_positional_encode(seqs, pad=(0, 0)):
    return torch.stack([positional_encode(seq, pad) for seq in seqs])


def get_ic_weights(df, ics_dict: dict, max_len=None, seq_col='Peptide', hla_col='HLA', mask=False,
                   invert=False, threshold=0.2):
    """

    Args:
        df:
        ics_dict:
        max_len:
        seq_col:
        hla_col:
        invert: Invert the behaviour; for KL/Shannon, will take IC instead of 1-IC as weight
                For Mask, will amplify MIA positions (by 1.3) instead of setting to 0

    Returns:

    """
    # if 'len' not in df.columns:

    df['len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('len<=@max_len')
    else:
        max_len = df['len'].max()
    # Weighting the encoding wrt len and HLA
    lens = df['len'].values
    pads = [max_len - x for x in lens]
    hlas = df[hla_col].str.replace('*', '').str.replace(':', '').values
    # If mask is true, then the weight is just a 0-1 mask filter
    # Using the conserved / MIAs positions instead of the ICs
    if mask:
        # Get mask for where the values should be thresholded to 0 and 1
        weights = np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(1, 1)) \
                            for l, hla, pad in zip(lens, hlas, pads)])
        # IC > 0.2 goes to 0 because anchor position
        # IC <= 0.2 goes to 1 because rest position
        idx_min = (weights > threshold)
        idx_max = (weights <= threshold)
        if invert:
            weights[idx_min] = 1
            weights[idx_max] = 0
        else:
            weights[idx_min] = 0
            weights[idx_max] = 1

    else:
        if invert:  # If invert, then uses the actual IC as weight
            weights = np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(0, 1)) \
                                for l, hla, pad in zip(lens, hlas, pads)])
        else:  # Else we get the weight with the 1-IC depending on the IC dict provided
            weights = 1 - np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(1, 1)) \
                                    for l, hla, pad in zip(lens, hlas, pads)])

    weights = np.expand_dims(weights, axis=2).repeat(len(AA_KEYS), axis=2)
    return weights




def pad_tensor(tensor, max_len=12, pad_scale=0, how='right'):
    return F.pad(tensor,
                 pad={'right': (0, 0, 0, max_len - tensor.shape[0]), 'left': (0, 0, max_len - tensor.shape[0])}[how],
                 value=pad_scale)


def get_positional_encoding(input_tensor, pad_scale=-15, n=10000):
    batch_size, max_seq_len, n_features = input_tensor.size()

    pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    mask = (~(input_tensor == pad_scale).all(dim=2)).float()
    pos = pos * mask

    div_term = torch.exp(torch.arange(0, n_features, 2, dtype=torch.float32) * (-math.log(n) / n_features))

    pe = torch.zeros(batch_size, max_seq_len, n_features)
    pe[:, :, 0::2] = torch.sin(pos.unsqueeze(2) * div_term)
    pe[:, :, 1::2] = torch.cos(pos.unsqueeze(2) * div_term)

    return pe
