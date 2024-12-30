import glob
import os
from typing import Dict

from src.utils import mkdirs
from torch.nn import LeakyReLU, ELU, SELU, ReLU
import json
from src.models import *
from src.conv_models import *

ACT_DICT = {'SELU': nn.SELU(), 'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(), 'ELU': nn.ELU()}


def paired_batch_generator(x_tensor_a, x_tensor_b, batch_size):
    """
    Generator that yields sequential batches from x_tensor, correctly handling the last batch.

    Args:
    - x_tensor (Tensor): The input tensor.
    - batch_size (int): The size of each batch.

    Yields:
    - Tensor: A batch of data.
    """
    assert x_tensor_a.size(0) == x_tensor_b.size(0), f'Size mismatch! {x_tensor_a.size(0)}, {x_tensor_b.size(0)}'
    num_samples = x_tensor_a.size(0)
    for start_idx in range(0, num_samples, batch_size):
        # This automatically adjusts to return all remaining data for the last batch
        yield x_tensor_a[start_idx:start_idx + batch_size], x_tensor_b[start_idx:start_idx + batch_size]


def batch_generator(x_tensor, batch_size):
    """
    Generator that yields sequential batches from x_tensor, correctly handling the last batch.

    Args:
    - x_tensor (Tensor): The input tensor.
    - batch_size (int): The size of each batch.

    Yields:
    - Tensor: A batch of data.
    """
    num_samples = x_tensor.size(0)
    for start_idx in range(0, num_samples, batch_size):
        # This automatically adjusts to return all remaining data for the last batch
        yield x_tensor[start_idx:start_idx + batch_size]


def get_available_device():
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Using CPU.")
        return 'cpu'

    print(f"Number of available GPUs: {num_gpus}")

    # Check if GPUs are currently in use
    in_use = [torch.cuda.memory_allocated(i) > 100 for i in range(num_gpus)]

    # Select the first available GPU that is not in use
    for i in range(num_gpus):
        if not in_use[i]:
            print(f"Using GPU {i}")
            return f'cuda:{i}'

    # If all GPUs are in use, fall back to CPU
    print("All GPUs are in use. Using CPU.")
    return 'cpu'


def mask_modality(tensor, mask, fill_value: float = 0.):
    """
    Check that the shapes match, and broadcast if needed in a very crude manner
    Used for example to mask inputs or loss for datapoints with missing modalities
    Assumes (and asserts) that mask has a dimensions <= tensor's dim
    Args:
        tensor: tensor to mask
        mask: Binary mask, should be binary and at least have the same number of elements as tensor (shape[0])
        fill_value (float): value with which to fill the masked version of the tensor. Use 0 when disabling gradients
                            Could be another value to make it easier to mask a reconstructed sequence tensor.
                            For example (ex set to -99) then use tensor[tensor!=-99] to index
    Returns:
        masked_tensor: same tensor but with some elements set to zero
    """
    assert len(mask.shape) <= len(tensor.shape) and mask.shape[0] == tensor.shape[
        0], f'Check mask/tensor dimensions! Mask: {mask.shape} ; Tensor : {tensor.shape}'
    if mask.shape != tensor.shape:
        while len(mask.shape) < len(tensor.shape):
            mask = mask.unsqueeze(1)

    return torch.where(mask.bool(), tensor, torch.full_like(tensor, fill_value))


def filter_modality(tensor, mask, fill_value=-99):
    masked_tensor = mask_modality(tensor, mask, fill_value)
    new_mask = (masked_tensor != fill_value).bool()
    while len(new_mask.shape) > 1:
        new_mask = new_mask[:, 0]
    return masked_tensor[new_mask]


def get_model(folder, **kwargs):
    pt = glob.glob(folder + '/*checkpoint_best*.pt')
    pt = [x for x in pt if 'interval' not in x][0]
    js = glob.glob(folder + '/*checkpoint*.json')[0]
    model = load_model_full(pt, js, **kwargs)
    return model


def load_model_full(checkpoint_filename, json_filename, dir_path=None,
                    return_json=False, verbose=True, return_best_dict=False, **kwargs):
    """
    Instantiate and loads a model directly from a checkpoint and json filename
    Args:
        checkpoint_filename:
        json_filename:
        dir_path:
    Returns:

    """
    dict_kwargs = load_json(json_filename, dir_path, **kwargs)
    assert 'constructor' in dict_kwargs.keys(), f'No constructor class name provided in the dict_kwargs keys! {dict_kwargs.keys()}'
    constructor = dict_kwargs.pop('constructor')
    if constructor == 'BimodalVAEClassifier':
        constructor = 'TwoStageVAECLF'
    if 'activation' in dict_kwargs:
        dict_kwargs['activation'] = eval(dict_kwargs['activation'])()
    for k in dict_kwargs:
        if type(dict_kwargs[k]) == dict:
            for l in dict_kwargs[k]:
                if l == 'activation':
                    dict_kwargs[k]['activation'] = eval(dict_kwargs[k]['activation'])()
    model = eval(constructor)(**dict_kwargs)
    model = load_checkpoint(model, checkpoint_filename, dir_path, verbose, return_best_dict, **kwargs)
    if return_best_dict:
        model, best = model
        dict_kwargs['best'] = best
    if return_json:
        return model, dict_kwargs
    else:
        return model


def save_model_full(model, checkpoint_filename='checkpoint.pt', dir_path='./', verbose=False, best_dict=None,
                    dict_kwargs=None, json_filename=None):
    """
    Saves a torch model (.pt) along with its init parameters in a JSON file
    Args:
        model: Model object
        checkpoint_filename:
        dir_path:
        verbose:
        best_dict:
        dict_kwargs:
        json_filename:

    Returns:

    """
    if json_filename is None:
        json_filename = f'{checkpoint_filename.split(".pt")[-2]}_JSON_kwargs.json' if checkpoint_filename.endswith(
            '.pt') \
            else f'{checkpoint_filename}_JSON_kwargs.json'

    save_checkpoint(model, checkpoint_filename, dir_path, verbose, best_dict)
    if 'constructor' not in dict_kwargs.keys():
        dict_kwargs['constructor'] = model.__class__.__name__
    save_json(dict_kwargs, json_filename, dir_path)
    print(
        f'Model weights saved at {os.path.abspath(os.path.join(dir_path, checkpoint_filename))} ' \
        f'and JSON at {os.path.abspath(os.path.join(dir_path, json_filename))}')


def load_json(filename, dir_path=None, **kwargs):
    """
    Loads a dictionary from a .json file and returns it
    Args:
        filename:

    Returns:
        dict_kwargs: A dictionary containing the kwargs necessary to instantiate a given model
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    with open(filename, 'r') as json_file:
        dict_kwargs = json.load(json_file)
    return dict_kwargs


def save_json(dict_kwargs, filename, dir_path='./'):
    """
    Saves a dictionary to a .json file
    When saving a model, should try to ensure that the model's constructor // class name exists
    Args:
        dict_kwargs:
        filename:
        dir_path:

    Returns:

    """
    savepath = os.path.join(dir_path, filename)
    for k in dict_kwargs:
        if type(dict_kwargs[k]) == dict:
            for l in dict_kwargs[k]:
                if issubclass(type(dict_kwargs[k][l]), nn.Module):
                    dict_kwargs[k][l] = dict_kwargs[k][l].__class__.__name__
        if issubclass(type(dict_kwargs[k]), nn.Module):
            dict_kwargs[k] = dict_kwargs[k].__class__.__name__
    # Write the dictionary to a JSON file
    with open(savepath, 'w') as json_file:
        json.dump(dict_kwargs, json_file)
    print(f"JSON data has been written to {savepath}")


def save_checkpoint(model, filename: str = 'checkpoint.pt', dir_path: str = './',
                    verbose=False,
                    best_dict: dict = None):
    """
    Saves a single torch model, with some sanity checks
    Args:
        model: torch model (i.e. anything that inherits from nn.Module and has a state_dict())
        name: the name itself (ex: CNN_model_t0_v1 for a CNN model from the test fold 0, validation fold 1)
        dir_path: path to the directory where the models should be saved

    Returns:
        nothing.
    """
    # Small sanity checks
    assert hasattr(model, 'state_dict'), f'Object of type {type(model)} has no state_dict and can\'t be saved!'
    # Bad practice but fuck it
    if not os.path.exists(dir_path):
        mkdirs(dir_path)
        if verbose:
            print(f'Creating {dir_path}; The provided dir path {dir_path} did not exist!')
    if dir_path.startswith('../output/../output/'):
        dir_path = dir_path[:10]+dir_path[20:]
    savepath = os.path.join(dir_path, filename)
    checkpoint = model.state_dict()
    if best_dict is not None and type(best_dict) == dict:
        checkpoint['best'] = best_dict

    torch.save(checkpoint, savepath)


    if verbose:
        print(f'Model saved at {os.path.abspath(savepath)}')


def load_checkpoint(model, filename: str, dir_path: str = None, verbose=True,
                    return_dict=False,
                    **kwargs):
    """
    Loads a model
    Args:
        model:
        name:
        dir_path:

    Returns:
        model: Loaded model, in eval mode.
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    try:
        if 'map_location' not in kwargs:
            DEVICE = get_available_device()
            kwargs['map_location'] = DEVICE
        checkpoint = torch.load(filename, **kwargs)
        if 'best' in checkpoint.keys():
            best = checkpoint.pop('best')
            if verbose:
                print('Reloading best model:')
                for k, v in best.items():
                    print(f'{k}: {v}')
        model.load_state_dict(checkpoint)
    except:
        st = torch.load(filename)
        print(st.keys())
        raise ValueError()
    model.eval()
    if return_dict:
        return model, best
    return model
