'''
Helper functions for tensors and numpy arrays
'''
import torch
import numpy as np

def to_tensor(x, device='cuda', dtype=torch.float32):
    '''
    Convert numpy array or torch.Tensor to torch.Tensor with consistent device and dtype.

    Params:
        x (np.ndarray or torch.Tensor): Input data
        device (str): 'cuda' or 'cpu'
        dtype(torch.dtype): torch.float32 or torch.float64
    Return:
        torch.Tensor: Tensor on specified device and dtype
    '''
    assert torch.cuda.is_available(), "No GPU Available!"
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got{type(x)}")

    return x.to(device=device, dtype=dtype)

def to_numpy(x):
    '''
    Converts torch.Tensory to np.array
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
