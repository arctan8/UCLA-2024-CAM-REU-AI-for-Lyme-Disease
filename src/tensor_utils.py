
'''
Helper functions for tensors and numpy arrays
'''
import subprocess
import os
from collections import namedtuple

Gpu = namedtuple('GPU', ['index','name','memory_used', 'memory_total', 'memory_frac', 'utilization'])

def get_gpu_stats():
    '''
    Gets a list of Gpu objects.

    Params: None
    Returns: 
    gpus (list): Gpu objects
    '''
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'])
    result = result.decode('utf-8').strip().split('\n')

    gpus = []
    for line in result:
        # index, name, mem_used, mem_total, util = map(int, line.strip().split(', '))
        index, name, mem_used, mem_total, util = map(str.strip, line.split(',', 4))
        index = int(index)
        mem_used = int(mem_used)
        mem_total = int(mem_total)
        util = int(util)
        gpus.append(Gpu(index=index,
                        name=name,
                        memory_used=mem_used,
                        memory_total=mem_total,
                        memory_frac=mem_used / mem_total,
                        utilization=util))
    return gpus

def choose_best_gpu(gpus, max_util=80, max_mem_frac=0.8):
    '''
    Chooses best GPU, i.e lowest utilization and, failing that, lowest memory percent used.

    Params:
    gpus (list): list of Gpu namedtuples
    max_util (int): maximum utilization percentage
    max_mem_frac (float): maximum memory fraction used
    
    Return:
    Gpu (namedtuple): best GPU

    Raises:
    RuntimeError: if all GPUs at maximum capacity
    '''
    candidates = [gpu for gpu in gpus if gpu.utilization < max_util and gpu.memory_frac < max_mem_frac]

    if not candidates:
        raise RuntimeError("All GPUs at Max Capacity! Please use CPU-based programs instead!")

    candidates.sort(key=lambda x: (x.utilization, x.memory_frac))
    
    return candidates[0]

def assign_best_gpu(set_visible=True, verbose=True):
    '''
    Picks best GPU and optionally sets CUDA_VISIBLE_DEVICES.

    Params:
    set_visible (bool): set CUDA_VISIBLE_DEVICES evironment variable to the highest performing GPU. 
        Caution: does not ensure that this GPU is always used when a tensor is created. Must specify "cuda: <CUDA device number>".
    verbose (bool): prints out relevant GPU information.

    Returns:
        int: device index (CUDA GPU index)
    '''
    gpus = get_gpu_stats()
    best_gpu = choose_best_gpu(gpus)

    physical_vs_cuda = get_physical_vs_cuda(verbose)
    if verbose: print(f'Physical GPU index: {best_gpu.index}')
    
    cuda_index = physical_vs_cuda[best_gpu.index]
    if verbose: print(f'CUDA GPU index: {cuda_index}')
    
    if set_visible:        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_index) 
        
    if verbose:
        memory_used_percent = round(best_gpu.memory_frac * 100, 1)
        print(f"Assigned GPU: {best_gpu.index} (Utilization: {best_gpu.utilization}%, Memory: {memory_used_percent}%)")

    return cuda_index

import torch # Caution: must import torch after assigning os.environ in assign_best_gpu

def get_physical_vs_cuda(verbose=False):
    '''
    Prints out gpu mapping that relates cuda gpu index to nvidia-smi (physical) gpu index.
    
    Return:
    gpu_mapping (dict): keys=physical gpu index, values=cuda index
    '''
    if verbose: print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    smi_map = get_gpu_stats()
    
    if verbose: print("Matching PyTorch CUDA devices to nvidia-smi GPUs:\n")

    matched_indices = set()
    gpu_mapping = dict()

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_total = int(props.total_memory // (1024 ** 2))  # in MB
        name = props.name.strip()
        
        # Match by closest memory and name
        possible_matches = [
            gpu for gpu in smi_map
            if gpu.name == name and abs(gpu.memory_total - mem_total) < 1500
        ]

        # Try to find an unmatched one
        match = None
        for gpu in possible_matches:
            if gpu.index not in matched_indices:
                match = gpu
                matched_indices.add(gpu.index)
                break

        if match:
            if verbose: print(f"cuda:{i} → physical GPU {match.index}: {match.name} ({match.memory_total} MB)")
            gpu_mapping[match.index] = i
        else:
            if verbose: print(f"cuda:{i} → physical GPU (unknown match), name = {name}, mem = {mem_total} MB")

    return gpu_mapping

import numpy as np
import gc

def to_tensor(x, device='cuda', dtype=torch.float32):
    '''
    Convert numpy array or torch.Tensor to torch.Tensor with consistent device and dtype.

    Params:
        x (np.ndarray or torch.Tensor): Input data
        device (str): 'cuda' or 'cpu'. Caution: 'cuda' automatically assigns 'cuda:0'. Please use assign_best_gpu to find best CUDA device number, n. 
            Specify device='cuda:n'.
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

def clear_memory(device=0):
    '''
    Clear CUDA memory: GPU cache, Python garbage collection, and CUDA memory if previous run crashed.

    Params:
    device (int): CUDA device number
    '''
    assert torch.cuda.is_available(), "No GPU Available!"

    print(f'Device: {torch.cuda.get_device_name(device)}')

    print(f'Before Clear: GPU Memory allocated: {torch.cuda.memory_allocated(device)}, Memory reserved: {torch.cuda.memory_reserved()}, Total memory: {torch.cuda.get_device_properties(device).total_memory}')
    
    torch.cuda.empty_cache()
    gc.collect()

    try:
        from numba import cuda
        cuda.select_device(device)
        cuda.close()
    except:
        print(f"Cannot close CUDA device {device}")
        
    print(f'After Clear: GPU Memory allocated: {torch.cuda.memory_allocated(device)}, Memory reserved: {torch.cuda.memory_reserved()}, Total memory: {torch.cuda.get_device_properties(device).total_memory}')


def check_leaks(device=0):
    '''
    Checks for memory leaks on the specified CUDA device, i.e unused tensor objects.

    Params:
    device (int, default=0): CUDA device number. Not same as physical device number.

    Returns: None
    '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(type(obj), obj.size(), obj.device())
        except:
            pass
    
def list_all_processes(device=0):
    '''
    Lists all processes running on the specified physical device.

    Params:
    device (int, default=0): physical device number

    Returns:
    None
    '''
    uuid_cmd = "nvidia-smi --query-gpu=uuid --format=csv,noheader"
    gpu_uuids = subprocess.check_output(uuid_cmd, shell=True).decode().strip().split('\n')
    target_uuid = gpu_uuids[device]

    query_cmd = "nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader,nounits"
    output = subprocess.check_output(query_cmd, shell=True).decode().strip().split('\n')

    print(f"Processes using GPU {device} (UUID: {target_uuid}")

    found = False

    for line in output:
        pid_str, uuid, proc_name = [item.strip() for item in line.split(',')]

        if uuid == target_uuid:
            found = True
            pid = int(pid_str)
            try:
                user = pwd.getpwuid(os.stat(f'/proc/{pid}').st_uid).pw_name
            except Exception:
                user = 'Unknown'
            print(f"PID: {pid} | User: {user} | Process: {proc_name}")

    if not found:
        print("No processes found on this GPU")
