import gc
import torch

def free_gpu():
    """Release cached GPU memory held by PyTorch, if available.

    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()     # releases PyTorch cached blocks to CUDA
        torch.cuda.ipc_collect()     # helps clean up inter-process allocations sometimes

# usage
#del calc
#free_gpu()
