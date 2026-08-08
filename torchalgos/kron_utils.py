"""stuff for kronecker-factored preconditioners"""
from collections.abc import Iterable, Sequence

import numpy as np
import torch


def merge_small_dims(tensors: Sequence[torch.Tensor], max_dim: int, whitelist: int | Sequence[int] | None, blacklist: int | Sequence[int] | None):
    """Merges small dims. The merged dims will always be the last dimension in returned tensor, while ordering of
    unmerged dims relative to each other won't be affected.

    Args:
        tensors: input tensors.
        max_dim: total size of merged dimensions won't exceed this.
        whitelist: dimensions that are allowed to be merged, if None, all dimensions are allowed to be merged.
        blacklist: dimensions that are not allowed to be merged, if None, all dimensions are allowed to be merged.

    Returns:
        tuple `(x, merge_sizes, permute_dims, n_full_dims)`, can be unmerged using `unmerge_dims`.
    """
    assert not isinstance(tensors, torch.Tensor)
    x = tensors[0]

    # Create a list of dimensions that can be merged
    dim_indexes = [i for i,s in enumerate(x.shape) if s <= max_dim]

    if whitelist is not None:
        if isinstance(whitelist, int): whitelist = [whitelist]
        whitelist = [i if i >= 0 else x.ndim - i for i in whitelist]
        dim_indexes = [i for i in dim_indexes if i in whitelist]

    if blacklist is not None:
        if isinstance(blacklist, int): blacklist = [blacklist]
        blacklist = [i if i >= 0 else x.ndim - i for i in blacklist]
        dim_indexes = [i for i in dim_indexes if i not in blacklist]

    if len(dim_indexes) <= 1:
        return tensors, None, None, None

    # Find dims that are small enough to be merged
    merge_total_size = 1
    merge_dim_indexes = []
    merge_sizes = []

    for index in np.argsort(x.shape).tolist():
        if index not in dim_indexes: continue
        size = x.size(index)
        merge_total_size = merge_total_size * size
        if merge_total_size <= max_dim:
            merge_dim_indexes.append(index)
            merge_sizes.append(size)

    if len(merge_dim_indexes) <= 1:
        return tensors, None, None, None

    # Move small dims to the end and flatten
    dims = [i for i in range(x.ndim) if i not in merge_dim_indexes]
    n_full_dims = len(dims)
    permute_dims = dims + merge_dim_indexes
    tensors = [t.permute(permute_dims).flatten(n_full_dims, -1) for t in tensors]

    return tensors, merge_sizes, permute_dims, n_full_dims

def unmerge_small_dims(x: torch.Tensor, merge_sizes: list[int] | None, permute_dims: list[int] | None, n_full_dims: int | None):
    """Unmerge dims of tensor merged via `merge_dims`.

    Args:
        x: input tensor.
        merge_sizes: returned by `merge_dims`.
        permute_dims: returned by `merge_dims`.
        n_full_dims: returned by `merge_dims`.
    """
    if merge_sizes is None: return x

    assert permute_dims is not None
    assert n_full_dims is not None

    x = x.unflatten(dim=n_full_dims, sizes=merge_sizes)
    return x.permute(*np.argsort(permute_dims).tolist())


def make_kron_param_groups_for_emb(model: torch.nn.Module):
    # set embeddings to use diagonal preconditioner
    kron_params = []
    diag_params = []
    seen = set()

    for module in model.modules():
        for p in module.parameters(recurse=False):
            if id(p) not in seen:
                seen.add(id(p))
                if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                    diag_params.append(p)
                else:
                    kron_params.append(p)

    params = [
        {"params": kron_params},
        {"params": diag_params, "precond_dims": None},
    ]

    return [d for d in params if len(d["params"]) > 0]


def kronecker_memory(params: torch.nn.Module | torch.Tensor | Iterable[torch.Tensor], merge_small:bool=True, max_dim:int=10_000, whitelist: int | list[int] | None = None, blacklist: int | list[int] | None = 0):
    """computes total size of tensors required to store kronecker-factored preconditioners"""
    if isinstance(params, torch.nn.Module): params = params.parameters()
    if isinstance(params, torch.Tensor): params = [params,]
    params = list(params)

    memory = 0
    for p in params:

        if merge_small:
            (p, ), *_ = merge_small_dims([p], max_dim, whitelist=whitelist, blacklist=blacklist)

        for dim in p.size():
            if dim > max_dim: memory += dim
            else: memory += dim**2

    return memory

