import torch


def make_muon_param_groups(params):
    """makes param groups for `pytorch_optimizer.Muon`"""

    if isinstance(params, torch.nn.Module):
        adam_params = []
        muon_params = []
        seen = set()

        for module in params.modules():
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                        adam_params.append(p)
                    else:
                        if p.ndim <= 1:
                            adam_params.append(p)
                        else:
                            muon_params.append(p)

        params_groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adam_params, "use_muon": False},
        ]
        return [g for g in params_groups if len(g["params"]) > 0]


    params = list(params)
    if isinstance(params[0], dict):
        return params

    params_groups = [
        {"params": [p for p in params if p.ndim >= 2], "use_muon": True},
        {"params": [p for p in params if p.ndim < 2], "use_muon": False},
    ]
    return [g for g in params_groups if len(g["params"]) > 0]

