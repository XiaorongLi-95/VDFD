
from importlib import import_module
import torch

def factory(subdir, module_name, func):
    module = import_module(
        '.' + module_name, package=subdir
    )
    model = getattr(module, func)
    return model


def count_parameter(model):
    clf_param_num = sum(p.numel() for p in model.parameters())
    return clf_param_num


def tensor_diag(a, device):
    # a is 2D matrix
    # return 3D tensor
    b = torch.eye(a.size(1)).to(device)
    c = a.unsqueeze(2).expand(*a.size(), a.size(1))
    d = c * b
    return d



