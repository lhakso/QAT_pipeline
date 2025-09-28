# fold_grace.py (update the path helper)

import re
import torch
import torch.nn as nn

_BRACKET_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)(\[(?P<idx>\d+)\])?$")

def find_parent_and_name(root, dotted_path: str):
    """
    Walk paths like:
      distilbert.transformer.layer[5].ffn.lin2
    or:
      distilbert.transformer.layer.5.ffn.lin2  (if layer is a ModuleList)
    Returns (parent_module, attr_name) so that getattr(parent, attr_name) is the target.
    """
    parts = dotted_path.split(".")
    parent = root

    for i, token in enumerate(parts[:-1]):
        m = _BRACKET_RE.match(token)
        if m:
            name = m.group("name")
            idx  = m.group("idx")
            parent = getattr(parent, name)  # e.g., transformer -> Module or ModuleList
            if idx is not None:
                parent = parent[int(idx)]   # e.g., layer[5]
        else:
            # numeric segment for ModuleList indexing (e.g., ".5.")
            if token.isdigit():
                parent = parent[int(token)]
            else:
                parent = getattr(parent, token)

    # final token can also have [idx]
    last = parts[-1]
    m = _BRACKET_RE.match(last)
    if m and m.group("idx") is not None:
        # target is an indexed child; return its parent and ensure attr exists
        name, idx = m.group("name"), int(m.group("idx"))
        container = getattr(parent, name)
        # expose as attribute-like by returning the container and index tuple
        return (container, idx)  # caller must handle integer 'name'
    else:
        return parent, last

@torch.no_grad()
def fold_adapter_to_linear(model, layer_path: str, seq_len: int, samples: int = 4096):
    root = getattr(model, "model", model)

    parent, name = find_parent_and_name(root, layer_path)

    # Support when the last segment was an index (ModuleList); 'name' will be int
    if isinstance(name, int):
        module = parent[name]
        set_module = lambda new: parent.__setitem__(name, new)
    else:
        module = getattr(parent, name)
        set_module = lambda new: setattr(parent, name, new)

    if isinstance(module, nn.Linear):
        return {'changed': False, 'max_abs_err': 0.0, 'how': 'already_linear'}

    if not hasattr(module, "layer") or not isinstance(module.layer, nn.Linear):
        raise TypeError(f"{layer_path} is not a GRACE adaptor wrapping nn.Linear (got {type(module)})")

    adaptor = module
    base = adaptor.layer
    dev, dt = base.weight.device, base.weight.dtype
    H, O = base.in_features, base.out_features

    N, S = samples, seq_len
    X = torch.randn(N, S, H, device=dev, dtype=dt)   # (N,S,H)
    Y = adaptor(X)                                   # (N,S,O)

    X2 = X.reshape(-1, H)                            # (N*S,H)
    Y2 = Y.reshape(-1, O)                            # (N*S,O)

    ones = torch.ones(X2.size(0), 1, device=dev, dtype=dt)
    X_aug = torch.cat([X2, ones], dim=1)            # (N*S,H+1)

    theta, *_ = torch.linalg.lstsq(X_aug, Y2)       # (H+1,O)
    Wp = theta[:-1].T.contiguous()                  # (O,H)
    bp = theta[-1]                                  # (O,)

    merged = nn.Linear(H, O, bias=True).to(dev).to(dt)
    merged.weight.data.copy_(Wp)
    merged.bias.data.copy_(bp)

    # quick sanity check
    X_test = torch.randn(256, S, H, device=dev, dtype=dt)
    err = (adaptor(X_test) - merged(X_test)).abs().max().item()

    set_module(merged)  # swap adaptor → baked Linear
    return {'changed': True, 'max_abs_err': err, 'how': 'least_squares'}