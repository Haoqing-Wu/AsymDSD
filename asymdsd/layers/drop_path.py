import torch
from torch import nn

from ..components.common_types import LayerFn


def drop_path(
    x: torch.Tensor,
    drop_p: float = 0.1,
    training: bool = False,
) -> torch.Tensor:
    if drop_p == 0.0 or not training:
        return x

    keep_p = 1 - drop_p

    # Unsqueezes to match dim of input
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    drop_batch = x.new_empty(shape).bernoulli_(keep_p)

    # Scaling to keep expected value of x the same
    x = x * drop_batch / keep_p

    return x


class DropPath(nn.Module):
    # Wrapper class for path_fn
    def __init__(
        self,
        drop_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert 0.0 <= drop_p <= 1.0
        self.drop_p = drop_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, drop_p=self.drop_p, training=self.training)


def drop_path_efficient(
    x: torch.Tensor,
    *layer_args,
    path_fn: LayerFn,
    drop_p: float = 0.1,
    training: bool = False,
    residual_add: bool = True,
    **layer_kwargs,
) -> torch.Tensor:
    if not training or drop_p == 0.0:
        residual = path_fn(x, **layer_kwargs)
        return x + residual if residual_add else residual

    keep_p = 1 - drop_p
    batch_size = x.shape[0]

    # To fixed batch size (Alternative would yield variable batch size)
    keep_batch_size = max(1, round(keep_p * batch_size))
    keep_indices = torch.randperm(batch_size, device=x.device)[:keep_batch_size]

    layer_args = list(layer_args)
    for i, arg in enumerate(layer_args):
        if isinstance(arg, torch.Tensor) and arg.shape[0] == batch_size:
            layer_args[i] = arg[keep_indices]

    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
            layer_kwargs[k] = v[keep_indices]

    residual = path_fn(x[keep_indices], *layer_args, **layer_kwargs) / keep_p

    if residual_add:
        x = x.clone()
        x[keep_indices] = x[keep_indices] + residual
    else:
        zeros = torch.zeros_like(x)
        zeros[keep_indices] = residual
        x = zeros

    return x
