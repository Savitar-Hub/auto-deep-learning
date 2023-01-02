from typing import Any, Tuple

from torch import optim

from auto_deep_learning.utils.config import ConfigurationObject

conf_obj = ConfigurationObject()


def weight_decay_params(params) -> Tuple[Any, Any]:
    """Get for which parameters we can have weight decay.

    Args:
        params (params): the parameters of the model

    Returns:
        wd_params: the parameters of the model for which we can have weight decay
        not_wd_params: the parameters of the model for which we can't have weight decay
    """

    wd_params, not_wd_params = [], []

    for param in params:
        param_list = not_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, not_wd_params


def get_optimizer(
    model,
    lr: float = 1e-4,
    filter_by_requires_grad=False,
    betas: Tuple[float, float] = (0.9, 0.99),
    eps: float = 1e-8,
    wd: float = 1e-2,
):
    """Get optimizer for the model parameters

    Args:
        model (Model): the model for which we want to apply an optimizer
        lr (float, optional): learning rate of the optimizer. Defaults to 1e-4.
        filter_by_requires_grad (bool, optional): if we want to filter by the params that have gradient desc activated.
            Defaults to False.
        betas (Tuple[float], optional): the betas for the Adam optimizer. Defaults to (.9, .99).
        eps (float, optional): the epsilon for the Adam optimizer. Defaults to 1e-8.
        wd (float, optional): weight decay of the learning rate. Defaults to 1e-2.

    Returns:
        optimizer: the optimizer for the model
    """

    params = model.parameters()

    # If we want to filter by the params that require gradient
    if filter_by_requires_grad:
        params = list(filter(lambda p: p.required_grad, params))

    if conf_obj.optimizer == "adam":
        if wd == 0:
            return optim.Adam(params, lr=lr, betas=betas, eps=eps)

        # Get which parameters can have weight decay
        wd_params, not_wd_params = weight_decay_params(params)

        params = [
            {"params": wd_params},
            {
                "params": not_wd_params,
                "weight_decay": 0,
            },  # For the ones that do not have wd
        ]

        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
