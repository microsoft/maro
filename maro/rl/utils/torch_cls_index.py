# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn, optim
from torch.optim import lr_scheduler

# For details, see https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
TORCH_ACTIVATION = {
    "elu": nn.ELU,
    "hard_shrink": nn.Hardshrink,
    "hard_sigmoid": nn.Hardsigmoid,
    "hard_tanh": nn.Hardtanh,
    "hardswish": nn.Hardswish,
    "leaky_relu": nn.LeakyReLU,
    "log_sigmoid": nn.LogSigmoid,
    "multihead_attention": nn.MultiheadAttention,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "soft_plus": nn.Softplus,
    "soft_shrink": nn.Softshrink,
    "soft_sign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanh_shrink": nn.Tanhshrink,
    "threshold": nn.Threshold
}

# For details, see https://pytorch.org/docs/stable/nn.html#loss-functions
TORCH_LOSS = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "ctc": nn.CTCLoss,
    "nll": nn.NLLLoss,
    "poisson_nll": nn.PoissonNLLLoss,
    "kl": nn.KLDivLoss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "margin_ranking": nn.MarginRankingLoss,
    "hinge_embedding": nn.HingeEmbeddingLoss,
    "multi_label_margin": nn.MultiLabelMarginLoss,
    "multi_label_soft_margin": nn.MultiLabelSoftMarginLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "soft_margin": nn.SoftMarginLoss,
    "cosine_embedding": nn.CosineEmbeddingLoss,
    "multi_margin": nn.MultiMarginLoss,
    "triplet_margin": nn.TripletMarginLoss,
}

# For details, see https://pytorch.org/docs/stable/optim.html
TORCH_OPTIM = {
    "sgd": optim.SGD,
    "asgd": optim.ASGD,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamax": optim.Adamax,
    "adamw": optim.AdamW,
    "sparse_adam": optim.SparseAdam,
    "lbfgs": optim.LBFGS,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop
}

# For details, see https://pytorch.org/docs/stable/optim.html
TORCH_LR_SCHEDULER = {
    "lambda": lr_scheduler.LambdaLR,
    "multiplicative": lr_scheduler.MultiplicativeLR,
    "step": lr_scheduler.StepLR,
    "multi_step": lr_scheduler.MultiStepLR,
    "exponential": lr_scheduler.ExponentialLR,
    "cosine_annealing": lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": lr_scheduler.ReduceLROnPlateau,
    "cyclic": lr_scheduler.CyclicLR,
    "one_cycle": lr_scheduler.OneCycleLR,
    "cosine_annealing_warm_restarts": lr_scheduler.CosineAnnealingWarmRestarts
}


def get_torch_activation_cls(activation_type):
    if isinstance(activation_type, str):
        if activation_type not in TORCH_ACTIVATION:
            raise KeyError(f"A string activation_type must be one of {list(TORCH_ACTIVATION.keys())}.")
        return TORCH_ACTIVATION[activation_type]

    return activation_type


def get_torch_loss_cls(loss_type):
    if isinstance(loss_type, str):
        if loss_type not in TORCH_LOSS:
            raise KeyError(f"A string loss_type must be one of {list(TORCH_LOSS.keys())}.")
        return TORCH_LOSS[loss_type]

    return loss_type


def get_torch_optim_cls(optim_type):
    if isinstance(optim_type, str):
        if optim_type not in TORCH_OPTIM:
            raise KeyError(f"A string optim_type must be one of {list(TORCH_OPTIM.keys())}.")
        return TORCH_OPTIM[optim_type]

    return optim_type


def get_torch_lr_scheduler_cls(lr_scheduler_type):
    if isinstance(lr_scheduler_type, str):
        if lr_scheduler_type not in TORCH_LR_SCHEDULER:
            raise KeyError(f"A string lr_scheduler_type must be one of {list(TORCH_LR_SCHEDULER.keys())}.")
        return TORCH_LR_SCHEDULER[lr_scheduler_type]

    return lr_scheduler_type
