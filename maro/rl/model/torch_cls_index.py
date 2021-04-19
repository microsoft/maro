from torch import optim
from torch.optim import lr_scheduler

# For details, see https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
TORCH_ACTIVATION_CLS = {
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

# For details, see https://pytorch.org/docs/stable/optim.html
TORCH_OPTIM_CLS = {
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
TORCH_LR_SCHEDULER_CLS = {
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
