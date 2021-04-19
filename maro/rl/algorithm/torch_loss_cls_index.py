# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn

# For details, see https://pytorch.org/docs/stable/nn.html#loss-functions
TORCH_LOSS_CLS = {
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