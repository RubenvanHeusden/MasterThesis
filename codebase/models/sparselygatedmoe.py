import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class SparselyGatedMoE(nn.Module):
    """
    This class implements the sparsely gated Mixture of Experts model as described
    in (Shazeer et al., 2017).

    """

    def __init__(self):
        super(SparselyGatedMoE, self).__init__()

    def forward(self, x):
        pass
