import torch
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
#from networkv5.deformableconv import *
from network.PSA import *
from network.Equioffset import *


class Prepare_equi_offset(nn.Module):
    def __init__(self, b, k, h, w):
        super(Prepare_equi_offset, self).__init__()
        self.b = b
        self.k = k
        self.h = h
        self.w = w

    def forward(self, x):
        x = x.view(self.b, self.k, -1, self.h, self.w)
        x = x.permute(0, 1, 3, 4, 2)
        return x