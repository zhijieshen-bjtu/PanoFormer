import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from network.GridGenerator import GridGenerator


def genSamplingPattern(h, w, kh, kw, stride=1):
    gridGenerator = GridGenerator(h, w, (kh, kw), stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    # lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    # lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = LonLatSamplingPattern#np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      grid = torch.FloatTensor(grid)
      grid.requires_grad = False

    return grid