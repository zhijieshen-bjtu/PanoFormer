from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(10)
torch.cuda.manual_seed(10)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None, d_map=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        #valid_mask = (target > 0).detach()
        #if mask is not None:
            #valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        #diff = diff[valid_mask]
        #d_map = d_map[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
