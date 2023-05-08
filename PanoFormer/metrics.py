import os
import torch
#==========================
# Depth Prediction Metrics
#==========================

def compute_depth_metrics(gt, pred, mask=None, median_align=True):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt
    pred_depth = pred

    gt_depth[gt_depth<=0.1] = 0.1
    pred_depth[pred_depth<=0.1] = 0.1
    gt_depth[gt_depth >= 16] = 16
    pred_depth[pred_depth >= 16] = 16


    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################

    rmse = (gt_depth - pred_depth) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt_depth) - torch.log10(pred_depth)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt_depth - pred_depth))

    abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth)

    sq_rel = torch.mean((gt_depth - pred_depth) ** 2 / gt_depth)

    log10 = torch.mean(torch.abs(torch.log10(pred / gt_depth)))

    mae = torch.mean(torch.abs((pred_depth - gt_depth)) / gt_depth)
    
    mre = torch.mean(((pred_depth - gt_depth)** 2) / gt_depth)
    #mae = (gt_depth - pred_depth).abs().mean()
    #mre = ((gt_depth - pred_depth).abs() / gt).mean()

    return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):

    def __init__(self, median_align=True):

        self.median_align = median_align
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/mre"] = AverageMeter()
        self.metrics["err/mae"] = AverageMeter()
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/mre"].reset()
        self.metrics["err/mae"].reset()
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

        self.metrics["err/mre"].update(mre, N)
        self.metrics["err/mae"].update(mae, N)
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/mre"].avg)
        avg_metrics.append(self.metrics["err/mae"].avg)
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 11).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3"), file=f)
                print(("&  {: 8.5f} " * 11).format(*avg_metrics), file=f)
