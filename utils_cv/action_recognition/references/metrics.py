# From https://github.com/feiyunzhang/i3d-non-local-pytorch/blob/master/main.py

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals=[]

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), clean_pred=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        _, pred_no_adv = clean_pred.topk(maxk, 1, True, True)
        pred_no_adv = pred_no_adv.t()
        correct_no_adv = pred_no_adv.eq(target.view(1, -1).expand_as(pred_no_adv))

        res = []
        for k in topk:
            correct_k = (correct[:k] * correct_no_adv[:k]).view(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / correct_no_adv[:k].view(-1).float().sum()))
        return res



