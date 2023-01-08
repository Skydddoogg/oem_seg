import torch
import torch.nn as nn
from . import metrics

# -------------------
# --- JaccardLoss ---
# -------------------
class JaccardLoss(nn.Module):
    def __init__(self, class_weights=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "Jaccard"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        return losses

class GroupedJaccardLoss(nn.Module):
    def __init__(self, class_group, class_weights=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "GroupedJaccard"
        self.class_group = class_group

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)

        total_loss = 0

        # calculate average loss of majority samples
        losses = 0
        for i in self.class_group['majority']:
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        total_loss += losses

        # calculate average loss of minority samples
        losses = 0
        for i in self.class_group['minority'][1:]:  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        total_loss += losses

        return total_loss


# ----------------
# --- DiceLoss ---
# ----------------
class DiceLoss(nn.Module):
    def __init__(self, class_weights=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "Dice"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.iou(ypr, ygt)
        return losses


# ------------------------
# --- CEWithLogitsLoss ---
# ------------------------
class CEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = torch.from_numpy(weight).float() if weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CE"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss


# -----------------
# --- FocalLoss ---
# -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", None]
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.name = "Focal"

    def forward(self, input, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            input, target.float(), reduction="none"
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()
        return focal_loss

class HybridFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.name = 'HybridFocalLoss'

    def forward(self, input, target):

        losses = 0
        for i in range(input.shape[1]):

            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]

            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                ypr, ygt.float(), reduction="none"
            )

            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            losses += focal_loss.mean()
        return losses

class GroupedFocalLoss(nn.Module):
    def __init__(self, class_group, alpha=1, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.name = 'GroupedFocalLoss'

        self.class_group = class_group

    def forward(self, input, target):

        total_loss = 0

        losses = 0
        for i in self.class_group['majority']:

            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]

            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                ypr, ygt.float(), reduction="none"
            )

            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            losses += focal_loss.mean()
        total_loss += losses

        losses = 0
        for i in self.class_group['minority'][1:]:  # background is not included

            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]

            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                ypr, ygt.float(), reduction="none"
            )

            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            losses += focal_loss.mean()
        total_loss += losses

        return total_loss

# ---------------
# --- MCCLoss ---
# ---------------
class MCCLoss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.name = "MCC"

    def forward(self, input, target):
        bs = target.shape[0]

        input = torch.sigmoid(input)

        target = target.view(bs, 1, -1)
        input = input.view(bs, 1, -1)

        tp = torch.sum(torch.mul(input, target)) + self.eps
        tn = torch.sum(torch.mul((1 - input), (1 - target))) + self.eps
        fp = torch.sum(torch.mul(input, (1 - target))) + self.eps
        fn = torch.sum(torch.mul((1 - input), target)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss

class HybridMCCLoss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.name = "HybridMCC"

    def forward(self, input, target):
        losses = 0
        for i in range(input.shape[1]):

            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            
            bs = ygt.shape[0]

            ypr = torch.sigmoid(ypr)

            ygt = ygt.view(bs, 1, -1)
            ypr = ypr.view(bs, 1, -1)

            tp = torch.sum(torch.mul(ypr, ygt)) + self.eps
            tn = torch.sum(torch.mul((1 - ypr), (1 - ygt))) + self.eps
            fp = torch.sum(torch.mul(ypr, (1 - ygt))) + self.eps
            fn = torch.sum(torch.mul((1 - ypr), ygt)) + self.eps

            numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
            denominator = torch.sqrt(
                torch.add(tp, fp)
                * torch.add(tp, fn)
                * torch.add(tn, fp)
                * torch.add(tn, fn)
            )

            mcc = torch.div(numerator.sum(), denominator.sum())
            loss = 1.0 - mcc

            losses += loss

        return losses

# ----------------
# --- OHEMLoss ---
# ----------------
class OHEMBCELoss(nn.Module):
    """
    Taken and modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(OHEMBCELoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.name = "OHEM"

    def forward(self, input, target):

        probs = torch.sigmoid(input)[:, 0, :, :].float()
        ygt = target[:, 0, :, :].float()

        # keep hard examples
        kept_flag = torch.zeros_like(probs).bool()
        # foreground pixels with low foreground probability
        kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
        # background pixel with high foreground probability
        kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

        if kept_flag.sum() < self.min_kept:
            # hardest examples have a probability closest to 0.5.
            # The network is very unsure whether they belong to the foreground
            # prob=1 or background prob=0
            hardest_examples = torch.argsort(
                torch.abs(probs - 0.5).contiguous().view(-1)
            )[: self.min_kept]
            kept_flag.contiguous().view(-1)[hardest_examples] = True
        return self.criterion(input[kept_flag, 0], target[kept_flag, 0].float())

class OHEMFLLoss(nn.Module):
    """
    Taken and modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(OHEMFLLoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = FocalLoss(gamma=2, alpha=0.25)
        self.name = "OHEMFL"

    def forward(self, input, target):

        probs = torch.sigmoid(input)[:, 0, :, :].float()
        ygt = target[:, 0, :, :].float()

        # keep hard examples
        kept_flag = torch.zeros_like(probs).bool()
        # foreground pixels with low foreground probability
        kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
        # background pixel with high foreground probability
        kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

        if kept_flag.sum() < self.min_kept:
            # hardest examples have a probability closest to 0.5.
            # The network is very unsure whether they belong to the foreground
            # prob=1 or background prob=0
            hardest_examples = torch.argsort(
                torch.abs(probs - 0.5).contiguous().view(-1)
            )[: self.min_kept]
            kept_flag.contiguous().view(-1)[hardest_examples] = True
        return self.criterion(input[kept_flag, 0], target[kept_flag, 0].float())


class HybridOHEMBCELoss(nn.Module):
    """
    Taken and modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(HybridOHEMBCELoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.name = "HybridOHEM"

    def forward(self, input, target):
        losses = 0
        for i in range(1, input.shape[1]):

            class_wise_input = input[:, [0, i], :, :]
            class_wise_target = target[:, [0, i], :, :]

            probs = torch.sigmoid(class_wise_input)[:, 0, :, :].float()
            ygt = class_wise_target[:, 0, :, :].float()

            # keep hard examples
            kept_flag = torch.zeros_like(probs).bool()
            # foreground pixels with low foreground probability
            kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
            # background pixel with high foreground probability
            kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

            if kept_flag.sum() < self.min_kept:
                # hardest examples have a probability closest to 0.5.
                # The network is very unsure whether they belong to the foreground
                # prob=1 or background prob=0
                hardest_examples = torch.argsort(
                    torch.abs(probs - 0.5).contiguous().view(-1)
                )[: self.min_kept]
                kept_flag.contiguous().view(-1)[hardest_examples] = True
            losses += self.criterion(class_wise_input[kept_flag, 0], class_wise_target[kept_flag, 0].float())
        
        return losses

class HybridOHEMFLLoss(nn.Module):
    """
    Taken and modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(HybridOHEMFLLoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = FocalLoss(gamma=2, alpha=0.25)
        self.name = "HybridOHEMFL"

    def forward(self, input, target):
        losses = 0
        for i in range(1, input.shape[1]):

            class_wise_input = input[:, [0, i], :, :]
            class_wise_target = target[:, [0, i], :, :]

            probs = torch.sigmoid(class_wise_input)[:, 0, :, :].float()
            ygt = class_wise_target[:, 0, :, :].float()

            # keep hard examples
            kept_flag = torch.zeros_like(probs).bool()
            # foreground pixels with low foreground probability
            kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
            # background pixel with high foreground probability
            kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

            if kept_flag.sum() < self.min_kept:
                # hardest examples have a probability closest to 0.5.
                # The network is very unsure whether they belong to the foreground
                # prob=1 or background prob=0
                hardest_examples = torch.argsort(
                    torch.abs(probs - 0.5).contiguous().view(-1)
                )[: self.min_kept]
                kept_flag.contiguous().view(-1)[hardest_examples] = True
            losses += self.criterion(class_wise_input[kept_flag, 0], class_wise_target[kept_flag, 0].float())
        
        return losses

