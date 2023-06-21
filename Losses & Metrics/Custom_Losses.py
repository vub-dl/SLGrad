from Requirements import *


class AbsLoss(object):
    r"""An abstract class for loss functions.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        r"""Calculate the loss.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        Return:
            torch.Tensor: The loss.
        """
        pass

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record * bs).sum() / bs.sum()

    def _reinit(self):
        self.record = []
        self.bs = []


class CELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """

    def __init__(self):
        super(CELoss, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class KLDivLoss(AbsLoss):
    r"""The Kullback-Leibler divergence loss function.
    """

    def __init__(self):
        super(KLDivLoss, self).__init__()

        self.loss_fn = nn.KLDivLoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class L1Loss(AbsLoss):
    r"""The Mean Absolute Error (MAE) loss function.
    """

    def __init__(self):
        super(L1Loss, self).__init__()

        self.loss_fn = nn.L1Loss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class MSELoss(AbsLoss):
    r"""The Mean Squared Error (MSE) loss function.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class SegLoss(AbsLoss):
    def __init__(self, reduction='mean'):
        super(SegLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction=self.reduction)

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())


class DepthLoss(AbsLoss):
    def __init__(self, reduction='mean'):
        super(DepthLoss, self).__init__()
        self.reduction = reduction

    def compute_loss(self, pred, gt):
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        print(pred.device)
        if self.reduction == 'mean':
            loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
                0)  # take sum of Absolute error (of non zero gt pixels) and divide it by number of non zero gt pixels =MAE
        elif self.reduction == 'none':
            loss = torch.abs(
                pred - gt) * binary_mask  # keep loss for all non zero points (watch out might not be equal to batchsize)

        return loss


class NormalLoss(AbsLoss):
    def __init__(self, reduction='mean'):
        super(NormalLoss, self).__init__()
        self.reduction = reduction

    def compute_loss(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)

        if self.reduction == 'mean':
            loss = 1 - torch.sum((pred * gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
                0)  # take sum of Absolute error (of non zero gt pixels) and divide it by number of non zero gt pixels =MAE
        elif self.reduction == 'none':
            # print((pred*gt)*binary_mask)
            # print(1-(pred*gt)*binary_mask)
            loss = 1 - (pred * gt * binary_mask).mean(dim=1)
        return loss




def nll(pred, gt, val=False, reduce="mean"):
            if reduce =='none':
              return F.nll_loss(pred, gt, size_average=False, reduction=reduce)
            if val:
              return F.nll_loss(pred, gt, size_average=False)
            else:
              return F.nll_loss(pred, gt)

