from Requirements import *


class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task.
    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).
        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []


# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """

    def __init__(self):
        super(AccMetric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(sum(self.record) / sum(self.bs))]


# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """

    def __init__(self):
        super(L1Metric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / (sum(batch_size))]

# seg
class SegMetric(AbsMetric):
    def __init__(self):
        super(SegMetric, self).__init__()

        self.num_classes = 13
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update_fun(self, pred, gt):
        self.record = self.record.to(pred.device)
        pred = pred.softmax(1).argmax(1).flatten()
        gt = gt.long().flatten()
        k = (gt >= 0) & (gt < self.num_classes)
        inds = self.num_classes * gt[k].to(torch.int64) + pred[k]
        self.record += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def score_fun(self):
        h = self.record.float()

        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc = torch.diag(h).sum() / h.sum()

        return torch.mean(iu)  # return only MIOU

    def reinit(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)


# depth
class DepthMetric(AbsMetric):
    def __init__(self):
        super(DepthMetric, self).__init__()

    def update_fun(self, pred, gt):
        device = pred.device
        self.bs = pred.size()[0]
        binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)
        pred = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / gt
        self.abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0))
        self.rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0))
        # self.abs_recordappend(abs_err)
        # self.rel_record.append(rel_err)


    def score_fun(self):
        return self.rel_err  # return only the relative error


# normal
class NormalMetric(AbsMetric):
    def __init__(self):
        super(NormalMetric, self).__init__()

    def update_fun(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(pred * gt, 1).masked_select(binary_mask), -1, 1)).detach()
        error = torch.rad2deg(error)
        self.record = error

    def score_fun(self):
        return torch.mean(self.record)  # return only the mean