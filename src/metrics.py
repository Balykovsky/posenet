import torch

from torch.nn import functional as F
from torch.nn import PairwiseDistance

from neural_pipeline.train_config import AbstractMetric, MetricsProcessor, MetricsGroup


class PoseNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_loss = PairwiseDistance()
        self.qtn_loss = PairwiseDistance()

    def forward(self, outputs, targets):
        pos_loss = self.pos_loss(outputs[0], targets[:, :3])
        qtn_loss = self.qtn_loss(outputs[1], targets[:, 3:])
        losses = 0.5*pos_loss + 0.5*qtn_loss
        return losses.sum()/len(losses)


class PosMetric(AbstractMetric):
    def __init__(self):
        super().__init__('position')

    def calc(self, output: tuple, target: torch.Tensor):
        pos = F.pairwise_distance(output[0], target[:, :3]).cpu().data
        return pos.numpy().copy()


class QtnMetric(AbstractMetric):
    def __init__(self):
        super().__init__('quaternion')

    def calc(self, output: tuple, target: torch.Tensor):
        qtn = F.pairwise_distance(output[1], target[:, 3:]).cpu().data
        return qtn.numpy().copy()


class PosQtnMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str):
        super().__init__()
        self.add_metrics_group(MetricsGroup(stage_name).add(PosMetric()).add(QtnMetric()))
