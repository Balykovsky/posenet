import torch.nn as nn
import torch.nn.functional as F


class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate = dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.qtn2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        x = self.base_model(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.view(x.size(0), -1)
        pos = self.pos2(x)
        qtn = self.qtn2(x)
        return pos, qtn
