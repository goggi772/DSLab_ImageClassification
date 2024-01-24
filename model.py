import torch
import torch.nn as nn
from torchvision import models



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # pretrained model

        # for param in self.resnet18.parameters():
        #     param.requires_grad = False

        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, 2)

    def forward(self, x):
        x = self.resnet34(x)

        return x