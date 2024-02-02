import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)


    def forward(self, x):
        x = self.model(x)

        return x