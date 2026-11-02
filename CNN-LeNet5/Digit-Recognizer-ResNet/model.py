import torch.nn as nn
import torchvision.models as models


class LeafResNet(nn.Module):
    def __init__(self, num_classes=176, freeze_backbone=True):
        super(LeafResNet, self).__init__()

        self.backbone = models.resnet50(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_in = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(num_in, num_classes)

    def forward(self, x):
        return self.backbone(x)
