import torch.nn as nn
import torch

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # skip connection (appelé downsample quand on print out resnet donc je garde le meme nom)
    self.downsample = nn.Identity()
    if stride != 1 or in_channels != out_channels:
        self.downsample = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
        )

  def forward(self, x):
    out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
    return self.relu(out + self.downsample(x))

class ClassificationNetwork(nn.Module):
  def __init__(self, input_channels, n_classes):
    super().__init__()

    # la tête du model
    self.head = nn.Sequential(
      nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False), # on garde la taille de l'image car elles sont déjà petites (comparé au 224x224 de base de ResNet)
      nn.BatchNorm2d(16),
      nn.ReLU(),
    ) # 53 x 53 x 16

    # resNet: quand la résolution coupe de moitié (stride=2), on double les channels
    self.layer1 = BasicBlock(16, 16, stride=1) # 53 x 53 x 16
    self.layer2 = BasicBlock(16, 32, stride=2) # 27 x 27 x 32
    self.layer3 = BasicBlock(32, 64, stride=2) # 14 x 14 x 64
    self.layer4 = BasicBlock(64, 64, stride=2) # 7 x 7 x 64

    self.avgpool = nn.AdaptiveAvgPool2d(1)

    # la queue pour faire la classification
    self.tail = nn.Sequential(
        nn.Linear(in_features=64, out_features=n_classes),
        nn.Sigmoid()
    ) # 3 classes en output

  def forward(self, x):
    o_head = self.head(x)
    o_1 = self.layer1(o_head)
    o_2 = self.layer2(o_1)
    o_3 = self.layer3(o_2)
    o_4 = self.layer4(o_3)
    pooled = torch.flatten(self.avgpool(o_4), 1)
    out = self.tail(pooled)
    return out

  def criterion(self):
    return nn.BCELoss()