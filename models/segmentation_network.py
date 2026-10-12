import torch.nn as nn
import torch


class DownBlock(nn.Module):
    def __init__(self, input_channels, out_channels, maxpool=True, lastDown = False):
        super().__init__()

        self.maxpool = nn.Identity()
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)

        lastDownFactor = 2 if lastDown else 1

        self.conv_1 = nn.Conv2d(input_channels, out_channels * lastDownFactor, kernel_size=3, padding=1, stride=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_channels * lastDownFactor, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        return self.relu_2(self.conv_2(self.relu_1(self.conv_1(self.maxpool(x)))))

class UpBlock(nn.Module):
    def __init__(self, channels, lastUp = False):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=2, padding=0, stride=2)
        self.conv_1 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, stride=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(channels, channels if lastUp else channels//2, kernel_size=3, padding=1, stride=1)
        self.relu_2 = nn.ReLU()

    def forward(self, x, skip):
        ct = self.upsample(x, output_size=skip.shape[2:]) # la taille impaire des images brise le concat de base (53x53)
        cat = torch.cat((ct, skip), dim=1)
        return self.relu_2(self.conv_2(self.relu_1(self.conv_1(cat))))

class SegmentationNetwork(nn.Module):
    def __init__(self, input_channels, n_classes, hidden = 16):
        super(SegmentationNetwork, self).__init__()
        self.hidden = hidden

        self.down_1 = DownBlock(input_channels, self.hidden, maxpool=False)
        self.down_2 = DownBlock(self.hidden, self.hidden*2)
        self.down_3 = DownBlock(self.hidden*2, self.hidden*4)
        self.down_4 = DownBlock(self.hidden*4, self.hidden*4, lastDown=True)

        self.up_5 = UpBlock(self.hidden*4)
        self.up_6 = UpBlock(self.hidden*2)
        self.up_7 = UpBlock(self.hidden, True)

        self.output = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        d_1 = self.down_1(x)
        d_2 = self.down_2(d_1)
        d_3 = self.down_3(d_2)
        d_4 = self.down_4(d_3)

        u_5 = self.up_5(d_4, d_3)
        u_6 = self.up_6(u_5, d_2)
        u_7 = self.up_7(u_6, d_1)

        return self.output(u_7)

    def criterion(self):
        # plusieurs classes possibles, une seule active (pour un même pixel)
        return nn.CrossEntropyLoss() # applique le softmax directement à l'interne
