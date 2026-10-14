import torch.nn as nn
import torch

class DetectionNetwork(nn.Module):
  def __init__(self, in_channels, n_classes):
    super().__init__()
    self.n_classes = n_classes

    kernel_size = 7
    padding = (kernel_size - 1) // 2
    out_channels = 32

    self.bloc1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding), # 27x27x32
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.1), # viens du paper de YOLO
      nn.MaxPool2d(kernel_size=2, stride=2)
    )

    kernel_size = 3
    padding = (kernel_size - 1) // 2

    self.bloc2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels*2, kernel_size=kernel_size, padding=padding), # 13x13x64
      nn.BatchNorm2d(out_channels*2),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.bloc3 =  nn.Sequential(
      nn.Conv2d(out_channels*2, out_channels, kernel_size=1), #6x6x32
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.1),
      nn.Conv2d(out_channels, out_channels*4, kernel_size=3, padding=padding), # 6x6x128
      nn.BatchNorm2d(out_channels*4),
      nn.LeakyReLU(0.1),
      nn.Conv2d(out_channels*4, out_channels*2, kernel_size=1), #6x6x64
      nn.BatchNorm2d(out_channels*2),
      nn.LeakyReLU(0.1),
      nn.Conv2d(out_channels*2, out_channels*8, kernel_size=3, padding=padding), # 6x6x256
      nn.BatchNorm2d(out_channels*8),
      nn.LeakyReLU(0.1),
      nn.Conv2d(out_channels*8, out_channels*4, kernel_size=1), #6x6x128
      nn.BatchNorm2d(out_channels*4),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 3x3x128

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(3*3*out_channels*4, out_channels*2),
      nn.LeakyReLU(0.1),
      nn.Linear(out_channels*2, n_classes*4), # [conf, x , y, size] pour chaque classe
      nn.Sigmoid()
    )

  def forward(self, x):
    N = x.shape[0]
    model_out = self.fc(self.bloc3(self.bloc2(self.bloc1(x)))).reshape(-1, self.n_classes, 4)
    onehot = torch.eye(self.n_classes, device=x.device).unsqueeze(0).expand(N, -1, -1)

    # slot 0 (circle):   [conf, x, y, size, 1, 0, 0]
    # slot 1 (triangle): [conf, x, y, size, 0, 1, 0]
    # slot 2 (cross):    [conf, x, y, size, 0, 0, 1]
    cat_out = torch.cat([model_out, onehot], dim=-1)
    order = cat_out[..., 0].argsort(dim=1, descending=True) # La metric a besoin d'avoir les pred en ordre de confiance sinon elle assigne les mauvaises box

    # recommandé par Claude pour conserver les gradients
    return torch.gather(cat_out, 1, order.unsqueeze(-1).expand(-1, -1, 7))
    

  def criterion(self, alpha = 5):
    bce = nn.BCELoss()
    def loss(pred, boxes):
      N = pred.shape[0]
      class_of_slot = pred[..., 4:].argmax(dim=1)
      order = class_of_slot.argsort(dim=1)
      pred = torch.gather(pred, 1, order.unsqueeze(-1).expand(-1, -1, 7))

      target = torch.zeros(N, 3, 4, device=pred.device)

      for n in range(N):
        for j in range(self.n_classes):
          k = boxes[n, j]
          if k[0]:
            #valid row
            class_idx = int(k[-1])
            target[n, class_idx] = k[:-1]

      confidence = bce(pred[..., 0], target[..., 0])

      present = target[..., 0]
      se = ((pred[..., 1:4] - target[..., 1:4]) ** 2).sum(dim=-1)

      position = (se * present).sum() / present.sum().clamp(min=1)
      
      return confidence + alpha * position

    return loss