from models.detection_network import DetectionNetwork
import torch

print(DetectionNetwork(1, 3)(torch.zeros(2, 1, 53, 53)).shape)