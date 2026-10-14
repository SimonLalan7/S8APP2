import torch
from models.detection_network import DetectionNetwork

model = DetectionNetwork(1, 3)
criterion = model.criterion()

boxes = torch.tensor([[[1, 0.2, 0.3, 0.4, 2],
                       [1, 0.7, 0.5, 0.3, 0],
                       [0, 0,   0,   0,   0]]], dtype=torch.float32)
boxes_before = boxes.clone()

def make_pred(rows):
    return torch.cat([torch.tensor([rows], dtype=torch.float32),
                      torch.eye(3).unsqueeze(0)], dim=-1)

perfect = make_pred([[1, 0.7, 0.5, 0.3],
                     [0, 0.9, 0.9, 0.9],
                     [1, 0.2, 0.3, 0.4]])
swapped = make_pred([[1, 0.2, 0.3, 0.4],
                     [0, 0.0, 0.0, 0.0],
                     [1, 0.7, 0.5, 0.3]])
missed = make_pred([[0, 0.7, 0.5, 0.3],
                    [0, 0.0, 0.0, 0.0],
                    [1, 0.2, 0.3, 0.4]])

print("1. perfect (expect 0):        ", criterion(perfect, boxes).item())
print("2. swapped boxes (expect > 0):", criterion(swapped, boxes).item())
print("3. missed circle (expect big):", criterion(missed, boxes).item())

pred = model(torch.rand(4, 1, 53, 53))
print("4. output shape (expect [4, 3, 7]):", pred.shape)

loss = criterion(pred, boxes.expand(4, -1, -1))
loss.backward()
print("5. backward OK, loss =", loss.item())

print("6. boxes unchanged:", torch.equal(boxes, boxes_before))
print("7. params (expect < 400k):", sum(p.numel() for p in model.parameters()))