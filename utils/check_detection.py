import argparse
import os

import torch
from torchvision import transforms

from dataset import ConveyorSimulator
from metrics import MeanAveragePrecisionMetric, detection_intersection_over_union
from models.detection_network import DetectionNetwork
from visualizer import Visualizer
from main import CONFIDENCE_THRESHOLD, INTERSECTION_OVER_UNION_THRESHOLD, CLASS_PROBABILITY_THRESHOLD

CLASS_NAMES = ['circle', 'triangle', 'cross']


def evaluate(model, split, n_images, batch_size):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset = ConveyorSimulator(os.path.join(dir_path, 'data', split), transforms.Compose([transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    metric = MeanAveragePrecisionMetric(3, INTERSECTION_OVER_UNION_THRESHOLD)
    ious = {c: [] for c in range(3)}
    conf_correct = 0
    conf_total = 0
    saved = 0

    with torch.no_grad():
        for image, _, boxes, _ in loader:
            pred = model(image)
            metric.accumulate(pred, boxes)

            for n in range(boxes.shape[0]):
                present = torch.zeros(3)
                for j in range(boxes.shape[1]):
                    if boxes[n, j, 0] == 1:
                        k = int(boxes[n, j, 4])
                        present[k] = 1
                        ious[k].append(detection_intersection_over_union(pred[n, k, 1:4].numpy(), boxes[n, j, 1:4].numpy()))
                predicted_present = (pred[n, :, 0] > CONFIDENCE_THRESHOLD).float()
                conf_correct += (predicted_present == present).sum().item()
                conf_total += 3

                if saved < n_images:
                    visualizer = Visualizer(f'check_{split}_{saved}', 'detection', CLASS_PROBABILITY_THRESHOLD,
                                            CONFIDENCE_THRESHOLD, 0)
                    visualizer.show_detection_prediction(image[n].numpy(), pred[n].numpy(), boxes[n].numpy())
                    saved += 1

    print(f'\n=== {split} ({len(dataset)} images) ===')
    print(f'mAP: {metric.get_value():.3f}')
    print(f'presence accuracy (conf > {CONFIDENCE_THRESHOLD}): {conf_correct / conf_total:.3f}')
    for k, name in enumerate(CLASS_NAMES):
        values = ious[k]
        if values:
            mean_iou = sum(values) / len(values)
            hit_rate = sum(v > INTERSECTION_OVER_UNION_THRESHOLD for v in values) / len(values)
            print(f'{name:>8}: mean IoU {mean_iou:.3f} | IoU > {INTERSECTION_OVER_UNION_THRESHOLD}: {hit_rate:.3f} ({len(values)} objects)')
    print(f'saved {saved} images to figures/check_{split}_*_detection_prediction.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    model = DetectionNetwork(1, 3)
    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'task_detection_best.pt')
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.eval()

    evaluate(model, 'training', args.n_images, args.batch_size)
    evaluate(model, 'test', args.n_images, args.batch_size)
