import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch
pred_path = "/Users/ben/Desktop/Programing/Python/evaluate/pred.json"
target_path = "/Users/ben/Desktop/Programing/Python/evaluate/target.json"

def get_gt_data(annotations):
    gt_data = {
               'boxes':[],
               'labels':[]
               }
    
    # gt_data['image_id'] = imageID
    gt_data['boxes'] = annotations['boxes']
    gt_data['labels'] = annotations['labels']
    
    return gt_data

with open(pred_path, 'r') as f:
    preds = json.load(f)

with open(target_path, 'r') as f:
    gt = json.load(f)

fname_to_imageID = {}

tabels = {f"WASTE_{i+1}": i for i in range(0, 20)}

for image in gt.keys():
    fname_to_imageID[image] = image

metric = MeanAveragePrecision()
device = 'cpu'

for fname, pred in tqdm(preds.items()):
    pred['labels'] = [tabels[name] for name in pred['labels']]
    pred = [{k: torch.tensor(v).to(device) for k, v in pred.items()}]
    imageID = fname_to_imageID[fname]
    target = get_gt_data(gt[imageID])
    target['labels'] = [tabels[name] for name in target['labels']]
    target = [{k: torch.tensor(v).to(device) for k, v in target.items()}]
    metric.update(pred, target)

result = metric.compute()
print(result)
