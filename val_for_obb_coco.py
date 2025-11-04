import sys
import io
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
try:
    from detectron2.evaluation.rotated_coco_evaluation import RotatedCOCOeval
except ImportError:
    print("Detectron2 is not installed. Please install it to use this script.")
    print("run:")
    print("python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    sys.exit(1)


def polygon_to_rotated_bbox(poly):

    pts = np.array(poly, dtype=np.float32).reshape((4, 2))
    rect = cv2.minAreaRect(pts)
    (xc, yc), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    return [xc, yc, w, h, angle]

def convert_detection_results(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    new_results = []
    for item in data:
        poly = item['poly']
        bbox = polygon_to_rotated_bbox(poly)

        new_item = {
            "image_id": item['image_id'],
            "category_id": item['category_id'],
            #"category_id": item['category_id']-1,
            "score": item['score'],
            "bbox": [round(x, 2) for x in bbox]
        }
        new_results.append(new_item)

    with open(output_path, 'w') as f:
        json.dump(new_results, f, indent=2)

    print(f"save path: {output_path}")

if __name__ == "__main__":
    convert_detection_results(
        "predictions.json", 
        "predictions_new.json")
    
    coco_gt = COCO("GT/ship9_test_coco.json")     
    coco_dt = coco_gt.loadRes("predictions_new.json") 

    coco_eval = RotatedCOCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()