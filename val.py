import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/runs/prune/yolov8nDB-lamp1.4-finetune/weights/best.pt')
    model.val(data='/root/autodl-tmp/ultralytics-main1/dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=32,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='demo222',
              )