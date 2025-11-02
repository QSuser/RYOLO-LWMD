import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/Ryolo-LWMD.yaml',task='obb')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/to/your/datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                batch=32,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path-
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='demo',
                )