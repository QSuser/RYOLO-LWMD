import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/ultralytics-main0123456/ultralytics/cfg/models/v8/Ryolo-LWMD.yaml',task='obb')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/ultralytics-main0123456/ultralytics/cfg/datasets/AShipClass9.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
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