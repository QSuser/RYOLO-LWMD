Implementation of the paper " RYOLO-LWMD-Lite: a Lightweight Rotating Ship Target Detection Model for Optical Remote Sensing Images".

Download address for the dataset used: https://ieee-dataport.org/documents/ashipclass9

# Create Environment

```
// Using conda
conda create -n [env-name] python=3.8
conda activate [env-name]

// Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
// venv\Scripts\activate  # Windows
```

# Install Dependencies

```
// CUDA 10.2
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

// CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch


pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

// for compress model
pip install torch-pruning==1.5.1 tensorboard dill

// for rotated coco evaluation
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

# train

```
python train.py
```

# compress

```
python transform_weight.py //get model that can be pruned
python compress.py
```

# val

```
python val.py // get prediction.json
python val_for_obb_coco.py
```
