# RetinaFace in PyTorch

This repository was forked from https://github.com/biubug6/Pytorch_Retinaface
Inference-only code, simplified installation dependencies

## Requirements

- python 3.8+
- torch >=1.1.0
- torchvision >=0.3.0
- opencv-python

## Installation

This project requires downloading the weights from the following link and organizing your directory structure as follows:
https://github.com/biubug6/Pytorch_Retinaface#training

```bash
  ~/.Pytorch_Retinaface
      mobilenet0.25_Final.pth
      Resnet50_Final.pth
```

```bash
# pip
pip install git+https://github.com/chatflip/Pytorch_Retinaface.git
# poetry
poetry add git+https://github.com/chatflip/Pytorch_Retinaface.git
# and install torch, torchvision, opencv-python
```

## Usage

```python
from retinaface import RetinaFace
image_path = "path to image"
image = cv2.imread(image_path)
model = RetinaFace(backbone="resnet50", device="cuda")
result = model(image)
visualize_image = model.visualize(image, result)
```


## References
- [RetinaFace in PyTorch](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
