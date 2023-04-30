import os
import sys
import cv2
sys.path.append("retinaface")
from RetinaFace import RetinaFace

def test_mobilenet():
    image_path = os.path.join("curve", "test.jpg")
    image = cv2.imread(image_path)
    model = RetinaFace(backbone="mobile0.25", device="cuda")
    result = model(image)
    visualize_image = model.visualize(image, result)
    assert visualize_image is not None
    
def test_resnet50():
    image_path = os.path.join("curve", "test.jpg")
    image = cv2.imread(image_path)
    model = RetinaFace(backbone="resnet50", device="cuda")
    result = model(image)
    visualize_image = model.visualize(image, result)
    assert visualize_image is not None