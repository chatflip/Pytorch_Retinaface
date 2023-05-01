import os

import cv2
from retinaface.RetinaFace import RetinaFace


def test_mobilenet() -> None:
    image_path = os.path.join("curve", "test.jpg")
    image = cv2.imread(image_path)
    model = RetinaFace(backbone="mobile0.25", device="cuda")
    result = model(image)
    visualize_image = model.visualize(image, result)
    assert visualize_image is not None


def test_resnet50() -> None:
    image_path = os.path.join("curve", "test.jpg")
    image = cv2.imread(image_path)
    model = RetinaFace(backbone="resnet50", device="cuda")
    result = model(image)
    visualize_image = model.visualize(image, result)
    assert visualize_image is not None
