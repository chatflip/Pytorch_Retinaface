import os

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from utils.load_model import load_model


class RFace:
    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "cuda",
        weight_root: str = "~/.Pytorch_Retinaface",
    ):
        assert backbone in [
            "resnet50",
            "mobile0.25",
        ], f"No such backbone {backbone}. Please choose from [resnet50, mobile0.25]"
        weight_root = os.path.expanduser(weight_root)
        os.makedirs(weight_root, exist_ok=True)

        if backbone == "mobile0.25":
            cfg = cfg_mnet
            weight_path = os.path.join(weight_root, "mobilenet0.25_Final.pth")
        elif backbone == "resnet50":
            cfg = cfg_re50
            weight_path = os.path.join(weight_root, "Resnet50_Final.pth")
        cfg["pretrain"] = False
        model = RetinaFace(
            cfg=cfg,
            phase="eval",
        )
        is_cpu = device == "cpu"
        self.model = load_model(model, weight_path, is_cpu)
        self.model.eval()

    def __call__(self, image):
        pass


if __name__ == "__main__":
    # net and model
    backbone = "resnet50"
    model = RFace(backbone=backbone)
