import os

import cv2
import numpy as np
import numpy.typing as npt
import torch
from retinaface.data.config import cfg_mnet, cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.models._retinaface import _RetinaFace
from retinaface.type import Bbox, Face
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.load_model import load_model
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms


class RetinaFace:
    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "cuda",
        weight_root: str = "~/.Pytorch_Retinaface",
        conf_thre: float = 0.02,
        nmf_thre: float = 0.4,
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
        model = _RetinaFace(
            cfg=cfg,
            phase="test",
        )
        is_cpu = device == "cpu"
        model = load_model(model, weight_path, is_cpu)
        model.eval()
        self.device = torch.device(device)
        self.model = model.to(device)
        self.cfg = cfg
        self.conf_thre = conf_thre
        self.nmf_thre = nmf_thre

    def __call__(self, raw_image: npt.NDArray[np.uint64]) -> list[Face]:
        img = np.float32(raw_image)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min: np.int64 = np.min(im_shape[0:2])
        im_size_max: np.int64 = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        # force original size
        resize = 1

        if resize != 1:
            img = cv2.resize(
                img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            )
        im_height, im_width = img.shape[:2]  # type: ignore
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])  # type: ignore
        img -= (104, 117, 123)  # type: ignore
        img = img.transpose(2, 0, 1)  # type: ignore
        img = torch.from_numpy(img).unsqueeze(0)  # type: ignore
        img = img.to(self.device)  # type: ignore
        scale = scale.to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.model(img)
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))  # type: ignore
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(
            landms.data.squeeze(0), prior_data, self.cfg["variance"]
        ).cpu()
        scale1 = torch.Tensor(
            [
                img.shape[3],  # type: ignore
                img.shape[2],  # type: ignore
                img.shape[3],  # type: ignore
                img.shape[2],  # type: ignore
                img.shape[3],  # type: ignore
                img.shape[2],  # type: ignore
                img.shape[3],  # type: ignore
                img.shape[2],  # type: ignore
                img.shape[3],  # type: ignore
                img.shape[2],  # type: ignore
            ]
        )
        scale1 = scale1.cpu()
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thre)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nmf_thre)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        results: list[Face] = []
        for det in dets:
            top_right = np.array([det[0], det[1]], dtype=np.int64)
            bottom_left = np.array([det[2], det[3]], dtype=np.int64)
            bbox = Bbox(top_right, bottom_left)
            confidence = float(det[4])
            left_eye = np.array([det[5], det[6]], dtype=np.int64)
            right_eye = np.array([det[7], det[8]], dtype=np.int64)
            nose = np.array([det[9], det[10]], dtype=np.int64)
            mouth_right = np.array([det[11], det[12]], dtype=np.int64)
            mouth_left = np.array([det[13], det[14]], dtype=np.int64)

            results.append(
                Face(
                    bbox=bbox,
                    confidence=confidence,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=nose,
                    mouth_right=mouth_right,
                    mouth_left=mouth_left,
                )
            )
        return results

    def visualize(
        self, image: npt.NDArray[np.uint64], faces: list[Face], vis_thr: float = 0.6
    ) -> npt.NDArray[np.uint64]:
        vis_image = image.copy()
        for face in faces:
            if vis_thr > face.confidence:
                continue
            bbox = face.bbox
            vis_image = cv2.rectangle(
                vis_image, bbox.top_right, bbox.bottom_left, (0, 0, 255), 2
            )
            vis_image = cv2.circle(vis_image, face.left_eye, 1, (0, 0, 255), 4)
            vis_image = cv2.circle(vis_image, face.right_eye, 1, (0, 255, 255), 4)
            vis_image = cv2.circle(vis_image, face.nose, 1, (255, 0, 255), 4)
            vis_image = cv2.circle(vis_image, face.mouth_right, 1, (0, 255, 0), 4)
            vis_image = cv2.circle(vis_image, face.mouth_left, 1, (255, 0, 0), 4)
        return vis_image


if __name__ == "__main__":
    # net and model
    backbone = "resnet50"
    model = RetinaFace(backbone=backbone)
    image_path = os.path.join("..", "curve", "test.jpg")
    image = cv2.imread(image_path)
    results = model(image)
    result_image = model.visualize(image, results)
    # cv2.imwrite("results.png", result_image)
