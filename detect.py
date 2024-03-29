from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

# from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync
import json
import os
import base64


class imageDetect:
    def __init__(self, source, device, model, conf):
        self.source = source
        self.model = model
        self.device = device
        self.conf = conf

    def preProcess(self, im0s, target_size=128):
        self.base64image = self.convertImg2b64(im0s)
        self.filename = os.path.basename(im0s)
        im0s = cv2.imread(im0s)
        self.source = im0s
        im = letterbox(im0s, target_size, stride=32, auto=True)
        im = np.array(im[0], dtype=np.uint8)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        return im

    def convertImg2b64(self, imgpath):
        with open(imgpath, "rb") as img_file:
            imgstring = base64.b64encode(img_file.read())
        return imgstring

    def detect(
        self, half=False, imgsz=(320,320), iou_thres=0.7, max_det=100, classes=None
    ):
        # pt_detections = []
        source = self.source
        model = self.model
        conf_thres = self.conf
        device = select_device(self.device)

        # preprocess
        im0s = source
        im = self.preProcess(im0s)

        # load model

        stride, names, pt = model.stride, model.names, model.pt
        model.warmup(imgsz=(1, 3, *imgsz))
        seen, windows = 0, []
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        gn = torch.tensor(self.source.shape)[[1, 0, 1, 0]]
        if len(im.shape) == 3:
            im = im[None]

        shapes = []
        # inference
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, False, max_det=max_det
        )
        for i, det in enumerate(pred):
            im0 = self.source.copy()
            # puts all points into a shapes array
            if len(det):
                seen += 1
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    xywh = torch.tensor(xyxy).tolist()
                    points0 = [xywh[0], xywh[1]]
                    points1 = [xywh[2], xywh[3]]
                    pointcombined = [points0, points1]
                    points = {
                        "label": names[int(cls)],
                        "points": pointcombined,
                        "group_id": f"{conf}",
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                    shapes.append(points)
            labelmeFormat = {
                "version": "5.1.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": self.filename,
                "imageData": self.base64image.decode("utf-8"),
                "imageHeight": self.source.shape[0],
                "imageWidth": self.source.shape[1],
            }
            # print(labelmeFormat)
            return labelmeFormat
