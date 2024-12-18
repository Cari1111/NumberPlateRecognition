from .utils import *

from PIL import Image
from pathlib import Path
from copy import deepcopy
from IPython.display import clear_output

import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Boxes, Results
import numpy as np
import torch
Img = str|Path|int|Image.Image|list|tuple|np.ndarray|torch.Tensor

class ObjectDetection():
    def __init__(self, file_path: str = "."):
        self.result = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.pt"))
        self.add_model(os.path.join(os.path.abspath("."),"models","yolo11n.pt"))
        self.conf_interval = 0.5


    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)


    def analyze(self, image: Img, classes: list[str]|str, verbose = False) -> Results:
        if type(classes) is str: classes = [classes]
        # select with what model which classes are detected
        mdl_clss: dict[int, list] = {}
        for mdl_idx in range(len(self.models)):
            clss = self.models[mdl_idx].names
            mdl_clss[mdl_idx] = []
            for cls in classes.copy():
                if len(classes) == 0: break
                try:
                    cls_idx = get_key(clss, cls)
                    classes.remove(cls)
                    mdl_clss[mdl_idx].append(cls_idx)
                except Exception as e:
                    if verbose: print(e)
            else: continue
            break
        if len(classes) > 0: print(f"Could not find models for the following classes: {classes}")
        # detect and combine results
        self.result = None
        for mdl,clss in mdl_clss.items():
            if len(clss) == 0: continue
            res = self.models[mdl](image, classes = clss)[0].cpu().numpy()
            if self.result is None:
                self.result = res
            else:
                self.result = merge_results(self.result, res)
        if not verbose: clear_output()
        return self.result


    def get_classes(self) -> list[str]: return np.concatenate([list(m.names.values()) for m in self.models])


    def blur_image(self, image: Img, results: Results, classes: list[str]|str) -> np.ndarray:
        if type(classes) is str: classes = [classes]
        image = image.copy()
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if not results.names[cls] in classes: continue
            cropped_img = self.crop_image(image, box)
            blurred_img = cv2.GaussianBlur(cropped_img, (25,25),0)
            x1,y1,x2,y2 = box.astype("int")
            image[y1:y2,x1:x2] = blurred_img
        return image


    def crop_image(self, image: Img, xyxy: np.ndarray) -> np.ndarray:
        assert len(xyxy) == 4, "Array must have exactly 4 entries"
        x1,y1,x2,y2 = xyxy.astype("int")
        return image[y1:y2,x1:x2]


    def chained_detection(self, image: Img, cls1: str, cls2: str) -> Results:
        cls1_results = self.analyze(image, cls1)
        cls1_id = get_key(cls1_results.names, cls1)

        cls1_boxes = cls1_results.boxes[cls1_results.boxes.cls == cls1_id]
        cls1_boxes = cls1_boxes[cls1_boxes.conf >= self.conf_interval]

        merged_results = deepcopy(cls1_results)

        for box in cls1_boxes.xyxy:
            x1, y1, _, _ = box.astype("int")
            cropped_image = self.crop_image(image, box)

            cls2_results = self.analyze(cropped_image, cls2)
            if cls2_results.boxes.data.size > 0:
                cls2_results.boxes.data[:, :4] += [x1, y1, x1, y1]

            merged_results = merge_results(merged_results, cls2_results)
        self.result = merged_results
        return self.result