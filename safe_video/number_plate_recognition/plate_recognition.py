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
Img = str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor


class ObjectDetection():
    def __init__(self, file_path: str = ".", conf=0.5, iou=0.7, vid_stride=1, stream_buffer=False):
        self.result = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."), "models", "first10ktrain", "weights", "best.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))

        # TODO change with conf
        self.conf_interval = conf
        self.iou = iou
        self.vid_stride = vid_stride
        self.stream_buffer = stream_buffer

    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)

    def choose_model(self, classes: list[str], verbose=False) -> dict[int, list[int]]:
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
        return mdl_clss

    def analyze(self, image: Img, mdl_clss: dict[int, list[int]], verbose=False) -> Results:
        # detect and combine results
        self.result = None
        for mdl, clss in mdl_clss.items():
            if len(clss) == 0: continue
            res = self.models[mdl](image, classes=clss)[0].cpu().numpy()
            if self.result is None: self.result = res
            else: self.result = merge_results(self.result, res)
        if not verbose: clear_output()
        return self.result

    def get_classes(self) -> list[str]: return np.concatenate([list(m.names.values()) for m in self.models])

    def blur_image(self, image: Img, results: Results, classes: list[str] | str) -> np.ndarray:
        if type(classes) is str: classes = [classes]
        image = image.copy()
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if results.names[cls] not in classes: continue
            cropped_img = self.crop_image(image, box)
            blurred_img = cv2.GaussianBlur(cropped_img, (25, 25), 0)
            x1, y1, x2, y2 = box.astype("int")
            image[y1:y2, x1:x2] = blurred_img
        return image

    def crop_image(self, image: Img, xyxy: np.ndarray) -> np.ndarray:
        assert len(xyxy) == 4, "Array must have exactly 4 entries"
        x1, y1, x2, y2 = xyxy.astype("int")
        return image[y1:y2, x1:x2]

    def chained_detection(self, image: Img, mdl_clss1: dict[int, list[int]], mdl_clss2: dict[int, list[int]], verbose = False) -> Results:
        cls1_results = self.analyze(image, mdl_clss1, verbose)
        merged_results = deepcopy(cls1_results)

        for box in cls1_results.boxes.xyxy:
            x1, y1, _, _ = box.astype("int")
            cropped_image = self.crop_image(image, box)

            cls2_results = self.analyze(cropped_image, mdl_clss2)
            if cls2_results.boxes.data.size > 0:
                cls2_results.boxes.data[:, :4] += [x1, y1, x1, y1]

            merged_results = merge_results(merged_results, cls2_results)
        self.result = merged_results
        return self.result

    def detect_image(self, image: Img, class1: list[str] | str, class2: list[str] | str = None, verbose = False) -> Results:

        if type(class1) is str: class1 = [class1]
        if type(class2) is str: class2 = [class2]

        if class2 is None:
            mdl_clss = self.choose_model(class1, False)
            detections = self.analyze(image, mdl_clss, verbose)
        else:
            mdl_clss1 = self.choose_model(class1, False)
            mdl_clss2 = self.choose_model(class2, False)
            detections = self.chained_detection(image, mdl_clss1, mdl_clss2, verbose)

        return detections

    def detect_video(self, video_path: str, class1: list[str] | str, class2: list[str] | str = None, vid_stride: int = 1, verbose=False):

        if type(class1) is str: class1 = [class1]
        if type(class2) is str: class2 = [class2]

        frame_count = 0
        if class2 is None:
            mdl_clss1 = self.choose_model(class1, verbose)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                if frame_count % vid_stride != 0:
                    frame_count += 1
                    continue
                detections = self.analyze(frame, mdl_clss1, verbose)

                # TODO delete later is for testing
                frame = detections.plot()
                cv2.imshow("frame", cv2.resize(frame, (1200, 800)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            mdl_clss1 = self.choose_model(class1, verbose)
            mdl_clss2 = self.choose_model(class2, verbose)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                if frame_count % vid_stride != 0:
                    frame_count += 1
                    continue
                detections = self.chained_detection(frame, mdl_clss1, mdl_clss2, verbose)

                # TODO delete later is for testing
                frame = detections.plot()
                cv2.imshow("frame", cv2.resize(frame, (1200, 800)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
