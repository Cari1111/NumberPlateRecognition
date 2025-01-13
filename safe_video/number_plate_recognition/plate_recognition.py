from .utils import *
from IPython.display import clear_output
import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results
import numpy as np
import torch


class ObjectDetection():
    def __init__(self, file_path: str = "."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."), "models", "first10ktrain", "weights", "best.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))

    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)

    def get_classes(self) -> list[str]: return np.concatenate([list(model.names.values()) for model in self.models])

    def map_classes_to_models(self, classes: list[str]) -> dict[int, list[int]]:
        classes = classes.copy()
        classes_len = len(classes)
        if classes_len == 0: raise ValueError("No classes provided")
        # select the right model for each class
        model_class_dict: dict[int, list] = {}
        for model_idx in range(len(self.models)):
            model_classes = self.models[model_idx].names
            model_classes_vals = model_classes.values()
            model_class_dict[model_idx] = []
            for target_class in classes.copy():
                if len(classes) == 0: break
                if target_class in model_classes_vals:
                    class_idx = find_key_by_value(model_classes, target_class)
                    classes.remove(target_class)
                    model_class_dict[model_idx].append(class_idx)
            else: continue
            break
        if len(classes) > 0: print(f"Could not find models for the following classes: {classes}")
        if len(classes) == classes_len: raise ValueError("No classes found for any model")
        return model_class_dict

    def detect_objects(self, image: ImageInput, model_class_dict: dict[int, list[int]], conf_thresh: float = 0.25, augment: bool = False, verbose: bool = False) -> Results:
        # detect and combine results
        result = None
        for model_idx, class_indices in model_class_dict.items():
            if len(class_indices) == 0: continue
            detection_results = self.models[model_idx](image, classes=class_indices, conf=conf_thresh, augment=augment, half=True)[0].cpu().numpy()
            if result is None: result = detection_results
            else: result = merge_results(result, detection_results)
        if not verbose: clear_output()
        return result

    def chain_detection(self, image: ImageInput, class_dicts: list[dict[int, list[int]]],
                        conf_thresh: float = 0.25, augment: bool = False, verbose: bool = False) -> Results:
        results = [self.detect_objects(image, class_dicts[0], conf_thresh=conf_thresh,
                                       augment=augment, verbose=verbose)]

        for class_dict in class_dicts[1:]:
            names = {}
            for model_idx, class_indices in class_dict.items():
                if len(class_indices) == 0: continue
                names.update(self.models[model_idx].names)
            results.append(Results(results[-1].orig_img, results[-1].path, names, np.empty((0, 6))))

            if results[-2].boxes.data.size > 0:
                for bbox in results[-2].boxes.xyxy:
                    x1, y1, _, _ = bbox.astype("int")
                    cropped_image = crop_image(image, bbox)
                    result = self.detect_objects(cropped_image, class_dict, conf_thresh=conf_thresh, augment=augment, verbose=verbose)
                    if result.boxes.data.size > 0: result.boxes.data[:, :4] += [x1, y1, x1, y1]

                    if results[-1] is None: results[-1] = result
                    else: results[-1] = merge_results(results[-1], result)
        return results

    def process_image(self, image: ImageInput, classes: str | list[str | list[str]],
                      remap_classes: bool = True, conf_thresh: float = 0.25, augment: bool = False, verbose: bool = False) -> list[Results]:
        if classes is None: raise ValueError("Primary classes must be provided")
        if issubclass(type(classes), str): classes = [classes]

        if (remap_classes):
            self._class_mappings = []
            for cls in classes:
                if issubclass(type(cls), str): cls = [cls]
                self._class_mappings.append(self.map_classes_to_models(cls))
        return self.chain_detection(image, self._class_mappings, conf_thresh=conf_thresh, augment=augment, verbose=verbose)

    def process_video(self, video_path: str, classes: str | list[str | list[str]],
                      conf_thresh: float = 0.25, iou_threshold: float = 0.7, video_stride: int = 1,
                      enable_stream_buffer: bool = False, augment: bool = False,
                      debug: bool = False, verbose: bool = False) -> list[tuple[int, Results]]:
        def debug_show_video(frame: ImageInput) -> bool:
            height, width = frame.shape[:2]
            cv2.imshow("frame", cv2.resize(frame, (int(width / 2), int(height / 2))))
            return cv2.waitKey(1) & 0xFF == ord('q')

        detections_in_frames = []
        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            if frame_counter % video_stride != 0:
                frame_counter += 1
                continue

            detections = self.process_image(frame, classes, frame_counter == 0,
                                            conf_thresh=conf_thresh, augment=augment, verbose=verbose)
            detections_in_frames.append((frame_counter, merge_results_list(detections)))

            # TODO delete later is for testing
            if debug:
                frame = merge_results_list(detections).plot()
                if debug_show_video(frame): break
            frame_counter += 1

        cap.release()
        if debug: cv2.destroyAllWindows()

        return detections_in_frames
