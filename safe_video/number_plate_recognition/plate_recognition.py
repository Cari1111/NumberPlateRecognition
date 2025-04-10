from .utils import *
import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results
import numpy as np
import torch
import warnings


class ObjectDetection():
    """
    A class for managing multiple YOLO object detection models and performing detection
    on images and videos with class-specific model selection and chaining capabilities.
    """

    def __init__(self):
        """
        Initializes the ObjectDetection class with a given file path and adds default models.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."), "models", "first10ktrain", "weights", "licensePlate.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))

    def add_model(self, path: str):
        """
        Loads a YOLO model from a specified path and adds it to the active model list.

        Args:
            path (str): File path to the YOLO model weights.

        Raises:
            ValueError: If a model with the same name already exists or if all classes in the model are already covered.
            Warning: If the model introduces classes already present in existing models.
        """
        if os.path.basename(path) in self.get_names():
            raise ValueError(f"Model with name {os.path.basename(path)} already exists")
        model = YOLO(path, task="detect")
        if set(model.names.values()).issubset(self.get_classes()):
            raise ValueError(f"All classes from the new model already exist: {list(model.names.values())}")
        intersection = list(set(model.names.values()) & set(self.get_classes()))
        if len(intersection) > 0:
            warnings.warn(f"Following new classes will not already exist: {intersection}")
        model.to(self.device)
        self.models.append(model)

    def del_model(self, id: str | int):
        """
        Removes a model by its name or index.

        Args:
            id (str | int): Name or index of the model to remove.
        """
        if issubclass(type(id), str): id = self.get_names().index(id)
        self.models.pop(id)

    def get_classes(self) -> list[str]:
        """
        Returns:
            l (list[str]): A flattened list of all unique class names across loaded models.
        """
        if len(self.models) == 0: return []
        return np.concatenate([list(model.names.values()) for model in self.models])

    def get_names(self) -> list[str]:
        """
        Returns:
            l (list[str]): Names of all currently loaded models.
        """
        return [os.path.basename(model.model_name) for model in self.models]

    def get_names_with_classes(self) -> list[str, list[str]]:
        """
        Returns:
            l (list[tuple[str, list[str]]]): A list of tuples containing model names and their associated class names.
        """
        return list(zip(self.get_names(), [list(model.names.values()) for model in self.models]))

    def map_classes_to_models(self, classes: list[str]) -> dict[int, list[int]]:
        """
        Maps each class to a model that can detect it.

        Args:
            classes (list[str]): A list of class names to map.

        Returns:
            d (dict[int, list[int]]): Dictionary mapping model indices to a list of class indices within those models.

        Raises:
            ValueError: If no classes are provided or none of the classes are found in any models.
        """
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
        if len(classes) == classes_len: raise ValueError(f"The classes: {classes} could not be found in any model")
        return model_class_dict

    def chain_detection(self, image: ImageInput, class_dicts: list[dict[int, list[int]]],
                        conf_thresh: float = 0.25, augment: bool = False) -> Results:
        """
        Performs chained detection where the output of each step is croped and used as input for the next step.

        Args:
            image (ImageInput): The input image to process.
            class_dicts (list[dict]): List of model-to-class mappings for each detection stage.
            conf_thresh (float): Confidence threshold for detections. Defaults to 0.25.
            augment (bool): Whether to use data augmentation. Defaults to False.

        Returns:
            results (Results): Final detection results after all chained stages.
        """
        def detect_objects(crp_img: ImageInput, class_dict: dict[int, list[int]]) -> Results:
            result = None
            for model_idx, class_indices in class_dict.items():
                if len(class_indices) == 0: continue
                detection_results = self.models[model_idx](
                    crp_img, classes=class_indices, conf=conf_thresh, augment=augment, half=True)[0].cpu().numpy()
                if result is None: result = detection_results
                else: result = merge_results(result, detection_results)
            return result

        results = [detect_objects(image, class_dicts[0])]

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
                    result = detect_objects(cropped_image, class_dict)
                    if result.boxes.data.size > 0: result.boxes.data[:, :4] += [x1, y1, x1, y1]

                    if results[-1] is None: results[-1] = result
                    else: results[-1] = merge_results(results[-1], result)
        return results

    def process_image(self, image: ImageInput, classes: str | list[str | list[str]],
                      remap_classes: bool = True, conf_thresh: float = 0.25, augment: bool = False, verbose: bool = False) -> list[Results]:
        """
        Processes a single image through one or more detection steps.

        Args:
            image (ImageInput): The input image to detect objects in.
            classes (str | list): A single class name, list of class names, or list of lists for chaining.
            remap_classes (bool): Whether to recompute the class-to-model mappings. Defaults to True.
            conf_thresh (float): Confidence threshold. Defaults to 0.25.
            augment (bool): Use augmentation. Defaults to False.
            verbose (bool): Whether to show verbose output. Defaults to False.

        Returns:
            l (list[Results]): A list of detection results for each chain stage.
        """
        if classes is None: raise ValueError("Primary classes must be provided")
        if issubclass(type(classes), str): classes = [classes]

        if (remap_classes):
            self._class_mappings = []
            for cls in classes:
                if issubclass(type(cls), str): cls = [cls]
                self._class_mappings.append(self.map_classes_to_models(cls))
        return self.chain_detection(image, self._class_mappings, conf_thresh=conf_thresh, augment=augment)

    def process_video(self, video_path: str, classes: str | list[str | list[str]],
                      conf_thresh: float = 0.25, iou_threshold: float = 0.7, video_stride: int = 1,
                      enable_stream_buffer: bool = False, augment: bool = False,
                      debug: bool = False, verbose: bool = False) -> list[tuple[int, Results]]:
        """
        Processes a video frame-by-frame, applying object detection at a fixed stride.

        Args:
            video_path (str): Path to the input video file.
            classes (str | list): Class or nested class list to detect.
            conf_thresh (float): Detection confidence threshold. Defaults to 0.25.
            iou_threshold (float): IOU threshold for merging boxes. (Currently unused).
            video_stride (int): Number of frames to skip between detections. Defaults to 1.
            enable_stream_buffer (bool): Placeholder for future streaming use. Defaults to False.
            augment (bool): Whether to use augmentations during detection. Defaults to False.
            debug (bool): If True, displays detection output for debugging. Defaults to False.
            verbose (bool): If True, enables verbose output. Defaults to False.

        Returns:
            l (list[tuple[int, Results]]): List of tuples containing frame indices and their corresponding detection results.
        """
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
