from ultralytics.engine.results import Boxes, Results
from copy import deepcopy
from PIL import Image, ImageColor
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
import torch
import ffmpeg
ImageInput = str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor


def merge_results(result1: Results, result2: Results) -> Results:
    """
    Merges the bounding boxes of two YOLO results and also updates the class mapping.

    Args:
        result1 (Results): First YOLO result
        result2 (Results): Second YOLO result

    Returns:
        Results: Merged YOLO result containing all bounding boxes from both results and also updated class mapping
    """
    if result1 is None: return result2
    if result2 is None: return result1
    boxes1: Boxes = result1.boxes
    boxes2: Boxes = deepcopy(result2.boxes)
    merged_result: Results = deepcopy(result1)
    updated_class_mapping: dict[int, str] = {}
    current_max_class_idx: int = max(result1.names.keys(), default=-1)

    # check for existing classes in results1 and append new classes from results2
    for class_id2, class_name2 in result2.names.items():
        existing_class_id = next((k for k, v in result1.names.items() if v == class_name2), None)
        if existing_class_id is not None:
            updated_class_mapping[class_id2] = existing_class_id
        else:
            current_max_class_idx += 1
            updated_class_mapping[class_id2] = current_max_class_idx
            merged_result.names[current_max_class_idx] = class_name2

    # remap classes in second results
    if boxes2.data.size > 0:
        for i, class_id in enumerate(boxes2.data[:, -1]):
            boxes2.data[i, -1] = updated_class_mapping[int(class_id)]

    if boxes1.data.size > 0 and boxes2.data.size > 0: merged_data = np.vstack([boxes1.data, boxes2.data])
    elif boxes1.data.size > 0: merged_data = boxes1.data
    else: merged_data = boxes2.data
    merged_result.boxes.data = merged_data
    return merged_result


def merge_results_list(results: list[Results]) -> Results:
    """
    Merges the bounding boxes of multiple YOLO results and also updates the class mapping.

    Args:
        results (list[Results]): List of YOLO results

    Returns:
        Results: Merged YOLO result containing all bounding boxes from all results and also updated class mapping
    """
    merged_result: Results = None
    for result in results: merged_result = merge_results(merged_result, result)
    return merged_result


def find_key_by_value(dictionary: dict, value: str) -> int:
    return list(dictionary.keys())[list(dictionary.values()).index(value)]


def filter_results(results: Results, class_filter: list[str] | str = None, conf_thresh: float = None) -> Results:
    if issubclass(type(class_filter), str): class_filter = [class_filter]
    results = deepcopy(results)

    if class_filter is not None:
        class_filter = [find_key_by_value(results.names, cls) for cls in class_filter]
        results.boxes.data = np.array([res for res in results.boxes.data if res[-1] in class_filter])

    if conf_thresh is not None:
        results.boxes.data = np.array([res for res in results.boxes.data if res[-2] > conf_thresh])

    return results


class Censor:
    def blur(image: np.ndarray, **kwargs) -> np.ndarray:
        def pixelate_region(region, pixel_size=10):
            height, width = region.shape[:2]

            # Ensure width and height do not become zero
            small_width = max(1, width // pixel_size)
            small_height = max(1, height // pixel_size)

            # Resize to a smaller size
            small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            # Scale back to the original size
            pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            return pixelated

        height, width = image.shape[:2]
        kernel_size = max(3, min(height, width))  # Adjust based on size
        if kernel_size % 2 == 0: kernel_size += 1  # Kernel size must be odd
        blurred_region = pixelate_region(image, pixel_size=10)
        blurred_region = cv2.GaussianBlur(blurred_region, (kernel_size, kernel_size), 0)
        return blurred_region

    def solid(color: tuple|str, **kwargs) -> np.ndarray:
        if type(color) is str:
            color = ImageColor.getcolor(color, "RGB")
        return color

    def overlay(image: np.ndarray, overlayImage: np.ndarray|str, **kwargs) -> np.ndarray:
        if type(overlayImage) is str:
            overlayImage = cv2.imread(overlayImage)[:, :, ::-1]
        return cv2.resize(overlayImage, image.shape[:2][::-1])


def apply_censorship(image: ImageInput, detection_results: Results,
                     action: Callable = Censor.blur, **kwargs) -> np.ndarray:
    if detection_results.boxes.data.size == 0: return image
    image_copy = image.copy()
    for bbox in detection_results.boxes.xyxy:
        x1, y1, x2, y2 = bbox.astype("int")
        modifiedRegion = action(image=image_copy[y1:y2, x1:x2], **kwargs)
        image_copy[y1:y2, x1:x2] = modifiedRegion
    return image_copy


def crop_image(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
    assert len(bbox) == 4, "Array must have exactly 4 entries"
    x1, y1, x2, y2 = bbox.astype("int")
    return image[y1:y2, x1:x2]


def save_result_as_video(results: list[tuple[int, Results]], output_path: str, original_video_path, codec: str = "mp4v", class_filter: list[str] | str = None,
                         conf_thresh: float = None, censorship: Callable = None, copy_audio: bool = False, **kwargs):
    def valid_codec(codec: str) -> bool:
        try:
            cv2.VideoWriter_fourcc(*codec)
            return True
        except cv2.error: return False

    if valid_codec(codec) is False: raise ValueError("Invalid codec provided")

    fps = round(cv2.VideoCapture(original_video_path).get(cv2.CAP_PROP_FPS))
    frame_size = results[0][1].orig_img.shape[:2][::-1]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # Create a temporary video file to store the processed frames and then copy the audio from the original video to the processed video
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, frame_size)
    for frame_counter, detection in results:
        frame = detection.orig_img
        if frame.shape[:2] != frame_size: frame = cv2.resize(frame, frame_size)

        detection = filter_results(detection, class_filter, conf_thresh)
        if censorship is not None: frame = apply_censorship(frame, detection, censorship, **kwargs)

        video_writer.write(frame)
    video_writer.release()

    if original_video_path and copy_audio:
        try:
            input_video = ffmpeg.input(original_video_path)
            input_audio = input_video.audio
            input_temp_video = ffmpeg.input(temp_output_path)
            ffmpeg.output(input_temp_video.video, input_audio, output_path, vcodec='copy',
                          acodec='aac', strict='experimental').run(overwrite_output=True)
        except ffmpeg.Error as e:
            print("Conversion failed:", e)
        finally:
            Path(temp_output_path).unlink(missing_ok=True)
    else:
        Path(temp_output_path).replace(output_path)
