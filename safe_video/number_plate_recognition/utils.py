from ultralytics.engine.results import Boxes, Results
from copy import deepcopy
import numpy as np


def merge_results(result1: Results, result2: Results) -> Results:
    """
    Merges the bounding boxes of two YOLO results and also updates the class mapping.

    Args:
        result1 (Results): First YOLO result
        result2 (Results): Second YOLO result

    Returns:
        Results: Merged YOLO result containing all bounding boxes from both results and also updated class mapping
    """
    boxes1: Boxes = result1.boxes
    boxes2: Boxes = deepcopy(result2.boxes)
    merged_result: Results = deepcopy(result1)
    new_class_mapping: dict[int, str] = {}
    current_max_class_idx: int = max(result1.names.keys(), default=-1)

    # check for existing classes in results1 and append new classes from results2
    for cls_id2, cls_name2 in result2.names.items():
        existing_cls_id = next((k for k, v in result1.names.items() if v == cls_name2), None)
        if existing_cls_id is not None:
            new_class_mapping[cls_id2] = existing_cls_id
        else:
            current_max_class_idx += 1
            new_class_mapping[cls_id2] = current_max_class_idx
            merged_result.names[current_max_class_idx] = cls_name2

    # remap classes in second results
    for i, cls_id in enumerate(boxes2.data[:, -1]):
        boxes2.data[i, -1] = new_class_mapping[int(cls_id)]

    merged_data = np.vstack([boxes1.data, boxes2.data]) if boxes1.data.size > 0 else boxes2.data
    merged_result.boxes.data = merged_data
    return merged_result


def get_key(dct: dict, val: str) -> int:
    return list(dct.keys())[list(dct.values()).index(val)]


def filter_results(results: Results, class_filter: list[str]|str) -> Results:
    if type(class_filter) is str: class_filter = [class_filter]

    class_filter = [get_key(results.names, cls) for cls in class_filter]
    results.boxes.data = np.array([d for d in results.boxes.data if d[-1] in class_filter])
    return results
