import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

class DetectionResults():
    def __init__(self, boxes, conf, cls, classlegend):
        self.boxes = boxes
        self.conf = conf
        self.cls = cls
        self.clslgd = classlegend

    def __str__(self):
        ret = ""
        for a,b in self.__dict__.items(): ret = f"{ret}{a}: {b}\n"
        return ret

class NumberPlateRecognition():
    def __init__(self, file_path: str = "."):
        self.file_path = file_path
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.onnx")
        self.model = YOLO(model_path, task='detect')

    def analyze(self, image):
        result = self.model(image)[0]
        data: Boxes = result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,result.names)

    def blur_image(self):
        print(f'Image at {self.file_path} has to be blurred')