from typing import List

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from inference.base_detector import BaseDetector


class YoloV8(BaseDetector):
    def __init__(
        self,
        model_path: str = None,
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will use the yolov8n.pt model in the project
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("models/yolo/yolov8n.pt")

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        predictions = self.model.predict(input_image, imgsz=640)
        result = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])
        if len(predictions) > 0:
            boxes = predictions[0].boxes
            result = pd.DataFrame(boxes.xyxy, columns=["xmin", "ymin", "xmax", "ymax"])
            result["confidence"] = boxes.conf
            result["class"] = [int(cls) for cls in boxes.cls]
            result["name"] = [predictions[0].names[cls] for cls in result["class"]]
        return result
