# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from pathlib import Path
import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import pandas as pd

class SAHIInference:
    """
    Runs Ultralytics YOLO11 and SAHI for object detection on video with options to view, save, and track results.

    This class integrates SAHI (Slicing Aided Hyper Inference) with YOLO11 models to perform efficient object detection
    on large images by slicing them into smaller pieces, running inference on each slice, and then merging the results.

    Attributes:
        detection_model (AutoDetectionModel): The loaded YOLO11 model wrapped with SAHI functionality.

    Methods:
        load_model: Loads a YOLO11 model with specified weights.
        inference: Runs object detection on a video using the loaded model.
        parse_opt: Parses command line arguments for the inference process.
    """

    def __init__(self):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None

    def load_model(self, weights: str) -> None:
        """
        Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
        """

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=weights, device="cuda"
        )

    def inference(
        self,
        weights: str = "yolo11n.pt",
        input_dir: str = "test.mp4",
        output_dir: str = "",
        exist_ok: bool = False,
    ) -> None:

        # Output setup
        output_image_dir = os.path.join(output_dir, 'image')
        output_pred_dir = os.path.join(output_dir, 'pred')
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_pred_dir, exist_ok=True)

        # Load model
        self.load_model(weights)
        img_list = os.listdir(input_dir)[2890:]

        for idx, img_name in enumerate(img_list):
            print(f'{idx}/{len(img_list)} -- {img_name} predict...')
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print('ERROR: img None!\n')
                continue
            annotator = Annotator(img)  # Initialize annotator for plotting detection results

            # Perform sliced prediction using SAHI
            results = get_sliced_prediction(
                img[..., ::-1],  # Convert BGR to RGB
                self.detection_model,
                slice_height=640,
                slice_width=640,
            )

            # Extract detection data from results
            detection_data = [
                (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
                for det in results.object_prediction_list
            ]

            # Annotate frame with detection results
            df = pd.DataFrame(None, columns=['class_id', 'x', 'y', 'h', 'w'])
            for det in detection_data:
                annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))

            output_image_path = os.path.join(output_image_dir, img_name)
            annotator.save(output_image_path)
            print(f'finished with {len(detection_data)} objects\n')

if __name__ == "__main__":
    inference = SAHIInference()
    # inference.inference(**vars(inference.parse_opt()))
    inference.inference(
        r'runs/segment/billboard_seg_389_c618/weights/best.pt',
        r'/localnvme/data/billboard/ps_data/0516/selected_img_filter2',
        r'/localnvme/data/billboard/ps_data/0516/selected_img_filter2/slice_pred',
    )