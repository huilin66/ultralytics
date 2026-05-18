# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


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

    def __init__(self, classes, class_path=None):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None
        self.classes = classes
        # self.class_path = class_path
        # self._load_class()

    def _load_class(self):
        df = pd.read_csv(self.class_path, header=None, index_col=False, names=["class"])
        self.classes = df["class"].to_list()

    def load_model(self, weights: str) -> None:
        """
        Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
        """

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=weights, device="cuda"
        )

    def inference(
        self,
        weights: str = "yolo11n.pt",
        input_dir: str = "test.mp4",
        output_dir: str = "",
        exist_ok: bool = False,
    ) -> None:
        output_image_dir = output_dir
        # Output setup
        # output_image_dir = os.path.join(output_dir, 'image')
        output_pred_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_pred_dir, exist_ok=True)

        # Load model
        self.load_model(weights)
        img_list = os.listdir(input_dir)

        for idx, img_name in enumerate(img_list):
            if img_name.endswith(".db"):
                continue
            print(f"{idx}/{len(img_list)} -- {img_name} predict...")
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            h_img, w_img = img.shape[:2]
            if img is None:
                print("ERROR: img None!\n")
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
                (
                    det.category.name,
                    det.category.id,
                    det.score.value,
                    (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy),
                )
                for det in results.object_prediction_list
            ]

            # Annotate frame with detection results
            # df = pd.DataFrame(None, columns=['class_id', 'x', 'y', 'h', 'w'])
            for det in detection_data:
                # if int(det[1]) != 6:
                #     continue
                annotator.box_label(det[3], label=f"{det[0]} {det[2]:.2f}", color=colors(int(det[1]), True))
                txt_name = Path(img_name).stem
                txt_path = os.path.join(output_pred_dir, f"{txt_name}.txt")
                with open(txt_path, "w") as f:
                    for det in detection_data:
                        cls_name = det[0]
                        cls_id = self.classes.index(cls_name)
                        score = det[2]
                        bbox = det[3]

                        box_w = bbox[2] - bbox[0]
                        box_h = bbox[3] - bbox[1]
                        box_cx = bbox[0] + box_w / 2
                        box_cy = bbox[1] + box_h / 2

                        norm_w = box_w / w_img
                        norm_h = box_h / h_img
                        norm_cx = box_cx / w_img
                        norm_cy = box_cy / h_img

                        line = f"{cls_id} {norm_cx} {norm_cy} {norm_w} {norm_h} {score}\n"
                        f.write(line)

            output_image_path = os.path.join(output_image_dir, img_name)
            annotator.save(output_image_path)
            print(f"finished with {len(detection_data)} objects\n")


if __name__ == "__main__":
    # inference = SAHIInference()
    # inference.inference(**vars(inference.parse_opt()))
    # inference.inference(
    #     r'/data/huilin/projects/ultralytics/runs/detect/train29/weights/best.pt',
    #     r'/data/huilin/data/BDD/BD_Drone_merge',
    #     r'/data/huilin/data/BDD/BD_Drone_merge_pred2',
    # )

    # inference.inference(
    #     r'/data/huilin/projects/ultralytics/runs/detect/train29/weights/best.pt',
    #     r'/data/huilin/data/BDD/demo',
    #     r'/data/huilin/data/BDD/demo_pred',
    # )
    model_path = r"C:\Users\USER\Downloads\model.pt"
    data_path = r"C:\Users\USER\Downloads\demo"
    infer_path = r"C:\Users\USER\Downloads\demo_infer"
    inference = SAHIInference(["FACE"])
    inference.inference(
        model_path,
        data_path,
        infer_path,
    )
    # import os
    #
    # class_path = r'/scrinvme/huilin/bdd/collected_data/HMT_data/dataset/rgb_selected_3_p12_v2/class_train.txt'
    # root_dir = r'/scrinvme/huilin/bdd/collected_data/HMT_data/data_split/visible_views'
    # infer_dir = root_dir + '_infer1'
    # data_list = os.listdir(root_dir)
    # os.makedirs(infer_dir, exist_ok=True)
    # inference = SAHIInference(class_path)
    # for data_name in data_list:
    #     data_path = os.path.join(root_dir, data_name)
    #     infer_path = os.path.join(infer_dir, data_name)
    #     if os.path.isdir(data_path) and len(os.listdir(data_path)) > 0:
    #         # demo_base.model_predict('hmt_t_p123_v41-[yolo11x]',
    #         #                         data_path,
    #         #                         name=infer_path,
    #         #                         batch=32, save_conf=True)
    #         inference.inference(
    #             r'/localnvme/project/ultralytics/runs/detect/hmt_rgb_p12_v2_s640-[yolov8x]5/weights/best.pt',
    #             data_path,
    #             infer_path,

    class_path = None
    # root_dir = r'/scrinvme/huilin/bdd/collected_data/20260211_HMT_data/data_anno/rgb_selected_yolo/images'
    root_dir = r"/scrinvme/huilin/bdd/collected_data/20260211_HMT_data/data_anno/rgb_align_selected_yolo/images"
    infer_dir = root_dir + "_infer"
    data_list = os.listdir(root_dir)
    os.makedirs(infer_dir, exist_ok=True)
    inference = SAHIInference(class_path)

    inference.inference(
        r"/localnvme/project/ultralytics/runs/detect/hmt_rgb_p12_v2_s640-[yolov8x]5/weights/best.pt",
        root_dir,
        infer_dir,
    )
