"""True multi-label YOLO detection and image-classification extensions.

The package is intentionally separate from the repository's existing
detection, multi-attribute detection, and segmentation implementations.  The
detection path keeps one physical object box and represents its labels with an
n-hot vector.  The classification path represents one image (or one crop)
with one n-hot vector.
"""

from .codec import CombinationCodec, normalize_combination
from .dataset import MultiLabelLabelError, MultiLabelYOLODataset
from .loss import MultiLabelDetectionLoss
from .model import MultiLabelDetectionModel
from .predictor import MultiLabelDetectionPredictor
from .tal import MultiLabelTaskAlignedAssigner
from .trainer import MultiLabelDetectionTrainer
from .validator import MultiLabelDetectionValidator
from .classification_dataset import (
    MultiLabelClassificationDataset,
    MultiLabelClassificationLabelError,
    load_multilabel_classification_data,
    parse_image_multilabel_label_file,
)
from .classification_loss import MultiLabelClassificationLoss
from .classification_metrics import MultiLabelClassificationMetrics
from .classification_model import MultiLabelClassificationModel
from .classification_pipeline import classify_detection_crops
from .classification_predictor import MultiLabelClassificationPredictor
from .classification_trainer import MultiLabelClassificationTrainer
from .classification_validator import MultiLabelClassificationValidator

__all__ = (
    "CombinationCodec",
    "MultiLabelClassificationDataset",
    "MultiLabelClassificationLabelError",
    "MultiLabelClassificationLoss",
    "MultiLabelClassificationMetrics",
    "MultiLabelClassificationModel",
    "classify_detection_crops",
    "MultiLabelClassificationPredictor",
    "MultiLabelClassificationTrainer",
    "MultiLabelClassificationValidator",
    "MultiLabelDetectionLoss",
    "MultiLabelDetectionModel",
    "MultiLabelDetectionPredictor",
    "MultiLabelDetectionTrainer",
    "MultiLabelDetectionValidator",
    "MultiLabelLabelError",
    "MultiLabelTaskAlignedAssigner",
    "MultiLabelYOLODataset",
    "load_multilabel_classification_data",
    "normalize_combination",
    "parse_image_multilabel_label_file",
)
