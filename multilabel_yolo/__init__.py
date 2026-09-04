"""True multi-label YOLO detection extensions.

The package is intentionally separate from the repository's existing detection,
multi-attribute detection, and segmentation implementations.  It keeps one
physical object box in the dataset and represents its labels with an n-hot
vector.
"""

from .codec import CombinationCodec, normalize_combination
from .dataset import MultiLabelLabelError, MultiLabelYOLODataset
from .loss import MultiLabelDetectionLoss
from .model import MultiLabelDetectionModel
from .predictor import MultiLabelDetectionPredictor
from .tal import MultiLabelTaskAlignedAssigner
from .trainer import MultiLabelDetectionTrainer
from .validator import MultiLabelDetectionValidator

__all__ = (
    "CombinationCodec",
    "MultiLabelDetectionLoss",
    "MultiLabelDetectionModel",
    "MultiLabelDetectionPredictor",
    "MultiLabelDetectionTrainer",
    "MultiLabelDetectionValidator",
    "MultiLabelLabelError",
    "MultiLabelTaskAlignedAssigner",
    "MultiLabelYOLODataset",
    "normalize_combination",
)
