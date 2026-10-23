# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Raw dataset -> ``<name>_mdet`` conversion helpers.

Kept outside the ``ultralytics`` package so upstream code stays untouched; the
only contract shared with it is the dataset YAML written by :mod:`schema`.
"""

from .schema import DatasetPlan, ModalitySpec

__all__ = ["DatasetPlan", "ModalitySpec"]
