"""Small helpers shared by mdet evaluation scripts.

The checked-in dataset YAMLs use the original training-machine path.  These
helpers allow an evaluator to keep the YAML semantics unchanged while
redirecting only the dataset root at run time.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import Iterator

from ultralytics.utils import yaml_load, yaml_save


@contextmanager
def resolved_data_yaml(data_yaml: str | Path, data_root: str | Path | None = None) -> Iterator[str]:
    """Yield a usable dataset YAML, optionally replacing its root directory."""

    if data_root is None:
        yield str(data_yaml)
        return

    data = yaml_load(data_yaml)
    data_root = Path(data_root)
    data["path"] = str(data_root)
    with TemporaryDirectory(prefix="mdet_data_") as temp_dir:
        # The original split files contain absolute paths from the training
        # machine (for example /localnvme/...).  Rebuild only these temporary
        # manifests so the evaluator uses the backup data without modifying
        # the source dataset or its cached labels.
        temp_dir = Path(temp_dir)
        for split in ("train", "val", "test", "minival"):
            split_value = data.get(split)
            if not isinstance(split_value, str):
                continue
            source_manifest = data_root / split_value
            if not source_manifest.exists() or source_manifest.suffix.lower() not in {".txt", ".list"}:
                continue
            rewritten = []
            for line in source_manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_name = PurePosixPath(line.replace("\\", "/")).name
                candidate = data_root / "images" / image_name
                rewritten.append(str(candidate))
            manifest = temp_dir / f"{split}.txt"
            manifest.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
            data[split] = str(manifest)
        resolved = Path(temp_dir) / "data.yaml"
        yaml_save(resolved, data)
        yield str(resolved)
