from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FIELDS = {
    "object_id",
    "object_name",
    "joint_name",
    "joint_type_original",
    "joint_type_normalized",
    "object_root",
    "rgb_dir",
    "mask_dir",
    "depth_dir",
    "camera_metadata_path",
    "colmap_path",
    "joint_metadata_path",
    "trajectory_path",
    "trajectory_joint_column",
    "gaussian_model_path",
    "available_cameras",
    "notes",
}


def normalize_joint_type(joint_type: str) -> str:
    value = str(joint_type).strip().lower()
    mapping = {
        "prismatic": "prismatic",
        "revolute": "revolute",
        "continuous": "revolute",
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported joint type {joint_type!r}; expected prismatic, revolute, or continuous"
        )
    return mapping[value]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class DatasetObject:
    object_id: str
    object_name: str
    joint_name: str
    joint_type_original: str
    joint_type_normalized: str
    object_root: Path
    rgb_dir: Path
    mask_dir: Path
    depth_dir: Path
    camera_metadata_path: Path
    colmap_path: Path
    joint_metadata_path: Path
    trajectory_path: Path
    trajectory_joint_column: str
    gaussian_model_path: Path | None
    available_cameras: tuple[str, ...]
    static_part_ids: tuple[int, ...]
    moving_part_ids: tuple[int, ...]
    notes: str

    def require_gaussian_model(self, override: str | Path | None = None) -> Path:
        path = resolve_repo_path(override) if override is not None else self.gaussian_model_path
        if path is None:
            raise FileNotFoundError(
                f"Object {self.object_id!r} has no trained enriched Gaussian model yet. "
                "Train and enrich a real frame-0 3DGS model, or pass --gaussian-model-override."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Gaussian model for object {self.object_id!r} does not exist: {path}"
            )
        return path

    def validate_paths(self, require_gaussian: bool = False) -> list[str]:
        errors: list[str] = []
        required_paths = {
            "object_root": self.object_root,
            "rgb_dir": self.rgb_dir,
            "mask_dir": self.mask_dir,
            "depth_dir": self.depth_dir,
            "camera_metadata_path": self.camera_metadata_path,
            "colmap_path": self.colmap_path,
            "joint_metadata_path": self.joint_metadata_path,
            "trajectory_path": self.trajectory_path,
        }
        for label, path in required_paths.items():
            if not path.exists():
                errors.append(f"{self.object_id}: missing {label}: {path}")
        if require_gaussian:
            try:
                self.require_gaussian_model()
            except FileNotFoundError as exc:
                errors.append(str(exc))
        elif self.gaussian_model_path is not None and not self.gaussian_model_path.exists():
            errors.append(
                f"{self.object_id}: configured Gaussian model does not exist: "
                f"{self.gaussian_model_path}"
            )
        for camera_id in self.available_cameras:
            if not (self.rgb_dir / camera_id).is_dir():
                errors.append(f"{self.object_id}: missing RGB camera directory {camera_id}")
            if not (self.mask_dir / camera_id).is_dir():
                errors.append(f"{self.object_id}: missing mask camera directory {camera_id}")
        return errors


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    version: int
    objects: tuple[DatasetObject, ...]

    def get(self, object_key: str) -> DatasetObject:
        wanted = object_key.strip().lower()
        matches = [
            obj
            for obj in self.objects
            if obj.object_id.lower() == wanted or obj.object_name.lower() == wanted
        ]
        if not matches:
            available = ", ".join(
                f"{obj.object_id} ({obj.object_name})" for obj in self.objects
            )
            raise KeyError(f"Unknown object {object_key!r}; available objects: {available}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous object key {object_key!r}")
        return matches[0]


def _parse_object(raw: dict[str, Any], index: int) -> DatasetObject:
    missing = sorted(REQUIRED_FIELDS - set(raw))
    if missing:
        raise ValueError(f"Manifest object #{index} is missing required fields: {missing}")

    original = str(raw["joint_type_original"]).strip().lower()
    normalized = str(raw["joint_type_normalized"]).strip().lower()
    expected_normalized = normalize_joint_type(original)
    if normalized != expected_normalized:
        raise ValueError(
            f"{raw['object_id']}: joint_type_normalized={normalized!r} does not match "
            f"normalization of {original!r} ({expected_normalized!r})"
        )

    cameras = tuple(str(value) for value in raw["available_cameras"])
    if not cameras:
        raise ValueError(f"{raw['object_id']}: available_cameras must not be empty")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"{raw['object_id']}: available_cameras contains duplicates")

    gaussian_value = raw["gaussian_model_path"]
    return DatasetObject(
        object_id=str(raw["object_id"]),
        object_name=str(raw["object_name"]),
        joint_name=str(raw["joint_name"]),
        joint_type_original=original,
        joint_type_normalized=normalized,
        object_root=resolve_repo_path(raw["object_root"]),
        rgb_dir=resolve_repo_path(raw["rgb_dir"]),
        mask_dir=resolve_repo_path(raw["mask_dir"]),
        depth_dir=resolve_repo_path(raw["depth_dir"]),
        camera_metadata_path=resolve_repo_path(raw["camera_metadata_path"]),
        colmap_path=resolve_repo_path(raw["colmap_path"]),
        joint_metadata_path=resolve_repo_path(raw["joint_metadata_path"]),
        trajectory_path=resolve_repo_path(raw["trajectory_path"]),
        trajectory_joint_column=str(raw["trajectory_joint_column"]),
        gaussian_model_path=(
            None if gaussian_value in {None, ""} else resolve_repo_path(gaussian_value)
        ),
        available_cameras=cameras,
        static_part_ids=tuple(int(value) for value in raw.get("static_part_ids", [0])),
        moving_part_ids=tuple(int(value) for value in raw.get("moving_part_ids", [1])),
        notes=str(raw["notes"]),
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = resolve_repo_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    if manifest_path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported manifest format {manifest_path.suffix!r}; this minimal loader expects JSON"
        )
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest root must be a JSON object")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Dataset manifest must contain a non-empty objects list")
    objects = tuple(_parse_object(raw, index) for index, raw in enumerate(raw_objects))
    object_ids = [obj.object_id.lower() for obj in objects]
    object_names = [obj.object_name.lower() for obj in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Dataset manifest contains duplicate object_id values")
    if len(set(object_names)) != len(object_names):
        raise ValueError("Dataset manifest contains duplicate object_name values")
    return DatasetManifest(
        path=manifest_path,
        version=int(payload.get("version", 1)),
        objects=objects,
    )
