"""Survey objects and registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - used in minimal local runtimes.
    yaml = None

from .paths import DEFAULT_METADATA_FILE, DEFAULT_PROJECT_ROOT, PACKAGE_ROOT


CATALOGUE_TYPES = {"catalogue", "binned_cmf", "digitized_curve"}


@dataclass(frozen=True)
class Survey:
    """One CMF survey/sample described by metadata plus a data table."""

    key: str
    metadata: dict[str, Any]
    project_root: Path = field(default=PACKAGE_ROOT)

    @property
    def name(self) -> str:
        return str(self.metadata.get("label", self.key))

    @property
    def short_name(self) -> str:
        return str(self.metadata.get("short_label", self.key))

    @property
    def data_type(self) -> str:
        return str(self.metadata.get("data_type", "catalogue"))

    @property
    def data_file(self) -> Path:
        data_file = Path(str(self.metadata["data_file"]))
        if data_file.is_absolute():
            return data_file
        return self.project_root / data_file

    @property
    def mass_column(self) -> str:
        return str(self.metadata.get("mass_column", "mass_msun"))

    @property
    def radius_column(self) -> str:
        return str(self.metadata.get("radius_column", self.metadata.get("size_column", "Size")))

    @property
    def completeness_msun(self) -> float | None:
        return _optional_float(self.metadata.get("completeness_msun"))

    @property
    def fit_mass_min_msun(self) -> float | None:
        return _optional_float(self.metadata.get("fit_mass_min_msun"))

    @property
    def fit_mass_max_msun(self) -> float | None:
        return _optional_float(self.metadata.get("fit_mass_max_msun"))

    @property
    def reference(self) -> str | None:
        value = self.metadata.get("reference")
        return None if value in {None, "", "null"} else str(value)

    @property
    def has_data(self) -> bool:
        return self.data_file.exists()

    def get(self, field_name: str, default: Any = None) -> Any:
        """Return an arbitrary metadata field."""

        return self.metadata.get(field_name, default)

    def load_table(self) -> pd.DataFrame:
        """Load this survey's table and apply its metadata selection query."""

        table = _load_table(self.data_file)
        query = self.metadata.get("selection_query")
        if query not in {None, "", "null"}:
            table = table.query(str(query)).copy()
        return table

    def masses(self) -> np.ndarray:
        """Return finite positive core masses for catalogue-like surveys."""

        table = self.load_table()
        if self.mass_column not in table.columns:
            raise KeyError(
                f"Mass column {self.mass_column!r} is not in {self.data_file}. "
                f"Available columns: {list(table.columns)}"
            )
        masses = pd.to_numeric(table[self.mass_column], errors="coerce").to_numpy(dtype=float)
        return clean_masses(masses)

    def radii(self) -> np.ndarray:
        """Return finite positive core radii/sizes for catalogue-like surveys."""

        table = self.load_table()
        if self.radius_column not in table.columns:
            raise KeyError(
                f"Radius column {self.radius_column!r} is not in {self.data_file}. "
                f"Available columns: {list(table.columns)}"
            )
        radii = pd.to_numeric(table[self.radius_column], errors="coerce").to_numpy(dtype=float)
        return clean_masses(radii)

    def mass_radius_table(self) -> pd.DataFrame:
        """Return rows with valid mass and radius columns."""

        table = self.load_table()
        missing = [
            column
            for column in (self.mass_column, self.radius_column)
            if column not in table.columns
        ]
        if missing:
            raise KeyError(f"{self.key}: missing required columns: {missing}")
        result = table.copy()
        result["mass_msun"] = pd.to_numeric(result[self.mass_column], errors="coerce")
        result["radius_au"] = pd.to_numeric(result[self.radius_column], errors="coerce")
        ok = (
            result["mass_msun"].notna()
            & result["radius_au"].notna()
            & (result["mass_msun"] > 0)
            & (result["radius_au"] > 0)
        )
        return result.loc[ok].reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for notebooks and logs."""

        summary = {
            "key": self.key,
            "name": self.name,
            "short_name": self.short_name,
            "data_type": self.data_type,
            "data_file": str(self.data_file),
            "mass_column": self.mass_column,
            "radius_column": self.radius_column,
            "completeness_msun": self.completeness_msun,
            "fit_mass_min_msun": self.fit_mass_min_msun,
            "fit_mass_max_msun": self.fit_mass_max_msun,
            "reference": self.reference,
        }
        if self.data_type == "catalogue":
            masses = self.masses()
            summary.update(
                {
                    "n_cores": int(masses.size),
                    "mass_min_msun": float(masses.min()) if masses.size else None,
                    "mass_max_msun": float(masses.max()) if masses.size else None,
                }
            )
        return summary

    def validate(self) -> None:
        """Raise a clear exception if this survey cannot be loaded."""

        if self.data_type not in CATALOGUE_TYPES:
            raise ValueError(
                f"{self.key}: unknown data_type {self.data_type!r}. "
                f"Expected one of {sorted(CATALOGUE_TYPES)}."
            )
        if not self.data_file.exists():
            raise FileNotFoundError(f"{self.key}: data file does not exist: {self.data_file}")
        table = self.load_table()
        if self.data_type == "catalogue" and self.mass_column not in table.columns:
            raise KeyError(f"{self.key}: missing mass column {self.mass_column!r}.")


class SurveyRegistry:
    """Access surveys by their metadata key or short label."""

    def __init__(
        self,
        surveys: Iterable[Survey],
        metadata_file: str | Path | None = None,
        project_root: str | Path | None = None,
    ):
        self._surveys = {survey.key: survey for survey in surveys}
        self._aliases = self._make_aliases(self._surveys.values())
        self.metadata_file = None if metadata_file is None else Path(metadata_file)
        self.project_root = None if project_root is None else Path(project_root)

    @classmethod
    def from_default(cls) -> "SurveyRegistry":
        """Load the repository's bundled example survey metadata."""

        return cls.from_yaml(DEFAULT_METADATA_FILE, project_root=DEFAULT_PROJECT_ROOT)

    @classmethod
    def from_directory(
        cls,
        project_root: str | Path,
        metadata_file: str | Path = "data/metadata/surveys.yaml",
    ) -> "SurveyRegistry":
        """Load surveys from a project directory.

        Relative ``data_file`` entries in the metadata are resolved against
        ``project_root``. If ``metadata_file`` is relative, it is also resolved
        against ``project_root``.
        """

        root = Path(project_root)
        metadata_path = Path(metadata_file)
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        return cls.from_yaml(metadata_path, project_root=root)

    @classmethod
    def from_yaml(
        cls,
        metadata_file: str | Path,
        project_root: str | Path | None = None,
    ) -> "SurveyRegistry":
        """Load surveys from a YAML metadata file.

        If ``project_root`` is omitted and the metadata file lives at
        ``PROJECT/data/metadata/surveys.yaml``, relative ``data_file`` entries
        are resolved against ``PROJECT``. Otherwise they are resolved against
        the metadata file's parent directory.
        """

        metadata_file = Path(metadata_file)
        root = Path(project_root) if project_root is not None else _infer_project_root(metadata_file)
        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = _load_yaml_mapping(handle.read(), metadata_file)
        surveys = [
            Survey(key=key, metadata=dict(value), project_root=root)
            for key, value in metadata.items()
        ]
        return cls(surveys, metadata_file=metadata_file, project_root=root)

    @property
    def keys(self) -> list[str]:
        return sorted(self._surveys)

    def __len__(self) -> int:
        return len(self._surveys)

    def __iter__(self):
        for key in self.keys:
            yield self._surveys[key]

    def get(self, key_or_alias: str) -> Survey:
        """Return a survey by key, label, short label, or case-insensitive alias."""

        if key_or_alias in self._surveys:
            return self._surveys[key_or_alias]
        normalized = _normalize_alias(key_or_alias)
        try:
            return self._surveys[self._aliases[normalized]]
        except KeyError as exc:
            available = ", ".join(self.keys)
            raise KeyError(f"Unknown survey {key_or_alias!r}. Available surveys: {available}") from exc

    def available(self) -> list[Survey]:
        """Return surveys whose data files currently exist."""

        return [survey for survey in self if survey.has_data]

    def missing_data(self) -> list[Survey]:
        """Return registered surveys whose data files are not present yet."""

        return [survey for survey in self if not survey.has_data]

    def validate(self, require_data: bool = True) -> None:
        """Validate registered surveys.

        If ``require_data`` is False, surveys without local catalogue files are
        skipped. This is useful while metadata is being drafted ahead of data
        ingestion.
        """

        for survey in self:
            if not require_data and not survey.has_data:
                continue
            survey.validate()

    def summary(self, require_data: bool = False) -> pd.DataFrame:
        """Return one-row-per-survey summary metadata."""

        surveys = self.available() if require_data else list(self)
        rows = []
        for survey in surveys:
            if survey.has_data:
                rows.append(survey.summary())
            else:
                rows.append(
                    {
                        "key": survey.key,
                        "name": survey.name,
                        "short_name": survey.short_name,
                        "data_type": survey.data_type,
                        "data_file": str(survey.data_file),
                        "has_data": False,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_aliases(surveys: Iterable[Survey]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for survey in surveys:
            values = {
                survey.key,
                survey.name,
                survey.short_name,
                survey.key.replace("_all", ""),
                survey.short_name.replace("-", ""),
            }
            for value in values:
                aliases[_normalize_alias(value)] = survey.key
        return aliases


def clean_masses(masses: Iterable[float]) -> np.ndarray:
    """Return sorted finite positive masses."""

    values = np.asarray(list(masses), dtype=float)
    return np.sort(values[np.isfinite(values) & (values > 0)])


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".fit", ".fits"}:
        try:
            from astropy.table import Table
        except ImportError as exc:
            raise ImportError("Install cmf4all[fits] to load FITS survey tables.") from exc
        return Table.read(path).to_pandas()
    raise ValueError(f"Unsupported table format: {path}")


def _load_yaml_mapping(text: str, path: Path) -> dict[str, dict[str, Any]]:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    surveys: dict[str, dict[str, Any]] = {}
    current_key: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError(f"{path}:{line_number}: expected a top-level survey key.")
            current_key = line[:-1].strip()
            surveys[current_key] = {}
            continue
        if current_key is None or ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected an indented metadata field.")
        key, value = line.strip().split(":", 1)
        surveys[current_key][key.strip()] = _parse_yaml_scalar(value.strip())
    return surveys


def _parse_yaml_scalar(value: str) -> Any:
    if value == "":
        return ""
    value = value.strip()
    lowered = value.casefold()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _infer_project_root(metadata_file: Path) -> Path:
    if len(metadata_file.parts) >= 3 and metadata_file.parts[-3:-1] == ("data", "metadata"):
        return metadata_file.parents[2]
    return metadata_file.parent


def _optional_float(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    return float(value)


def _normalize_alias(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
