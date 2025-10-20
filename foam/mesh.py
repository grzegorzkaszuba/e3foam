"""Mesh utilities for bridging OpenFOAM geometry with tensor representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import importlib

import numpy as np
import torch

from tensors.base import TensorData

_meshio_spec = importlib.util.find_spec("meshio")
if _meshio_spec is not None:
    meshio = importlib.import_module("meshio")  # type: ignore
else:
    meshio = None  # type: ignore

_torch_geometric_spec = importlib.util.find_spec("torch_geometric.data")
if _torch_geometric_spec is not None:
    Data = importlib.import_module("torch_geometric.data").Data  # type: ignore
else:
    Data = None  # type: ignore


@dataclass
class SignalGenerator:
    """Base class for creating synthetic signals on a mesh."""

    def generate(self, mesh: "Mesh") -> TensorData:
        raise NotImplementedError


@dataclass
class ZeroSignalGenerator(SignalGenerator):
    """Generate a zero-valued scalar field."""

    def generate(self, mesh: "Mesh") -> TensorData:
        zeros = torch.zeros(mesh.num_points, 1, dtype=torch.float32)
        return TensorData(zeros, rank=0)


@dataclass
class RadialSignalGenerator(SignalGenerator):
    """Generate a scalar proportional to the distance from the origin."""

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def generate(self, mesh: "Mesh") -> TensorData:
        diffs = mesh.points - np.asarray(self.origin, dtype=np.float64)
        distances = np.linalg.norm(diffs, axis=1, keepdims=True)
        tensor = torch.from_numpy(distances.astype(np.float32))
        return TensorData(tensor, rank=0)


@dataclass
class Mesh:
    """Container for OpenFOAM mesh geometry."""

    points: np.ndarray
    faces: Optional[List[Sequence[int]]] = None
    raw_mesh: Optional["meshio.Mesh"] = None

    @classmethod
    def from_foam_case(cls, case_path: Path) -> "Mesh":
        base = Path(case_path) / "constant" / "polyMesh"
        points_path = base / "points"
        if not points_path.exists():
            raise FileNotFoundError(f"Missing points file at {points_path}")
        points = cls._parse_points(points_path)
        faces_path = base / "faces"
        faces = cls._parse_faces(faces_path) if faces_path.exists() else None
        mesh_obj = None
        if meshio is not None:
            mesh_obj = meshio.Mesh(points=points, cells={"polygon": np.asarray(faces)}) if faces else meshio.Mesh(points=points, cells=[])
        return cls(points=points, faces=faces, raw_mesh=mesh_obj)

    @staticmethod
    def _parse_points(path: Path) -> np.ndarray:
        raw = _read_parenthesised_block(path)
        data = []
        for line in raw:
            values = [float(v) for v in line.strip().strip("()").split()]
            if not values:
                continue
            if len(values) == 2:
                values.append(0.0)
            data.append(values)
        return np.asarray(data, dtype=np.float64)

    @staticmethod
    def _parse_faces(path: Path) -> List[Sequence[int]]:
        raw = _read_parenthesised_block(path)
        faces: List[Sequence[int]] = []
        for line in raw:
            cleaned = line.strip().rstrip(")")
            if "(" in cleaned:
                cleaned = cleaned.split("(", 1)[1]
            cleaned = cleaned.replace(")", "").replace("(", "")
            if not cleaned:
                continue
            faces.append(tuple(int(idx) for idx in cleaned.split()))
        return faces

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def _edge_index(self) -> torch.Tensor:
        if self.faces:
            edges = set()
            for face in self.faces:
                if len(face) < 2:
                    continue
                for i, start in enumerate(face):
                    end = face[(i + 1) % len(face)]
                    edge = tuple(sorted((start, end)))
                    edges.add(edge)
        else:
            edges = self._knn_edges(k=3)
        if not edges:
            raise ValueError("Mesh does not contain connectivity information")
        edge_array = np.array(list(edges), dtype=np.int64)
        # undirected graph => add both directions
        sources = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        targets = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
        return torch.from_numpy(np.vstack([sources, targets]))

    def _knn_edges(self, k: int) -> set:
        edges = set()
        for i, point in enumerate(self.points):
            distances = np.linalg.norm(self.points - point, axis=1)
            neighbour_indices = np.argsort(distances)[1 : k + 1]
            for j in neighbour_indices:
                if i == j:
                    continue
                edge = tuple(sorted((i, j)))
                edges.add(edge)
        return edges

    def to_graph(self, field: Optional[TensorData] = None):
        if Data is None:
            raise ImportError(
                "torch_geometric is required to build graph objects. Install torch_geometric to use this method."
            )
        node_features: torch.Tensor
        if field is None:
            node_features = torch.zeros(self.num_points, 1, dtype=torch.float32)
        else:
            tensor = field.tensor
            if tensor.shape[0] != self.num_points:
                raise ValueError("TensorData does not align with mesh points")
            node_features = tensor.float()
        pos = torch.from_numpy(self.points.astype(np.float32))
        edge_index = self._edge_index()
        return Data(x=node_features, pos=pos, edge_index=edge_index)

    def to_artificial_graph(self, signal_generator: SignalGenerator):
        field = signal_generator.generate(self)
        return self.to_graph(field)

    def tensor_to_field(
        self,
        tensor: TensorData,
        field_type: str,
        dimensions: Sequence[int],
        object_name: str,
        boundary_field=None,
    ):
        values = tensor.tensor.reshape(self.num_points).tolist()
        return FoamField(
            field_type=field_type,
            dimensions=list(dimensions),
            internal_field=values,
            boundary_field=boundary_field,
            location=object_name,
        )


from foam.foamfield import FoamField


def _read_parenthesised_block(path: Path) -> Iterable[str]:
    lines = Path(path).read_text().splitlines()
    data_started = False
    buffer: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not data_started:
            if stripped.isdigit():
                continue
            if stripped == "(":
                data_started = True
            continue
        if stripped == ")":
            break
        if stripped:
            buffer.append(stripped)
    return buffer


__all__ = [
    "Mesh",
    "SignalGenerator",
    "ZeroSignalGenerator",
    "RadialSignalGenerator",
]
