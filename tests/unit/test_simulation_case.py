from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close

from foam.interface import read_simulation_case


def _write_scalar_field(path: Path, values) -> None:
    n_values = len(values)
    value_lines = "\n".join(f"{v:.6f}" for v in values)
    content = f"""FoamFile
{{
    version     2;
    format      ascii;
    class       volScalarField;
    location    \"0\";
    object      T;
}}

dimensions      [ 0 0 0 0 0 0 0 ];

internalField   nonuniform List<scalar>
{n_values}
(
{value_lines}
)
;
"""
    path.write_text(content)


def test_read_simulation_case_selects_first_numeric_time(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    (case_dir / "abc").mkdir(parents=True)
    for time_label, values in {"0": [1.0, 2.0], "0.5": [3.0, 4.0]}.items():
        time_dir = case_dir / time_label
        time_dir.mkdir(parents=True, exist_ok=True)
        _write_scalar_field(time_dir / "T", values)

    tensor_data = read_simulation_case(str(case_dir), ["T"])
    assert tensor_data.tensor.shape == (2, 1)
    assert_close(tensor_data.tensor, torch.tensor([[1.0], [2.0]], dtype=torch.float32))


def test_read_simulation_case_explicit_time(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    for time_label, values in {"0": [1.0, 2.0], "1": [5.0, 6.0]}.items():
        time_dir = case_dir / time_label
        time_dir.mkdir(parents=True, exist_ok=True)
        _write_scalar_field(time_dir / "T", values)

    tensor_data = read_simulation_case(str(case_dir), ["T"], time_step="1")
    assert_close(tensor_data.tensor, torch.tensor([[5.0], [6.0]], dtype=torch.float32))
