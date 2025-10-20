from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close

from foam.foamfield import parse_foam_file
from foam.interface import read_foam_field


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "files" / "identity" / "examples" / "TinyS_NB"


EXPECTED_VALUES = torch.tensor([
    [3.75273e-04, -8.11028e-01, -9.61560e-23, -4.07937e-04, -1.46684e-24, 6.15812e-51],
    [-2.77022e+00, -9.51032e-01, -1.14151e-19, 2.69102e+00, -1.64148e-19, 0.00000e+00],
    [-4.40095e+00, 7.85760e-01, -5.30074e-17, 4.24940e+00, -3.38788e-17, -2.64852e-35],
    [-1.80567e+00, 3.52294e+00, -4.37852e-20, 1.89468e+00, 3.57386e-20, 2.41209e-38],
    [-3.22473e-02, 2.08805e+00, -7.72513e-20, 5.87137e-02, -4.84568e-19, 5.14402e-39],
], dtype=torch.float32)


def test_foam_field_to_tensor_matches_expected() -> None:
    foam_field = parse_foam_file(str(FIXTURE_PATH))
    tensor = foam_field.to_tensor()

    assert tensor.shape == EXPECTED_VALUES.shape
    assert_close(tensor, EXPECTED_VALUES)


def test_read_foam_field_populates_tensor_metadata() -> None:
    tensor_data = read_foam_field(str(FIXTURE_PATH))

    assert tensor_data.tensor.shape == EXPECTED_VALUES.shape
    assert_close(tensor_data.tensor, EXPECTED_VALUES)

    assert tensor_data.rank.values.tolist() == [2]
    assert tensor_data.symmetry.values.tolist() == [1]
