from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.testing import assert_close

from foam.foamfield import parse_foam_file


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "files" / "identity" / "examples"


@pytest.mark.parametrize("fixture_name", ["TinyS_NB"])
def test_parse_and_inject_round_trip(tmp_path: Path, fixture_name: str) -> None:
    source_path = FIXTURE_ROOT / fixture_name
    original = parse_foam_file(str(source_path))

    round_trip_path = tmp_path / fixture_name
    original.inject(str(round_trip_path))

    regenerated = parse_foam_file(str(round_trip_path))

    assert regenerated.field_type == original.field_type
    assert regenerated.dimensions == original.dimensions
    assert regenerated.boundary_field == original.boundary_field

    assert_close(regenerated.to_tensor(), original.to_tensor())


@pytest.mark.parametrize("fixture_name", ["TinyS_NB"])
def test_round_trip_preserves_serialization(tmp_path: Path, fixture_name: str) -> None:
    source_path = FIXTURE_ROOT / fixture_name

    original_text = source_path.read_text()
    foam_field = parse_foam_file(str(source_path))

    injected_path = tmp_path / fixture_name
    foam_field.inject(str(injected_path))

    regenerated_text = injected_path.read_text()

    # Normalise whitespace to ignore trailing spaces introduced by formatting
    normalise = lambda text: "\n".join(line.rstrip() for line in text.splitlines())
    assert normalise(regenerated_text) == normalise(original_text)
