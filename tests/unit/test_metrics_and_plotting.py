from pathlib import Path
from typing import List

import pytest

torch = pytest.importorskip("torch")

from foam.foamfield import FoamField, compare_foam_fields, parse_foam_file, plot_case


IDENTITY_FIXTURE = Path(__file__).resolve().parents[2] / "files" / "identity" / "examples" / "TinyS_NB"
PLOTTING_CASE = Path(__file__).resolve().parents[2] / "files" / "plotting" / "examples"
PLOTTING_TRUTH = Path(__file__).resolve().parents[2] / "files" / "plotting" / "truth" / "SimpleS"


def _copy_with_delta(field: FoamField, delta: float) -> FoamField:
    perturbed_values: List[List[float]] = []
    for row in field.internal_field:
        perturbed_row = [value + delta for value in row]
        perturbed_values.append(perturbed_row)

    return FoamField(
        field_type=field.field_type,
        dimensions=list(field.dimensions),
        internal_field=perturbed_values,
        boundary_field=field.boundary_field,
        format_version=field.format_version,
        format_type=field.format_type,
    )


def test_compare_foam_fields_reports_expected_metrics(tmp_path: Path) -> None:
    ground_truth = parse_foam_file(str(IDENTITY_FIXTURE))
    prediction = _copy_with_delta(ground_truth, delta=0.1)

    prediction_path = tmp_path / "prediction"
    prediction.inject(str(prediction_path))

    metrics = compare_foam_fields(str(IDENTITY_FIXTURE), str(prediction_path))

    gt_tensor = ground_truth.to_tensor()
    pred_tensor = prediction.to_tensor()
    abs_error = torch.abs(gt_tensor - pred_tensor)
    squared_error = abs_error ** 2
    scale = torch.mean(torch.abs(gt_tensor))
    denom = torch.maximum(torch.abs(gt_tensor), torch.full_like(gt_tensor, 1e-10 * scale))
    relative_error = abs_error / denom

    expected_metrics = {
        "MSE": torch.mean(squared_error).item(),
        "RMSE": torch.sqrt(torch.mean(squared_error)).item(),
        "MAE": torch.mean(abs_error).item(),
        "MAPE": torch.mean(relative_error * 100).item(),
        "max_error": torch.max(abs_error).item(),
        "max_relative_error": torch.max(relative_error * 100).item(),
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)


def test_plot_case_filters_timesteps_and_saves_plot(tmp_path: Path) -> None:
    truth_tensor = parse_foam_file(str(PLOTTING_TRUTH)).to_tensor()

    def mean_absolute_error(vars_tensors, const_tensors):
        return torch.mean(torch.abs(vars_tensors[0] - const_tensors[0]))

    plot_path = tmp_path / "metrics.png"
    times, metric_values = plot_case(
        case_path=str(PLOTTING_CASE),
        variables=["SimpleS"],
        constants=[truth_tensor],
        metrics=[mean_absolute_error],
        show=False,
        save_path=str(plot_path),
    )

    assert plot_path.exists()
    assert list(times) == [0.0, 7.47, 15.0]
    assert len(metric_values) == 1
    assert len(metric_values[0]) == 3
    assert all(value >= 0 for value in metric_values[0])
