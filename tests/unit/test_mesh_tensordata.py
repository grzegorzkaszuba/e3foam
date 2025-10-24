import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from foam.foamfield import parse_foam_file
from foam.mesh import (
    GaussianMixtureSignalGenerator,
    Mesh,
    RadialSignalGenerator,
    ZeroSignalGenerator,
)

EDDY_VISCOSITY_DIMENSIONS = [0, 2, -1, 0, 0, 0, 0]

POINTS_TEMPLATE = """FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}

4
(
(0 0 0)
(1 0 0)
(1 1 0)
(0 1 0)
)
"""

FACES_TEMPLATE = """FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    location    "constant/polyMesh";
    object      faces;
}

1
(
(0 1 2 3)
)
"""


@pytest.fixture()
def simple_case(tmp_path_factory):
    case_dir = tmp_path_factory.mktemp("simple_case")
    poly_mesh = case_dir / "constant" / "polyMesh"
    poly_mesh.mkdir(parents=True)
    (poly_mesh / "points").write_text(POINTS_TEMPLATE)
    (poly_mesh / "faces").write_text(FACES_TEMPLATE)
    return case_dir


def load_mesh(case_dir):
    return Mesh.from_foam_case(case_dir)


def test_zero_signal_generator_writes_openfoam(tmp_path, simple_case):
    mesh = load_mesh(simple_case)
    generator = ZeroSignalGenerator()
    field = generator.generate(mesh)
    foam_field = mesh.tensor_to_field(
        tensor=field,
        field_type="volScalarField",
        dimensions=EDDY_VISCOSITY_DIMENSIONS,
        object_name="nut",
    )
    output_path = tmp_path / "nut"
    foam_field.inject(str(output_path))

    parsed = parse_foam_file(str(output_path))
    assert parsed.dimensions == EDDY_VISCOSITY_DIMENSIONS
    assert all(value == 0.0 for value in parsed.internal_field)


def test_radial_signal_generator_matches_mesh_geometry(tmp_path, simple_case):
    mesh = load_mesh(simple_case)
    generator = RadialSignalGenerator()
    field = generator.generate(mesh)
    foam_field = mesh.tensor_to_field(
        tensor=field,
        field_type="volScalarField",
        dimensions=EDDY_VISCOSITY_DIMENSIONS,
        object_name="nut",
    )
    output_path = tmp_path / "nutRadial"
    foam_field.inject(str(output_path))

    parsed = parse_foam_file(str(output_path))
    values = np.asarray(parsed.internal_field)
    expected = np.linalg.norm(mesh.points, axis=1)
    assert np.allclose(values, expected)


@pytest.mark.parametrize("origin", [(0.5, 0.5, 0.0), (-1.0, 0.0, 0.0)])
def test_radial_signal_generator_custom_origin(simple_case, origin):
    mesh = load_mesh(simple_case)
    generator = RadialSignalGenerator(origin=origin)
    field = generator.generate(mesh)
    expected = np.linalg.norm(mesh.points - np.asarray(origin), axis=1)
    assert torch.allclose(field.tensor.squeeze(), torch.tensor(expected, dtype=torch.float32))


def test_gaussian_mixture_signal_generator_custom_centers(simple_case):
    mesh = load_mesh(simple_case)
    centers = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    generator = GaussianMixtureSignalGenerator(
        num_components=2,
        mean_variance=0.01,
        variance_std=0.0,
        centers=centers,
    )
    field = generator.generate(mesh)
    values = field.tensor.squeeze().numpy()
    assert values[0] > values[1]
    assert values[2] > values[3]


def test_visualize_gaussian_signal_with_matplotlib(tmp_path, simple_case):
    plt = pytest.importorskip("matplotlib.pyplot")
    mesh = load_mesh(simple_case)
    generator = GaussianMixtureSignalGenerator(num_components=2, random_state=42)
    field = generator.generate(mesh)

    fig, ax = plt.subplots()
    scatter = ax.scatter(mesh.points[:, 0], mesh.points[:, 1], c=field.tensor.squeeze().numpy())
    ax.set_title("Gaussian mixture signal on mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(scatter, ax=ax)

    output_path = tmp_path / "gaussian_signal.png"
    fig.savefig(output_path)
    plt.close(fig)

    assert output_path.exists()
