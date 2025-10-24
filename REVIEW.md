# e3foam Codebase Review

This document captures a first-pass review of the current e3foam repository with a focus on how
OpenFOAM data structures are mapped into tensor representations for use with e3nn. The intent is to
summarize the main data abstractions, document notable design decisions, and point out potential
improvements or correctness risks in each script.

## Core Data Structures

### `tensors/base.py`
- `TensorData` is the central abstraction: it keeps a 2D tensor (batch, features) alongside
  metadata (`TensorIndex` instances) describing field boundaries (`ptr`), per-field tensor ranks,
  symmetries, and parities.【F:tensors/base.py†L87-L166】【F:tensors/base.py†L233-L314】 The class
  handles both single-field tensors (automatic rank inference with optional symmetry) and multi-field
  compositions (`TensorData.cat`, `TensorData.append`). It also provides flattening helpers that map
  rank-2 tensors with symmetry into minimal component sets.
- `TensorIndex` (near the bottom of the file) tracks structured metadata with pointer arrays similar
  to CSR indices, supporting alignment operations and safe slicing across concatenated fields.
- Utility helpers (`safe_divide`, projection helpers) and adapter accessors (`get_irrep_rizzler`,
  `get_cartesian_rizzler`) bridge to `e3nn` irreps space via the adapter classes in
  `tensors/adapters.py` and `tensors/utils.py`.

### `tensors/adapters.py` and `tensors/utils.py`
- `IrrepAdapter` and `CartesianAdapter` wrap `TensorData` into e3nn-compatible carriers, caching
  representations and providing callable interfaces to transform between Cartesian and irrep forms.
  The adapters assume consistent metadata coming from `TensorData` and expose hooks to cache reduced
  tensor products for repeated use.【F:tensors/adapters.py†L1-L191】
- `project_tensor_to_2d` and plotting utilities in `tensors/utils.py` primarily serve visualization
  and dimensionality reduction tasks while respecting tensor symmetries.【F:tensors/utils.py†L1-L217】

## Foam I/O Layer

### `foam/foamfield.py`
- `FoamField` dataclass wraps parsed OpenFOAM field files, exposing convenience helpers like
  `.to_tensor()` and `.inject()` to convert between plain lists and `torch.Tensor`/OpenFOAM ASCII
  formats.【F:foam/foamfield.py†L17-L118】 The parser scans raw files line by line, captures internal
  fields, and keeps boundary dictionaries for later reinjection.【F:foam/foamfield.py†L121-L208】
- `compare_foam_fields` and `plot_case` provide evaluation and visualization utilities over time
  directories, leveraging `TensorData` where possible.【F:foam/foamfield.py†L210-L344】 The comparison
  routine safeguards relative-error divisions by scaling with the mean magnitude of the field.
- **Observations / Suggestions:**
  - `parse_foam_file` treats every non-parenthesized numeric line inside `internalField` as data. It
    skips entries that begin with `(`, which is correct for scalar fields but means vector/tensor
    entries must be on single lines. Consider handling multi-line tuples or embedded comments to make
    parsing more robust.
  - `inject` assumes boundary values are already formatted strings; providing structure-aware
    formatting (e.g., for vector-valued boundary conditions) could improve round-tripping.
  - For large cases, loading everything into Python lists before tensor conversion may be heavy;
    exploring streaming or memory-mapped conversions could improve scalability.

### `foam/interface.py`
- High-level glue translating between `FoamField` and `TensorData`. `read_foam_field` maps OpenFOAM
  field classes (`volScalarField`, etc.) to rank/symmetry metadata, while `read_simulation_case`
  batches multiple files into a concatenated `TensorData` bundle.【F:foam/interface.py†L1-L77】
- `tensor_to_foam`/`foam_to_tensor` provide dictionary-based serializations useful for templating or
  service interfaces.【F:foam/interface.py†L79-L145】
- **Observations:**
  - `read_simulation_case` picks the lexicographically smallest time directory; when case folders mix
    numeric and non-numeric names, the filtering logic (`d[0].isdigit()`) might skip valid steps such
    as `.0005`. Reusing the more robust numeric detection from `plot_case` could help.
  - `tensor_to_foam` returns flattened symmetric tensors (length 6), yet `format_tensor_field` in
    `foam/writers.py` expands them again. Ensuring these pathways agree on orientation order would
    avoid subtle mismatches.

### `foam/writers.py`
- Focused on reinjecting `TensorData` back into OpenFOAM ASCII files. `format_tensor_field` expands
  by rank/symmetry, and `inject_tensor_field` rewrites `internalField` sections in-place.
- **Observations:**
  - `inject_tensor_field` searches for the first closing parenthesis after `internalField`; this can
    misidentify nested parentheses (e.g., boundary lists that follow immediately). Parsing sections
    with a simple state machine, similar to the reader, would make the writer more resilient.
  - Template-based multi-field injection (`inject_tensor_fields`) expects predefined template files
    named by field type and writes outputs as `field_{i}`. Documenting the expected template folder
    structure would ease reuse.【F:foam/writers.py†L1-L94】

### `foam/foamfield_new.py`
- Experimental refactor that combines parsing, type inference, and mapping to tensor metadata. The
  module introduces `FoamMesh`, `FoamFieldMeta`, and `FoamDataset` to better represent complete case
  hierarchies, but key methods remain incomplete and marked with TODOs.【F:foam/foamfield_new.py†L1-L240】
  As noted, the implementation does not yet support round-tripping or complex boundary handling.
  Treat it as a design sketch for future work.

## Preprocessing

### `preprocessing/scalers.py`
- Defines `EquivariantScalerWrapper` (delegating to an underlying scaler per field) and specific
  scaler implementations (`MeanScaler`, `StandardScalerWrapper`). Each scaler respects tensor ranks
  and applies operations field-wise via `TensorIndex` iteration.【F:preprocessing/scalers.py†L1-L266】
- Potential enhancements include caching fitted statistics on `TensorData` metadata (to avoid
  recomputing per batch) and validating that transformed tensors preserve expected shapes.

## Development Utilities

### `dev/datamodule.py`
- PyTorch Lightning `LightningDataModule` for CFD cases: loads cases, concatenates fields with
  `TensorData.cat`, scales via equivariant scalers, and exposes rizzlers for irrep/cartesian
  conversions.【F:dev/datamodule.py†L1-L166】
- **Observations:**
  - The module assumes a `.env` variable `PROJECT_ROOT`; failing to set it will raise a KeyError.
    Consider providing a clearer error message or allowing relative paths.
  - `submodule_root` calculation (`os.path.join(this_dir, 'e3foam')`) looks suspicious—when running
    inside the package it creates a `.../dev/e3foam` path that likely does not exist. Simplifying the
    import strategy (e.g., using relative imports) would reduce fragility.
  - `TensorData.append` is used for dataset aggregation; confirm that appended cases maintain
    consistent batch sizes, otherwise Lightning dataloaders may mix different mesh sizes.

## Tutorials and Tests

- Tutorials (e.g., `tutorials/handy_injection.py`) demonstrate end-to-end usage of readers, scalers,
  and writers. Keeping these notebooks/scripts in sync with the core APIs helps new users understand
  the workflow.【F:tutorials/handy_injection.py†L1-L200】

### OpenFOAM Test Assets and Coverage

- Sample OpenFOAM fields live under `files/`: identity/round-trip fixtures (`TinyS`, `TinyS_NB`),
  plotting references (`plotting/truth/SimpleS`), and comparison cases (`comparison/examples/Rrans`,
  `Rdns`). They offer small symmetric tensor fields that are cheap to parse and inject during
  testing.【F:files/identity/TinyS†L1-L37】【F:files/identity/examples/TinyS_NB†L1-L24】【F:files/plotting/truth/SimpleS†L1-L24】【F:files/comparison/examples/Rrans†L1-L24】
- `tests/test_foam_identity.py` exercises read→write round-tripping but currently only iterates over
  examples without asserting file equality (the `check_file_identity` assertion is commented out), so
  regressions would pass silently.【F:tests/test_foam_identity.py†L7-L26】
- `tests/test_foam_to_array.py` confirms tensor conversion by printing metadata; however, the lack of
  assertions or golden tensors prevents automated verification.【F:tests/test_foam_to_array.py†L6-L21】
- `tests/test_plotting.py` and `tests/test_reynolds_stress.py` drive plotting and comparison
  utilities but swallow exceptions and only log results, again leaving correctness unchecked by the
  test runner.【F:tests/test_plotting.py†L36-L61】【F:tests/test_reynolds_stress.py†L20-L82】

**Suggested enhancements for a more comprehensive testing framework:**

1. Restore strict round-trip checks by diffing regenerated files byte-for-byte (or parsing and
   asserting tensor equality with tolerances) and expand fixtures to cover scalar, vector, and tensor
   boundary conditions.【F:tests/test_foam_identity.py†L20-L26】
2. Replace print-based checks with `pytest` assertions—e.g., compare `.to_tensor()` outputs to saved
   `torch.Tensor` fixtures or `numpy` arrays; validate metadata (dimensions, component order) against
   expected dictionaries.【F:tests/test_foam_to_array.py†L10-L17】
3. Parameterize plotting/comparison tests with temporary directories so metrics and artifacts are
   validated (e.g., ensure `plot_case` raises for missing fields, verify produced PNG hashes, confirm
   `compare_foam_fields` error magnitudes).【F:tests/test_plotting.py†L36-L61】【F:tests/test_reynolds_stress.py†L20-L82】
4. Introduce synthetic multi-time-step cases to exercise `read_simulation_case` and writers across
   multiple timesteps, including boundary patches and mixed field types. This will stress IO helpers
   beyond the minimal identity fixtures.【F:foam/interface.py†L1-L118】

## General Recommendations

1. **Consistent Import Paths:** Several modules manipulate `sys.path` or rely on package-relative
   imports. As the project matures, consider enforcing package-relative imports and supplying a
   `pyproject.toml` to manage dependencies cleanly.
2. **Robust Parsing/Writing:** The reader and writer both rely on simple string matching. Investing in
   a shared parser (possibly leveraging `pyfoam` or a lightweight grammar) would reduce duplication
   and handle edge cases like comments, scientific notation with uppercase `E`, or multi-line entries.
3. **Metadata Validation:** Provide helper assertions that verify `TensorData` objects coming from
   OpenFOAM maintain consistent ranks, symmetries, and component counts. This would catch mis-tagged
   tensors early in preprocessing.
4. **Performance Considerations:** For large CFD datasets, consider lazy loading or memory-mapped
   tensors instead of reading entire fields into Python lists before converting to torch tensors.
5. **Document Future Work:** `foamfield_new.py` outlines a promising direction. Annotating it with a
   roadmap (e.g., mesh handling, boundary condition typing) will help prioritize implementation steps.

