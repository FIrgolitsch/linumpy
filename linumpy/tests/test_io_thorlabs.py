"""Tests for linumpy/io/thorlabs.py — Thorlabs PSOCT OCT data handling.

The ThorOCT class reads from a zip archive containing a ``Header.xml`` metadata
file and ``data/Complex.data`` / ``data/Complex_Cam1.data`` binary blobs. These
tests build a minimal synthetic zip in ``tmp_path`` so no real subject data is
needed.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from linumpy.io.thorlabs import PreprocessingConfig, ThorOCT

# ---------------------------------------------------------------------------
# Synthetic Thorlabs zip builder
# ---------------------------------------------------------------------------


def _header_xml(
    size_x: int = 4,
    size_y: int = 4,
    size_z: int = 8,
    range_x: float = 1.0,
    range_y: float = 1.0,
    range_z: float = 1.0,
    ascan_averaging: int = 1,
) -> str:
    """Build a minimal Thorlabs Header.xml matching the parser's expectations."""
    data_attr = f'SizeZ="{size_z}" SizeX="{size_x}" SizeY="{size_y}" RangeX="{range_x}" RangeY="{range_y}" RangeZ="{range_z}"'
    return f"""<?xml version="1.0"?>
<ThorImage>
  <AScans>{ascan_averaging}</AScans>
  <DataFile {data_attr}>data\\Complex.data</DataFile>
</ThorImage>
"""


def _make_synthetic_oct_zip(
    path: Path,
    size_x: int = 4,
    size_y: int = 4,
    size_z: int = 8,
    range_x: float = 1.0,
    range_y: float = 1.0,
    range_z: float = 1.0,
    ascan_averaging: int = 1,
    include_pol1: bool = True,
    include_pol2: bool = True,
) -> None:
    """Write a synthetic .oct zip with Header.xml + complex data blobs."""
    header = _header_xml(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        range_x=range_x,
        range_y=range_y,
        range_z=range_z,
        ascan_averaging=ascan_averaging,
    )
    # Complex data: shape (size_x, size_y, size_z), complex64.
    data = np.ones((size_x, size_y, size_z), dtype=np.complex64)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Header.xml", header)
        if include_pol1:
            zf.writestr("data/Complex.data", data.tobytes())
        if include_pol2:
            zf.writestr("data/Complex_Cam1.data", data.tobytes())


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------


def test_preprocessing_config_defaults():
    """The PreprocessingConfig class has the documented defaults.

    Note: ``return_complex`` is declared as a bare class-level annotation
    (no default value), so it is NOT set on instances. The test sets it
    explicitly before checking the other defaults.
    """
    cfg = PreprocessingConfig()
    cfg.return_complex = False
    assert cfg.return_complex is False
    assert cfg.crop_first_index == 320
    assert cfg.crop_second_index == 750
    assert cfg.erase_raw_data is True
    assert cfg.erase_polarization_1 is False
    assert cfg.erase_polarization_2 is True


# ---------------------------------------------------------------------------
# ThorOCT.load + header extraction
# ---------------------------------------------------------------------------


def test_thoroct_load_extracts_dimensions(tmp_path):
    """load() parses the header and populates size_x/y/z + resolution."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=4, size_y=6, size_z=8, range_x=2.0, range_y=3.0, range_z=4.0)
    cfg = PreprocessingConfig()
    cfg.return_complex = True
    cfg.crop_first_index = 0
    cfg.crop_second_index = 8  # full depth, no crop
    cfg.erase_polarization_2 = True
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.size_x == 4
    assert thor.size_y == 6
    assert thor.size_z == 8
    # resolution = range / size for each axis.
    np.testing.assert_allclose(thor.resolution[0], 2.0 / 4, rtol=1e-9)
    np.testing.assert_allclose(thor.resolution[1], 3.0 / 6, rtol=1e-9)
    np.testing.assert_allclose(thor.resolution[2], 4.0 / 8, rtol=1e-9)


def test_thoroct_load_extracts_ascan_averaging(tmp_path):
    """load() extracts the AScan averaging value from the header."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, ascan_averaging=2)
    cfg = PreprocessingConfig()
    cfg.return_complex = True
    cfg.crop_first_index = 0
    cfg.crop_second_index = 8
    cfg.erase_polarization_2 = True
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.ascan_averaging_value == 2


def test_thoroct_load_without_config_raises(tmp_path):
    """load() without a config raises (the assert in load fires)."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path)
    thor = ThorOCT(path=oct_path, config=None)
    with pytest.raises(AssertionError):
        thor.load()


def test_thoroct_load_without_data_source_raises():
    """load() with no path and no compressed_data raises ValueError."""
    thor = ThorOCT(path=None, config=PreprocessingConfig())
    with pytest.raises(ValueError, match="No valid data source"):
        thor.load()


def test_thoroct_load_missing_header_raises(tmp_path):
    """load() with a zip missing Header.xml raises FileNotFoundError."""
    oct_path = tmp_path / "bad.oct"
    with zipfile.ZipFile(oct_path, "w") as zf:
        zf.writestr("dummy.txt", "no header here")
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    with pytest.raises(FileNotFoundError, match="Error loading header"):
        thor.load()


# ---------------------------------------------------------------------------
# Polarization data loading
# ---------------------------------------------------------------------------


def test_thoroct_load_magnitude_returns_float(tmp_path):
    """return_complex=False produces a float64 magnitude array."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=4)
    cfg = PreprocessingConfig()
    cfg.return_complex = False
    cfg.crop_first_index = 0
    cfg.crop_second_index = 4
    cfg.erase_polarization_2 = True
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.first_polarization is not None
    assert thor.first_polarization.dtype == np.float64
    # magnitude of complex ones = 1.0.
    np.testing.assert_allclose(thor.first_polarization, 1.0)


def test_thoroct_load_complex_returns_complex(tmp_path):
    """return_complex=True keeps the complex64 data."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=4)
    cfg = PreprocessingConfig()
    cfg.return_complex = True
    cfg.crop_first_index = 0
    cfg.crop_second_index = 4
    cfg.erase_polarization_2 = True
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.first_polarization is not None
    assert np.iscomplexobj(thor.first_polarization)


def test_thoroct_load_both_polarizations(tmp_path):
    """Both polarizations are loaded when neither is erased."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=4)
    cfg = PreprocessingConfig()
    cfg.return_complex = True
    cfg.crop_first_index = 0
    cfg.crop_second_index = 4
    cfg.erase_polarization_1 = False
    cfg.erase_polarization_2 = False
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.first_polarization is not None
    assert thor.second_polarization is not None


def test_thoroct_load_erase_raw_data_clears_zip(tmp_path):
    """erase_raw_data=True sets compressed_data to None after load."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=4)
    cfg = PreprocessingConfig()
    cfg.return_complex = True
    cfg.crop_first_index = 0
    cfg.crop_second_index = 4
    cfg.erase_raw_data = True
    cfg.erase_polarization_2 = True
    thor = ThorOCT(path=oct_path, config=cfg)
    thor.load()
    assert thor.compressed_data is None


# ---------------------------------------------------------------------------
# _crop_z
# ---------------------------------------------------------------------------


def test_crop_z_returns_cropped_volume(tmp_path):
    """_crop_z crops along the Z-axis between the two indices."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=8)
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    thor._extract_oct_header()
    thor._extract_complex_dimensions()
    data = np.arange(2 * 2 * 8, dtype=np.float64).reshape(2, 2, 8)
    cropped = thor._crop_z(data, index1=2, index2=6)
    assert cropped.shape == (2, 2, 4)
    assert thor.size_z == 4


def test_crop_z_invalid_indices_raises(tmp_path):
    """_crop_z raises ValueError for invalid indices."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=2, size_y=2, size_z=8)
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    thor._extract_oct_header()
    thor._extract_complex_dimensions()
    data = np.zeros((2, 2, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="Invalid indices"):
        thor._crop_z(data, index1=5, index2=3)  # index1 >= index2
    with pytest.raises(ValueError, match="Invalid indices"):
        thor._crop_z(data, index1=-1, index2=3)  # index1 < 0
    with pytest.raises(ValueError, match="Invalid indices"):
        thor._crop_z(data, index1=0, index2=99)  # index2 > shape


# ---------------------------------------------------------------------------
# _stack_tiles_vertically
# ---------------------------------------------------------------------------


def test_stack_tiles_vertically_stacks_along_z(tmp_path):
    """_stack_tiles_vertically groups tiles and stacks them along the z-axis.

    The function concatenates ``ascan_averaging`` tiles along axis=0 (the
    first axis of the (SizeX, SizeY, SizeZ) data), then reverses that axis.
    With 6 tiles of shape (4, 2) and ascan_averaging=2, each group of 2
    tiles becomes (8, 2) and 3 groups are stacked -> (3, 8, 2).
    """
    oct_path = tmp_path / "synthetic.oct"
    # ascan_averaging=2 -> 6 tiles grouped into 3 stacks of 2.
    _make_synthetic_oct_zip(oct_path, size_x=6, size_y=4, size_z=2, ascan_averaging=2)
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    thor._extract_oct_header()
    thor._extract_complex_dimensions()
    data = np.arange(6 * 4 * 2, dtype=np.float64).reshape(6, 4, 2)
    stacked = thor._stack_tiles_vertically(data)
    # 6 tiles / 2 = 3 groups; each group stacks 2 tiles of (4, 2) along axis 0 -> (8, 2).
    assert stacked.shape == (3, 8, 2)


def test_stack_tiles_vertically_indivisible_raises(tmp_path):
    """_stack_tiles_vertically raises when tiles aren't divisible by ascan_avg."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=5, size_y=4, size_z=2, ascan_averaging=2)
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    thor._extract_oct_header()
    thor._extract_complex_dimensions()
    data = np.zeros((5, 4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="must be divisible"):
        thor._stack_tiles_vertically(data)


# ---------------------------------------------------------------------------
# orient_volume_psoct
# ---------------------------------------------------------------------------


def test_orient_volume_psoct_transposes_to_ras():
    """orient_volume_psoct transposes (X, Y, Z) -> (Z, X, Y) for RAS orientation."""
    vol = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = ThorOCT.orient_volume_psoct(vol)
    assert out.shape == (4, 2, 3)
    # The transpose is (2, 0, 1): out[z, x, y] = vol[x, y, z].
    np.testing.assert_array_equal(out[0], vol[:, :, 0])


# ---------------------------------------------------------------------------
# extract_positions_from_scan
# ---------------------------------------------------------------------------


def test_extract_positions_from_scan_remaps_indices(tmp_path):
    """extract_positions_from_scan remaps raw x/y to grid indices."""
    scan_path = tmp_path / "tiles.scan"
    scan_path.write_text("Positions\n0.1, 0.3\n0.2, 0.3\n0.1, 0.2\n0.2, 0.2\n")
    new_data, raw_positions = ThorOCT.extract_positions_from_scan(str(scan_path))
    # 4 raw positions, z=0.
    assert len(raw_positions) == 4
    assert all(pos[2] == 0 for pos in raw_positions)
    # x remapped to ascending index, y to descending index.
    xs = {pos[0] for pos in new_data}
    ys = {pos[1] for pos in new_data}
    assert xs == {0, 1}
    assert ys == {0, 1}


def test_extract_positions_from_scan_none_path_returns_empty():
    """A None scan path returns empty position lists."""
    new_data, raw_positions = ThorOCT.extract_positions_from_scan(None)
    assert new_data == []
    assert raw_positions == []


# ---------------------------------------------------------------------------
# get_psoct_tiles_ids
# ---------------------------------------------------------------------------


def test_get_psoct_tiles_ids_groups_by_angle(tmp_path):
    """get_psoct_tiles_ids groups .oct files by angle and reads positions from .scan."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # Write a .scan file.
    (tiles_dir / "tile.scan").write_text("Positions\n0.1, 0.1\n0.2, 0.2\n")
    # Write 4 .oct files.
    for i in range(4):
        _make_synthetic_oct_zip(tiles_dir / f"tile_{i}.oct", size_x=2, size_y=2, size_z=2)
    grouped_files, positions = ThorOCT.get_psoct_tiles_ids(tiles_dir, number_of_angles=2)
    assert len(grouped_files) == 2
    assert len(grouped_files[0]) == 2  # files 0, 2
    assert len(grouped_files[1]) == 2  # files 1, 3
    assert len(positions) == 2


def test_get_psoct_tiles_ids_no_oct_files_raises(tmp_path):
    """get_psoct_tiles_ids raises ValueError when no .oct files are found."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    (tiles_dir / "tile.scan").write_text("Positions\n0.1, 0.1\n")
    with pytest.raises(ValueError, match=r"No \.oct files"):
        ThorOCT.get_psoct_tiles_ids(tiles_dir)


def test_get_psoct_tiles_ids_invalid_dir_raises(tmp_path):
    """get_psoct_tiles_ids raises ValueError for a non-directory path."""
    with pytest.raises(ValueError, match="not a valid directory"):
        ThorOCT.get_psoct_tiles_ids(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# _load_raw_data
# ---------------------------------------------------------------------------


def test_load_raw_data_returns_correct_shape(tmp_path):
    """_load_raw_data reads the complex blob and reshapes to (size_x, size_y, size_z)."""
    oct_path = tmp_path / "synthetic.oct"
    _make_synthetic_oct_zip(oct_path, size_x=3, size_y=5, size_z=7)
    thor = ThorOCT(path=oct_path, config=PreprocessingConfig())
    thor._extract_oct_header()
    thor._extract_complex_dimensions()
    raw = thor._load_raw_data("data/Complex.data")
    assert raw.shape == (3, 5, 7)
    assert raw.dtype == np.complex64
