"""Tests for lidar_to_minecraft conversion."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import lidar_to_minecraft
sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_to_minecraft import (
    MinecraftWorldGenerator,
    create_heightmap,
    read_laz_file,
)


def test_read_laz_file():
    """Test reading a LAZ file."""
    laz_file = Path(__file__).parent.parent / "dars" / "USGS_LPC_CA_SanFrancisco_B23_05300270.laz"

    if not laz_file.exists():
        pytest.skip(f"LAZ file not found: {laz_file}")

    x, y, z = read_laz_file(str(laz_file))

    assert len(x) > 0
    assert len(y) > 0
    assert len(z) > 0
    assert len(x) == len(y) == len(z)


def test_create_heightmap():
    """Test creating a heightmap from point cloud data."""
    import numpy as np

    # Create sample point cloud data
    n_points = 1000
    x = np.random.rand(n_points) * 1000
    y = np.random.rand(n_points) * 1000
    z = np.random.rand(n_points) * 100 + 50  # Elevation between 50-150

    heightmap = create_heightmap(x, y, z, resolution=64)

    assert heightmap.shape == (64, 64)
    assert not np.isnan(heightmap).any()


def test_minecraft_world_generator():
    """Test Minecraft world generator."""
    import numpy as np

    # Create a simple heightmap
    heightmap = np.random.rand(128, 128) * 50 + 100

    generator = MinecraftWorldGenerator("test_world", output_dir="test_output")
    block_heights = generator.heightmap_to_blocks(heightmap)

    assert block_heights.shape == heightmap.shape
    assert block_heights.min() >= 64
    assert block_heights.max() <= 320


@pytest.mark.slow
def test_full_conversion(tmp_path):
    """Test the full conversion process with a small sample."""
    laz_file = Path(__file__).parent.parent / "dars" / "USGS_LPC_CA_SanFrancisco_B23_05300270.laz"

    if not laz_file.exists():
        pytest.skip(f"LAZ file not found: {laz_file}")

    # Read LAZ file
    x, y, z = read_laz_file(str(laz_file))

    # Create a small heightmap for testing (lower resolution for speed)
    heightmap = create_heightmap(x, y, z, resolution=128)

    # Generate world in temporary directory
    output_dir = tmp_path / "output"
    generator = MinecraftWorldGenerator("test_world", output_dir=str(output_dir))
    generator.generate_world(heightmap)

    # Verify world files were created
    world_dir = output_dir / "test_world"
    assert world_dir.exists()
    level_dat = world_dir / "level.dat"
    assert level_dat.exists()

    # Verify level.dat is GZIP compressed (Minecraft requires this)
    import gzip

    try:
        with gzip.open(level_dat, "rb") as f:
            f.read(1)  # Try to read at least one byte
        is_gzipped = True
    except (gzip.BadGzipFile, OSError):
        is_gzipped = False

    assert is_gzipped, "level.dat must be GZIP compressed for Minecraft to load it"

    assert (world_dir / "region").exists()
    assert (world_dir / "region").is_dir()

    # Check that at least one region file was created
    region_files = list((world_dir / "region").glob("r.*.mca"))
    assert len(region_files) > 0
