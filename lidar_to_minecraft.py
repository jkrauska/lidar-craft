#!/usr/bin/env python3
"""
Convert LAZ LiDAR files or GeoTIFF elevation data to Minecraft Java Edition worlds.
"""

import argparse
import cProfile
import functools
import hashlib
import multiprocessing
import os
import pstats
import struct
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import diskcache as dc
except ImportError:
    print("Error: diskcache is not installed. Please run: uv sync")
    sys.exit(1)

try:
    import laspy
except ImportError:
    print("Error: laspy is not installed. Please run: uv sync")
    sys.exit(1)

try:
    import rasterio
except ImportError:
    print("Error: rasterio is not installed. Please run: uv sync")
    sys.exit(1)

try:
    from nbtlib import Byte, ByteArray, Compound, File, Int, List, Long, Short, String

    try:
        from nbtlib import Boolean
    except ImportError:
        # Boolean might not be available, use Byte instead
        Boolean = None
except ImportError:
    print("Error: nbtlib is not installed. Please run: uv sync")
    sys.exit(1)

# Initialize disk cache for expensive operations
CACHE_DIR = Path.home() / ".cache" / "lidar-craft"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = dc.Cache(str(CACHE_DIR), size_limit=2**30)  # 1GB cache limit

# Global profiling state
_profiler = None
_timings = {}


def timing_decorator(func_name: str | None = None):
    """Decorator to time function execution."""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if name not in _timings:
                    _timings[name] = []
                _timings[name].append(elapsed)
        return wrapper
    return decorator


@contextmanager
def timing_context(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name not in _timings:
            _timings[name] = []
        _timings[name].append(elapsed)


class MinecraftWorldGenerator:
    """Generate Minecraft Java Edition world from heightmap data."""

    def __init__(self, world_name: str, output_dir: str = "output"):
        self.world_name = world_name
        self.output_dir = Path(output_dir)
        self.world_dir = self.output_dir / world_name
        self.region_dir = self.world_dir / "region"

        # Create directories
        self.world_dir.mkdir(parents=True, exist_ok=True)
        self.region_dir.mkdir(parents=True, exist_ok=True)

    def create_level_dat(self, block_heights: np.ndarray):
        """Create level.dat file for Minecraft world."""
        # Calculate world bounds
        # block_heights shape: (rows, cols) = (Minecraft Z, Minecraft X)
        # First dimension = Minecraft Z (north-south)
        # Second dimension = Minecraft X (east-west)
        height, width = block_heights.shape

        # Spawn at center of terrain in Minecraft world coordinates
        # World coordinates start at (0, 0) for the first block
        spawn_x = width // 2
        spawn_z = height // 2

        # Find spawn Y at the terrain height at spawn location
        # Ensure spawn is on solid ground (not air)
        spawn_terrain_height = int(block_heights[spawn_z, spawn_x])
        spawn_y = max(spawn_terrain_height + 2, 64)  # At least 2 blocks above terrain, minimum Y=64

        # Verify spawn Y is reasonable
        if spawn_y < -64 or spawn_y > 320:
            print(f"WARNING: Spawn Y={spawn_y} is outside valid Minecraft range (-64 to 320)!")

        # Create level.dat structure
        level_data = Compound(
            {
                "Data": Compound(
                    {
                        "LevelName": String(self.world_name),
                        "generatorName": String("flat"),  # Use flat generator with empty layers
                        "generatorVersion": Int(1),
                        "generatorOptions": String(""),
                        "MapFeatures": Byte(0),  # Disable structures
                        "hardcore": Byte(0),
                        "allowCommands": Byte(1),
                        "GameType": Int(1),  # Creative
                        "Difficulty": Byte(2),  # Normal
                        "SpawnX": Int(spawn_x),
                        "SpawnY": Int(spawn_y),
                        "SpawnZ": Int(spawn_z),
                        "Time": Long(0),
                        "DayTime": Long(6000),
                        "LastPlayed": Long(0),
                        "SizeOnDisk": Long(0),
                        "RandomSeed": Long(0),
                        "version": Int(19133),  # Minecraft 1.20.1
                        "DataVersion": Int(3465),  # 1.20.1
                        "raining": Byte(0),
                        "rainTime": Int(0),
                        "thundering": Byte(0),
                        "thunderTime": Int(0),
                        "DifficultyLocked": Byte(0),
                        "GameRules": Compound(
                            {
                                "showCoordinates": String("true"),
                            }
                        ),
                        "DataPacks": Compound(
                            {
                                "Enabled": List([String("vanilla")]),
                                "Disabled": List([]),
                            }
                        ),
                        "WorldGenSettings": Compound(
                            {
                                "seed": Long(0),
                                "generate_features": Byte(0),  # false - don't generate structures
                                "bonus_chest": Byte(0),  # false
                                "dimensions": Compound(
                                    {
                                        "minecraft:overworld": Compound(
                                            {
                                                "type": String("minecraft:overworld"),
                                                "generator": Compound(
                                                    {
                                                        "type": String("minecraft:flat"),
                                                        "settings": Compound(
                                                            {
                                                                "layers": List(
                                                                    []
                                                                ),  # Empty layers = void world
                                                                "biome": String("minecraft:plains"),
                                                                "lakes": Byte(0),  # false
                                                                "features": Byte(0),  # false
                                                            }
                                                        ),
                                                        "biome_source": Compound(
                                                            {
                                                                "type": String("minecraft:fixed"),
                                                                "biome": String("minecraft:plains"),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                        "minecraft:the_nether": Compound(
                                            {
                                                "type": String("minecraft:the_nether"),
                                                "generator": Compound(
                                                    {
                                                        "type": String("minecraft:noise"),
                                                        "settings": String("minecraft:nether"),
                                                        "biome_source": Compound(
                                                            {
                                                                "type": String(
                                                                    "minecraft:multi_noise"
                                                                ),
                                                                "preset": String(
                                                                    "minecraft:nether"
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                        "minecraft:the_end": Compound(
                                            {
                                                "type": String("minecraft:the_end"),
                                                "generator": Compound(
                                                    {
                                                        "type": String("minecraft:noise"),
                                                        "settings": String("minecraft:end"),
                                                        "biome_source": Compound(
                                                            {
                                                                "type": String("minecraft:the_end"),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                )
            }
        )

        # Write level.dat (must be GZIP compressed)
        level_file = self.world_dir / "level.dat"
        nbt_file = File(level_data, filename=str(level_file), gzipped=True)
        nbt_file.save()
        print(f"Created {level_file}")

    @timing_decorator("heightmap_to_blocks")
    def heightmap_to_blocks(self, heightmap: np.ndarray, scale: float = 1.0, z_scale: float = 100.0) -> np.ndarray:
        """
        Convert heightmap to block heights, scaling appropriately.
        Minecraft has a height limit of 320 blocks (1.18+), so we need to scale.

        Args:
            heightmap: 2D numpy array of elevation values
            scale: Legacy parameter (unused)
            z_scale: Vertical scale percentage (0-100, default: 100 = no scaling)
        """
        # Identify NoData values (0.0) which should map to Minecraft Y=64 (sea level)
        # Note: 0.0 might also be a valid elevation, but we'll treat it as NoData for now
        # A more sophisticated approach would track which pixels were originally NoData
        nodata_mask = heightmap == 0.0

        # Exclude NoData values from min/max calculation for proper scaling
        # NoData values are set to 0.0 for GeoTIFF files
        valid_mask = ~nodata_mask
        if np.any(valid_mask):
            valid_heights = heightmap[valid_mask]
            min_height = np.min(valid_heights)
            max_height = np.max(valid_heights)
        else:
            # Fallback if all values are NoData
            min_height = np.min(heightmap)
            max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            # All values are the same - scale the single value to target range
            # If it's a high value, scale it to max; if low, scale to min
            # For simplicity, scale single value to middle-high range
            target_min = 64
            target_max = 172  # Scaled down by half: (280 - 64) / 2 + 64 = 172
            # Use the value itself as a hint - if it's > 100, scale to high; otherwise to low
            if min_height > 100:
                normalized = np.full_like(heightmap, target_max, dtype=np.int32)
            else:
                normalized = np.full_like(heightmap, target_min, dtype=np.int32)
        else:
            # Scale to fit between sea level (64) and reduced max height
            # Vertical height scaled down by half
            target_min = 64
            base_target_max = 172  # Scaled down by half: (280 - 64) / 2 + 64 = 172

            # Apply z_scale to the target range
            # If z_scale is 50%, the height range should be 50% of normal
            scale_factor = z_scale / 100.0
            height_range_blocks = (base_target_max - target_min) * scale_factor
            target_max = target_min + height_range_blocks

            normalized = (
                (heightmap - min_height) / height_range * (target_max - target_min) + target_min
            ).astype(np.int32)

            if z_scale != 100.0:
                print(f"Applied z-scale {z_scale}%: Minecraft height range is {target_min}-{target_max:.0f} blocks")

        # Set NoData values to Minecraft's sea level (Y=64)
        # This must be done after scaling to avoid affecting the scaling calculation
        normalized[nodata_mask] = 64

        return normalized

    @timing_decorator("encode_block_states")
    def encode_block_states(
        self, chunk_heights: np.ndarray, chunk_classifications: np.ndarray, section_y: int
    ) -> tuple[List, List, int]:
        """
        Encode block states for a chunk section.
        Returns (palette, data, block_count) where:
        - palette: list of block state compounds
        - data: list of longs encoding block indices
        - block_count: number of non-air blocks (Short)
        """
        y_start = section_y * 16

        # Block type mapping based on LAS classification codes:
        # 0 = air
        # 1 = grass_block (ground/unclassified)
        # 2 = stone (buildings/structures)
        # 3 = water
        # 4 = stone_bricks (high confidence buildings)
        palette = [
            Compound({"Name": String("minecraft:air")}),
            Compound({"Name": String("minecraft:grass_block")}),
            Compound({"Name": String("minecraft:stone")}),
            Compound({"Name": String("minecraft:water")}),
            Compound({"Name": String("minecraft:stone_bricks")}),
        ]

        # Vectorized block state encoding
        # Create 3D coordinate arrays using broadcasting
        # Shape: (16, 16, 16) for (local_y, z, x)
        local_y_arr = np.arange(16)[:, None, None]  # Shape: (16, 1, 1)
        
        # Broadcast world_y values: (16, 1, 1) -> (16, 16, 16)
        world_y_3d = (y_start + local_y_arr).astype(np.int32)
        
        # Broadcast heights and classifications: (16, 16) -> (1, 16, 16) -> (16, 16, 16)
        heights_3d = chunk_heights[None, :, :]  # Shape: (1, 16, 16)
        classifications_3d = chunk_classifications[None, :, :]  # Shape: (1, 16, 16)
        
        # Determine air vs solid blocks (vectorized)
        # Air where world_y > height
        is_air = world_y_3d > heights_3d
        
        # Determine surface blocks (where world_y == height)
        is_surface = world_y_3d == heights_3d
        
        # Vectorized classification to block type mapping
        # Initialize block types array
        blocks_3d = np.zeros((16, 16, 16), dtype=np.uint16)
        
        # Water (classification == 9)
        water_mask = (classifications_3d == 9) & ~is_air
        blocks_3d[water_mask] = 3
        
        # Buildings (classification == 6 or >= 20)
        building_mask = ((classifications_3d == 6) | (classifications_3d >= 20)) & ~is_air & ~water_mask
        # Stone bricks on surface, stone below
        blocks_3d[building_mask & is_surface] = 4
        blocks_3d[building_mask & ~is_surface] = 2
        
        # Ground (classification == 2)
        ground_mask = (classifications_3d == 2) & ~is_air & ~water_mask & ~building_mask
        blocks_3d[ground_mask] = 1
        
        # Unclassified/other (classification == 1 or other) -> grass_block
        other_mask = ~is_air & ~water_mask & ~building_mask & ~ground_mask
        blocks_3d[other_mask] = 1
        
        # Air blocks are already 0 from initialization, but set explicitly for clarity
        blocks_3d[is_air] = 0
        
        # Flatten to 1D array (4096 blocks)
        # Order: local_y * 256 + z * 16 + x
        blocks = blocks_3d.reshape(4096)

        # Count non-air blocks (blocks that are not 0)
        block_count = int(np.count_nonzero(blocks))

        # Encode into compact long array (bits per entry = ceil(log2(len(palette))))
        bits_per_entry = max(4, (len(palette) - 1).bit_length())
        entries_per_long = 64 // bits_per_entry

        data = []
        for i in range(0, 4096, entries_per_long):
            value = 0
            for j in range(entries_per_long):
                if i + j < 4096:
                    value |= int(blocks[i + j]) << (j * bits_per_entry)
            data.append(Long(value))

        return palette, data, block_count

    @timing_decorator("create_chunk_nbt")
    def create_chunk_nbt(
        self,
        chunk_x: int,
        chunk_z: int,
        block_heights: np.ndarray,
        classification_map: np.ndarray,
        world_offset_x: int = 0,
        world_offset_z: int = 0,
    ) -> bytes:
        # Store block_heights shape for debug output
        self._block_heights_shape = block_heights.shape
        """
        Create a Minecraft chunk NBT structure with block data.
        A chunk is 16x16 blocks in X and Z, and up to 384 blocks tall (1.18+).
        """
        chunk_size_x = 16
        chunk_size_z = 16

        # Calculate world block coordinates for this chunk
        # chunk_x and chunk_z are global chunk coordinates
        # Each chunk is 16x16 blocks
        world_chunk_start_x = chunk_x * chunk_size_x
        world_chunk_start_z = chunk_z * chunk_size_z

        # Get chunk height data and classification from arrays using vectorized operations
        # block_heights and classification_map are indexed as [z, x] (rows, cols)
        # Calculate world coordinates for the entire chunk using broadcasting
        world_z_coords = np.arange(world_chunk_start_z, world_chunk_start_z + chunk_size_z)[:, None]  # Shape: (chunk_size_z, 1)
        world_x_coords = np.arange(world_chunk_start_x, world_chunk_start_x + chunk_size_x)[None, :]  # Shape: (1, chunk_size_x)
        
        # Create boolean masks for valid coordinates (broadcast to (chunk_size_z, chunk_size_x))
        valid_z_mask = (world_z_coords >= 0) & (world_z_coords < block_heights.shape[0])  # Shape: (chunk_size_z, 1)
        valid_x_mask = (world_x_coords >= 0) & (world_x_coords < block_heights.shape[1])  # Shape: (1, chunk_size_x)
        valid_mask = valid_z_mask & valid_x_mask  # Shape: (chunk_size_z, chunk_size_x)
        
        # Initialize arrays
        chunk_heights = np.zeros((chunk_size_z, chunk_size_x), dtype=np.int32)
        chunk_classifications = np.zeros((chunk_size_z, chunk_size_x), dtype=np.uint8)
        
        # Extract valid data using vectorized indexing
        # Use np.where to get indices where valid_mask is True
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > 0:
            # Extract the actual world coordinates for valid positions
            valid_z = world_z_coords[valid_indices[0], 0]  # Extract z coordinates (flattened)
            valid_x = world_x_coords[0, valid_indices[1]]  # Extract x coordinates (flattened)
            # Assign values using the valid mask
            chunk_heights[valid_indices] = block_heights[valid_z, valid_x]
            chunk_classifications[valid_indices] = classification_map[valid_z, valid_x]
        
        # Fill out-of-bounds areas with default values
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            min_terrain_height = int(np.min(block_heights)) if block_heights.size > 0 else 64
            default_height = max(min_terrain_height, 64)
            invalid_indices = np.where(invalid_mask)
            chunk_heights[invalid_indices] = default_height
            chunk_classifications[invalid_indices] = 1  # Default to unclassified

        # Create sections (16-block tall slices)
        # Minecraft 1.18+: Y ranges from -64 to 320
        # Section Y values: section -4 = Y -64 to -49, section 0 = Y 0 to 15, section 4 = Y 64 to 79
        sections = []
        max_height = int(np.max(chunk_heights))

        # Calculate section range based on terrain height
        # Section Y = floor(world_y / 16)
        # We need to fill from the bottom (Y=64, section 4) up to the terrain surface
        # Section 4 = Y 64 to 79, section 19 = Y 304 to 319
        min_section = 4  # Always start from section 4 (Y=64) to ensure solid ground
        max_section = max_height // 16  # Up to highest terrain

        # Clamp to valid section range (-4 to 19)
        # Section -4 = Y -64 to -49, section 19 = Y 304 to 319
        min_section = max(-4, min_section)
        max_section = min(19, max(max_section, min_section))  # Ensure max >= min

        for section_y in range(min_section, max_section + 1):
            palette, block_data, block_count = self.encode_block_states(
                chunk_heights, chunk_classifications, section_y
            )

            # Lighting data: 16x16x16 = 4096 blocks, 2 nibbles per block = 2048 bytes
            # Packed as byte array: 2048 bytes total
            # Each byte contains 2 light levels (upper and lower nibble)
            # 0xFF = 15 in both nibbles = full light
            # Use numpy uint8 array for ByteArray (nbtlib expects numpy array)
            block_light_array = np.array(
                [0] * 2048, dtype=np.uint8
            )  # All blocks have no block light
            sky_light_array = np.array(
                [255] * 2048, dtype=np.uint8
            )  # All blocks have full sky light (0xFF = 15 in each nibble)

            section_data = Compound(
                {
                    "Y": Byte(section_y),
                    "block_states": Compound({"palette": List(palette), "data": List(block_data)}),
                    "biomes": Compound(
                        {
                            "palette": List([String("minecraft:plains")]),
                            "data": List([Long(0)] * 64),
                        }
                    ),
                    "BlockLight": ByteArray(block_light_array),
                    "SkyLight": ByteArray(sky_light_array),
                    "block_count": Short(block_count),  # Required: number of non-air blocks
                }
            )
            sections.append(section_data)

        # Create heightmap (used for lighting and generation)
        # Heightmap is stored as a compact long array, 9 bits per entry
        # For 16x16 = 256 entries, we need 36 longs (256 * 9 / 64 = 36)
        def encode_heightmap(heights):
            bits_per_entry = 9
            entries_per_long = 64 // bits_per_entry
            num_entries = chunk_size_x * chunk_size_z  # 256
            num_longs = (num_entries * bits_per_entry + 63) // 64  # 36

            result = []
            for i in range(num_longs):
                value = 0
                for j in range(entries_per_long):
                    entry_idx = i * entries_per_long + j
                    if entry_idx < num_entries:
                        z_idx = entry_idx // chunk_size_x
                        x_idx = entry_idx % chunk_size_x
                        height = int(heights[z_idx, x_idx])
                        value |= (height & ((1 << bits_per_entry) - 1)) << (j * bits_per_entry)
                result.append(Long(value))
            return result

        heightmap_motion = encode_heightmap(chunk_heights)
        heightmap_surface = encode_heightmap(chunk_heights)

        # Create the chunk NBT structure
        chunk_data = Compound(
            {
                "DataVersion": Int(3465),  # Minecraft 1.20.1
                "xPos": Int(chunk_x),
                "zPos": Int(chunk_z),
                "Status": String("full"),
                "LastUpdate": Long(0),
                "InhabitedTime": Long(0),
                "sections": List(sections),
                "Heightmaps": Compound(
                    {
                        "MOTION_BLOCKING": List(heightmap_motion),
                        "WORLD_SURFACE": List(heightmap_surface),
                    }
                ),
                "biomes": Compound(
                    {"palette": List([String("minecraft:plains")]), "data": List([Long(0)] * 1024)}
                ),
            }
        )

        # Serialize to NBT and compress
        # Try to use in-memory serialization if supported, otherwise fall back to tempfile
        try:
            # Attempt in-memory serialization using BytesIO
            nbt_buffer = BytesIO()
            nbt_file = File(chunk_data)
            # Some versions of nbtlib support fileobj parameter
            try:
                nbt_file.save(gzipped=False, fileobj=nbt_buffer)
                nbt_bytes = nbt_buffer.getvalue()
            except (TypeError, AttributeError):
                # Fall back to using a temporary file (but in memory if possible)
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    nbt_file = File(chunk_data, filename=tmp_file.name)
                    nbt_file.save(gzipped=False)
                    tmp_file.seek(0)
                    nbt_bytes = tmp_file.read()
        except Exception:
            # Final fallback: use temporary file
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                nbt_file = File(chunk_data, filename=tmp_file.name)
                nbt_file.save(gzipped=False)
                tmp_file.seek(0)
                nbt_bytes = tmp_file.read()

        compressed = zlib.compress(nbt_bytes, level=6)

        return compressed

    @timing_decorator("create_region_file")
    def create_region_file(
        self,
        region_x: int,
        region_z: int,
        block_heights: np.ndarray,
        classification_map: np.ndarray,
        world_offset_x: int,
        world_offset_z: int,
    ):
        """
        Create a Minecraft region file (.mca format).
        A region file contains 32x32 chunks (512x512 blocks).
        """
        region_file = self.region_dir / f"r.{region_x}.{region_z}.mca"

        # Region header: 8KB (1024 entries of 4 bytes each)
        # Format: [offset: 3 bytes][sector_count: 1 byte] for each chunk
        header = bytearray(8192)

        # Timestamp table: 4KB (1024 entries of 4 bytes each)
        timestamps = bytearray(4096)

        chunks_data = []
        chunk_offsets = []

        # Generate chunks for this region (32x32 chunks)
        # Region file layout: header (8KB = 2 sectors) + timestamps (4KB = 1 sector) + chunk data
        # So chunks start at sector 3 (byte 12288)
        sector_offset = 3  # Start after header (2 sectors) + timestamps (1 sector)

        for local_chunk_z in range(32):
            for local_chunk_x in range(32):
                # Calculate global chunk coordinates
                # Each region contains 32x32 chunks
                chunk_x = region_x * 32 + local_chunk_x
                chunk_z = region_z * 32 + local_chunk_z

                # Create chunk NBT
                # Note: world_offset_x and world_offset_z are not needed anymore
                # as we calculate world coordinates directly from chunk coordinates
                compressed_data = self.create_chunk_nbt(
                    chunk_x, chunk_z, block_heights, classification_map, 0, 0
                )

                # Region file format requires:
                # - 4 bytes: chunk length (big-endian, includes compression byte)
                # - 1 byte: compression type (2 = ZLIB)
                # - N bytes: compressed chunk data
                chunk_length = len(compressed_data) + 1  # +1 for compression type byte
                chunk_bytes = (
                    struct.pack(">I", chunk_length) + b"\x02" + compressed_data
                )  # 2 = ZLIB

                # Calculate size in sectors (4096 bytes per sector)
                chunk_size = len(chunk_bytes)
                sectors_needed = (chunk_size + 4095) // 4096

                # Pad chunk to sector boundary
                chunk_bytes += b"\x00" * (sectors_needed * 4096 - chunk_size)

                # Store chunk
                chunks_data.append(chunk_bytes)
                chunk_offsets.append((sector_offset, sectors_needed))

                # Update header entry
                chunk_index = local_chunk_z * 32 + local_chunk_x
                offset_bytes = struct.pack(">I", (sector_offset << 8) | sectors_needed)
                header[chunk_index * 4 : (chunk_index + 1) * 4] = offset_bytes

                # Update timestamp (current time in seconds since epoch)
                import time

                timestamp = int(time.time())
                timestamp_bytes = struct.pack(">I", timestamp)
                timestamps[chunk_index * 4 : (chunk_index + 1) * 4] = timestamp_bytes

                sector_offset += sectors_needed

        # Write region file
        with open(region_file, "wb") as f:
            f.write(header)
            f.write(timestamps)
            for chunk_data in chunks_data:
                f.write(chunk_data)

        print(f"Created region file: {region_file} ({len(chunks_data)} chunks)")

    def generate_world(
        self, heightmap: np.ndarray, classification_map: np.ndarray, block_size: int = 1, z_scale: float = 100.0, num_workers: int = 1
    ):
        """
        Generate the complete Minecraft world from heightmap and classification map.

        Args:
            heightmap: 2D numpy array of elevation values
            classification_map: 2D numpy array of classification codes
            block_size: How many real-world units per Minecraft block
            z_scale: Vertical scale percentage (0-100, default: 100 = no scaling)
            num_workers: Number of worker processes for parallel region creation
        """
        print(f"Generating Minecraft world: {self.world_name}")
        print(f"Heightmap shape: {heightmap.shape}")

        # Convert heightmap to block heights
        block_heights = self.heightmap_to_blocks(heightmap, z_scale=z_scale)

        # Create level.dat
        self.create_level_dat(block_heights)

        # Create region files
        # Minecraft regions are 32x32 chunks = 512x512 blocks
        region_size = 512

        height, width = block_heights.shape

        # Calculate number of regions needed
        regions_x = (width + region_size - 1) // region_size
        regions_z = (height + region_size - 1) // region_size

        total_regions = regions_x * regions_z
        print(f"Creating {regions_x}x{regions_z} region files ({total_regions} total)...")

        if num_workers > 1 and total_regions > 1:
            # Parallel processing
            print(f"Using {num_workers} worker processes for parallel region creation...")
            region_tasks = [
                (rx, rz, str(self.region_dir), block_heights, classification_map)
                for rx in range(regions_x)
                for rz in range(regions_z)
            ]

            regions_created = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_region = {
                    executor.submit(_create_region_file_worker, task): (task[0], task[1])
                    for task in region_tasks
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_region):
                    region_file, chunk_count = future.result()
                    regions_created += 1
                    if regions_created % 4 == 0 or regions_created == total_regions:
                        print(f"Progress: {regions_created}/{total_regions} region files created...")
        else:
            # Sequential processing (original code)
            regions_created = 0
            for rx in range(regions_x):
                for rz in range(regions_z):
                    self.create_region_file(
                        rx, rz, block_heights, classification_map, rx * region_size, rz * region_size
                    )
                    regions_created += 1
                    if regions_created % 4 == 0 or regions_created == total_regions:
                        print(f"Progress: {regions_created}/{total_regions} region files created...")

        print(f"\nGenerated {total_regions} region files ({regions_x}x{regions_z})")
        print(f"World generated at: {self.world_dir}")
        print("To use this world:")
        print(f"1. Copy '{self.world_dir}' to your Minecraft saves directory")
        print("2. Minecraft saves directory is typically at:")
        print("   - macOS: ~/Library/Application\\ Support/minecraft/saves/")
        print("   - Windows: %appdata%/.minecraft/saves/")
        print("   - Linux: ~/.minecraft/saves/")


def _create_region_file_worker(args):
    """
    Worker function for parallel region file creation.
    This function must be at module level to be picklable for multiprocessing.
    """
    (
        region_x,
        region_z,
        region_dir,
        block_heights,
        classification_map,
    ) = args
    
    # Import here to avoid issues with multiprocessing
    # Create a minimal generator instance to use its methods
    # We only need the methods, not the full initialization
    temp_gen = MinecraftWorldGenerator("temp", output_dir=str(Path(region_dir).parent))
    
    region_file = Path(region_dir) / f"r.{region_x}.{region_z}.mca"
    
    # Region header: 8KB (1024 entries of 4 bytes each)
    header = bytearray(8192)
    timestamps = bytearray(4096)
    chunks_data = []
    chunk_offsets = []
    sector_offset = 3
    
    for local_chunk_z in range(32):
        for local_chunk_x in range(32):
            chunk_x = region_x * 32 + local_chunk_x
            chunk_z = region_z * 32 + local_chunk_z
            
            compressed_data = temp_gen.create_chunk_nbt(
                chunk_x, chunk_z, block_heights, classification_map, 0, 0
            )
            
            chunk_length = len(compressed_data) + 1
            chunk_bytes = struct.pack(">I", chunk_length) + b"\x02" + compressed_data
            chunk_size = len(chunk_bytes)
            sectors_needed = (chunk_size + 4095) // 4096
            chunk_bytes += b"\x00" * (sectors_needed * 4096 - chunk_size)
            
            chunks_data.append(chunk_bytes)
            chunk_offsets.append((sector_offset, sectors_needed))
            
            chunk_index = local_chunk_z * 32 + local_chunk_x
            offset_bytes = struct.pack(">I", (sector_offset << 8) | sectors_needed)
            header[chunk_index * 4 : (chunk_index + 1) * 4] = offset_bytes
            
            timestamp = int(time.time())
            timestamp_bytes = struct.pack(">I", timestamp)
            timestamps[chunk_index * 4 : (chunk_index + 1) * 4] = timestamp_bytes
            
            sector_offset += sectors_needed
    
    # Write region file
    region_file.parent.mkdir(parents=True, exist_ok=True)
    with open(region_file, "wb") as f:
        f.write(header)
        f.write(timestamps)
        for chunk_data in chunks_data:
            f.write(chunk_data)
    
    return str(region_file), len(chunks_data)


def auto_resolution(
    x: np.ndarray, y: np.ndarray, point_count: int, min_res: int = 512, max_res: int = 4096
) -> int:
    """
    Auto-detect optimal resolution based on point density and spatial extent.

    Args:
        x, y: Point cloud coordinates
        point_count: Total number of points
        min_res: Minimum resolution (default: 512)
        max_res: Maximum resolution (default: 4096)

    Returns:
        Recommended resolution that balances detail vs performance
    """
    if point_count == 0:
        return min_res

    # Calculate spatial extent
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Calculate area covered by point cloud
    area = (x_max - x_min) * (y_max - y_min)

    if area <= 0:
        return min_res

    # Target: ~1-2 points per grid cell for good detail
    # Resolution = sqrt(points / target_points_per_cell)
    # Using 1.5 points per cell as target
    target_points_per_cell = 1.5
    recommended_res = int(np.sqrt(point_count / target_points_per_cell))

    # Clamp to reasonable bounds
    resolution = max(min_res, min(max_res, recommended_res))

    # Round to nearest power of 2 for better performance (optional but helpful)
    # Find nearest power of 2
    power_of_2 = 2 ** int(np.log2(resolution))
    next_power = power_of_2 * 2

    # Choose closest power of 2
    if abs(resolution - power_of_2) < abs(resolution - next_power):
        resolution = power_of_2
    else:
        resolution = next_power

    # Ensure it's within bounds
    resolution = max(min_res, min(max_res, resolution))

    return resolution


def read_laz_file(laz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read LAZ file and extract point cloud data.
    Results are cached based on file path and modification time.

    Returns:
        Tuple of (x, y, z, classification) coordinate arrays
    """
    laz_file = Path(laz_path).resolve()

    # Create cache key based on file path and modification time
    try:
        mtime = os.path.getmtime(laz_file)
    except OSError:
        mtime = 0

    cache_key = f"laz_file:{laz_file}:{mtime}"

    # Check cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Using cached LAZ data for: {laz_file}")
        return cached_result

    print(f"Reading LAZ file: {laz_file}")

    las = laspy.read(str(laz_file))

    x = las.x
    y = las.y
    z = las.z

    # Get classification data if available
    if hasattr(las, "classification"):
        classification = las.classification
    else:
        # Fallback: try to get from points
        classification = np.zeros(len(x), dtype=np.uint8)

    print(f"Loaded {len(x)} points")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y range: {y.min():.2f} to {y.max():.2f}")
    print(f"Z range: {z.min():.2f} to {z.max():.2f}")

    # Show classification distribution
    unique_classes = np.unique(classification)
    print(f"Classifications found: {unique_classes}")
    for cls in unique_classes:
        count = np.sum(classification == cls)
        print(f"  Class {cls}: {count:,} points")

    # Convert to regular numpy arrays (laspy returns ScaledArrayView which can't be hashed)
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    z_array = np.asarray(z)
    classification_array = np.asarray(classification)

    result = (x_array, y_array, z_array, classification_array)

    # Cache the result
    cache.set(cache_key, result)

    return result


@timing_decorator("read_geotiff_file")
def read_geotiff_file(
    geotiff_path: str, target_resolution: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read GeoTIFF file and extract elevation data as a heightmap.
    Results are cached based on file path, modification time, and target resolution.

    Args:
        geotiff_path: Path to the GeoTIFF file
        target_resolution: Optional target resolution to resample to (if None, uses native resolution)

    Returns:
        Tuple of (heightmap, classification_map) as 2D numpy arrays
        - heightmap: elevation values in meters
        - classification_map: classification codes (defaults to unclassified/ground)
    """
    geotiff_file = Path(geotiff_path).resolve()

    # Create cache key based on file path, modification time, and target resolution
    try:
        mtime = os.path.getmtime(geotiff_file)
    except OSError:
        mtime = 0

    cache_key = f"geotiff_file:{geotiff_file}:{mtime}:{target_resolution}"

    # Check cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Using cached GeoTIFF data for: {geotiff_file}")
        return cached_result

    print(f"Reading GeoTIFF file: {geotiff_file}")

    with rasterio.open(str(geotiff_file)) as dataset:
        original_height, original_width = dataset.height, dataset.width
        print(f"Original GeoTIFF size: {original_width}x{original_height} pixels")
        print(f"CRS: {dataset.crs}")
        print(f"Bounds: {dataset.bounds}")

        # Determine target dimensions
        spatial_scale_factor = 1.0  # Track horizontal scale factor for 3D consistency
        if target_resolution is not None:
            # Calculate scale factor to maintain aspect ratio
            spatial_scale_factor = target_resolution / max(original_width, original_height)
            target_width = int(original_width * spatial_scale_factor)
            target_height = int(original_height * spatial_scale_factor)
            print(f"Resampling to {target_width}x{target_height} pixels (target resolution: {target_resolution})")
            print(f"Spatial scale factor: {spatial_scale_factor:.4f} (elevations will be scaled by same factor for 3D consistency)")
        else:
            target_width = original_width
            target_height = original_height
            print(f"Using native resolution: {target_width}x{target_height} pixels")

        # Resample if needed
        if target_resolution is not None and (target_width != original_width or target_height != original_height):
            # Calculate new transform for resampling
            from rasterio import warp
            from rasterio.enums import Resampling
            from rasterio.transform import from_bounds

            # Get bounds
            left, bottom, right, top = dataset.bounds

            # Create new transform for target resolution
            new_transform = from_bounds(left, bottom, right, top, target_width, target_height)

            # Resample elevation data (first band)
            # Initialize with NaN so areas outside bounds remain NaN
            heightmap_dest = np.full((target_height, target_width), np.nan, dtype=np.float32)
            warp.reproject(
                source=dataset.read(1),
                destination=heightmap_dest,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=new_transform,
                dst_crs=dataset.crs,
                resampling=Resampling.bilinear,  # Use bilinear for smooth resampling
                src_nodata=dataset.nodata,  # Tell rasterio what NoData value to expect
                dst_nodata=np.nan,  # Use NaN as NoData in destination
            )
            heightmap = heightmap_dest.astype(np.float32)

            # Scale elevation values by the same factor as spatial resampling to maintain 3D scale
            # If we resample from 2000x2000 to 1000x1000 (0.5x), elevations should also be 0.5x
            if spatial_scale_factor != 1.0:
                valid_mask = ~np.isnan(heightmap)
                if valid_mask.any():
                    heightmap[valid_mask] = heightmap[valid_mask] * spatial_scale_factor
                    print(f"Scaled elevation values by {spatial_scale_factor:.4f} to maintain 3D scale consistency")

            # Any remaining NaN values are either original NoData or outside bounds
            # These will be set to 0 later

            # Resample classification data if available (second band)
            if dataset.count >= 2:
                try:
                    # Initialize with 1 (unclassified) as default
                    classification_dest = np.ones((target_height, target_width), dtype=np.uint8)
                    # Read second band and check if it has NoData
                    classification_band = dataset.read(2)
                    classification_nodata = None
                    if hasattr(dataset, "nodata") and dataset.nodata is not None:
                        # Try to use the same NoData value, or check if band 2 has its own
                        classification_nodata = dataset.nodata

                    warp.reproject(
                        source=classification_band,
                        destination=classification_dest,
                        src_transform=dataset.transform,
                        src_crs=dataset.crs,
                        dst_transform=new_transform,
                        dst_crs=dataset.crs,
                        resampling=Resampling.nearest,  # Use nearest neighbor for classification
                        src_nodata=classification_nodata,  # Handle NoData in classification band
                    )
                    classification_map = np.clip(classification_dest, 0, 255).astype(np.uint8)
                    # Set classification to 1 (unclassified) for areas where heightmap is NaN
                    classification_map[np.isnan(heightmap)] = 1
                    print("Using second band for classification (resampled)")
                except Exception as e:
                    print(f"Warning: Could not read/resample classification from second band: {e}")
                    classification_map = np.ones((target_height, target_width), dtype=np.uint8)
                    # Ensure NaN areas (outside bounds) have proper classification
                    classification_map[np.isnan(heightmap)] = 1
            else:
                classification_map = np.ones((target_height, target_width), dtype=np.uint8)  # Default to unclassified (1)
                # Set classification to 1 for areas where heightmap is NaN (outside bounds)
                classification_map[np.isnan(heightmap)] = 1
        else:
            # Read at native resolution
            heightmap = dataset.read(1).astype(np.float32)

            # Handle NoData values
            if dataset.nodata is not None:
                heightmap[heightmap == dataset.nodata] = np.nan

            # Get dimensions
            height, width = heightmap.shape

            # Create classification map (default to ground/unclassified)
            classification_map = np.ones((height, width), dtype=np.uint8)  # Default to unclassified (1)

            # If there's a second band, try to use it for classification
            if dataset.count >= 2:
                try:
                    classification_band = dataset.read(2).astype(np.uint8)
                    classification_map = np.clip(classification_band, 0, 255).astype(np.uint8)
                    print("Using second band for classification")
                except Exception as e:
                    print(f"Warning: Could not read classification from second band: {e}")

        print(f"Elevation range: {np.nanmin(heightmap):.2f} to {np.nanmax(heightmap):.2f}")

    # Note: z-scale is not applied here - it will be applied after normalization
    # in heightmap_to_blocks to preserve the effect on final Minecraft block heights

    # Fill NaN values (NoData) with 0 for GeoTIFF files
    mask = np.isnan(heightmap)
    if mask.any():
        heightmap[mask] = 0.0
        classification_map[mask] = 1  # Default to unclassified for NoData areas
        print(f"Set {np.sum(mask):,} NoData pixels to 0.0")

    result = (heightmap, classification_map)

    # Cache the result
    cache.set(cache_key, result)

    return result


def create_heightmap(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    resolution: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert point cloud to a regular grid heightmap and classification map.
    Results are cached based on input data hash and resolution.

    Args:
        x, y, z: Point cloud coordinates
        classification: Point classification codes
        resolution: Grid resolution (pixels per dimension)

    Returns:
        Tuple of (heightmap, classification_map) as 2D numpy arrays
    """
    # Ensure we have regular numpy arrays (in case they're views from laspy)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    classification = np.asarray(classification)

    # Create cache key based on input data hash and resolution
    # Use a sample of points for hashing to avoid huge hash calculations
    sample_size = min(10000, len(x))
    if len(x) > 0:
        sample_indices = np.linspace(0, len(x) - 1, sample_size, dtype=int)
        x_sample = x[sample_indices].tobytes()
        y_sample = y[sample_indices].tobytes()
        z_sample = z[sample_indices].tobytes()
    else:
        x_sample = b""
        y_sample = b""
        z_sample = b""

    data_hash = hashlib.sha256(
        x_sample + y_sample + z_sample + np.array([len(x), resolution], dtype=np.int64).tobytes()
    ).hexdigest()

    cache_key = f"heightmap:{data_hash}:{resolution}"

    # Check cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Using cached heightmap (resolution {resolution}x{resolution})")
        return cached_result

    print(f"Creating heightmap with resolution {resolution}x{resolution}...")

    # Normalize coordinates to grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Create grid
    x_norm = ((x - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    y_norm = ((y - y_min) / (y_max - y_min) * (resolution - 1)).astype(int)

    # Clip to valid range
    x_norm = np.clip(x_norm, 0, resolution - 1)
    y_norm = np.clip(y_norm, 0, resolution - 1)

    # Create heightmap and classification map by taking maximum Z value in each grid cell
    # Mapping: LiDAR (x, y, z) -> Minecraft (X, Z, Y)
    # heightmap is indexed as [Minecraft Z, Minecraft X] = [LiDAR y, LiDAR x]
    # First dimension (rows) = Minecraft Z (from LiDAR y)
    # Second dimension (cols) = Minecraft X (from LiDAR x)
    heightmap = np.full((resolution, resolution), np.nan, dtype=np.float32)
    classification_map = np.zeros((resolution, resolution), dtype=np.uint8)

    # Use a simple approach: bin the points and take max height per cell
    # For classification, use the classification of the highest point in each cell
    for i in range(len(x)):
        grid_x = x_norm[i]  # LiDAR x -> Minecraft X (column index)
        grid_y = y_norm[i]  # LiDAR y -> Minecraft Z (row index)
        if np.isnan(heightmap[grid_y, grid_x]) or z[i] > heightmap[grid_y, grid_x]:
            heightmap[grid_y, grid_x] = z[i]  # LiDAR z -> stored as height value
            classification_map[grid_y, grid_x] = classification[i]

    # Fill any remaining NaN values
    mask = np.isnan(heightmap)
    if mask.any():
        # Use mean of non-NaN values
        mean_val = np.nanmean(heightmap)
        heightmap[mask] = mean_val if not np.isnan(mean_val) else 64.0
        # For classification, default to unclassified (1) for empty cells
        classification_map[mask] = 1

    # Optional: smooth the heightmap
    try:
        from scipy import ndimage

        heightmap = ndimage.gaussian_filter(heightmap, sigma=1)
    except ImportError:
        pass  # scipy not required, but helps with smoothing

    print(f"Heightmap created: {heightmap.shape}")
    print(f"Elevation range: {np.nanmin(heightmap):.2f} to {np.nanmax(heightmap):.2f}")

    # Cache the result
    result = (heightmap, classification_map)
    cache.set(cache_key, result)

    return result


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert LAZ LiDAR files or GeoTIFF elevation data to Minecraft Java Edition worlds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run lidar_to_minecraft.py dars/USGS_LPC_CA_SanFrancisco_B23_05300270.laz
  uv run lidar_to_minecraft.py dars/data.laz san_francisco 1024
  uv run lidar_to_minecraft.py elevation.tif my_world
  uv run lidar_to_minecraft.py elevation.tif my_world 1024 --z-scale 50
        """,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (LAZ or GeoTIFF)",
    )
    parser.add_argument(
        "world_name",
        type=str,
        nargs="?",
        default="lidar_world",
        help="Name for the generated Minecraft world (default: lidar_world)",
    )
    parser.add_argument(
        "resolution",
        type=int,
        nargs="?",
        default=None,
        help="Heightmap resolution in pixels (for LAZ: grid resolution; for GeoTIFF: max dimension to resample to)",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=100.0,
        help="Vertical scale percentage for GeoTIFF files (0-100, default: 100 = no scaling)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling and save profile data to {world_name}.prof",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel region processing (default: CPU count)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Detect file type based on extension
    input_path = Path(args.input_file)
    file_ext = input_path.suffix.lower()

    # Supported GeoTIFF extensions
    geotiff_extensions = {".tif", ".tiff", ".gtif", ".geotiff"}
    # Supported LAZ extensions
    laz_extensions = {".laz", ".las"}

    if file_ext in geotiff_extensions:
        # Read GeoTIFF file (already a heightmap)
        print("Detected GeoTIFF file")

        # Validate resolution if provided
        if args.resolution is not None:
            if args.resolution <= 0:
                print(f"Error: Resolution must be positive, got: {args.resolution}", file=sys.stderr)
                sys.exit(1)
            print(f"Resampling GeoTIFF to target resolution: {args.resolution}x{args.resolution} (max dimension)")

        # Validate z_scale
        if args.z_scale < 0 or args.z_scale > 100:
            print(
                f"Error: z-scale must be between 0 and 100, got: {args.z_scale}",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.z_scale != 100.0:
            print(f"Using z-scale: {args.z_scale}%")

        heightmap, classification_map = read_geotiff_file(
            args.input_file, target_resolution=args.resolution
        )

    elif file_ext in laz_extensions:
        # Read LAZ file (point cloud)
        print("Detected LAZ/LAS file")
        x, y, z, classification = read_laz_file(args.input_file)

        # Auto-detect resolution if not specified
        if args.resolution is None:
            resolution = auto_resolution(x, y, len(x))
            print(f"Auto-detected resolution: {resolution}x{resolution} (from {len(x):,} points)")
        else:
            resolution = args.resolution
            if resolution <= 0:
                print(f"Error: Resolution must be positive, got: {resolution}", file=sys.stderr)
                sys.exit(1)
            print(f"Using specified resolution: {resolution}x{resolution}")

        # Create heightmap and classification map from point cloud
        heightmap, classification_map = create_heightmap(x, y, z, classification, resolution=resolution)

    else:
        print(
            f"Error: Unsupported file type: {file_ext}",
            file=sys.stderr,
        )
        print(
            f"Supported formats: LAZ/LAS ({', '.join(laz_extensions)}) or GeoTIFF ({', '.join(geotiff_extensions)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Setup profiling if requested
    global _profiler
    if args.profile:
        _profiler = cProfile.Profile()
        _profiler.enable()
        print(f"Profiling enabled - profile will be saved to {args.world_name}.prof")

    # Generate Minecraft world
    generator = MinecraftWorldGenerator(args.world_name)
    # Pass z_scale for GeoTIFF files, use 100% (no scaling) for LAZ files
    z_scale = args.z_scale if file_ext in geotiff_extensions else 100.0
    num_workers = args.workers if args.workers is not None else multiprocessing.cpu_count()
    generator.generate_world(heightmap, classification_map, z_scale=z_scale, num_workers=num_workers)

    # Save profile and print timing summary
    if args.profile and _profiler:
        _profiler.disable()
        profile_file = f"{args.world_name}.prof"
        _profiler.dump_stats(profile_file)
        print(f"\nProfile saved to {profile_file}")
        print(f"To view: python -m pstats {profile_file} | sort cumulative | head -20")
        
        # Print top functions by cumulative time
        stats = pstats.Stats(_profiler)
        stats.sort_stats("cumulative")
        print("\nTop 20 functions by cumulative time:")
        stats.print_stats(20)

    # Print timing summary
    if _timings:
        print("\n" + "=" * 60)
        print("Timing Summary")
        print("=" * 60)
        total_time = sum(sum(times) for times in _timings.values())
        for name, times in sorted(_timings.items(), key=lambda x: sum(x[1]), reverse=True):
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(f"{name:50s} {total:8.2f}s ({count:4d} calls, avg {avg:.4f}s, {pct:5.1f}%)")
        print(f"{'Total':50s} {total_time:8.2f}s")
        print("=" * 60)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
