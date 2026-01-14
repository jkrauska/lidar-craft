#!/usr/bin/env python3
"""
Convert LAZ LiDAR files to Minecraft Java Edition worlds.
"""

import argparse
import hashlib
import os
import struct
import sys
import tempfile
import zlib
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

        # Debug: print terrain height statistics
        min_terrain = int(np.min(block_heights))
        max_terrain = int(np.max(block_heights))
        print(f"Terrain height range: Y={min_terrain} to {max_terrain}")
        print(f"Spawn point set to: X={spawn_x}, Y={spawn_y}, Z={spawn_z}")
        print(f"Terrain height at spawn: {spawn_terrain_height}")
        print(f"Terrain bounds: X=0-{width - 1}, Z=0-{height - 1}")

        # Verify spawn Y is reasonable
        if spawn_y < -64 or spawn_y > 320:
            print(f"WARNING: Spawn Y={spawn_y} is outside valid Minecraft range (-64 to 320)!")

        # Create level.dat structure
        level_data = Compound(
            {
                "Data": Compound(
                    {
                        "LevelName": String(self.world_name),
                        "generatorName": String(
                            "flat"
                        ),  # Keep as flat but configure to not interfere
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
                                                                ),  # Empty layers - no terrain generation
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

    def heightmap_to_blocks(self, heightmap: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Convert heightmap to block heights, scaling appropriately.
        Minecraft has a height limit of 320 blocks (1.18+), so we need to scale.
        """
        # Normalize heightmap to fit within Minecraft's height range
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            # All values are the same - scale the single value to target range
            # If it's a high value, scale it to max; if low, scale to min
            # For simplicity, scale single value to middle-high range
            target_min = 64
            target_max = 280
            # Use the value itself as a hint - if it's > 100, scale to high; otherwise to low
            if min_height > 100:
                normalized = np.full_like(heightmap, target_max, dtype=np.int32)
            else:
                normalized = np.full_like(heightmap, target_min, dtype=np.int32)
        else:
            # Scale to fit between sea level (64) and max build height (320)
            # Reserve some space at the top
            target_min = 64
            target_max = 280
            normalized = (
                (heightmap - min_height) / height_range * (target_max - target_min) + target_min
            ).astype(np.int32)

        return normalized

    def encode_block_states(
        self, chunk_heights: np.ndarray, section_y: int
    ) -> tuple[List, List, int]:
        """
        Encode block states for a chunk section.
        Returns (palette, data, block_count) where:
        - palette: list of block state compounds
        - data: list of longs encoding block indices
        - block_count: number of non-air blocks (Short)
        """
        chunk_size = 16
        y_start = section_y * 16

        # Simple palette: air (0), stone (1)
        palette = [
            Compound({"Name": String("minecraft:air")}),
            Compound({"Name": String("minecraft:stone")}),
        ]

        # Create block array (16x16x16 = 4096 blocks)
        blocks = np.zeros(4096, dtype=np.uint16)

        for local_y in range(16):
            world_y = y_start + local_y
            for z in range(chunk_size):
                for x in range(chunk_size):
                    height = chunk_heights[z, x]

                    block_index = local_y * 256 + z * 16 + x

                    if world_y > height:
                        blocks[block_index] = 0  # Air
                    else:
                        blocks[block_index] = 1  # Stone (both surface and below)

        # Count non-air blocks (blocks that are not 0)
        block_count = int(np.count_nonzero(blocks))

        # Debug: verify stone blocks are being set
        stone_count = int(np.sum(blocks == 1))
        if stone_count > 0 and section_y == (y_start // 16):
            print(
                f"    Section {section_y}: {stone_count} stone blocks, {block_count} total non-air"
            )

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

    def create_chunk_nbt(
        self,
        chunk_x: int,
        chunk_z: int,
        block_heights: np.ndarray,
        world_offset_x: int = 0,
        world_offset_z: int = 0,
    ) -> bytes:
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

        # Get chunk height data from block_heights array
        # block_heights is indexed as [z, x] (rows, cols)
        chunk_heights = np.zeros((chunk_size_z, chunk_size_x), dtype=np.int32)

        for local_z in range(chunk_size_z):
            for local_x in range(chunk_size_x):
                # World block coordinates
                world_x = world_chunk_start_x + local_x
                world_z = world_chunk_start_z + local_z

                # Map to heightmap array (indexed as [z, x])
                if 0 <= world_x < block_heights.shape[1] and 0 <= world_z < block_heights.shape[0]:
                    chunk_heights[local_z, local_x] = block_heights[world_z, world_x]
                else:
                    # For out-of-bounds, use minimum terrain height or 64, whichever is higher
                    # This ensures we don't have gaps
                    min_terrain_height = (
                        int(np.min(block_heights)) if block_heights.size > 0 else 64
                    )
                    chunk_heights[local_z, local_x] = max(min_terrain_height, 64)

        # Create sections (16-block tall slices)
        # Minecraft 1.18+: Y ranges from -64 to 320
        # Section Y values: section -4 = Y -64 to -49, section 0 = Y 0 to 15, section 4 = Y 64 to 79
        sections = []
        max_height = int(np.max(chunk_heights))
        min_height = int(np.min(chunk_heights))

        # Calculate section range based on terrain height
        # Section Y = floor(world_y / 16)
        min_section = min_height // 16  # Start from lowest terrain
        max_section = max_height // 16  # Up to highest terrain

        # Ensure we create at least section 4 (Y=64) if terrain is low
        # But don't go below section -4 (Y=-64) or above section 19 (Y=304)
        min_section = max(
            -4, min(min_section, 4)
        )  # At least section 4 if terrain is low, but allow down to -4
        max_section = min(19, max(max_section, 4))  # At least section 4, max section 19 (Y=304)

        print(
            f"  Chunk ({chunk_x}, {chunk_z}): terrain Y={min_height} to {max_height}, sections {min_section} to {max_section}"
        )

        for section_y in range(min_section, max_section + 1):
            palette, block_data, block_count = self.encode_block_states(chunk_heights, section_y)

            # Debug: verify blocks are being generated
            if block_count > 0:
                y_start = section_y * 16
                y_end = y_start + 15
                print(f"  Section {section_y} (Y={y_start} to {y_end}): {block_count} stone blocks")

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
        # Use a temporary file to serialize NBT data
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            nbt_file = File(chunk_data, filename=tmp_file.name)
            nbt_file.save(gzipped=False)
            tmp_file.seek(0)
            nbt_bytes = tmp_file.read()

        compressed = zlib.compress(nbt_bytes, level=6)

        return compressed

    def create_region_file(
        self,
        region_x: int,
        region_z: int,
        block_heights: np.ndarray,
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
        sector_offset = 2  # Start after header (2 sectors = 8192 bytes)

        for local_chunk_z in range(32):
            for local_chunk_x in range(32):
                # Calculate global chunk coordinates
                # Each region contains 32x32 chunks
                chunk_x = region_x * 32 + local_chunk_x
                chunk_z = region_z * 32 + local_chunk_z

                # Create chunk NBT
                # Note: world_offset_x and world_offset_z are not needed anymore
                # as we calculate world coordinates directly from chunk coordinates
                chunk_bytes = self.create_chunk_nbt(chunk_x, chunk_z, block_heights, 0, 0)

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

    def generate_world(self, heightmap: np.ndarray, block_size: int = 1):
        """
        Generate the complete Minecraft world from heightmap.

        Args:
            heightmap: 2D numpy array of elevation values
            block_size: How many real-world units per Minecraft block
        """
        print(f"Generating Minecraft world: {self.world_name}")
        print(f"Heightmap shape: {heightmap.shape}")

        # Convert heightmap to block heights
        block_heights = self.heightmap_to_blocks(heightmap)

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

        # For each region, create the region file
        regions_created = 0
        for rx in range(regions_x):
            for rz in range(regions_z):
                self.create_region_file(rx, rz, block_heights, rx * region_size, rz * region_size)
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


def read_laz_file(laz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read LAZ file and extract point cloud data.
    Results are cached based on file path and modification time.

    Returns:
        Tuple of (x, y, z) coordinate arrays
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

    print(f"Loaded {len(x)} points")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y range: {y.min():.2f} to {y.max():.2f}")
    print(f"Z range: {z.min():.2f} to {z.max():.2f}")

    # Convert to regular numpy arrays (laspy returns ScaledArrayView which can't be hashed)
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    z_array = np.asarray(z)

    result = (x_array, y_array, z_array)

    # Cache the result
    cache.set(cache_key, result)

    return result


def create_heightmap(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, resolution: int = 512
) -> np.ndarray:
    """
    Convert point cloud to a regular grid heightmap.
    Results are cached based on input data hash and resolution.

    Args:
        x, y, z: Point cloud coordinates
        resolution: Grid resolution (pixels per dimension)

    Returns:
        2D numpy array representing elevation
    """
    # Ensure we have regular numpy arrays (in case they're views from laspy)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

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

    # Create heightmap by taking maximum Z value in each grid cell
    # Mapping: LiDAR (x, y, z) -> Minecraft (X, Z, Y)
    # heightmap is indexed as [Minecraft Z, Minecraft X] = [LiDAR y, LiDAR x]
    # First dimension (rows) = Minecraft Z (from LiDAR y)
    # Second dimension (cols) = Minecraft X (from LiDAR x)
    heightmap = np.full((resolution, resolution), np.nan, dtype=np.float32)

    # Use a simple approach: bin the points and take max height per cell
    for i in range(len(x)):
        grid_x = x_norm[i]  # LiDAR x -> Minecraft X (column index)
        grid_y = y_norm[i]  # LiDAR y -> Minecraft Z (row index)
        if np.isnan(heightmap[grid_y, grid_x]) or z[i] > heightmap[grid_y, grid_x]:
            heightmap[grid_y, grid_x] = z[i]  # LiDAR z -> stored as height value

    # Fill any remaining NaN values
    mask = np.isnan(heightmap)
    if mask.any():
        # Use mean of non-NaN values
        mean_val = np.nanmean(heightmap)
        heightmap[mask] = mean_val if not np.isnan(mean_val) else 64.0

    # Optional: smooth the heightmap
    try:
        from scipy import ndimage

        heightmap = ndimage.gaussian_filter(heightmap, sigma=1)
    except ImportError:
        pass  # scipy not required, but helps with smoothing

    print(f"Heightmap created: {heightmap.shape}")
    print(f"Elevation range: {np.nanmin(heightmap):.2f} to {np.nanmax(heightmap):.2f}")

    # Cache the result
    cache.set(cache_key, heightmap)

    return heightmap


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert LAZ LiDAR files to Minecraft Java Edition worlds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run lidar_to_minecraft.py dars/USGS_LPC_CA_SanFrancisco_B23_05300270.laz
  uv run lidar_to_minecraft.py dars/data.laz san_francisco 1024
        """,
    )
    parser.add_argument(
        "laz_file",
        type=str,
        help="Path to the input LAZ file",
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
        help="Heightmap resolution in pixels (default: auto-detect based on point density)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.laz_file):
        print(f"Error: File not found: {args.laz_file}", file=sys.stderr)
        sys.exit(1)

    # Read LAZ file
    x, y, z = read_laz_file(args.laz_file)

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

    # Create heightmap
    heightmap = create_heightmap(x, y, z, resolution=resolution)

    # Generate Minecraft world
    generator = MinecraftWorldGenerator(args.world_name)
    generator.generate_world(heightmap)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
