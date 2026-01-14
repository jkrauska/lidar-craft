# lidar-craft
Generate Minecraft Worlds from Lidar Sources

This tool converts LAZ (LiDAR) files into playable Minecraft Java Edition worlds.

## Installation

1. Install [uv](https://github.com/astral-sh/uv) (fast Python package installer)
2. Install Python 3.13 and dependencies:
```bash
uv sync
```

This will create a virtual environment and install all dependencies, including the `lazrs` backend required for reading LAZ files.

## Usage

Using `uv`:
```bash
uv run lidar_to_minecraft.py <laz_file> [world_name] [resolution]
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
python lidar_to_minecraft.py <laz_file> [world_name] [resolution]
```

### Arguments
- `laz_file`: Path to the input LAZ file
- `world_name`: (Optional) Name for the generated Minecraft world (default: "lidar_world")
- `resolution`: (Optional) Heightmap resolution in pixels (default: auto-detect based on point density)

### Example

```bash
python lidar_to_minecraft.py dars/USGS_LPC_CA_SanFrancisco_B23_05300270.laz san_francisco 1024
```

## How It Works

1. **Read LAZ File**: Extracts point cloud data (x, y, z coordinates) from the LAZ file
2. **Create Heightmap**: Converts the point cloud into a regular grid heightmap
3. **Generate Minecraft World**: Creates a complete Minecraft world with:
   - `level.dat`: World metadata and settings
   - Region files (`.mca`): Contains chunks with terrain blocks
   - Proper block placement based on elevation data

## Installing the World in Minecraft

After generation, copy the world folder to your Minecraft saves directory:

- **macOS**: `~/Library/Application\ Support/minecraft/saves/`
- **Windows**: `%appdata%/.minecraft/saves/`
- **Linux**: `~/.minecraft/saves/`

Then launch Minecraft Java Edition and select the world from your world list.

## Notes

- The tool scales elevation data to fit within Minecraft's build height limits (64-320 blocks)
- Terrain is generated using dirt blocks
- **Auto-resolution**: If resolution is not specified, the tool automatically calculates an optimal resolution based on point density (targeting ~1.5 points per grid cell). This ensures dense LiDAR data is properly represented.
- Large LAZ files may take some time to process
- Higher resolutions produce more detailed terrain but require more processing time and disk space
- The world uses a flat world generator to prevent Minecraft from overwriting your custom terrain

## Caching

The tool uses `diskcache` to cache expensive operations:
- **LAZ file reading**: Cached based on file path and modification time
- **Heightmap generation**: Cached based on input data hash and resolution

Cache is stored in `~/.cache/lidar-craft/` with a 1GB size limit. Subsequent runs with the same input will be much faster. To clear the cache, simply delete the cache directory.

## Testing

Run the test suite to verify everything works:

```bash
uv run pytest tests/ -v
```

By default, slow tests (like the full conversion test) are skipped for faster feedback. To run all tests including slow ones:

```bash
uv run pytest tests/ -m slow -v
```

The tests include:
- Reading LAZ files
- Creating heightmaps from point cloud data
- Generating Minecraft world files
- Full end-to-end conversion test (marked as slow, ~24s)
