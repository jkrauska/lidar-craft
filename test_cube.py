#!/usr/bin/env python3
"""
Simple test to generate a 10x10x10 cube of stone blocks.
This verifies that blocks are being written correctly to the world.
"""

import sys
from pathlib import Path

# Add parent directory to path to import lidar_to_minecraft
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from lidar_to_minecraft import MinecraftWorldGenerator


def create_test_cube():
    """Create a simple 10x10x10 cube of stone at Y=100, centered at origin."""
    print("Creating test cube world...")

    # Create a small heightmap - make it 16x16 (one chunk) for simplicity
    size = 16  # Exactly one chunk
    # Make the ENTIRE area raised so it's impossible to miss
    # Set everything to Y=100 (after scaling)
    # To get Y=100: we need heightmap value such that it scales to 100
    # Formula: (100 - 64) / (280 - 64) * (max - min) + min = 100
    # If min=64, we want: 36/216 * (max-64) + 64 = 100
    # 36/216 * (max-64) = 36, so max-64 = 216, so max = 280
    # But we want a simpler approach - just make everything high
    heightmap = np.full((size, size), 200.0, dtype=np.float32)  # Entire area at Y=280

    print(f"Heightmap shape: {heightmap.shape}")
    print(f"Heightmap range: {heightmap.min()} to {heightmap.max()}")
    print("Expected: ENTIRE 16x16 area filled with stone from Y=64 to Y=280")

    # Generate world
    generator = MinecraftWorldGenerator("test_cube", output_dir="output")
    generator.generate_world(heightmap)

    print("\nTest cube world generated!")
    print(f"World location: {generator.world_dir}")
    print("\nThe ENTIRE 16x16 area should be filled with stone blocks!")
    print("  X: 0 to 15")
    print("  Z: 0 to 15")
    print("  Y: 64 to 280 (216 blocks tall)")
    print("\nSpawn is at the center. You should be standing ON a solid stone platform!")
    print("If you don't see blocks, there's an issue with block generation or saving.")


if __name__ == "__main__":
    create_test_cube()
