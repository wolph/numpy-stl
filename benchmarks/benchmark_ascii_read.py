#!/usr/bin/env python
"""Benchmark ASCII STL read with and without speedups.

Downloads the Stanford Dragon PLY model on first run,
converts to ASCII STL, then times reads.

Usage:
    python benchmarks/benchmark_ascii_read.py
    python benchmarks/benchmark_ascii_read.py --iterations 10
    python benchmarks/benchmark_ascii_read.py --render
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import statistics
import sys
import time
import urllib.request

# Ensure the package is importable from repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from stl import Mode, mesh  # noqa: E402

CACHE_DIR = pathlib.Path(__file__).parent / '.cache'
DRAGON_URL = (
    'http://graphics.stanford.edu/pub/3Dscanrep/'
    'dragon/dragon_recon/dragon_vrip.ply.gz'
)
DRAGON_PLY = CACHE_DIR / 'dragon_vrip.ply'
DRAGON_ASCII_STL = CACHE_DIR / 'dragon_ascii.stl'


def download_dragon() -> pathlib.Path:
    """Download and decompress the Stanford Dragon."""
    if DRAGON_PLY.exists():
        return DRAGON_PLY

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = CACHE_DIR / 'dragon_vrip.ply.gz'

    print(f'Downloading Stanford Dragon from {DRAGON_URL}...')
    urllib.request.urlretrieve(DRAGON_URL, gz_path)

    print('Decompressing...')
    with gzip.open(gz_path, 'rb') as f_in, open(DRAGON_PLY, 'wb') as f_out:
        f_out.write(f_in.read())

    gz_path.unlink()
    print(f'Saved to {DRAGON_PLY}')
    return DRAGON_PLY


def convert_to_ascii_stl() -> pathlib.Path:
    """Convert the Dragon PLY to ASCII STL."""
    if DRAGON_ASCII_STL.exists():
        return DRAGON_ASCII_STL

    ply_path = download_dragon()
    print(f'Loading PLY from {ply_path}...')
    dragon = mesh.Mesh.from_ply_file(str(ply_path))
    print(f'Loaded {len(dragon.data)} triangles, saving as ASCII STL...')
    dragon.save(str(DRAGON_ASCII_STL), mode=Mode.ASCII)
    print(f'Saved to {DRAGON_ASCII_STL}')
    return DRAGON_ASCII_STL


def benchmark_read(
    stl_path: pathlib.Path,
    speedups: bool,
    iterations: int,
    warmup: int = 1,
) -> float:
    """Time ASCII STL reads, return median seconds."""
    for _ in range(warmup):
        mesh.Mesh.from_file(str(stl_path), speedups=speedups)

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        mesh.Mesh.from_file(str(stl_path), speedups=speedups)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return statistics.median(times)


def render_dragon(output_path: pathlib.Path) -> None:
    """Render the Dragon mesh to a PNG image."""
    try:
        import matplotlib as mpl

        mpl.use('Agg')
        from matplotlib import pyplot as plt
        from mpl_toolkits import mplot3d  # noqa: F401
    except ImportError:
        print('matplotlib not installed, skipping render')
        return

    ply_path = download_dragon()
    dragon = mesh.Mesh.from_ply_file(str(ply_path))

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(projection='3d')
    axes.add_collection3d(
        mplot3d.art3d.Poly3DCollection(
            dragon.vectors,
            edgecolor='none',
            facecolor='#4a90d9',
            alpha=0.7,
        )
    )

    scale = dragon.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title('Stanford Dragon')
    axes.view_init(elev=20, azim=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
    )
    plt.close()
    print(f'Render saved to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark ASCII STL read')
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of timed iterations (default: 5)',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Also render a Dragon PNG image',
    )
    parser.add_argument(
        '--render-output',
        type=str,
        default='docs/images/dragon_render.png',
        help='Output path for render image',
    )
    args = parser.parse_args()

    stl_path = convert_to_ascii_stl()

    dragon = mesh.Mesh.from_file(str(stl_path))
    tri_count = len(dragon.data)

    print(f'\nBenchmarking ASCII STL read ({tri_count:,} triangles)')
    print(f'Iterations: {args.iterations}')
    print()

    time_pure = benchmark_read(
        stl_path,
        speedups=False,
        iterations=args.iterations,
    )

    time_fast = benchmark_read(
        stl_path,
        speedups=True,
        iterations=args.iterations,
    )

    factor = time_pure / time_fast if time_fast > 0 else 0

    print(
        f'{"Model":<20} {"Triangles":>12} '
        f'{"Pure Python":>14} {"Speedups":>12} '
        f'{"Factor":>8}'
    )
    print('-' * 70)
    print(
        f'{"Stanford Dragon":<20} {tri_count:>12,} '
        f'{time_pure:>13.2f}s {time_fast:>11.2f}s '
        f'{factor:>7.1f}x'
    )

    if args.render:
        print()
        render_dragon(pathlib.Path(args.render_output))


if __name__ == '__main__':
    main()
