from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.exr_utils import compute_metrics, read_exr_rgb  # noqa: E402


EXPECTED_RESOLUTION = (192, 64)
ROI_CENTERS = ((60, 32), (78, 32), (96, 32), (114, 32), (132, 32))
ROI_RADIUS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the fixed OpenPBR white-furnace sphere ROIs."
    )
    parser.add_argument("image", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = read_exr_rgb(args.image)
    height, width, _ = image.shape
    if (width, height) != EXPECTED_RESOLUTION:
        print(
            f"error: expected {EXPECTED_RESOLUTION[0]}x{EXPECTED_RESOLUTION[1]}, "
            f"got {width}x{height}",
            file=sys.stderr,
        )
        return 2

    rois = []
    for center_x, center_y in ROI_CENTERS:
        rois.append(
            image[
                center_y - ROI_RADIUS : center_y + ROI_RADIUS + 1,
                center_x - ROI_RADIUS : center_x + ROI_RADIUS + 1,
                :,
            ]
        )
    actual = np.concatenate([roi.reshape(-1, 1, 3) for roi in rois], axis=0)
    metrics = compute_metrics(actual, np.ones_like(actual))
    max_bias = max(abs(value) for value in metrics["mean_rgb_bias"]["channels"])
    nrmse = metrics["nrmse"]
    print(f"OpenPBR white furnace: mean RGB bias={max_bias:.6f}, NRMSE={nrmse:.6f}")
    if max_bias > 0.02 or nrmse > 0.08:
        print("error: white-furnace ROI thresholds exceeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
