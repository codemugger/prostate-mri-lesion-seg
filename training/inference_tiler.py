from __future__ import annotations

import torch


def inference_ranges(length: int, multiple: int = 32) -> list[tuple[int, int]]:
    """Mirror the lesion app's full-multiple + tail range construction."""
    if length < multiple:
        raise ValueError(
            f"Lesion ROI dimension {length} is smaller than network multiple {multiple}"
        )
    full = length // multiple * multiple
    ranges = [(0, full)]
    if full < length:
        ranges.append((length - full, length))
    return ranges


def lesion_tiled_predict(
    model: torch.nn.Module, inputs: torch.Tensor, *, multiple: int = 32
) -> torch.Tensor:
    """Run RRUNet with the same tiling/overlap averaging as deployed inference."""
    if inputs.ndim != 5 or inputs.shape[0] != 1:
        raise ValueError("Lesion validation expects inputs shaped [1,C,X,Y,Z]")
    spatial = inputs.shape[-3:]
    ranges = [inference_ranges(int(length), multiple) for length in spatial]
    probability_sum = torch.zeros(
        (1, 2, *spatial), dtype=torch.float32, device=inputs.device
    )
    count = torch.zeros_like(probability_sum)
    for x_range in ranges[0]:
        for y_range in ranges[1]:
            for z_range in ranges[2]:
                slices = (
                    slice(x_range[0], x_range[1]),
                    slice(y_range[0], y_range[1]),
                    slice(z_range[0], z_range[1]),
                )
                patch = inputs[(slice(None), slice(None), *slices)]
                output = model(patch)
                probability_sum[(slice(None), slice(None), *slices)] += output.float()
                count[(slice(None), slice(None), *slices)] += 1.0
    if torch.any(count == 0):
        raise RuntimeError("Lesion tiler left uncovered voxels")
    return probability_sum / count
