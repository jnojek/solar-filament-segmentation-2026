"""Native-resolution tiled inference and connected-component instance extraction."""

import time

import numpy as np
from pycocotools import mask as coco
from scipy import ndimage as ndi
import torch

from .model import FilamentUNet


def normalization(image):
    height, width = image.shape
    central = image[height // 4:3 * height // 4:8, width // 4:3 * width // 4:8]
    lower, middle, upper = np.percentile(central, [10, 50, 90])
    return float(middle), max(float(upper - lower), 16.0)


def normalize(image, center, scale):
    return np.clip((image.astype(np.float32) - center) / scale, -4.0, 4.0)


def origins(length, tile, stride):
    if tile > length or not 0 < stride <= tile:
        raise ValueError("Tile geometry does not cover the image")
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


def axis_weight(start, length, tile, ramp):
    weight = np.ones(tile, dtype=np.float32)
    if ramp:
        taper = np.linspace(1.0 / (ramp + 1), 1.0, ramp, dtype=np.float32)
        if start > 0:
            weight[:ramp] = taper
        if start + tile < length:
            weight[-ramp:] = taper[::-1]
    return weight


@torch.inference_mode()
def probability_map(model, image, config, device):
    tile = config["inference_tile"]
    stride = config["inference_stride"]
    batch_size = config["inference_batch"]
    height, width = image.shape
    rows, cols = origins(height, tile, stride), origins(width, tile, stride)
    ramp = (tile - stride) // 2
    weights_y = {row: axis_weight(row, height, tile, ramp) for row in rows}
    weights_x = {col: axis_weight(col, width, tile, ramp) for col in cols}
    positions = [(row, col) for row in rows for col in cols]
    center, scale = normalization(image)
    plane = normalize(image, center, scale)
    prediction = np.zeros_like(plane)
    weight_sum = np.zeros_like(plane)
    for start in range(0, len(positions), batch_size):
        group = positions[start:start + batch_size]
        tiles = np.stack([plane[row:row + tile, col:col + tile] for row, col in group])
        inputs = torch.from_numpy(tiles[:, None]).to(device, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(inputs)
        probabilities = outputs.float().sigmoid().cpu().numpy()[:, 0]
        for (row, col), probability in zip(group, probabilities):
            block = np.s_[row:row + tile, col:col + tile]
            weight = weights_y[row][:, None] * weights_x[col][None, :]
            prediction[block] += probability * weight
            weight_sum[block] += weight
    if np.any(weight_sum <= 0):
        raise RuntimeError("Tiled inference left pixels uncovered")
    prediction /= weight_sum
    # Freeze the representation used for both model selection and matched postprocessing.
    return prediction.astype(np.float16)


def label_instances(probability, threshold, min_area):
    foreground = probability >= threshold
    labels, count = ndi.label(foreground, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    bounds = ndi.find_objects(labels)
    return labels, [(label, bounds[label - 1]) for label in range(1, count + 1)
                    if sizes[label] >= min_area]


def connected_instances(probability, threshold, min_area):
    labels, keep = label_instances(probability, threshold, min_area)
    for label, bounds in keep:
        mask = np.zeros(labels.shape, dtype=np.uint8, order="F")
        mask[bounds] = labels[bounds] == label
        yield mask


def instance_rles(probability, threshold, min_area):
    labels, keep = label_instances(probability, threshold, min_area)
    return [label_rle(labels, label, bounds) for label, bounds in keep]


def label_rle(labels, label, bounds):
    height, width = labels.shape
    rows, cols = bounds
    x, y = np.nonzero(labels[bounds].T == label)
    offsets = (x + cols.start).astype(np.int64) * height + y + rows.start
    breaks = np.flatnonzero(np.diff(offsets) != 1) + 1
    starts = offsets[np.r_[0, breaks]]
    ends = offsets[np.r_[breaks - 1, len(offsets) - 1]] + 1
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[::2] = np.r_[starts[0], starts[1:] - ends[:-1], height * width - ends[-1]]
    counts[1::2] = ends - starts
    if counts[-1] == 0:
        counts = counts[:-1]
    return coco.frPyObjects({"size": [height, width], "counts": counts.tolist()}, height, width)


def open_model(config, checkpoint, device):
    device = torch.device(device)
    model = FilamentUNet(config["channels"]).to(device, memory_format=torch.channels_last)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def profile_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def runtime_receipt(device, start):
    import psutil

    process = psutil.Process()
    memory = process.memory_info()
    return {
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "ram_rss_bytes": memory.rss,
        "ram_peak_bytes": getattr(memory, "peak_wset", memory.rss),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": str(torch.__version__),
    }
