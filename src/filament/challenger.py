"""Matched postprocessing for dark absorption relative to same-disk flanks.

All geometry and contrast come from the image being predicted. The filter
either keeps a complete baseline instance unchanged or discards it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.morphology import skeletonize

from .infer import label_instances, label_rle


def disk_geometry(image):
    """Estimate the limb by fitting steep outward intensity drops across radial rays."""
    length = min(image.shape)
    midpoint = (np.asarray(image.shape, dtype=float) - 1.0) / 2.0
    radii = np.arange(0.36 * length, 0.48 * length, 2.0)
    theta = np.arange(180) * (2.0 * np.pi / 180)
    directions = np.stack([np.sin(theta), np.cos(theta)])
    rays = midpoint[:, None, None] + directions[:, :, None] * radii[None, None, :]
    brightness = ndi.map_coordinates(image, rays, order=1, output=np.float32)
    outward_drop = -np.diff(ndi.gaussian_filter1d(brightness, sigma=1, axis=1), axis=1)
    edges = radii[outward_drop.argmax(axis=1)] + 1.0
    edge_points = midpoint[:, None] + directions * edges
    initial = np.r_[midpoint, np.median(edges)]

    def residual(circle):
        return np.linalg.norm(edge_points - circle[:2, None], axis=0) - circle[2]

    circle = least_squares(residual, initial, loss="soft_l1", f_scale=4.0).x
    support = np.mean(np.abs(residual(circle)) <= 0.01 * length)
    if (support < 0.65 or not (0.36 * length <= circle[2] <= 0.48 * length)
            or np.linalg.norm(circle[:2] - midpoint) > 0.035 * length):
        return None
    return tuple(map(float, circle))


def component_contrast(labels, label, bounds, image, disk):
    """Compare spine brightness with nearby unclaimed on-disk background."""
    if disk is None:
        return None, None
    rows, cols = bounds
    cy, cx, radius = disk
    pad = 40
    roi = (slice(max(0, rows.start - pad), min(image.shape[0], rows.stop + pad)),
           slice(max(0, cols.start - pad), min(image.shape[1], cols.stop + pad)))
    region = labels[roi]
    component = region == label
    spine = skeletonize(component)
    if not spine.any():
        spine = component
    width = float(2.0 * np.median(ndi.distance_transform_edt(component)[spine]))
    dist = ndi.distance_transform_edt(~component)
    yy, xx = np.ogrid[roi[0].start:roi[0].stop, roi[1].start:roi[1].stop]
    radial = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
    radial_median = float(np.median(radial[spine]))
    background = ((dist >= max(3.0, 0.45 * width))
                  & (dist <= min(39.0, max(7.0, 1.2 * width)))
                  & (region == 0) & (radial < 0.98))
    if np.count_nonzero(background) < 16 or radial_median >= 0.98:
        return None, radial_median
    grayscale = image[roi]
    flank = grayscale[background].astype(np.float32)
    background_center = float(np.median(flank))
    mad = float(np.median(np.abs(flank - background_center)))
    contrast = (background_center - float(np.median(grayscale[spine]))) / max(2.0, 1.4826 * mad)
    return contrast, radial_median


def scored_components(probability, image, threshold, min_area, config):
    """Return baseline RLE and the candidate's contrast/radius evidence for each instance."""
    if image.shape != probability.shape or image.ndim != 2:
        raise ValueError("Expected a native grayscale image and equal-size probability map")
    labels, kept = label_instances(probability, threshold, min_area)
    if not kept:
        return []
    disk = disk_geometry(image)
    return [(label_rle(labels, label, bounds),
             *component_contrast(labels, label, bounds, image, disk))
            for label, bounds in kept]


def contrast_accepts(contrast, radial, config):
    """Abstain when flank contrast is unavailable, keeping the baseline shape."""
    if config["method"] != "disk_aware_contrast":
        raise ValueError(f"Unsupported postprocess: {config['method']}")
    floor = float(config["contrast_floor"])
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("Contrast floor must be finite and nonnegative")
    return contrast is None or contrast >= floor


def candidate_rles(probability, image, threshold, min_area, config):
    return [rle for rle, contrast, radial in scored_components(probability, image, threshold, min_area, config)
            if contrast_accepts(contrast, radial, config)]
