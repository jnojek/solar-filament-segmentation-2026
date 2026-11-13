"""Exercise encoding, export readback and placement with asymmetric fixtures."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

from filament.format import decode_mask, encode_mask, write_submission
from filament.infer import normalization, normalize, probability_map
from verify_entry import HEADER, SHAPE, image_catalog, verify_submission


def check_tile_reconstruction():
    class PointwiseModel(torch.nn.Module):
        def forward(self, pixels):
            return pixels

    rows, columns = np.ogrid[:SHAPE[0], :SHAPE[1]]
    disk = (rows - 1024) ** 2 + (columns - 1024) ** 2 < 900 ** 2
    image = np.where(disk, (7 * rows + 3 * columns) % 150 + 90, 0).astype(np.uint8)
    config = json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tiled = probability_map(PointwiseModel(), image, config, torch.device("cpu"))
    expected = 1 / (1 + np.exp(-normalize(image, *normalization(image))))
    error = float(np.max(np.abs(tiled.astype(np.float32) - expected)))
    if tiled.shape != SHAPE or error > 0.00025:
        raise AssertionError(f"Native tile reconstruction produced seams or placement errors: {error}")
    return error


def main():
    small = np.zeros((4, 7), dtype=np.uint8)
    small[1, 4] = 1
    small[2, 5] = 1
    encoded = encode_mask(small)
    independent = coco_mask.decode({"size": [4, 7], "counts": encoded.encode("ascii")})
    np.testing.assert_array_equal(np.argwhere(independent), [[1, 4], [2, 5]])
    np.testing.assert_array_equal(decode_mask(encoded, small.shape), small)

    with TemporaryDirectory(prefix="filament-format-fixture-") as temp_dir:
        root = Path(temp_dir)
        images_dir = root / "images"
        images_dir.mkdir()
        predicted, empty = "20260101120000Mh", "20260111120000Bh"
        for image_id in (predicted, empty):
            Image.new("L", SHAPE).save(images_dir / f"{image_id}.jpeg")
        images = image_catalog(images_dir)

        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[41:44, 203:209] = 1
        second_mask = np.zeros(SHAPE, dtype=np.uint8)
        second_mask[51:61, 390:393] = 1
        path = root / "submission.csv"
        coverage_path = root / "coverage.json"
        write_submission(
            path,
            [(predicted, [mask, second_mask]), (empty, [])],
            coverage_path=coverage_path,
        )
        report = verify_submission(path, coverage_path, images)
        if (report["empty_image_ids"], report["instance_rows"], report["foreground_pixels"]) != (
            [empty], 2, 48
        ):
            raise AssertionError("Export fixture did not preserve image coverage")

        changed = root / "transposed.csv"
        with changed.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow((f"{predicted}_1", encode_mask(mask.T)))
            writer.writerow((f"{predicted}_2", encode_mask(second_mask)))
        try:
            verify_submission(changed, coverage_path, images)
        except ValueError as error:
            if "decoded mask placement differs" not in str(error):
                raise
        else:
            raise AssertionError("Verifier accepted a transposed prediction")

    report["asymmetric_non_square_fixture"] = "passed"
    report["transposed_mask_rejected"] = True
    report["native_tile_reconstruction_max_absolute_error"] = check_tile_reconstruction()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
