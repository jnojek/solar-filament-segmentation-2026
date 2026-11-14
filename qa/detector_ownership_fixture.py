"""Exercise detector ownership, connected residues, and unchanged parent RLE."""

import numpy as np
from pycocotools import mask as coco

from filament.detector_ownership import fuse_detector


def encode(mask):
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))


def main():
    shape = (96, 128)
    first = np.zeros(shape, dtype=bool)
    first[20:40, 10:35] = True
    second = np.zeros(shape, dtype=bool)
    second[20:40, 42:66] = True
    probability = np.zeros(shape, dtype=np.float16)
    probability[20:40, 10:66] = .9
    probability[20:45, 86:111] = .9
    detection = [(0.8, encode(second)), (0.9, encode(first))]

    strict = fuse_detector(probability, detection, mode="strict", confidence=.5)
    assert len(strict) == 2
    masks = coco.decode(strict).astype(bool)
    assert masks[25, 38, 0] and not masks[25, 38, 1]
    assert masks[25, 55, 1] and not masks[25, 93].any()
    assert not np.any(masks.sum(axis=2) > 1)

    parent = encode(first)
    obstruction = np.zeros(shape, dtype=bool)
    obstruction[20:40, 12:46] = True
    third = np.zeros(shape, dtype=bool)
    third[20:40, 36:65] = True
    appended = fuse_detector(probability, [(0.95, encode(obstruction)), (.9, encode(third))],
                             mode="append", confidence=.5, parent=[parent])
    assert appended[0] == parent and len(appended) == 2
    assert np.array_equal(coco.decode(appended[1]), third)

    disconnected = first.copy()
    disconnected[65:82, 60:82] = True
    standalone = fuse_detector(probability, [(0.9, encode(disconnected))], mode="detector", confidence=.5)
    assert len(standalone) == 1 and np.array_equal(coco.decode(standalone[0]), first)
    assert not fuse_detector(probability, detection, mode="strict", confidence=.95)
    print("Detector ownership, rejection without claims, disconnected pixels, and parent preservation passed.")


if __name__ == "__main__":
    main()
