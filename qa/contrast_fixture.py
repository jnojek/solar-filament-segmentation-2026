"""Check local contrast evidence and mask preservation on a controlled native disk."""
import numpy as np
from pycocotools import mask as coco

from filament.challenger import candidate_rles, contrast_accepts, disk_geometry, scored_components
from filament.infer import instance_rles

h = w = 2048
r, c = np.ogrid[:h, :w]
disk = (r - 1023.5)**2 + (c - 1023.5)**2 < 905**2
image = np.where(disk, 130, 10).astype(np.uint8)
p = np.zeros((h, w), dtype=np.float16)
strong = np.s_[890:902, 500:650]
weak = np.s_[1120:1132, 1100:1250]
p[strong] = p[weak] = 0.8
image[strong] = 110
image[weak] = 127
params = {"method": "disk_aware_contrast", "contrast_floor": 2.4}
assert abs(disk_geometry(image)[2] - 905) < 3
baseline = instance_rles(p, 0.65, 700)
scored = scored_components(p, image, 0.65, 700, params)
assert [x[0] for x in scored] == baseline
assert len(scored) == 2 and np.allclose([s[1] for s in scored], [10.0, 1.5])
retained = candidate_rles(p, image, 0.65, 700, params)
assert retained == baseline[:1]
expected = np.zeros((h, w), dtype=np.uint8)
expected[strong] = 1
assert np.array_equal(coco.decode(retained[0]), expected)
assert contrast_accepts(None, None, params)
assert contrast_accepts(2.4, 0.85, params)
print("Native disk fit, strong/weak contrast, abstention and exact mask preservation passed.")
