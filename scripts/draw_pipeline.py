"""Render a compact pipeline diagram for the organizer's two-column template."""
from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
canvas = Image.new("RGB", (1520, 420), "#ffffff")
draw = ImageDraw.Draw(canvas)
try:
    regular = ImageFont.truetype("DejaVuSans.ttf", 26)
    strong = ImageFont.truetype("DejaVuSans-Bold.ttf", 31)
    small = ImageFont.truetype("DejaVuSans.ttf", 23)
except OSError:
    regular = ImageFont.truetype("arial.ttf", 26)
    strong = ImageFont.truetype("arialbd.ttf", 31)
    small = ImageFont.truetype("arial.ttf", 23)
colors = {"text": "#163344", "muted": "#4b6270", "border": "#80abb5", "fill": "#edf6f7", "accent": "#166775"}
steps = [
    ("Official input", "Train: JPEG + polygons", "Test: JPEG only"),
    ("Native pixels", "Normalize grayscale", "Overlapping 384 px tiles"),
    ("Frozen U-Net", "Single foreground logit", "Native probability map"),
    ("Instances", "Threshold + components", "Compressed COCO RLE"),
]
for i, (title, one, two) in enumerate(steps):
    x = 16 + i * 380
    draw.rounded_rectangle((x, 110, x + 338, 319), radius=20, fill=colors["fill"], outline=colors["border"], width=2)
    draw.text((x + 24, 137), title, font=strong, fill=colors["text"])
    draw.text((x + 24, 211), one, font=regular, fill=colors["text"])
    draw.text((x + 24, 257), two, font=small, fill=colors["muted"])
    if i < len(steps) - 1:
        draw.line((x + 342, 214, x + 375, 214), fill=colors["accent"], width=5)
        draw.polygon([(x + 377, 214), (x + 365, 207), (x + 365, 221)], fill=colors["accent"])
draw.text((16, 37), "FROM OBSERVATION TO INSTANCE MASKS", font=strong, fill=colors["accent"])
draw.text((16, 354), "Train, develop and confirm on separate time groups. Independently decode the final native CSV.", font=regular, fill=colors["muted"])
canvas.save(out)
