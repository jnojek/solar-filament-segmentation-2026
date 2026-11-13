"""Generate a readable held-out example grid from the review overlays."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

src = Path("runs/contrast/rejected-floor-1.8-confirm")
dest = Path("report/figures-local")
dest.mkdir(exist_ok=True)
im = Image.new("RGB", (1440, 810), "#ffffff")
d = ImageDraw.Draw(im)
strong = ImageFont.truetype("arialbd.ttf", 28)
normal = ImageFont.truetype("arial.ttf", 24)
cases = [
    ("retained_match-01.png", "A  Retained match", "Both methods retain the annotated structure."),
    ("removed_unsupported_fp-01.png", "B  Unsupported detection removed", "The predicted patch has no overlap in this view."),
    ("lost_match-01.png", "C  Near-limb match lost", "The contrast filter removes a baseline match."),
    ("missed_by_both-01.png", "D  Missed by both", "The faint structure has no predicted overlap."),
]
for i, (name, title, subtitle) in enumerate(cases):
    x, y = 16 + (i % 2) * 718, 70 + (i // 2) * 362
    d.text((x, y), title, fill="#163344", font=strong)
    d.text((x, y + 38), subtitle, fill="#4b6270", font=normal)
    panel = Image.open(src / name).convert("RGB")
    panel = panel.crop((0, 0, panel.width, panel.height - 23))
    factor = min(698 / panel.width, 260 / panel.height)
    panel = panel.resize((round(panel.width * factor), round(panel.height * factor)), Image.Resampling.LANCZOS)
    im.paste(panel, (x, y + 76))
d.text((16, 16), "Confirmation examples", fill="#163344", font=strong)
d.line((918, 32, 958, 32), fill="#009eaa", width=6)
d.text((969, 17), "annotation", fill="#163344", font=normal)
d.line((1123, 32, 1163, 32), fill="#d08010", width=6)
d.text((1174, 17), "prediction", fill="#163344", font=normal)
im.save(dest / "examples.png")
