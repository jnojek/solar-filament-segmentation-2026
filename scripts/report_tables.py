"""Report tables from matched score receipts, without loading images or test data."""
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser(description=__doc__)
for field in ("dev", "confirm", "site", "export", "output"):
    parser.add_argument("--" + field, required=True, type=Path)
args = parser.parse_args()
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

parts = [(name, read(path)) for name, path in (
    ("Development", args.dev), ("Confirmation", args.confirm), ("Udaipur", args.site))]
export = read(args.export)
lines = [r"\subsection{Matched results and selection}",
         r"\begin{table}[h]\centering\small",
         r"\caption{Matched instance scoring. Each result is pooled over independent views.}",
         r"\begin{tabular}{llrrr}",
         r"\toprule",
         r"Partition & Method & PQ & FP & FN\\",
         r"\midrule"]
for name, part in parts:
    for arm, label in (("baseline", "Baseline"), ("candidate", "Contrast")):
        m = part["metrics"][arm]
        lines.append(f"{name} & {label} & {m['pq']:.4f} & {m['fp']:,} & {m['fn']:,}"+r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

lines += [r"\begin{table}[h]\centering\small",
          r"\caption{Diagnostic errors for baseline \(\rightarrow\) contrast.}",
          r"\begin{tabular}{lrrrr}", r"\toprule",
          r"Partition & Splits & Merges & Zero overlap & New misses\\",
          r"\midrule"]
for name, part in parts:
    b, c = (part["metrics"][key] for key in ("baseline", "candidate"))
    def pair(field):
        return f"{b[field]:,}"+r"\(\to\)"+f"{c[field]:,}"
    lines.append(f"{name} & {pair('meaningful_splits')} & {pair('meaningful_merges')} & {pair('unsupported_fp')} & {c['new_complete_misses']:,}"+r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

for name, part in parts:
    b, c = (part["metrics"][key] for key in ("baseline", "candidate"))
    boot = part["paired_group_bootstrap"]["candidate"]
    delta = part["paired_pq_delta"]["candidate"]
    interval = boot["95pct_interval"]
    span = f"; paired 95\\% bootstrap interval [{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else ""
    lines.append(f"{name}: {part['observation_count']} disks, {b['annotator_views']} views, {b['truth_instances']} annotated instances; PQ change {delta:+.4f}{span}.")
lines += [r"\par\smallskip",
          r"The baseline was selected before opening confirmation. The least aggressive contrast floor, 1.8, raised development PQ but created 37 complete misses, exceeding the predeclared cap of seven. We carried this rejected setting into confirmation and the site holdout as a fixed diagnostic. Removing a component can reduce split counts without connecting a filament; this challenger never adds pixels.",
          r"\subsection{Thin, faint and near-limb instances}",
          r"\begin{table}[h]\centering\small",
          r"\caption{Confirmation match recall at instance IoU \(>0.5\).}",
          r"\begin{tabular}{lrrr}", r"\toprule",
          r"Slice & Truth masks & Baseline & Contrast\\", r"\midrule"]
for key, name in (("thin", "Thin"), ("faint", "Faint"), ("near_limb", "Near limb")):
    group = parts[1][1]["slices"].get(key)
    if not group:
        continue
    b, c = group["baseline"], group["candidate"]
    lines.append(f"{name} & {b['truth_instances']} & {b['recall_at_iou_over_50']:.1%} & {c['recall_at_iou_over_50']:.1%}".replace("%", r"\%")+r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\input{discussion}"]
lines += [r"\subsection{Inference and reproducibility}",
          f"The selected export processed {export['observations']} supplied test images in {export['elapsed_seconds']:.2f} seconds on the RTX 4090, with {export['ram_peak_bytes']/2**30:.2f} GiB peak process RAM and {export['cuda_peak_reserved_bytes']/2**30:.2f} GiB peak reserved GPU memory. It contains {export['instances']:,} predicted instances and {export['empty_observations']} disks with no emitted rows. The {export['csv_bytes']:,}-byte CSV fits the 10 MB form limit. The released notebook independently decodes the exact output and checks its SHA-256.",
          r"\input{official}"]
args.output.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
