# Solar imaging rationale and challenger protocol

## Hypothesis

On the solar disk, filaments appear as dark H-alpha absorption structures. Their spines are often elongated, with local changes in apparent width, curvature, and contrast. Atmospheric seeing and changes in visible opacity can weaken a narrow part of a spine while leaving the portions on either side visible. Diercke et al. document a GONG observation in which a detector divided a large central filament into three detections. Their classical thresholding method also split filaments with uneven opacity. In the MAGFiLO annotation process, reviewers sometimes filled missing parts between fragments judged to belong to one filament. A thresholded foreground probability map can therefore break an annotated instance even when it detects much of its area. [MAGFiLO, pp. 8–9](https://www.nature.com/articles/s41597-024-03876-y.pdf) · [Diercke et al., sections 4.2–4.4](https://doi.org/10.1051/0004-6361/202348314)

The proposed change is to add a narrow bridge between nearby fragments when their terminal directions, widths, and the image or probability in the gap support the same projected spine. Qu et al. previously used spine orientation to link H-alpha filament fragments. The experiment here asks whether constraining that idea with frozen U-Net probabilities improves instance PQ relative to simple closing. [Qu et al., section 2.4.3](https://www.sidc.be/users/evarob/Literature/Papers/Solar%20Physics/2005%20Qu-filament%20detection%20SoPh%20preprint.pdf)

Nearby independent structures can align in projection. Sunspots, penumbral fibrils, and other dark features can resemble parts of filaments. Some true filament channels contain gaps or transverse segments. MAGFiLO also preserves cases where one annotator drew a single filament and another drew several. Treat a bridge as an estimate of the annotated projected structure. Require a fit to the full set of available development annotations when judging the effect. [MAGFiLO, pp. 8–9 and 11](https://www.nature.com/articles/s41597-024-03876-y.pdf) · [Diercke et al., section 4.3](https://doi.org/10.1051/0004-6361/202348314) · [Filippov 2017](https://doi.org/10.1093/mnras/stx2060)

## Evidence gate

Use only development observations to decide which challenger to test. For each annotation set, build a predicted-instance to true-instance overlap table before the official IoU matching step. A split has at least two predictions contributing substantial coverage to one true instance. A merge has one prediction covering substantial portions of at least two true instances. Record minimum overlap rules before inspecting candidate results and distinguish these diagnostics from PQ matches. Review a reproducible sample of official false positives, misses, splits, and merges across sites and dates. Record probable causes such as a faint gap, a dark nonfilament feature, a limb artifact, or an inference tile seam. Mark uncertain cases and annotation disagreements explicitly.

Estimate whether repair has room to help by counting fragmented cases with short gaps, compatible directions, and residual corridor signal. A fragmentation-related false positive is part of this opportunity, so raw false-positive totals alone cannot determine the challenger. For an additional diagnostic on development data, measure how often unioning fragments associated with the same annotated object could clear the official IoU threshold. This reference-informed union measures headroom and is never an inference method. Choose gap repair if repairable fragmentation is the dominant opportunity. If unsupported false positives dominate, choose a disk-aware local contrast challenger only. A disk-aware test should measure residual background variation in the supplied JPEGs, which already have large-scale intensity variation reduced. It should avoid treating all features at a given radius as equally trustworthy. [MAGFiLO, pp. 3–4](https://www.nature.com/articles/s41597-024-03876-y.pdf) · [NSO GONG image processing](https://nso.edu/data/nisp-data/h-alpha/)

## Matched test for orientation-aware repair

1. Freeze the checkpoint, full-resolution reconstruction, probability threshold, connected-component convention, and size rule. Start all three arms from the same foreground masks. Compare no repair, isotropic closing, and orientation-aware repair. Choose one closing radius and one repair configuration using development groups before confirmation.
2. Trace skeleton endpoints of distinct nearby components. Estimate outward tangents from short terminal branches and width from the foreground distance transform along each branch. Consider endpoint pairs that face each other, have comparable widths, and have an uncovered gap short in both native pixels and local widths. Require evidence along the gap itself, such as subthreshold filament probability across the corridor or darkness relative to flanking pixels. Exclude endpoints from the corridor statistic. Reject candidates crossing a third component, leaving the disk, or competing with another similarly plausible pairing. Join only unambiguous pairs and limit chaining.
3. Score each arm against every annotator separately and aggregate with the organizer scorer. Report PQ, TP, FP, FN, Dice/IoU diagnostics, splits, merges, and false merges introduced. Evaluate accepted bridges against each annotator using the overlap table and inspect examples of true repairs and new mistakes. Compare closing at a similar bridge distance and report how many bridges or added pixels each arm produced.
4. Freeze the development selection before opening confirmation groups. Compare changes on the same observation groups using paired group resampling if there are enough groups. Report the group and instance counts and intervals. Retain the baseline if the evidence is mixed.

The promotion rules were set before viewing model scores. The candidate must improve pooled PQ on development and add at least **0.005 PQ** on confirmation. Gap repair must reduce meaningful split counts on both partitions. The number of newly merged predictions may not exceed **0.5%** of the partition's annotator truth instances, rounded up. Count each candidate merge as new if the corresponding set of truth instances did not already share one baseline prediction under the meaningful-overlap rule. The number of previously unmatched truth instances recovered must be at least twice the number of newly merged predictions. If residual contrast is tested, it must reduce zero-overlap false positives on both partitions and introduce at most **0.5%** new complete misses, rounded up. Report the held-out observatory separately. A PQ decrease of **0.005** or more on that observatory vetoes promotion. Show observation and instance counts so readers can assess the precision of that comparison.

For descriptive slices, measure thinness from median distance-transform width on the annotated spine or skeleton. Estimate faintness from the difference between the spine and local flanks, scaled by local variation. Measure limb proximity using projected radius divided by the fitted disk radius. Fix slice cutoffs on development groups and report overlap among slices plus site and date support. MAGFiLO instructed annotators to omit very faint or small objects, small crowded groups, and very near-limb objects. Low agreement in these slices may reflect both visibility and label coverage. [MAGFiLO, pp. 8–9](https://www.nature.com/articles/s41597-024-03876-y.pdf)

## Published comparison

Reche and Cid (2026) reported mean IoU **0.70** (95% bootstrap CI 0.68–0.72) and Dice **0.81** (0.79–0.82) on their MAGFiLO test split. Their U-Net measured overlap in filament crops and used both MAGFiLO and SWEFil for training. Diercke et al. reported 92% detection accuracy using a protocol based on padded boxes and a 0.10 IoU threshold. These studies give context for filament detection and mask quality. Challenge PQ requires full-disk instance predictions evaluated under the organizer protocol. [Reche and Cid, Table 2](https://doi.org/10.1016/j.actaastro.2026.07.010) · [Diercke et al., section 4.1](https://doi.org/10.1051/0004-6361/202348314)

## Sprint result

The frozen baseline scored PQ 0.33654 on 115 development images with 186 annotator views. Its 655 unmatched predictions included 298 with zero annotation overlap. Of these, 189 were at projected radius 0.80–0.95 R, 101 were interior, and eight were at or beyond 0.95 R. The diagnostic found 73 meaningful splits, of which six had short, compatible endpoints and subthreshold support. One such pair would clear IoU 0.50 if its fragments were combined using the reference labels. These counts selected local contrast as the single challenger.

The contrast filter estimates an instance's median spine darkness relative to surrounding pixels inside the fitted solar disk. It scales the difference by a robust local intensity variation and keeps the baseline component whenever there are too few valid comparison pixels. The candidate only removes components, so the shape and placement of retained masks are identical. Four predeclared floors were compared on the same probability maps.

| Contrast floor | Development PQ | Zero-overlap FPs | New complete misses |
| --- | ---: | ---: | ---: |
| Baseline | 0.33654 | 298 | 0 |
| 1.8 | 0.34663 | 189 | 37 |
| 2.1 | 0.34398 | 149 | 79 |
| 2.4 | 0.34179 | 99 | 147 |
| 2.7 | 0.31952 | 62 | 272 |

Every floor exceeded the predeclared development limit of seven new complete misses. The baseline was selected before opening confirmation groups. Floor 1.8 was retained as a rejected diagnostic for the fixed comparison.

| Partition | Images | Baseline PQ | Floor 1.8 PQ | Paired change and 95% interval | New complete misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 115 | 0.33654 | 0.34663 | +0.01009 [0.00601, 0.01485] | 37 |
| Confirmation | 80 | 0.29314 | 0.29784 | +0.00471 [−0.00455, 0.01174] | 16 |
| Held-out Udaipur | 101 | 0.33854 | 0.35000 | +0.01146 [0.00641, 0.01703] | 27 |

On confirmation, floor 1.8 removed 51 zero-overlap false positives and lost 12 PQ matches. Only 40 of 221 faint reference filaments matched after filtering, compared with 49 for the baseline. Near-limb matches fell from 61 to 54 of 107 and thin matches fell from 57 to 56 of 209. The confirmation gain was below 0.005 PQ, and the 16 additional complete misses exceeded the limit of five. The paired intervals use 2,000 resamples of 27-day observation groups.
