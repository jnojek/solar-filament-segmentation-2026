# Detector-guided instance segmentation

The first six uploads scored 0.27–0.29 public PQ. The best was the 419-disk native U-Net with a small-proposal verifier. The original 115-disk development audit separated instance grouping from foreground errors. Connected components of exact annotator foreground yielded 0.9976 PQ with no area floor. Lowering the real prediction area floor from 700 to 150 px at the frozen 0.65 probability threshold gained 213 matches and added 1,119 false positives. A reference-assisted audit identified one match recoverable through supported short-gap repair. The detector test targets selectivity and missed objects directly.

## Public implementations

| Source | Mechanism | Author-reported evidence | Scope of reuse |
| --- | --- | --- | --- |
| [Pawar](https://github.com/ameyypawar/filament-seg) | Independent YOLO11-seg masks claim native-resolution U-Net foreground | 0.371 U-Net, 0.383 detector, 0.421 fusion on 144 validation observations and 0.37 public PQ | Cite the MIT implementation for the ownership design. Use only our own weights, annotations and predictions. |
| [Angulo de Lafuente](https://github.com/Agnuxo1/solar-filament-segmentation-2026) | Mask R-CNN proposals followed by crop refinement | 0.39 public PQ. Author reports the end-to-end JPEG replay differs from the accepted CSV at some instances. | Cite as a separate two-stage example. |
| [He](https://github.com/HeShen-1/filament-segmentation-2026) | Detection followed by seed-conditioned mask refinement | 0.3593 mean five-fold PQ for its single pipeline. The reported 0.35 public fusion includes a third-party model. | Cite the independent design evidence. |
| [Memari](https://github.com/FarnooshMemari/solar-filament-segmentation) | Scratch U-Net and connected components | 0.32 public PQ | Use for context on the original family. |
| [Cross-model consensus](https://github.com/ranjeet258/solar-filament-consensus-2026) | Independently trained detector and semantic models, with a learned proposal selector | 0.39 public PQ for the model-only entry | Cite the model-only result. The repository separately reports scores that use outside predictions. |
| [Diercke et al.](https://doi.org/10.1051/0004-6361/202348314) | Detection plus pixel segmentation in H-alpha | Published detection results use a different data and scoring protocol | Cite the solar precedent. |
| [Ahmadzadeh et al.](https://arxiv.org/abs/1912.02743) | Mask R-CNN on full-disk H-alpha images | Earlier detector precedent with BBSO observations and HEK labels | Use the published result as context. |

The public results use different fits, splits and in some cases pretrained weights. Their figures are context for choosing an experiment. Our comparison uses the same supplied training set, original JPEGs and frozen group memberships for each arm. No third-party predictions, pretrained weights or external MAGFiLO annotations are inputs.

## Comparison

Train one independent instance-mask detector on the 419 fit disks. Score detector-only masks, detector additions to the incumbent, and exclusive detector claims on the frozen U-Net foreground. Use the 32 development disks to choose one fusion rule. Assess that rule on the 80 confirmation and 101 Udaipur disks using pooled organizer PQ, matching counts and unsupported predictions. These folds are out of training for the detector and refit and have been used in previous evaluation. The small-proposal verifier fit included the 32 development disks through the older network's predictions.

Training-mask geometry motivates a resolution check. Of 3,087 rasterized polygons from the 419 selected training views, 38.78% have median skeleton diameter below four pixels after 2:1 downsampling. At native resolution the fraction is 0.10%. A 3-by-3 grid of native 1024-pixel tiles at origins 0, 512 and 1024 contains 99.74% of these truth bounding boxes in at least one tile. If recall losses concentrate in thin filaments, compare a fixed native-tile continuation on the same fit and evaluation splits. Include overlap and boundary diagnostics so the resolution change can be distinguished from tile clipping.

The original Kaggle selection, exact source and recorded organizer response serve as the current finalist package. A replacement needs a completed official score and a verified matching CSV, code, weights, notebook and four-page report. The recorded form has an edit action for revising the existing response.

## Full-disk detector

The frozen rule uses exclusive claims at detector confidence 0.30 and semantic probability 0.65. The 36-epoch scratch fit took 388.05 seconds with 7.16 GiB peak reserved VRAM on the RTX 4090.

| Arm | Confirmation PQ | Udaipur PQ | Pooled PQ |
| --- | ---: | ---: | ---: |
| Original baseline | .29314 | .33854 | .31886 |
| Plain 419-disk refit | .31466 | .34152 | .32975 |
| Refit and verifier incumbent | .32536 | .35107 | .33984 |
| Detector alone at .30 | .21592 | .25600 | .23824 |
| Add detector masks to incumbent at .30 | .32289 | .34948 | .33785 |
| Exclusive detector ownership | .33168 | .36679 | .35129 |

Exclusive ownership reduced zero-overlap predictions from 429 to 109, and meaningful splits from 157 to 57. It also reduced matches from 1,000 to 851 and increased complete misses from 506 to 764. Faint matches fell from 130 to 85, and near-limb matches fell from 138 to 84. The pooled PQ change was +0.01145 with a corrected 95% paired interval of [−0.00051, +0.02380] across 68 date blocks. This is a candidate for one official calibration. Its recall loss argues against selecting it as the finalist on the local result alone.

The raw detector dev diagnostic found that 46 of 89 thin truth filaments had a detector bounding box above 0.50 IoU, while only eight had a detector mask above 0.50 IoU. This provides a direct reason to compare a native detector with the half-resolution detector.

Kaggle scored the exact full-disk detector-guided export **0.31 public PQ** as submission **58330445**, compared with **0.29** for submission **58324985**. The native export covered 180 test images, with 878 masks and five images without predictions. Both stages together took 104.01 seconds and peaked at 5.44 GiB reserved GPU memory.

## Native-tile detector continuation

The native detector continues from our full-disk scratch checkpoint for eight epochs on nine overlapping 1024-pixel crops per observation. Test proposals are lifted to the native grid, rejected if they touch an internal tile edge within four pixels, and deduplicated by native mask IoU. The fixed comparison holds detector confidence 0.30 and native semantic threshold 0.65 constant. The continuation, cropping and tile consolidation are a combined change.

| Strict detector-guided arm | Confirmation PQ | Udaipur PQ | Pooled PQ |
| --- | ---: | ---: | ---: |
| Full disk at half resolution | .33168 | .36679 | .35129 |
| Native tiles | .35876 | .37473 | .36769 |

The paired pooled gain over the full-disk detector was +.01640, 95% interval [.00351, .02991]. The gain over the refit plus verifier was +.02785, interval [.01306, .04290]. Against that incumbent, native tiles added 43 matches and 58 thin matches and reduced zero-overlap predictions by 140. Faint matches still decreased by 12 and near-limb matches by 25. The native held-out result supports a distinct official calibration.
