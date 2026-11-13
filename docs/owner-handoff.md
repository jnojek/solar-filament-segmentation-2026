# Owner handoff

The signed-in Kaggle CLI returned an entered account, no submissions, and a prediction cutoff of **2026-11-15 06:00 UTC** (midnight at the end of Saturday in Chicago). Official rules cap uploads at five per day and allow up to two final selections. The organizer's Important Dates page gives **Nov. 15 at 11:59 p.m. CDT** for final reports and solutions. The signed-in Google Form did not display its own cutoff. Chicago is on CST that week, so use the earlier Kaggle cutoff for both deliverables. Local handoff target: **Saturday Nov. 14 at 6 p.m. CST** (00:00 UTC).

The Nov. 11 announcement says incomplete or hard-to-reproduce work will not be judged. The official report uses the supplied Overleaf template. Publish a public Git repo with pinned `requirements.txt`, a Jupyter notebook that runs end-to-end, and the approved trained weights so retraining is unnecessary. Verify anonymous read access. The general rules require public code sharing on Kaggle through a competition-associated notebook or discussion when sharing the competition code; include that as a separate owner action. The owner should be able to explain and reproduce the method under the AI-assisted work rule.

After a final package is verified:

1. On [Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026), submit the approved exact CSV. Record the submission reference and check the final selection before Kaggle closes. If the calibration artifact is selected as final, reuse its accepted submission.
2. Publish the reviewed code and approved weights. Check that the Git URL opens anonymously and that the notebook loads the checkpoint.
3. Open the [organizer form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform). Supply response email, full names of all team members (semicolon separated), team email addresses (at least one), optional affiliations, countries, and the public Git URL.
4. Upload the exact accepted CSV under **Submission File** and the report under **Technical Report**. Each file is capped at 10 MB. Submit the form and retain the emailed response and Kaggle receipt.

The signed-in form was read without entering any values. The Google account name, email and photo are recorded on upload and submission. The report's author name, city, country and contact details need owner review before release. The owner may complete these actions directly; any action by Codex requires approval for the exact artifact and destination.

Sources: [announcements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements), [rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules), [final submission instructions](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission), and [official report template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe).


## Exact local artifacts

- CSV: `output/submission.csv`, 324,704 bytes, SHA-256 `22f46dfefbc233ba1b5a033397f61225d7563af5c73b5d9c35628155e35536b4`.
- Report: `output/pdf/solar-filament-report.pdf`, exactly four pages, with the owner-supplied author details.
- Public-ready code bundle: `output/solar-filament-source.zip`, including `artifacts/model.pt`, `artifacts/selected.json`, pinned dependencies and `reproduce.ipynb`. Extract and push to the owner's chosen repository after review.
- Kaggle submission message: `Native U-Net baseline; grouped validation; frozen 180-image export`.
- Form entries: use the team/contact values you confirm. The report names James Nojek, Houston, Texas, United States. No affiliation was supplied. Paste your actual public Git repository URL.

The PDF and CSV are local deliverables. Verify the accepted Kaggle submission reference and public Git access before sending the form. Use the same CSV bytes for the Kaggle and form uploads.
