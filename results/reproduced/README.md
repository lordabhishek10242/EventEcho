# Current release outputs

Every file in this folder comes from **one** analysis run. The CI workflow
*Historical Rebuild* generated them and committed them here only after
`scripts/verify_release.py` passed.

How to check which run you are looking at:

| File | What it tells you |
|---|---|
| `manifest.json` | Run time (`generated_utc`), code commit (`code_commit`), snapshot ID, price SHA-256, price fields, calendar, parameters, development/holdout split, and the row count of every table. |
| `release_validation.json` | Result of every integrity check on these exact files (CAR sums, Holm recomputation, cost arithmetic, sample summaries, overlap symmetry, dashboard = CSVs). |
| `snapshot_metadata/manifest.json`, `snapshot_metadata/validation.json` | The frozen price snapshot's own provenance and data-quality diagnostics. |
| `eventecho_demo.html` | The dashboard. It embeds this folder's manifest and CSVs, so it works offline. |

The raw provider prices (`data/snapshots/<snapshot-id>/prices.csv`) are not
committed, because Yahoo Finance data may be subject to provider terms. They are
kept in the workflow artifact of the run that downloaded them. With that
snapshot in `data/snapshots/`, the same tables can be rebuilt offline:

```bash
python scripts/run_analysis.py --snapshot-id <snapshot-id> --offline --output-dir results/reproduced
python scripts/build_demo.py --analysis-dir results/reproduced
python scripts/verify_release.py --analysis-dir results/reproduced
```

The 2026-10-27 legacy exports are kept separately in
[`../legacy_2026-10-27/`](../legacy_2026-10-27/) and must not be cited as
current results. See [the final evidence audit](../../docs/final-evidence-audit-2026-10-29.md)
for the verified figures and their limits.
