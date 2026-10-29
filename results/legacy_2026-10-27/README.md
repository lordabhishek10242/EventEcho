# Legacy outputs (2026-10-27) — do not cite

> **Quarantined.** These files are the original 9-event / 5-stock exports from the supplied notebook archive. They predate the 21-event registry, the Close/Close price-return convention and the benchmark-session calendar fix. They are kept only for history. The current results are in [`../reproduced/`](../reproduced/), which carries its own manifest and integrity report.

## Original provenance note

The CSV files and `manifest.json` in this directory were extracted unchanged from `notebook/EventEcho_artifacts.zip`. They are the project's supplied legacy exports, generated in the original environment on 2026-10-27 according to the embedded manifest. They have not been independently regenerated in this checkout because no historical prices were supplied and the Python dependencies were not installed at intake.

Run `python scripts/run_analysis.py` to generate fresh outputs. The run writes a new manifest that identifies the input source, date range, parameters, event split, and generated files. Use `--output-dir` to avoid overwriting the archived outputs if comparing versions.

The legacy exports have these principal row counts: event master 9; multi-stock event study 495; multi-stock CAR 135; cross-stock consistency 27; trades 45; event trading results 9; significance tests 135; single-stock event study 66. The original notebook's single-stock example covers six events, while the multi-stock extension covers nine.

The supplied ZIP did not include raw historical prices. The CSV files are analysis outputs, not source price data. Their reported results should be treated as legacy and unverified until the analysis is rerun against an identified price snapshot.