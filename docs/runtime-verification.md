# Runtime verification record

This record distinguishes versions that were observed in a historical V3
environment from versions that were only tested as current compatibility
probes. It is intended to prevent an unverified modern package from being
described as the exact historical production dependency.

## Verified on Grid'5000

- `tabicl==2.0.2` was installed in the `tabicl-gpu` environment on `musa-2`.
  The package metadata did not retain a Git commit or a `direct_url.json`.
- `tabpfn==8.0.7` was installed in the historical `marta_tabpfn` environment.
- `pysr==1.5.9` was installed in the historical PySR environment used for
  the V3 seed runner.
- `SymbolicRegression.jl==1.11.3` was recovered from its Julia project
  metadata and declares Julia 1.10 compatibility.
- The currently tested XGBoost GPU compatibility version is `3.2.0`.
  It completed a small CUDA regression probe with V3-shaped data on an H100.
  The original XGBoost wheel used by every historical seed is not available.
- `musa-2` reported CUDA driver `580.95.05` and two H100 NVL GPUs with
  95,830 MiB each during verification.

## Data checksum

The currently accessible dataframe used in the GPU compatibility probe was:

```text
01A_table_anonymized_for_external_analysis.csv
size: 298403 bytes
sha256: 9be1cc63cbc8c5b57ed32d45159b69cd8198b8b376a7bff1c93b20dc0c062b17
```

This checksum identifies the available V2 shared-storage copy. It is not a
claim that the file is the final public data release. The public release must
recompute and publish checksums for every input file, including the taxa
dictionary.

## Missing artifacts

The named TabICLv2 classifier and regressor checkpoint files were not present
on the verification node, so their SHA-256 values remain unavailable. The
same applies to the upstream TabICLv2 Git commit and the historical XGBoost
wheel. The manifest records these limitations explicitly. A future release
can replace the `null` fields after the original artifacts are recovered,
without changing the analysis code or the tabular result schema.
