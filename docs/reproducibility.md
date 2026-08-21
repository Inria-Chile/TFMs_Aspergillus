# Reproducibility protocol

Each seed is an analysis replicate, not an independent biological sample. A run records the seed, fold definitions, resolved configuration, package versions, input SHA-256, host, model, scenario, task, metrics, and output checksums.

The public repository should contain scripts and small metadata only. Raw/private data and large result trees remain outside Git and are referenced by a DOI, access instructions, or an approved encrypted artifact. A release manifest must identify the exact final tables and figures used by the manuscript.

## Migration from V3

The source implementation is the tagged V3 release used to prepare the paper.
The final validated result tables are identified by the artifact
`403_final_tables_100_replicas_20260816`. Historical monitoring logs are
provenance, not runtime inputs. Generated figures are derived artifacts and
should be regenerated in CI or a documented Grid'5000 recipe rather than
committed in bulk.
