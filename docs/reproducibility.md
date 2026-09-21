# Reproducibility protocol

Each seed is an analysis replicate, not an independent biological sample. A
run records the seed, fold definitions, resolved configuration, package
versions, input SHA-256, host, model, scenario, task, metrics, and output
checksums.

The public repository contains scripts, configuration files, documentation,
and final consolidated tables. Non-release intermediate data and large result trees remain
outside Git and are referenced by access instructions or approved archived
artifacts. A release manifest identifies the exact final tables used by the
manuscript.

## Migration from V3

The source implementation is the validated V3 release used to prepare the
paper. The final validated result tables are identified by archived
consolidation batches, including `403_final_tables_100_replicas_20260816`
and the later public-release batches documented in
`docs/final-results-manifest.md`. Historical monitoring logs are provenance,
not runtime inputs. Generated figures are derived artifacts and should be
regenerated in CI or a documented Grid'5000 recipe rather than committed in
bulk.
