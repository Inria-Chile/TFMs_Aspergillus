# Scripts

The command-line scripts are intentionally thin orchestration layers.
Model-specific runners and plotting builders receive a resolved YAML
configuration and write seed-level outputs under a unique run directory. They
must never infer completion from directory names alone.

The validated V3 implementation is retained for migration provenance and
backward-compatible reconstruction. Public runners should prefer the
repository-relative entry points in `scripts/runners/`, with tests around
their export contracts.

The upstream Google Earth Engine scripts used to generate the environmental
covariates are retained separately in `scripts/gee_environmental_covariates/`
as provenance code. They document the remote-sensing/GIS processing layer but
are not executed by the repository CI because they require external Earth
Engine assets and credentials.
