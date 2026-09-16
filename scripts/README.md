# Scripts

The command-line scripts are intentionally thin orchestration layers.
Model-specific runners and plotting builders receive a resolved YAML
configuration and write seed-level outputs under a unique run directory. They
must never infer completion from directory names alone.

The validated V3 implementation is retained for migration provenance and
backward-compatible reconstruction. Public runners should prefer the
repository-relative entry points in `scripts/runners/`, with tests around
their export contracts.
