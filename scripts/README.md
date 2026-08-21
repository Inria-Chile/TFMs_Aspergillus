# Scripts

The command-line scripts are intentionally thin orchestration layers. Model-specific runners and plotting builders should receive a resolved YAML configuration and write seed-level outputs under a unique run directory. They must never infer completion from directory names alone.

The validated V3 implementation is retained as the migration source in the Nancy project. The next migration step is to move the final seed runner, TabPFN/TabICLv2 adapters, PySR runner, SHAP aggregation, and plot builders into this directory with tests around their export contracts.
