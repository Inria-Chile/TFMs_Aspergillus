# Optional-model compatibility probes

The repository includes scripts/smoke_test_models.py for a small real-data probe of TabPFN and PySR. It is a compatibility check, not a benchmark and not a substitute for the 100-seed experiments.

Run it on an allocated GPU node, never on a Grid'5000 frontend:

    /home/lvalenzuela/.conda-env/bin/python scripts/smoke_test_models.py \
      --data data/raw/aspergillus_predictors.csv \
      --output results/smoke_tests/musa-2-20260827

The probe uses target F_Aspergillus, three numeric predictors, seed 123, a stratified 80/20 split for TabPFN, and twelve complete rows for the short PySR regression fit. PySR is serial and bounded to two iterations, two populations, population size 20, and a 30-second timeout.

On musa-2.sophia.grid5000.fr (OAR job 3052916, checked 2026-08-27), both probes passed. TabPFN 8.0.7 ran on an NVIDIA H100 NVL with CUDA and returned ten test predictions (AUC 0.6666667; 2.093 seconds). PySR 1.5.9 used SymbolicRegression.jl 1.11.3, returned four equations, and took 11.461 seconds.

The command writes smoke_test_report.json and one JSON result per model. These probes establish that the backends can import, fit, predict, and finish a minimal run in the tested environment; they do not validate every checkpoint or production-scale configuration.
