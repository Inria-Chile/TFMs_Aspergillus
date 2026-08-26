# Historical runtime environments

The V3 experiments were executed in model-specific environments on
Grid'5000. This is intentional: TabPFN, TabICLv2, PySR, and the tree models
do not require the same Python or PyTorch stack. The versions below were read
from the environments that remain available on the allocated Sophia nodes.

| Workload | Environment | Python | Verified packages |
|---|---|---:|---|
| TabICLv2 and shared GPU utilities | `tabicl-gpu` | 3.10.19 | NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.2; scikit-learn 1.7.2; SHAP 0.49.1; PyTorch 2.5.1; `tabicl` 2.0.2; PySR 1.5.9 |
| TabPFN | `marta_tabpfn` | 3.10.20 | NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2; SHAP 0.49.1; PyTorch 2.12.0; `tabpfn` 8.0.7 |
| PySR seed runner | `cyt_models_nancy_clone_sophia` | 3.10.19 | NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2; SHAP 0.49.1; PyTorch 2.9.0; PySR 1.5.9 |
| Historical PySR comparison environment | `pysr_pheno_sophia` | 3.10.15 | NumPy 1.26.4; pandas 2.2.2; SciPy 1.13.1; scikit-learn 1.5.1; PyTorch 2.5.0; PySR 1.5.10 |

## XGBoost status

`xgboost==3.2.0` passed a current CUDA smoke test on an H100 using the V3
data schema. The historical production wheel used for every original V3
XGBoost seed could not be recovered from the remaining environments. It is
therefore recorded as a validated compatibility version, not as a claim
about the historical wheel. This distinction must remain in the paper
provenance and in any release note.

## Julia and PySR

The recovered Julia package is `SymbolicRegression.jl` version 1.11.3, from
the package project at `/home/lvalenzuela/.julia/packages/SymbolicRegression`.
Its project declares Julia 1.10 compatibility. The exact Julia executable
build and the package manifest commit were not retained and must be added
before claiming bitwise reproduction of PySR equations.

## Installation policy

The CPU utilities can be installed with `pip install -e '.[dev]'`. GPU
backends should be installed in separate environments using the versions in
this matrix. Do not merge these rows into a single environment lock unless a
fresh compatibility test confirms that the combined stack is supported.
