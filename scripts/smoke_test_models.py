#!/usr/bin/env python3
"""Small real-data compatibility probes for optional TabPFN and PySR backends."""
from __future__ import annotations
import argparse, json, platform, subprocess, time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return "not-installed"

def gpus():
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT)
        return [x.strip() for x in text.splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError): return []

def load_data(path, target, n_features):
    frame = pd.read_csv(path)
    if target not in frame: raise ValueError(f"Target column not found: {target}")
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    features = [x for x in numeric if x != target][:n_features]
    if len(features) < 2: raise ValueError("At least two numeric predictors are required")
    frame = frame[features + [target]].replace([np.inf, -np.inf], np.nan)
    frame[target] = pd.to_numeric(frame[target], errors="raise")
    return frame.dropna(subset=[target]), features

def tabpfn_probe(frame, features, target, out):
    import torch
    from tabpfn import TabPFNClassifier
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    X, y = frame[features].to_numpy(np.float32), (frame[target].to_numpy() > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=123)
    imp = SimpleImputer(strategy="median").fit(Xtr); Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    start = time.perf_counter(); model = TabPFNClassifier(device="cuda", random_state=123); model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    result = {"status":"passed", "model":"TabPFN", "task":"occurrence", "seed":123, "n_train":len(ytr), "n_test":len(yte), "n_features":len(features), "auc":float(roc_auc_score(yte,p)), "elapsed_seconds":round(time.perf_counter()-start,3), "device":"cuda"}
    (out / "tabpfn_probe.json").write_text(json.dumps(result, indent=2) + "\n"); return result

def pysr_probe(frame, features, target, out):
    from pysr import PySRRegressor
    small = frame[features + [target]].dropna().head(12); X = small[features].to_numpy(float); y = small[target].to_numpy(float)
    start = time.perf_counter(); model = PySRRegressor(niterations=2, populations=2, population_size=20, tournament_selection_n=5, timeout_in_seconds=30, parallelism="serial", procs=1, maxsize=8, random_state=123, verbosity=0, progress=False, output_directory=str(out / "pysr_runtime"), delete_tempfiles=True); model.fit(X,y)
    result = {"status":"passed", "model":"PySR", "task":"regression_probe", "seed":123, "n_samples":len(y), "n_features":len(features), "n_equations":len(model.equations_), "rmse":float(np.sqrt(mean_squared_error(y, model.predict(X))),), "elapsed_seconds":round(time.perf_counter()-start,3), "parallelism":"serial"}
    (out / "pysr_probe.json").write_text(json.dumps(result, indent=2) + "\n"); return result

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--data", type=Path, default=Path("data/raw/aspergillus_predictors.csv")); ap.add_argument("--target", default="F_Aspergillus"); ap.add_argument("--output", type=Path, default=Path("results/smoke_tests")); ap.add_argument("--features", type=int, default=3); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    frame, features = load_data(a.data, a.target, a.features); report = {"status":"passed", "python":platform.python_version(), "platform":platform.platform(), "packages":{n:pkg(n) for n in ["numpy","pandas","scikit-learn","torch","tabpfn","pysr"]}, "gpu_snapshot":gpus(), "data":str(a.data), "target":a.target, "features":features, "probes":{}}
    for key, fn in (("tabpfn", tabpfn_probe), ("pysr", pysr_probe)):
        try: report["probes"][key] = fn(frame, features, a.target, a.output)
        except Exception as exc: report["probes"][key] = {"status":"failed", "error":f"{type(exc).__name__}: {exc}"}
    if not all(x.get("status") == "passed" for x in report["probes"].values()): report["status"] = "failed"
    (a.output / "smoke_test_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2)); return 0 if report["status"] == "passed" else 2

if __name__ == "__main__": raise SystemExit(main())
