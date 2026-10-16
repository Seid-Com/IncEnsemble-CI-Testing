"""Finish remaining experiment families after main+ablation; merge into experiment_results.json."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

import run_experiments as R

RES = R.RESULTS_DIR

def main():
    results = {}
    main_df = pd.read_csv(os.path.join(RES, "main.csv"))
    results["main"] = main_df.to_dict(orient="records")
    results["statistics"] = R.compute_statistics(main_df)
    print("stats OK", flush=True)

    ablation_df = pd.read_csv(os.path.join(RES, "ablation.csv"))
    results["ablation"] = ablation_df.to_dict(orient="records")
    print("ablation loaded", flush=True)

    print("== Overhead ==", flush=True)
    results["overhead"] = R.measure_overhead()

    print("== Update latency ==", flush=True)
    upd_df = R.run_update_latency()
    upd_df.to_csv(os.path.join(RES, "update_latency.csv"), index=False)
    results["update_latency"] = upd_df.to_dict(orient="records")

    print("== Sensitivity ==", flush=True)
    spath = os.path.join(RES, "sensitivity.csv")
    if os.path.exists(spath) and os.path.getsize(spath) > 1000:
        sens_df = pd.read_csv(spath)
        print("loaded existing sensitivity.csv", flush=True)
    else:
        sens_df = R.run_sensitivity()
        sens_df.to_csv(spath, index=False)
    results["sensitivity"] = sens_df.to_dict(orient="records")

    print("== Imbalance ==", flush=True)
    ipath = os.path.join(RES, "imbalance.csv")
    if os.path.exists(ipath) and os.path.getsize(ipath) > 100:
        imb_df = pd.read_csv(ipath)
        print("loaded existing imbalance.csv", flush=True)
    else:
        imb_df = R.run_imbalance()
        imb_df.to_csv(ipath, index=False)
    results["imbalance"] = imb_df.to_dict(orient="records")

    print("== Scalability ==", flush=True)
    scpath = os.path.join(RES, "scalability.csv")
    if os.path.exists(scpath) and os.path.getsize(scpath) > 100:
        scale_df = pd.read_csv(scpath)
        print("loaded existing scalability.csv", flush=True)
    else:
        scale_df = R.run_scalability()
        scale_df.to_csv(scpath, index=False)
    results["scalability"] = scale_df.to_dict(orient="records")

    print("== Drift ==", flush=True)
    cpath = os.path.join(RES, "commit_fdr.csv")
    if os.path.exists(cpath) and os.path.getsize(cpath) > 1000:
        commits_df = pd.read_csv(cpath)
        print("loaded existing commit_fdr.csv", flush=True)
    else:
        commit_rows = []
        for seed in (0, 1, 2):
            data = R.build_project("Lang", seed=seed)
            summary, crows = R.run_single(data, 25, seed, collect_commit=True)
            for r in crows:
                r.update({"project": "Lang", "seed": seed, "budget": 25})
                commit_rows.append(r)
        commits_df = pd.DataFrame(commit_rows)
    if not commits_df.empty:
        # wide fdr_<method> columns -> long method/fdr
        fdr_cols = [c for c in commits_df.columns if c.startswith("fdr_")]
        if fdr_cols:
            commits_df = commits_df.melt(
                id_vars=[c for c in ["commit", "drift"] if c in commits_df.columns],
                value_vars=fdr_cols, var_name="method", value_name="fdr")
            commits_df["method"] = commits_df["method"].str.replace("fdr_", "", regex=False)
        drift = commits_df.groupby(["method", "drift"])["fdr"].mean().reset_index()
        results["drift"] = drift.to_dict(orient="records")
        drift.to_csv(os.path.join(RES, "drift.csv"), index=False)
        commits_df.to_csv(os.path.join(RES, "commit_fdr.csv"), index=False)

    print("== SHAP ==", flush=True)
    shap_df = R.run_shap()
    shap_df.to_csv(os.path.join(RES, "shap.csv"), index=False)
    results["shap"] = shap_df.to_dict(orient="records")

    with open(os.path.join(RES, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("Saved results/experiment_results.json", flush=True)


if __name__ == "__main__":
    main()
