import os
import io
import json
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def blob_client(conn_str: str) -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(conn_str)

def download_blob_bytes(bsc: BlobServiceClient, container: str, blob_path: str) -> bytes:
    return bsc.get_blob_client(container=container, blob=blob_path).download_blob().readall()

def upload_blob_bytes(bsc: BlobServiceClient, container: str, blob_path: str, payload: bytes) -> None:
    bsc.get_container_client(container).upload_blob(blob_path, payload, overwrite=True)
    print(f"Uploaded: {container}/{blob_path}")

@dataclass
class Config:
    dt: str
    cutoff_date: str = "2025-01-01"
    top_n: int = 100
    n_estimators: int = 300
    max_depth: int = 12
    random_state: int = 42

def compute_capture_metrics(scored_df: pd.DataFrame, target_col: str, score_col: str) -> List[dict]:
    scored_sorted = scored_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    baseline = float(scored_sorted[target_col].mean())

    out = []
    for pct in [0.05, 0.10, 0.20, 0.30]:
        k = max(1, int(len(scored_sorted) * pct))
        topk = scored_sorted.head(k)
        rate = float(topk[target_col].mean())
        out.append({
            "pct": pct,
            "k": k,
            "topk_rate": rate,
            "baseline_rate": baseline,
            "lift_vs_baseline": (rate / baseline) if baseline > 0 else None,
            "topk_positive_count": int(topk[target_col].sum()),
        })
    return out

def main():
    load_dotenv()

    conn_str = get_env("AZURE_STORAGE_CONNECTION_STRING")
    container = get_env("AZURE_BLOB_CONTAINER")
    bsc = blob_client(conn_str)

    # You can pass DT via env var too if you want
    dt = os.getenv("DT") or "20260217T153907Z"
    cfg = Config(dt=dt)

    # Paths
    gold_path = f"gold/features/dt={cfg.dt}/features_inspections_v1.parquet"
    silver_path = f"silver/food_inspections/dt={cfg.dt}/food_inspections_silver.parquet"

    print(f"Downloading gold: {container}/{gold_path}")
    gold_bytes = download_blob_bytes(bsc, container, gold_path)
    df = pd.read_parquet(io.BytesIO(gold_bytes))
    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure inspection_date is datetime
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    df = df[df["inspection_date"].notna()].copy()

    # Time split
    cutoff = pd.Timestamp(cfg.cutoff_date)
    df = df.sort_values("inspection_date")
    train = df[df["inspection_date"] < cutoff].copy()
    test = df[df["inspection_date"] >= cutoff].copy()

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train/test split. train={len(train)} test={len(test)}. "
                           f"Try adjusting cutoff_date in Config.")

    # Features/target
    target_col = "has_critical_violation"
    feature_cols = [
        "prev_inspection_count",
        "prev_has_violation_rate",
        "prev_has_critical_rate",
        "prev_fail_rate",
        "days_since_prev_inspection",
        "prev_critical_count_last_3",
        "prev_violation_count_last_3",
        "zip_prev_critical_rate",
        "zip_prev_fail_rate",
        "zip_prev_count",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing}")

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int)

    X_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    # Model pipeline
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("rf", RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1
        ))
    ])

    rf_pipe.fit(X_train, y_train)
    probs = rf_pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    print("RF Test ROC-AUC:", auc)

    # Score output
    scored = test.copy()
    scored["risk_score"] = probs
    scored["risk_rank"] = scored["risk_score"].rank(method="first", ascending=False).astype(int)
    scored["risk_decile"] = pd.qcut(scored["risk_score"], 10, labels=False, duplicates="drop").astype(int)

    # Join identity columns from silver
    print(f"Downloading silver: {container}/{silver_path}")
    silver_bytes = download_blob_bytes(bsc, container, silver_path)
    silver_df = pd.read_parquet(io.BytesIO(silver_bytes))
    silver_df.columns = [c.lower().strip() for c in silver_df.columns]

    id_cols = [
        "inspection_id", "dba_name", "aka_name", "address", "zip",
        "facility_type", "inspection_type", "results_norm"
    ]
    id_cols = [c for c in id_cols if c in silver_df.columns]

    # Avoid zip collision; keep zip from scored (gold) and only bring identity cols without zip
    id_cols_no_zip = [c for c in id_cols if c != "zip"]
    scored_out = scored.merge(silver_df[id_cols_no_zip], on="inspection_id", how="left")

    # Top N table
    topn = scored_out.sort_values("risk_score", ascending=False).head(cfg.top_n).copy()
    top_cols = [
        "risk_rank", "risk_score", "risk_decile",
        "inspection_date",
        "dba_name", "aka_name", "address", "zip",
        "facility_type", "inspection_type",
        target_col
    ]
    top_cols = [c for c in top_cols if c in topn.columns]

    # Operational metrics
    lift_table = (
        scored_out.groupby("risk_decile")[target_col]
        .agg(["mean", "count"])
        .sort_index(ascending=False)
        .reset_index()
    )
    baseline = float(scored_out[target_col].mean())
    lift_table["baseline"] = baseline
    lift_table["lift_vs_baseline"] = lift_table["mean"] / baseline if baseline > 0 else None

    capture_metrics = compute_capture_metrics(scored_out, target_col=target_col, score_col="risk_score")

    # Upload scored dataset artifacts
    scores_parquet_blob = f"gold/scores/dt={cfg.dt}/scored_test_set.parquet"
    upload_blob_bytes(bsc, container, scores_parquet_blob, scored_out.to_parquet(index=False))

    top_csv_blob = f"gold/scores/dt={cfg.dt}/top{cfg.top_n}_restaurants.csv"
    upload_blob_bytes(bsc, container, top_csv_blob, topn[top_cols].to_csv(index=False).encode("utf-8"))

    # Upload model artifact
    model_blob = f"gold/models/dt={cfg.dt}/rf_pipeline.joblib"
    buf = io.BytesIO()
    joblib.dump(rf_pipe, buf)
    buf.seek(0)
    upload_blob_bytes(bsc, container, model_blob, buf.getvalue())

    # Upload metrics
    metrics = {
        "dt": cfg.dt,
        "cutoff_date": cfg.cutoff_date,
        "model": "RandomForestClassifier (with SimpleImputer)",
        "roc_auc": auc,
        "baseline_rate": baseline,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_cols": feature_cols,
        "capture_metrics": capture_metrics,
        "lift_table": lift_table.to_dict(orient="records"),
    }
    metrics_blob = f"gold/models/dt={cfg.dt}/metrics.json"
    upload_blob_bytes(bsc, container, metrics_blob, json.dumps(metrics, indent=2).encode("utf-8"))

    print("Done.")

if __name__ == "__main__":
    main()
