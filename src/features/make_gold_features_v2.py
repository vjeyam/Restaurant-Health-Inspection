import os, io, re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def download_blob_to_bytes(conn_str: str, container: str, blob_path: str) -> bytes:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    return bsc.get_blob_client(container, blob_path).download_blob().readall()

def upload_bytes(conn_str: str, container: str, blob_path: str, payload: bytes) -> None:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    bsc.get_container_client(container).upload_blob(blob_path, payload, overwrite=True)
    print(f"Uploaded: {container}/{blob_path}")

def make_restaurant_key(df: pd.DataFrame) -> pd.Series:
    def norm(s):
        if not isinstance(s, str):
            return ""
        s = s.upper().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^A-Z0-9 ]+", "", s)
        return s

    name = df["dba_name"].fillna("").map(norm)
    addr = df["address"].fillna("").map(norm)
    zipc = df.get("zip", pd.Series([""] * len(df))).fillna("").astype(str)
    return (name + "|" + addr + "|" + zipc).astype(str)

def main():
    load_dotenv()
    conn_str = get_env("AZURE_STORAGE_CONNECTION_STRING")
    container = get_env("AZURE_BLOB_CONTAINER")

    dt = "20260217T153907Z"  # change if needed

    silver_path = f"silver/food_inspections/dt={dt}/food_inspections_silver.parquet"
    raw = download_blob_to_bytes(conn_str, container, silver_path)
    df = pd.read_parquet(io.BytesIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]

    # Basic sanity
    df = df[df["inspection_date"].notna()].copy()
    df = df.sort_values(["inspection_date", "inspection_id"])

    # Restaurant key
    df["restaurant_key"] = make_restaurant_key(df)

    # Ensure target columns exist
    for col in ["has_critical_violation", "has_violation", "is_fail"]:
        if col not in df.columns:
            df[col] = 0

    # ========= Restaurant-level lag features (leakage-safe) =========
    df = df.sort_values(["restaurant_key", "inspection_date", "inspection_id"]).copy()
    g = df.groupby("restaurant_key", group_keys=False)

    df["prev_inspection_count"] = g.cumcount()
    df["prev_has_violation_rate"] = g["has_violation"].apply(lambda s: s.shift(1).expanding().mean())
    df["prev_has_critical_rate"] = g["has_critical_violation"].apply(lambda s: s.shift(1).expanding().mean())
    df["prev_fail_rate"] = g["is_fail"].apply(lambda s: s.shift(1).expanding().mean())

    df["prev_inspection_date"] = g["inspection_date"].shift(1)
    df["days_since_prev_inspection"] = (df["inspection_date"] - df["prev_inspection_date"]).dt.days

    df["prev_critical_count_last_3"] = g["has_critical_violation"].apply(lambda s: s.shift(1).rolling(3).sum())
    df["prev_violation_count_last_3"] = g["has_violation"].apply(lambda s: s.shift(1).rolling(3).sum())

    # trend feature: last 3 critical rate vs prior historical rate
    df["prev_critical_rate_last_3"] = g["has_critical_violation"].apply(lambda s: s.shift(1).rolling(3).mean())
    df["prev_critical_rate_change"] = df["prev_critical_rate_last_3"] - df["prev_has_critical_rate"]

    # fill NaNs
    fill0 = [
        "prev_has_violation_rate", "prev_has_critical_rate", "prev_fail_rate",
        "days_since_prev_inspection",
        "prev_critical_count_last_3", "prev_violation_count_last_3",
        "prev_critical_rate_last_3", "prev_critical_rate_change"
    ]
    for c in fill0:
        df[c] = df[c].fillna(0)

    # ========= ZIP-level historical risk (leakage-safe) =========
    # For each zip, compute expanding mean of critical violations, shifted by 1.
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.extract(r"(\d{5})", expand=False)
        gz = df.groupby("zip", group_keys=False)
        df["zip_prev_critical_rate"] = gz["has_critical_violation"].apply(lambda s: s.shift(1).expanding().mean()).fillna(0)
        df["zip_prev_fail_rate"] = gz["is_fail"].apply(lambda s: s.shift(1).expanding().mean()).fillna(0)
        df["zip_prev_count"] = gz.cumcount()
    else:
        df["zip_prev_critical_rate"] = 0
        df["zip_prev_fail_rate"] = 0
        df["zip_prev_count"] = 0

    # Optional: keep only restaurants
    if "facility_type" in df.columns:
        df = df[df["facility_type"].fillna("").str.lower().str.contains("restaurant")].copy()

    # Select gold columns (now includes zip + facility_type)
    gold_cols = [
        "inspection_id", "restaurant_key", "inspection_date",
        "zip", "facility_type",
        "risk_num",
        "prev_inspection_count",
        "prev_has_violation_rate", "prev_has_critical_rate", "prev_fail_rate",
        "days_since_prev_inspection",
        "prev_critical_count_last_3", "prev_violation_count_last_3",
        "prev_critical_rate_last_3", "prev_critical_rate_change",
        "zip_prev_critical_rate", "zip_prev_fail_rate", "zip_prev_count",
        "has_critical_violation", "is_fail"
    ]
    gold_cols = [c for c in gold_cols if c in df.columns]
    gold = df[gold_cols].copy()

    gold_path = f"gold/features/dt={dt}/features_inspections_v2.parquet"
    out = gold.to_parquet(index=False)
    upload_bytes(conn_str, container, gold_path, out)

    print(f"Gold v2 rows: {len(gold):,}")
    print("Done.")

if __name__ == "__main__":
    main()
