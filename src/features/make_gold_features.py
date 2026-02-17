import os
import io
import re
import random
import time
from typing import List, Dict

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
    blob_client = bsc.get_blob_client(container=container, blob=blob_path)
    return blob_client.download_blob().readall()

def upload_bytes(conn_str: str, container: str, blob_path: str, payload: bytes) -> None:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)
    cc.upload_blob(name=blob_path, data=payload, overwrite=True)
    print(f"Uploaded: {container}/{blob_path}")

def make_restaurant_key(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic restaurant key (v1): normalized name + address + zip.
    (We can improve later with license_ and fuzzy matching.)
    """
    def norm(s):
        if not isinstance(s, str):
            return ""
        s = s.upper().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^A-Z0-9 ]+", "", s)
        return s

    name = df["dba_name"].fillna("").map(norm) if "dba_name" in df.columns else ""
    addr = df["address"].fillna("").map(norm) if "address" in df.columns else ""
    zipc = df["zip"].fillna("").astype(str) if "zip" in df.columns else ""
    return (name + "|" + addr + "|" + zipc).astype(str)

def main():
    load_dotenv()

    conn_str = get_env("AZURE_STORAGE_CONNECTION_STRING")
    container = get_env("AZURE_BLOB_CONTAINER")

    # Use the dt of your SILVER parquet (set this to your latest dt)
    dt = "20260217T153907Z"  # <-- change if your silver dt differs

    silver_path = f"silver/food_inspections/dt={dt}/food_inspections_silver.parquet"
    print(f"Downloading: {container}/{silver_path}")
    data = download_blob_to_bytes(conn_str, container, silver_path)

    df = pd.read_parquet(io.BytesIO(data))
    df.columns = [c.lower().strip() for c in df.columns]

    # Basic sanity filters
    df = df[df["inspection_date"].notna()].copy()
    df = df.sort_values(["inspection_date", "inspection_id"])

    # Restaurant key
    df["restaurant_key"] = make_restaurant_key(df)

    # Ensure targets exist
    for col in ["has_critical_violation", "has_violation", "is_fail"]:
        if col not in df.columns:
            df[col] = 0

    # -----------------------------
    # Restaurant-level lag features
    # -----------------------------
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

    # Fill NaNs produced by shifting/rolling
    fill_zero_cols = [
        "prev_has_violation_rate", "prev_has_critical_rate", "prev_fail_rate",
        "days_since_prev_inspection", "prev_critical_count_last_3", "prev_violation_count_last_3"
    ]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0)

    # --------------------------------
    # ZIP-level lag features (no leak)
    # --------------------------------
    if "zip" in df.columns:
        # normalize zip to 5 digits
        df["zip"] = df["zip"].astype(str).str.extract(r"(\d{5})", expand=False)

        # IMPORTANT: ensure time ordering for expanding stats
        df = df.sort_values(["inspection_date", "inspection_id"]).copy()

        gz = df.groupby("zip", group_keys=False)
        df["zip_prev_critical_rate"] = gz["has_critical_violation"].apply(lambda s: s.shift(1).expanding().mean()).fillna(0)
        df["zip_prev_fail_rate"] = gz["is_fail"].apply(lambda s: s.shift(1).expanding().mean()).fillna(0)
        df["zip_prev_count"] = gz.cumcount()
    else:
        df["zip_prev_critical_rate"] = 0
        df["zip_prev_fail_rate"] = 0
        df["zip_prev_count"] = 0

    # Select gold columns
    gold_cols = [
        "inspection_id", "restaurant_key", "inspection_date",
        "facility_type", "zip", "latitude", "longitude",
        "risk_num",
        "zip_prev_critical_rate", "zip_prev_fail_rate", "zip_prev_count",
        "prev_inspection_count",
        "prev_has_violation_rate", "prev_has_critical_rate", "prev_fail_rate",
        "days_since_prev_inspection",
        "prev_critical_count_last_3", "prev_violation_count_last_3",
        # targets
        "has_critical_violation", "is_fail"
    ]
    gold_cols = [c for c in gold_cols if c in df.columns]
    gold = df[gold_cols].copy()

    # Optional: drop non-restaurants
    if "facility_type" in gold.columns:
        gold = gold[gold["facility_type"].fillna("").str.lower().str.contains("restaurant")].copy()

    # Upload gold (overwrite v1)
    gold_path = f"gold/features/dt={dt}/features_inspections_v1.parquet"
    out_bytes = gold.to_parquet(index=False)
    upload_bytes(conn_str, container, gold_path, out_bytes)

    print(f"Gold rows: {len(gold):,}")
    print("Done.")

if __name__ == "__main__":
    main()
