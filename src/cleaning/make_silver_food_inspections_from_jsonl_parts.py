import os
import io
import re
import json
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def normalize_results(x: str) -> str:
    if not isinstance(x, str):
        return "UNKNOWN"
    s = x.strip().lower()
    if "pass w" in s:
        return "PASS_WITH_CONDITIONS"
    if s == "pass":
        return "PASS"
    if s == "fail":
        return "FAIL"
    if "out of business" in s:
        return "OUT_OF_BUSINESS"
    if "no entry" in s:
        return "NO_ENTRY"
    if "not located" in s:
        return "NOT_LOCATED"
    return s.upper().replace(" ", "_")

def parse_risk(x: str):
    if not isinstance(x, str):
        return None
    m = re.search(r"risk\s*(\d+)", x.lower())
    return int(m.group(1)) if m else None

def has_any_violation(v: str) -> int:
    return int(isinstance(v, str) and v.strip() != "")


def has_critical_violation(v: str) -> int:
    if not isinstance(v, str) or not v.strip():
        return 0
    critical_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 29, 33, 36, 38}
    nums = set(int(n) for n in re.findall(r"(?m)^\s*(\d+)\.", v))
    if nums.intersection(critical_nums):
        return 1
    keywords = ["rodent", "roach", "sewage", "vomit", "feces", "no hot water", "imminent health hazard"]
    lv = v.lower()
    return int(any(k in lv for k in keywords))

def main():
    load_dotenv()

    conn_str = get_env("AZURE_STORAGE_CONNECTION_STRING")
    container = get_env("AZURE_BLOB_CONTAINER")

    # use the dt from your by_date run
    dt = "20260217T153907Z"

    bronze_prefix = f"bronze/food_inspections/dt={dt}/by_month/"
    silver_path = f"silver/food_inspections/dt={dt}/food_inspections_silver.parquet"

    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)

    # List all jsonl parts
    part_names: List[str] = [b.name for b in cc.list_blobs(name_starts_with=bronze_prefix) if b.name.endswith(".jsonl")]
    part_names.sort()
    if not part_names:
        raise RuntimeError(f"No JSONL parts found under {bronze_prefix}")

    dfs = []
    total_rows = 0

    for name in part_names:
        print(f"Downloading part: {name}")
        blob_client = bsc.get_blob_client(container=container, blob=name)
        raw = blob_client.download_blob().readall().decode("utf-8")

        # JSONL -> list of dicts
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows:
            continue

        df = pd.DataFrame(rows)
        total_rows += len(df)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded rows from parts: {total_rows:,} | concat rows: {len(df):,}")

    # --- Clean + normalize ---
    df.columns = [c.strip().lower() for c in df.columns]

    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.extract(r"(\d{5})", expand=False)

    for col in ["dba_name", "aka_name", "address", "facility_type", "city", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["results_norm"] = df["results"].apply(normalize_results) if "results" in df.columns else "UNKNOWN"
    df["risk_num"] = df["risk"].apply(parse_risk) if "risk" in df.columns else None
    df["has_violation"] = df["violations"].apply(has_any_violation) if "violations" in df.columns else 0
    df["has_critical_violation"] = df["violations"].apply(has_critical_violation) if "violations" in df.columns else 0

    df["is_fail"] = (df["results_norm"] == "FAIL").astype(int)

    # Drop obvious duplicates
    if "inspection_id" in df.columns:
        df = df.sort_values(["inspection_id", "inspection_date"], ascending=[True, False])
        df = df.drop_duplicates(subset=["inspection_id"], keep="first")

    # Upload silver parquet
    out_bytes = df.to_parquet(index=False)
    cc.upload_blob(name=silver_path, data=out_bytes, overwrite=True)
    print(f"Uploaded: {container}/{silver_path}")
    print(f"Silver rows: {len(df):,}")


if __name__ == "__main__":
    main()
