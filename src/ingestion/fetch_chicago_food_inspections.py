import os, json, time, random
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from sodapy import Socrata
from azure.storage.blob import BlobServiceClient

def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def upload_bytes(conn_str: str, container: str, blob_path: str, payload: bytes) -> None:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)
    cc.upload_blob(name=blob_path, data=payload, overwrite=True)
    print(f"Uploaded: {container}/{blob_path}")

def socrata_get_with_retry(client: Socrata, dataset_id: str, **kwargs) -> List[Dict]:
    max_retries = 6
    for attempt in range(1, max_retries + 1):
        try:
            return client.get(dataset_id, **kwargs)
        except Exception as e:
            sleep_s = min(60, (2 ** attempt) + random.random())
            print(f"⚠️ Socrata request failed (attempt {attempt}/{max_retries}). "
                  f"Sleeping {sleep_s:.1f}s. Error: {type(e).__name__}: {e}")
            time.sleep(sleep_s)
    raise RuntimeError("Exceeded max retries")

def month_ranges(start: str, end: str):
    # start/end are YYYY-MM-DD
    cur = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    while cur < end_dt:
        # next month
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1)

        yield cur.date().isoformat(), min(nxt, end_dt).date().isoformat()
        cur = nxt

def fetch_window_all(client: Socrata, dataset_id: str, where: str, batch_size: int = 10_000) -> List[Dict]:
    offset = 0
    out: List[Dict] = []
    while True:
        batch = socrata_get_with_retry(
            client,
            dataset_id,
            where=where,
            limit=batch_size,
            offset=offset,
            order="inspection_date, inspection_id"
        )
        if not batch:
            break
        out.extend(batch)
        offset += batch_size
    return out

def main():
    load_dotenv()

    # Socrata
    domain = get_env("SOCRATA_DOMAIN")
    dataset_id = get_env("SOCRATA_DATASET_ID")
    app_token = os.getenv("SOCRATA_APP_TOKEN")

    # Azure
    conn_str = get_env("AZURE_STORAGE_CONNECTION_STRING")
    container = get_env("AZURE_BLOB_CONTAINER")  # keep your naming
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Choose a realistic modeling window (adjust as you like)
    # 5 years is usually plenty for strong signals + faster iteration
    START = "2021-01-01"
    END = datetime.now().date().isoformat()

    client = Socrata(domain, app_token)

    total = 0
    for s, e in month_ranges(START, END):
        where = f"inspection_date >= '{s}T00:00:00.000' AND inspection_date < '{e}T00:00:00.000'"
        print(f"\n=== Fetching window {s} -> {e} ===")
        rows = fetch_window_all(client, dataset_id, where=where, batch_size=10_000)
        if not rows:
            print("No rows in this window.")
            continue

        total += len(rows)
        print(f"Window rows: {len(rows):,} | Running total: {total:,}")

        # Upload this window as its own bronze part (JSONL)
        jsonl = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
        blob_path = f"bronze/food_inspections/dt={ts}/by_month/{s}_to_{e}.jsonl"
        upload_bytes(conn_str, container, blob_path, jsonl)

    # Write a tiny manifest you can use later in cleaning
    manifest = {
        "pulled_at_utc": ts,
        "start": START,
        "end": END,
        "notes": "Bronze data pulled in monthly chunks to avoid large offset pagination failures."
    }
    upload_bytes(conn_str, container, f"bronze/food_inspections/dt={ts}/manifest.json",
                 json.dumps(manifest, indent=2).encode("utf-8"))

    print(f"\nDone. Total rows fetched/uploaded: {total:,}")


if __name__ == "__main__":
    main()
