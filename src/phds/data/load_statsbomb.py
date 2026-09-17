"""Download and cache StatsBomb Open Data (events + 360 freeze frames).

Data: https://github.com/statsbomb/open-data. Non-commercial use, attribution
required (see ATTRIBUTION.md).

We fetch raw JSON straight from GitHub instead of using statsbombpy, because we
want the untouched 360 payloads and a transparent on-disk cache. Files are
stored gzip-compressed (~10x smaller) under data/raw/statsbomb/.

A competition's `match_available_360` flag doesn't mean every match has 360
data. We filter on the per-match `match_status_360 == "available"`.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "statsbomb"


def _fetch_json(url: str, retries: int = 8, timeout: int = 120):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))  # flaky DNS/network: back off up to 1 min
            continue
        return r.json()  # a complete response that fails to parse is corrupt upstream: no retry


def cached_json(rel_path: str, refresh: bool = False):
    """Return the JSON at `BASE_URL/rel_path`, using a gzip cache in RAW_DIR."""
    path = RAW_DIR / (rel_path + ".gz")
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = _fetch_json(f"{BASE_URL}/{rel_path}")  # raises ValueError on corrupt upstream JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)  # atomic: an interrupted download never leaves a half-written cache file
    return data


def competitions_with_360(refresh: bool = False) -> pd.DataFrame:
    comps = pd.DataFrame(cached_json("competitions.json", refresh=refresh))
    return comps[comps["match_available_360"].notna()].reset_index(drop=True)


def matches_with_360(refresh: bool = False) -> pd.DataFrame:
    """One row per match with 360 data, across all competitions that have any."""
    frames = []
    for c in competitions_with_360(refresh).itertuples():
        m = pd.json_normalize(
            cached_json(f"matches/{c.competition_id}/{c.season_id}.json", refresh=refresh), sep="_"
        )
        m["competition_gender"] = c.competition_gender
        frames.append(m)
    matches = pd.concat(frames, ignore_index=True)
    return matches[matches["match_status_360"] == "available"].reset_index(drop=True)


def download_all(max_workers: int = 8) -> pd.DataFrame:
    """Cache events and 360 frames for every match with 360 data. Safe to re-run."""
    matches = matches_with_360()
    jobs = [f"{kind}/{mid}.json" for mid in matches["match_id"] for kind in ("events", "three-sixty")]
    todo = [j for j in jobs if not (RAW_DIR / (j + ".gz")).exists()]
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Discard the parsed JSON: keeping it in the futures holds every file in memory.
        futures = {ex.submit(lambda j: cached_json(j) and None, j): j for j in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - report every failure at the end, don't abort
                failures.append((futures[fut], repr(e)))
    if failures:
        print(f"{len(failures)} downloads failed (re-run to retry):")
        for j, e in failures[:10]:
            print("  ", j, e)
    return matches


if __name__ == "__main__":
    m = download_all()
    print(f"{len(m)} matches with 360 data cached in {RAW_DIR}")
