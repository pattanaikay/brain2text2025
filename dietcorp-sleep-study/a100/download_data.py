"""
a100/download_data.py
---------------------
Download the T15 copy-task dataset from Dryad (doi:10.5061/dryad.dncjsxm85; Card et al., NEJM
2024) and unzip the per-session HDF5 files (neural + seq_class_ids + transcription) to data/sessions/.

This is the faithful data source that unblocks ALL conditions C0-C4 — the local
preprocessed_data.h5 has neural+transcription but NO seq_class_ids. The same files are produced
by the official nejm-brain-to-text `download_data.py`; this is a dependency-free standalone
equivalent (stdlib urllib + zipfile only).

NOTE: dncjsxm85 is the T15 set. The older doi:10.5061/dryad.x69p8czpq is T12 (Willett 2023) and
is NOT this data.

Usage:
  python a100/download_data.py --out data/sessions
  python a100/download_data.py --out data/sessions --file t15_copyTask_neuralData.zip
  python a100/download_data.py --url <direct_zip_url> --out data/sessions   # manual override
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile

DRYAD = "https://datadryad.org"
API = f"{DRYAD}/api/v2"
DOI = "doi:10.5061/dryad.dncjsxm85"


def _get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "dietcorp-sleep-study"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _latest_version_files(doi):
    """Resolve the latest dataset version and return its file objects (version-proof)."""
    enc = urllib.parse.quote(doi, safe="")
    versions = _get_json(f"{API}/datasets/{enc}/versions")
    vlist = versions["_embedded"]["stash:versions"]
    files_href = vlist[-1]["_links"]["stash:files"]["href"]   # /api/v2/versions/<id>/files
    files = _get_json(f"{DRYAD}{files_href}?per_page=100")
    return files["_embedded"]["stash:files"]


def _download(url, dest):
    print(f"[download] {url}\n        -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "dietcorp-sleep-study"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)                            # 1 MB chunks (file is ~11 GB)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.1f}%)",
                      end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sessions")
    ap.add_argument("--file", default="t15_copyTask_neuralData.zip",
                    help="substring of the Dryad file name to fetch")
    ap.add_argument("--url", default=None, help="direct zip URL (skips the Dryad API lookup)")
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, "_t15_download.zip")

    if args.url:
        _download(args.url, zip_path)
    else:
        files = _latest_version_files(DOI)
        match = next((f for f in files if args.file in f["path"]), None)
        if match is None:
            print(f"No Dryad file matching '{args.file}'. Available:")
            for f in files:
                print(f"  {f['path']} ({f.get('size', 0)/1e9:.2f} GB)")
            sys.exit(1)
        href = match["_links"]["stash:download"]["href"]      # /api/v2/files/<id>/download
        print(f"[dryad] {match['path']} ({match.get('size', 0)/1e9:.2f} GB)")
        _download(f"{DRYAD}{href}", zip_path)

    print(f"[unzip] -> {args.out}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    if not args.keep_zip:
        os.remove(zip_path)
    print(f"[done] data under {args.out} (expect per-session t15.YYYY.MM.DD/data_*.hdf5)")
    print(f"Next: python a100/prepare_data.py --data {args.out}")


if __name__ == "__main__":
    main()
