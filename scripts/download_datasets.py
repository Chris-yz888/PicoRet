#!/usr/bin/env python3
"""Download the public benchmark archives used by PicoRet."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import gdown


ARCHIVES = {
    "lolv1": ("1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu", "LOLv1.zip"),
    "lolv2": ("1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw", "LOLv2.zip"),
    "dicm": ("0B_FjaR958nw_eHhvUUN6MzBCQXc", "DICM.zip"),
    "mef": ("0B_FjaR958nw_NmdONjU1eDJsVW8", "MEF.zip"),
    "npe": ("0B_FjaR958nw_VC1ueFBVWXhkUTA", "NPE.zip"),
    "lime": ("0B_FjaR958nw_QUpMeFlmYW5MUVE", "LIME.zip"),
}

DIRECTORY_NAMES = {"lolv1": "LOLv1", "lolv2": "LOLv2", "dicm": "DICM", "mef": "MEF", "npe": "NPE", "lime": "LIME"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=[*ARCHIVES, "all"])
    parser.add_argument("--output", default="data")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    selected = list(ARCHIVES) if "all" in args.datasets else args.datasets
    root = Path(args.output)
    downloads = root / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name in selected:
        file_id, filename = ARCHIVES[name]
        archive = downloads / filename
        if not archive.exists():
            gdown.download(id=file_id, output=str(archive), quiet=False)
        destination = root / DIRECTORY_NAMES[name]
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        print(f"{name}: extracted to {destination}")
        if not args.keep_archives:
            archive.unlink()
    print("See data/README.md for the paired-directory layout expected by training and evaluation.")


if __name__ == "__main__":
    main()
