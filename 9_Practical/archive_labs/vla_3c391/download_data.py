#!/usr/bin/env python3
"""Download and verify the external NRAO 3C391 tutorial Measurement Set."""

import argparse
import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE_URL = "https://casa.nrao.edu/Data/EVLA/3C391/3c391_ctm_mosaic_10s_spw0.ms.tgz"
ARCHIVE_NAME = "3c391_ctm_mosaic_10s_spw0.ms.tgz"
MEASUREMENT_SET_NAME = "3c391_ctm_mosaic_10s_spw0.ms"
EXPECTED_BYTES = 3_271_507_505
EXPECTED_SHA256 = "9152b1ce8603a3b0ffd50ce3d3c57f53a82e6fde89fd6a4b5fa4f8c7d8481910"
CHUNK_BYTES = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path):
    path = Path(path)
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"archive length is {path.stat().st_size}, expected {EXPECTED_BYTES}"
        )
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"archive SHA-256 is {digest}, expected {EXPECTED_SHA256}")
    return digest


def download_archive(destination):
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_archive(destination)
        return destination

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > EXPECTED_BYTES:
        partial.unlink()
        offset = 0
    if offset == EXPECTED_BYTES:
        verify_archive(partial)
        os.replace(partial, destination)
        return destination
    request = Request(SOURCE_URL, headers={"Range": f"bytes={offset}-"})
    with urlopen(request) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            offset = 0
        with partial.open(mode) as output:
            transferred = offset
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                transferred += len(chunk)
                print(
                    f"\rDownloaded {transferred / 1024**3:.2f} / "
                    f"{EXPECTED_BYTES / 1024**3:.2f} GiB",
                    end="",
                    flush=True,
                )
    print()
    verify_archive(partial)
    os.replace(partial, destination)
    return destination


def extract_archive(archive, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_set = output_dir / MEASUREMENT_SET_NAME
    if measurement_set.is_dir():
        return measurement_set
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            target = (output_dir / member.name).resolve()
            if output_dir != target and output_dir not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdev() or member.issym() or member.islnk():
                raise ValueError(f"unsupported archive member: {member.name}")
        stream.extractall(output_dir)
    if not measurement_set.is_dir():
        raise ValueError(f"archive did not create {measurement_set}")
    return measurement_set


def parse_args():
    default = Path(tempfile.gettempdir()) / "vla_3c391_archive_lab"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    archive = args.output_dir / ARCHIVE_NAME
    if args.verify_only:
        verify_archive(archive)
        print(f"Verified {archive}")
        return
    archive = download_archive(archive)
    if args.download_only:
        print(f"Verified {archive}")
        return
    measurement_set = extract_archive(archive, args.output_dir)
    print(f"Ready: {measurement_set}")


if __name__ == "__main__":
    main()
