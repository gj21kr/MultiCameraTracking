"""Robust extractor for the WILDTRACK >4GB non-zip64 zip.

The official Wildtrack_dataset_full.zip is ~6.8 GB but was written WITHOUT zip64,
so the central-directory header offsets wrap at 4 GB and standard unzip tools
report "damaged archive". The local file headers and per-entry compressed sizes
(read from the central directory) are intact, so we ignore the broken offsets and
walk local headers sequentially from byte 0.

Usage:
    python scripts/extract_wildtrack.py <zip> <out_dir> [--filter Image_subsets]
"""

import sys
import zlib
import struct
import argparse
import zipfile
from pathlib import Path

LFH_SIG = b"PK\x03\x04"
CDH_SIG = b"PK\x01\x02"
DD_SIG = b"PK\x07\x08"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("out_dir")
    ap.add_argument("--filter", default="", help="only extract paths containing this substring")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (0=all)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)

    # Ordered metadata from the (readable) central directory.
    zf = zipfile.ZipFile(args.zip)
    infos = zf.infolist()
    meta = [
        (zi.filename, zi.compress_size, zi.compress_type, zi.flag_bits)
        for zi in infos
    ]
    zf.close()
    print(f"central directory entries: {len(meta)}")

    written = 0
    skipped = 0
    with open(args.zip, "rb") as f:
        for idx, (name, csize, method, flags) in enumerate(meta):
            sig = f.read(4)
            if sig == CDH_SIG:
                print("reached central directory; done.")
                break
            if sig != LFH_SIG:
                # Try to resync to next local header.
                raise RuntimeError(
                    f"entry {idx} ({name}): expected local header, got {sig.hex()} "
                    f"at pos {f.tell() - 4}"
                )
            # Local file header (after the 4-byte sig): 26 bytes fixed.
            hdr = f.read(26)
            (_ver, _flg, _meth, _t, _d, _crc, _csz, _usz, nlen, elen) = struct.unpack(
                "<HHHHHIIIHH", hdr
            )
            f.read(nlen)            # filename
            f.read(elen)            # extra field
            data = f.read(csize)    # compressed payload (size from central dir)

            # Optional data descriptor when flag bit 3 is set.
            if flags & 0x08:
                peek = f.read(4)
                if peek == DD_SIG:
                    f.read(12)      # crc(4)+csize(4)+usize(4)
                else:
                    f.read(8)       # crc(4)+csize(4); we already consumed 4 (usize tail)

            want = (not args.filter) or (args.filter in name)
            if not want or name.endswith("/"):
                skipped += 1
                continue

            if method == 0:
                raw = data
            elif method == 8:
                raw = zlib.decompress(data, -15)
            else:
                raise RuntimeError(f"unsupported method {method} for {name}")

            dest = out_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            written += 1
            if written % 200 == 0:
                print(f"  extracted {written} files...")
            if args.limit and written >= args.limit:
                print("hit --limit; stopping.")
                break

    print(f"DONE: wrote {written} files, skipped {skipped}.")


if __name__ == "__main__":
    main()
