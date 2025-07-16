#!/usr/bin/env python3
"""
compare_json.py  path1.json  path2.json

Reads two JSON files and prints every difference:
  • keys present in only one file
  • values that differ (recursively)
  • list length or item differences
"""

import json
import sys
from pathlib import Path

def compare(a, b, path="root"):
    """Recursively compare two JSON-compatible Python objects."""
    if type(a) is not type(b):
        print(f"{path}: TYPE mismatch → {type(a).__name__} vs {type(b).__name__}")
        return

    if isinstance(a, dict):
        # keys only in a
        for k in a.keys() - b.keys():
            print(f"{path}: key present only in first → {k}")
        # keys only in b
        for k in b.keys() - a.keys():
            print(f"{path}: key missing in first  → {k}")
        # keys in both
        for k in a.keys() & b.keys():
            compare(a[k], b[k], f"{path}.{k}")

    elif isinstance(a, list):
        if len(a) != len(b):
            print(f"{path}: LIST length differs → {len(a)} vs {len(b)}")
        # compare common indices
        for i, (ai, bi) in enumerate(zip(a, b)):
            compare(ai, bi, f"{path}[{i}]")
        # report extra tail elements
        if len(a) > len(b):
            for i in range(len(b), len(a)):
                print(f"{path}[{i}]: element present only in first → {a[i]}")
        elif len(b) > len(a):
            for i in range(len(a), len(b)):
                print(f"{path}[{i}]: element missing in first  → {b[i]}")

    else:  # primitive types
        if a != b:
            print(f"{path}: VALUE differs → {a!r} vs {b!r}")

def main(p1: Path, p2: Path):
    with p1.open() as f:
        j1 = json.load(f)
    with p2.open() as f:
        j2 = json.load(f)
    import pprint
    pprint.pprint(j1)
    pprint.pprint(j2)
    compare(j1, j2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python compare_json.py path1.json path2.json")
    main(Path(sys.argv[1]), Path(sys.argv[2]))