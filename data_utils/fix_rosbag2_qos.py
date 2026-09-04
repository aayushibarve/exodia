#!/usr/bin/env python3
"""
fix_rosbag2_qos.py

Normalizes a rosbag2 metadata.yaml so that older rosbag2_storage builds
(e.g. ROS 2 Humble) can read bags whose offered_qos_profiles were written
in the newer Jazzy-era format.

Fixes two independent problems, either or both of which may be present:

  1. STRUCTURE: offered_qos_profiles must be a single YAML *string* value
     containing embedded YAML text, not literal nested keys. Some tools /
     manual edits "unwrap" it into nested keys, which breaks parsing.

  2. CONTENT: newer rosbag2 (metadata version 9+) encodes the QoS enum
     fields (history, reliability, durability, liveliness) as human-
     readable strings ("keep_last", "reliable", ...). Older rosbag2_storage
     builds (e.g. Humble) only understand the original numeric codes.
     This converts any string enum values back to their numeric codes.

Usage:
    python3 fix_rosbag2_qos.py path/to/bag_folder
    python3 fix_rosbag2_qos.py path/to/bag_folder/metadata.yaml

By default this edits the file in place, after saving a timestamped
backup alongside it. Use --dry-run to preview without writing anything,
or --output to write to a different path instead of overwriting.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml

HISTORY = {"system_default": 0, "keep_last": 1, "keep_all": 2, "unknown": 3}
RELIABILITY = {
    "system_default": 0, "reliable": 1, "best_effort": 2,
    "unknown": 3, "best_available": 4,
}
DURABILITY = {
    "system_default": 0, "transient_local": 1, "volatile": 2,
    "unknown": 3, "best_available": 4,
}
LIVELINESS = {
    "system_default": 0, "automatic": 1, "manual_by_topic": 3,
    "unknown": 4, "best_available": 5,
}

ENUM_MAPS = {
    "history": HISTORY,
    "reliability": RELIABILITY,
    "durability": DURABILITY,
    "liveliness": LIVELINESS,
}


def resolve_metadata_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "metadata.yaml")
        if not os.path.isfile(candidate):
            sys.exit(f"ERROR: no metadata.yaml found in directory '{path}'")
        return candidate
    if os.path.isfile(path):
        return path
    sys.exit(f"ERROR: path '{path}' does not exist")


def normalize_qos_list(qos_list, changes):
    for profile in qos_list:
        for field, mapping in ENUM_MAPS.items():
            val = profile.get(field)
            if isinstance(val, str):
                if val not in mapping:
                    sys.exit(
                        f"ERROR: unrecognized {field} value '{val}' - "
                        f"expected one of {list(mapping)}"
                    )
                profile[field] = mapping[val]
                changes["numeric_conversions"] += 1
    return qos_list


def fix_metadata(data, changes):
    topics = data["rosbag2_bagfile_information"]["topics_with_message_count"]

    for entry in topics:
        tm = entry["topic_metadata"]
        qos = tm.get("offered_qos_profiles")
        if qos is None:
            continue

        was_string = isinstance(qos, str)
        qos_list = yaml.safe_load(qos) if was_string else qos

        if not was_string:
            changes["restructured"] += 1

        qos_list = normalize_qos_list(qos_list, changes)

        tm["offered_qos_profiles"] = yaml.dump(
            qos_list, default_flow_style=False, sort_keys=False
        )

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to a bag folder or its metadata.yaml")
    parser.add_argument("--output", help="Write to this path instead of overwriting in place")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would change without writing any file")
    parser.add_argument("--no-backup", action="store_true",
                         help="Skip creating a backup when overwriting in place")
    args = parser.parse_args()

    src = resolve_metadata_path(args.path)
    print(f"Reading: {src}")

    with open(src) as f:
        data = yaml.safe_load(f)

    changes = {"restructured": 0, "numeric_conversions": 0}
    data = fix_metadata(data, changes)

    print(f"  Topics restructured (string->nested fix): {changes['restructured']}")
    print(f"  QoS enum values converted to numeric:      {changes['numeric_conversions']}")

    if changes["restructured"] == 0 and changes["numeric_conversions"] == 0:
        print("Nothing to fix - offered_qos_profiles already looks normalized.")

    if args.dry_run:
        print("\n--dry-run set: no file written.")
        return

    dest = args.output if args.output else src

    if dest == src and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{src}.bak_{ts}"
        shutil.copy2(src, backup)
        print(f"Backup saved: {backup}")

    with open(dest, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=1000)

    print(f"Wrote: {dest}")


if __name__ == "__main__":
    main()