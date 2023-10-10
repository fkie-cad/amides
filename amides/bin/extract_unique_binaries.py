#!/usr/bin/env python3

import re
import argparse
import json


def extract_unique_binaries(cmdlines_file, out_file):
    unique_binaries = set()
    binary_regex = re.compile(r"(\")?((1)?.+?(?=\")|(?<=^)\S+|(?<=\s)\S+)")

    with open(cmdlines_file, "r", encoding="utf-8") as in_file:
        for line in in_file:
            decoded = json.loads(line)
            binary_name = re.findall(binary_regex, decoded)[0]
            unique_binaries.add(binary_name)

    with open(out_file, "w") as out_file:
        for binary in sorted(unique_binaries):
            out_file.write(f"{binary}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmdlines_file", type=str, action="store")
    parser.add_argument("out_file", type=str, action="store")

    args = parser.parse_args()

    extract_unique_binaries(args.cmdlines_file, args.out_file)


if __name__ == "__main__":
    main()
