#!/usr/bin/env/python3
"""Assuming samples are stored in .txt or .jsonl-files, this script can be used to randomly split the samples into half
and save them in two separate files (Assuming the given file contains one sample per line).
"""

import argparse
import random


def main():
    args = parse_args()
    lines = open(args.file, encoding="utf-8").readlines()
    random.shuffle(lines)
    half = int(len(lines) / 2)
    open(args.file + "_train", "w", encoding="utf-8").writelines(lines[:half])
    open(args.file + "_test", "w", encoding="utf-8").writelines(lines[half:])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly split text file into training and test files"
    )
    parser.add_argument("file", help="filename")
    return parser.parse_args()


if __name__ == "__main__":
    main()
