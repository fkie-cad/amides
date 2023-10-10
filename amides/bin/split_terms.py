import argparse
import random


def main():
    args = parse_args()
    lines = open(args.file).readlines()
    random.shuffle(lines)
    half = int(len(lines) / 2)
    open(args.file + "_train", "w").writelines(lines[:half])
    open(args.file + "_test", "w").writelines(lines[half:])


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly split text file into training and test files")
    parser.add_argument("file", help="filename")
    return parser.parse_args()


if __name__ == "__main__":
    main()
