#!/usr/bin/env python3


import sys
import argparse
import functools

from amides.persist import Dumper, PersistError
from amides.utils import get_logger, set_log_level


set_log_level("info")
logger = get_logger(__name__)

dumper = None


def init_dumper(out_dir):
    global dumper

    try:
        if not dumper:
            dumper = Dumper(out_dir)

    except OSError as err:
        logger.err(err)


def load_pickled_object(path):
    try:
        return dumper.load_object(path)
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)


def save_object(obj):
    dumper.save_object(obj)


def rsetattribute(obj, attribute, value):
    pre, _, post = attribute.rpartition(".")

    return setattr(rgetattribute(obj, pre) if pre else obj, post, value)


def rgetattribute(obj, attribute, *args):
    def _getattr(obj, attribute):
        return getattr(obj, attribute, *args)

    return functools.reduce(_getattr, [obj] + attribute.split("."))


def empty_attribute(object_paths, attribute, out_dir):
    init_dumper(out_dir)

    for object_path in object_paths:
        obj = load_pickled_object(object_path)

        try:
            rsetattribute(obj, attribute, None)
        except AttributeError as err:
            logger.error(err)
            sys.exit(1)

        save_object(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object", type=str, action="append", help="Path of the pickled object file"
    )
    parser.add_argument(
        "--attribute",
        type=str,
        action="store",
        help="Attribute which should be removed",
    )
    parser.add_argument(
        "--out-dir", type=str, action="store", help="Attribute which should be removed"
    )

    args = parser.parse_args()

    empty_attribute(args.object, args.attribute, args.out_dir)


if __name__ == "__main__":
    main()
