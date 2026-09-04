"""Small command-line compatibility helpers for older Python environments."""

import argparse


def add_bool_argument(parser, option, default=False, help=None):
    """Add ``--flag`` and ``--no-flag`` on Python 3.8 and newer.

    ``argparse.BooleanOptionalAction`` was introduced in Python 3.9.  The
    project still has environments based on Python 3.8, so use equivalent
    ``store_true``/``store_false`` actions when the newer class is absent.
    """
    option = option if option.startswith("--") else f"--{option}"
    dest = option[2:].replace("-", "_")
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(option, action=action, default=default, help=help)
    else:
        parser.add_argument(option, dest=dest, action="store_true", default=default, help=help)
        parser.add_argument(f"--no-{option[2:]}", dest=dest, action="store_false")

