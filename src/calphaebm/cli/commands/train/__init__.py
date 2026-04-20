"""Train command package entry.

This module provides the train command interface.
"""

from .config import build_parser
from .train_main import run


def register(subparsers):
    """Register train command with subparsers."""
    parser = build_parser(subparsers)
    parser.set_defaults(func=run)
    return parser


# Maintain backward compatibility for main.py that expects add_parser
add_parser = register
