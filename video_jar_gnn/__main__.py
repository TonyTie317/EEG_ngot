"""Command dispatcher for ``python -m video_jar_gnn``."""

from __future__ import annotations

import sys


HELP = """\
Video JAR GNN

Usage:
  python -m video_jar_gnn prepare [options]
  python -m video_jar_gnn extract [options]
  python -m video_jar_gnn train   [options]
  python -m video_jar_gnn train-advanced [options]
  python -m video_jar_gnn train-classical [options]
  python -m video_jar_gnn train-video-features [options]
  python -m video_jar_gnn train-expression [options]
  python -m video_jar_gnn audit-expression [options]

Run a command with --help for its full arguments.
"""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(HELP)
        return 0
    command, rest = arguments[0], arguments[1:]
    if command == "prepare":
        from .manifest import main as command_main
    elif command == "extract":
        from .extract import main as command_main
    elif command == "train":
        from .train import main as command_main
    elif command == "train-advanced":
        from .train_advanced import main as command_main
    elif command == "train-classical":
        from .train_classical import main as command_main
    elif command == "train-video-features":
        from .train_classical import video_feature_main as command_main
    elif command == "train-expression":
        from .train_expression import main as command_main
    elif command == "audit-expression":
        from .expression_audit import main as command_main
    else:
        print(f"Unknown command: {command!r}\n\n{HELP}", file=sys.stderr)
        return 2
    return int(command_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
