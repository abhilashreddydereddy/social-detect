from __future__ import annotations

"""Deprecated stub — use training.image_branch.scripts.train instead."""

import runpy
import sys


def main() -> None:
    print(
        "train_stub.py is deprecated. Forwarding to training.image_branch.scripts.train …",
        file=sys.stderr,
    )
    # Preserve argv (--config etc.)
    sys.argv[0] = "training.image_branch.scripts.train"
    runpy.run_module("training.image_branch.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
