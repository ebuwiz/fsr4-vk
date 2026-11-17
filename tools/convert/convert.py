# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.

"""Prepare the FSR 4.1.2 INT8 and FP8 asset pack."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from setup import main


if __name__ == "__main__":
    main()
