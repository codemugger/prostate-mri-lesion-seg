#!/usr/bin/env python3
"""Generate audit reports, a mixed SGH/ProstateX manifest, and locked splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.cohort import main


if __name__ == "__main__":
    raise SystemExit(main())
