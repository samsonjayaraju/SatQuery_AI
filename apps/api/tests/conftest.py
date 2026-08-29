from __future__ import annotations

import sys
import os
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
os.environ["MOCK_MODE"] = "true"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
