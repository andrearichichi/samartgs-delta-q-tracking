from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.diagnostics.manual_deltaq_check import main


if __name__ == "__main__":
    print("This legacy wrapper has moved to scripts/delta_q_tracking/wrappers/01_manual_deltaq_check.py")
    print("Preferred command: python scripts/delta_q_tracking/diagnostics/manual_deltaq_check.py ...")
    main()
