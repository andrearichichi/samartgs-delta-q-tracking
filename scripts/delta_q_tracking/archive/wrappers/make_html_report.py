from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.reporting.make_html_report import main


if __name__ == "__main__":
    print("This legacy wrapper has moved to scripts/delta_q_tracking/wrappers/make_html_report.py")
    print("Preferred command: python scripts/delta_q_tracking/reporting/make_html_report.py")
    main()
