#!/usr/bin/env python3
"""Run strict new-campaign preflight using the normal launcher options."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scsf.dsspmi_campaign import main
sys.argv.insert(1, "preflight")
if __name__ == "__main__":
    main()
