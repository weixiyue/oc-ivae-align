"""Compatibility entry point; plots and metrics use scripts.run_eval exclusively."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import main

if __name__ == '__main__':
    main()
