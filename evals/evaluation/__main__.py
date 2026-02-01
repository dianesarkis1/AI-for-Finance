"""
Entry point for running model_run as a module.

This allows the model_run script to be invoked using:
    python -m evals.evaluation.model_run
"""
# Import the main function and run it
import sys
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.evaluation.model_run import main

if __name__ == "__main__":
    main()
