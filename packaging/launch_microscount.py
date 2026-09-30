"""Entry point used by PyInstaller for the frozen application."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from microscount.cli import main

    sys.exit(main())
