#!/usr/bin/env python3
"""Panaesthesis desktop launcher: open the native app window.

Double-clicked via "Start Panaesthesis.bat". Launches the Tkinter desktop app
(hodos_monitor.app) from this checkout. Any startup failure is written to
console-jobs/app_launch.log so a windowless launch still leaves a trace.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _log(msg):
    try:
        d = REPO / "console-jobs"
        d.mkdir(exist_ok=True)
        (d / "app_launch.log").write_text(msg)
    except Exception:
        pass


def main():
    try:
        from hodos_monitor.app import main as app_main
        return app_main()
    except Exception:
        _log(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
