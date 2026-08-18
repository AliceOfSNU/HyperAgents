"""Runs export + upload on a loop. This is the thing you actually leave
running (nohup ... &) to keep the dashboard current.

    venv_nat/bin/python3 dashboard/scripts/run_loop.py

Export runs every EXPORT_INTERVAL_S; each JSON file gets re-uploaded every
cycle (cheap, small files). Log files are large and change less usefully
often, so drive_upload.py internally throttles each one to at most once per
LOG_UPLOAD_INTERVAL_S regardless of how often this loop runs.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import export_all  # noqa: E402
from drive_upload import upload_all  # noqa: E402

EXPORT_INTERVAL_S = 2 * 60


def main():
    while True:
        started = time.time()
        try:
            export_all()
            upload_all()
        except Exception:
            print("=== run_loop cycle failed, will retry next interval ===", flush=True)
            traceback.print_exc()
        elapsed = time.time() - started
        print(f"Cycle done in {elapsed:.1f}s", flush=True)
        time.sleep(max(0, EXPORT_INTERVAL_S - elapsed))


if __name__ == "__main__":
    main()
