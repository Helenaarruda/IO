"""Master script: solve all equilibria and create all requested figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    run("solve_equilibria.py")
    run("question_iii.py")
    run("plot_market_shares.py", "--case", "both", "--mu", "0")
    run("estimation.py")
    print("\nAll computational outputs have been generated.")


if __name__ == "__main__":
    main()
