import subprocess
import sys

import anomx


def test_version_is_exposed():
    assert anomx.__version__ == "0.2.17"


def test_cli_version_runs():
    result = subprocess.run(
        [sys.executable, "-m", "anomx", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "anomx 0.2.17" in result.stdout
