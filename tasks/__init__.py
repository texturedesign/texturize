# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import sys
from invoke import task


@task
def test(cmd):
    """Run the automated test suite using `pytest`.

    Any arguments that are after `--` are passed directly to pytest, useful for:
        * specifying which files or patterns to test
        * configuring the various plugins installed

    """
    try:
        idx = sys.argv.index("--")
        extra = " ".join(sys.argv[idx+1:])
    except ValueError:
        extra = ""

    cmd.run("poetry run pytest -c tasks/pytest.ini " + extra)
