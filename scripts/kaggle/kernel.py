import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import rich_click as click

from scripts.kaggle.shell import kaggle


def pip_freeze_on_kaggle() -> str:
    """Run `pip freeze` in a private Kaggle script, so it describes the image notebooks get."""
    match = re.search(r"username: (\S+)", kaggle("config", "view"))
    if match is None:
        raise click.ClickException("Could not determine Kaggle username from `kaggle config view`.")
    user = match[1]
    kernel = f"{user}/birdclef-image-freeze"
    metadata = {
        "id": kernel,
        "title": "birdclef-image-freeze",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": False,
        "enable_gpu": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        shutil.copyfile(Path(__file__).with_name("freeze_image.py"), folder / "script.py")
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        kaggle("kernels", "push", "-p", str(folder))
        wait_until_complete(kernel)
        kaggle("kernels", "output", kernel, "-p", str(folder / "out"))
        return (folder / "out" / "image-freeze.txt").read_text(encoding="utf-8")


def wait_until_complete(kernel: str) -> None:
    for _ in range(60):
        status = kaggle("kernels", "status", kernel).upper()
        if "COMPLETE" in status:
            return
        if "ERROR" in status or "CANCEL" in status:
            raise click.ClickException(status)
        time.sleep(10)
    raise click.ClickException(f"{kernel} did not finish in 10 minutes.")
