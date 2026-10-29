"""Download the pinned, self-contained Perch v2 ONNX backbone (about 45 MB)."""

import shutil
from pathlib import Path
from urllib.request import urlopen

import rich_click as click

from birdclef.cli_common import CONTEXT_SETTINGS
from birdclef.config import get_config
from birdclef.models.classifier.perch.encoder import file_sha256

REVISION = "7fb3a9838cd7c9577d69c461947c71e5e6687030"
SHA256 = "97a0f6c20e7892594c4811156b51d5a27172af44c54fde158e892ede6893b83c"
URL = (
    f"https://huggingface.co/biodiversica/Perch-onnx-backbone/resolve/{REVISION}/"
    "perch_v2_backbone.onnx"
)


def download_perch(path: Path) -> Path:
    if path.is_file() and file_sha256(path) == SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if file_sha256(temporary) != SHA256:
            raise ValueError("Downloaded Perch encoder checksum mismatch.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--output", type=click.Path(path_type=Path, dir_okay=False))
def main(output: Path | None) -> None:
    """Fetch public weights once; training and inference use local files."""
    path = output or get_config(verbose=False).data.cache_dir / "models/perch_v2_backbone.onnx"
    click.echo(f"Perch v2: {download_perch(path)}")


if __name__ == "__main__":
    main()
