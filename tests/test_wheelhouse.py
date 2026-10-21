import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import rich_click as click
from click.testing import CliRunner


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return {
        name: importlib.import_module(
            f"scripts.{name}" if name == "wheelhouse_common" else f"scripts.kaggle.{name}"
        )
        for name in ("wheelhouse_common", "image", "dataset", "kernel")
    }


@pytest.mark.parametrize("platform", ["colab", "kaggle"])
def test_wheelhouse_help_accepts_shared_rich_settings(scripts, platform):
    command = importlib.import_module(f"scripts.{platform}.wheelhouse").main
    result = CliRunner().invoke(command, ["-h"])
    assert result.exit_code == 0, result.output
    assert "Commands" in result.output


def test_snapshot_copies_standalone_script_and_reads_output(scripts, monkeypatch):
    kernel = scripts["kernel"]
    snapshot = "# Kaggle default image, Python 3.12.10, taken 2026-10-19\ntorch==2.8.0\n"

    def kaggle(*args):
        if args[:2] == ("config", "view"):
            return "username: test-user\n"
        if args[:2] == ("kernels", "push"):
            folder = Path(args[-1])
            assert (folder / "script.py").read_bytes() == Path(kernel.__file__).with_name(
                "freeze_image.py"
            ).read_bytes()
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            assert metadata["is_private"] and metadata["enable_gpu"]
            assert not metadata["enable_internet"]
        elif args[:2] == ("kernels", "status"):
            return 'Kernel has status "complete"'
        elif args[:2] == ("kernels", "output"):
            output = Path(args[-1])
            output.mkdir()
            (output / "image-freeze.txt").write_text(snapshot, encoding="utf-8")
        return ""

    monkeypatch.setattr(kernel, "kaggle", kaggle)
    assert kernel.pip_freeze_on_kaggle() == snapshot


def test_failed_kernel_stops_polling(scripts, monkeypatch):
    monkeypatch.setattr(scripts["kernel"], "kaggle", lambda *args: 'status "error"')
    with pytest.raises(click.ClickException, match="ERROR"):
        scripts["kernel"].wait_until_complete("user/kernel")


def test_resolution_and_download_target_snapshot_python(scripts, monkeypatch, tmp_path):
    image, dataset = scripts["image"], scripts["dataset"]
    freeze = tmp_path / "image-freeze.txt"
    freeze.write_text("# Kaggle default image, Python 3.12.10, taken 2026-10-19\n")
    monkeypatch.setattr(image, "FREEZE", freeze)
    compile_command = Mock(return_value="lightning==2.5.0\n")
    monkeypatch.setattr(image, "run", compile_command)
    assert image.resolve("pyproject.toml") == {"lightning": "2.5.0"}
    args = compile_command.call_args.args
    assert args[args.index("--python-version") + 1] == "3.12"
    assert "--no-sources" in args and "--no-config" in args
    assert args.count("--extra") == 2
    download = Mock()
    monkeypatch.setattr(dataset, "run", download)
    dataset.download_wheels({"lightning": "2.5.0"})
    args = download.call_args.args
    assert args[args.index("--python-version") + 1] == "3.12"
    download.reset_mock()
    dataset.download_wheels({})
    download.assert_not_called()


def test_no_torch_snapshot_is_rejected(scripts, monkeypatch, tmp_path):
    freeze = tmp_path / "image-freeze.txt"
    freeze.write_text("numpy==2.0.0\n")
    monkeypatch.setattr(scripts["image"], "FREEZE", freeze)
    with pytest.raises(click.ClickException, match="no PyTorch"):
        scripts["image"].missing_from_image()


def test_parse_freeze_wheel_url_and_normalize_names(scripts):
    assert scripts["image"].parse_freeze(
        "# comment\nTorch==2.8.0+cu128\n"
        "some_pkg @ https://example.com/some_pkg-1.2.3%2Blocal-py3-none-any.whl#sha256=abc\n"
    ) == {"torch": "2.8.0", "some-pkg": "1.2.3"}


def test_dataset_requirements_include_notebook_extras(scripts, monkeypatch, tmp_path):
    monkeypatch.setattr(scripts["dataset"], "STAGE", tmp_path)
    monkeypatch.setattr(scripts["dataset"], "DATASET", "user/wheels")
    requirements = scripts["dataset"].write_dataset_files("0.1.0", {"lightning": "2.5.0"})
    assert requirements == "birdclef[notebooks,deep-learning]==0.1.0\nlightning==2.5.0\n"
    assert json.loads((tmp_path / "dataset-metadata.json").read_text())["id"] == "user/wheels"


def test_build_rejects_paths_outside_build_folder(scripts, tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(click.ClickException, match="staging folder"):
        scripts["wheelhouse_common"].build_wheel(tmp_path)
    assert sentinel.read_text() == "keep"
