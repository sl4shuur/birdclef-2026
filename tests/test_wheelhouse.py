import importlib
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

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

    def compile_requirements(*args):
        output = Path(args[args.index("--output-file") + 1])
        assert output.read_text() == "lightning==2.4.0\n"
        assert "--constraints" not in args
        output.write_text("lightning==2.5.0\n")

    compile_command = Mock(side_effect=compile_requirements)
    monkeypatch.setattr(image, "run", compile_command)
    assert image.resolve("pyproject.toml", preferences="lightning==2.4.0\n") == {
        "lightning": "2.5.0"
    }
    args = compile_command.call_args.args
    assert args[args.index("--python-version") + 1] == "3.12"
    assert "--no-sources" in args and "--no-config" in args
    assert args.count("--extra") == 4
    assert "audio" in args
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


@pytest.mark.parametrize(
    "wandb_version, sdk_version, reuse", [("0.28.1", "1.42.1", False), ("0.31.0", "1.43.0", True)]
)
def test_incompatible_wandb_and_transitive_dependencies_are_upgraded(
    scripts, monkeypatch, tmp_path, wandb_version, sdk_version, reuse
):
    image = scripts["image"]
    freeze = tmp_path / "image-freeze.txt"
    freeze.write_text(
        f"# Kaggle default image, Python 3.13.15, taken 2026-10-19\n"
        f"torch==2.11.0+cu128\nwandb=={wandb_version}\nopentelemetry-sdk=={sdk_version}\n"
    )
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\ndependencies = ["wandb>=0.31.0"]\n[project.optional-dependencies]\n'
    )
    monkeypatch.setattr(image, "FREEZE", freeze)
    monkeypatch.setattr(image, "PYPROJECT", project)
    upgraded = {"wandb": "0.31.0", "opentelemetry-sdk": "1.43.0"}
    resolver = Mock(return_value=upgraded)
    monkeypatch.setattr(image, "resolve", resolver)
    assert image.shared_with_image() == (upgraded if reuse else {})
    assert image.missing_from_image() == ({} if reuse else upgraded)
    assert f"wandb=={wandb_version}" in resolver.call_args.kwargs["preferences"]
    assert f"opentelemetry-sdk=={sdk_version}" in resolver.call_args.kwargs["preferences"]
    assert resolver.call_args.kwargs["overrides"] == 'torch ; sys_platform == "never"\n'


def test_parse_freeze_wheel_url_and_normalize_names(scripts):
    assert scripts["image"].parse_freeze(
        "# comment\nTorch==2.8.0+cu128\n"
        "some_pkg @ https://example.com/some_pkg-1.2.3%2Blocal-py3-none-any.whl#sha256=abc\n"
    ) == {"torch": "2.8.0", "some-pkg": "1.2.3"}


def test_dataset_requirements_include_notebook_extras(scripts, monkeypatch, tmp_path):
    monkeypatch.setattr(scripts["dataset"], "STAGE", tmp_path)
    monkeypatch.setattr(scripts["dataset"], "DATASET", "user/wheels")
    requirements = scripts["dataset"].write_dataset_files("0.1.0", {"lightning": "2.5.0"})
    assert (
        requirements
        == "birdclef-2026[notebooks,audio,deep-learning,perch]==0.1.0\nlightning==2.5.0\n"
    )
    assert json.loads((tmp_path / "dataset-metadata.json").read_text())["id"] == "user/wheels"


def test_build_rejects_paths_outside_build_folder(scripts, tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(click.ClickException, match="staging folder"):
        scripts["wheelhouse_common"].build_wheel(tmp_path)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("create", [False, True])
def test_dataset_upload_rejects_api_errors_even_when_cli_exits_successfully(
    scripts, monkeypatch, create
):
    dataset = scripts["dataset"]
    command = Mock(return_value="Dataset creation error: Title is already in use.")
    monkeypatch.setattr(dataset, "kaggle", command)
    with pytest.raises(click.ClickException, match="Kaggle rejected"):
        dataset.upload("new version", create=create)
    args = command.call_args.args
    assert args[1] == ("create" if create else "version")
    assert "--public" not in args
    assert args[-3:] == ("-r", "zip", "-t")


def test_build_uses_normalized_wheel_filename_and_project_metadata(scripts, monkeypatch, tmp_path):
    common = scripts["wheelhouse_common"]
    monkeypatch.setattr(common, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "audio-project"\n')
    stage = tmp_path / "build" / "wheels"

    def build(*args):
        stage.mkdir(parents=True)
        (stage / "audio_project-0.1.0-py3-none-any.whl").write_bytes(b"wheel")

    monkeypatch.setattr(common, "run", build)
    assert common.build_wheel(stage) == "0.1.0"
    assert (
        common.project_requirement("0.1.0")
        == "audio-project[notebooks,audio,deep-learning,perch]==0.1.0"
    )


@pytest.mark.parametrize("tampered", [False, True])
def test_training_bundle_preserves_prepared_data_and_cached_weights(
    scripts, monkeypatch, tmp_path, tampered
):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    command = importlib.import_module("scripts.kaggle.wheelhouse")
    monkeypatch.setattr(command, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(command, "STAGE", tmp_path / "build" / "kaggle-run")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    original = b"filename\nfocal.ogg\n"
    (prepared / "focal.csv").write_bytes(b"changed" if tampered else original)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "artifacts_sha256": {"focal.csv": sha256(original).hexdigest()},
            }
        )
    )
    (prepared / "unrelated-secret.txt").write_text("do not package")
    cache = tmp_path / "cache"
    (cache / "checkpoints").mkdir(parents=True)
    (cache / "checkpoints/weights.pth").write_bytes(b"cached weights")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache))
    weights = Mock(url="https://example.com/weights.pth")
    monkeypatch.setattr(
        torchvision.models, "EfficientNet_B0_Weights", SimpleNamespace(IMAGENET1K_V1=weights)
    )

    def build_wheel(stage):
        stage.mkdir(parents=True)
        (stage / "birdclef_2026-0.1.0-py3-none-any.whl").write_bytes(b"current wheel")
        return "0.1.0"

    builder = Mock(side_effect=build_wheel)
    monkeypatch.setattr(command, "build_wheel", builder)
    result = CliRunner().invoke(command.main, ["build", "--prepared-dir", str(prepared)])
    if tampered:
        assert result.exit_code != 0 and "checksum mismatch" in result.output
        builder.assert_not_called()
        weights.get_state_dict.assert_not_called()
        return
    assert result.exit_code == 0, result.output
    weights.get_state_dict.assert_called_once_with(progress=True, check_hash=True)
    with ZipFile(tmp_path / "build/birdclef-kaggle.zip") as archive:
        assert archive.read("data_prep/focal.csv") == original
        assert archive.read("torch/hub/checkpoints/weights.pth") == b"cached weights"
        assert archive.read("birdclef_2026-0.1.0-py3-none-any.whl") == b"current wheel"
        assert b"audio" in archive.read("requirements.txt")
        assert all("secret" not in name for name in archive.namelist())
