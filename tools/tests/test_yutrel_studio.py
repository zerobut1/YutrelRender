from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pytest
from PySide6.QtCore import QProcess, QSettings

from yutrel_studio.model import (
    DEFAULT_RESOLUTION,
    UINT32_MAX,
    RenderMode,
    RenderOptions,
    RenderState,
    build_command,
    find_project_root,
    read_pbrt_resolution,
    validate_render_options,
)
from yutrel_studio.process import RenderProcessController
from yutrel_studio.window import MainWindow


def make_options(tmp_path: Path, **overrides: object) -> RenderOptions:
    scene = tmp_path / "scene.pbrt"
    scene.write_text("WorldBegin\nWorldEnd\n", encoding="utf-8")
    values: dict[str, object] = {
        "scene": scene,
        "output": tmp_path / "render.exr",
    }
    values.update(overrides)
    return RenderOptions(**values)


@pytest.mark.parametrize(
    ("mode", "flag"),
    [
        (RenderMode.OFFLINE, None),
        (RenderMode.HEADLESS, "--headless"),
        (RenderMode.INTERACTIVE, "--interactive"),
    ],
)
def test_build_command_modes(
    tmp_path: Path, mode: RenderMode, flag: str | None
) -> None:
    options = make_options(tmp_path, mode=mode)
    program, arguments = build_command(options)
    assert program == "xmake"
    assert arguments[:4] == ["run", "Yutrel", "dx", str(options.scene)]
    assert (flag in arguments) if flag else "--headless" not in arguments
    if flag is None:
        assert "--interactive" not in arguments
    assert arguments[-8:] == [
        "--spp",
        "16",
        "--seed",
        "20120712",
        "--resolution",
        "1920x1080",
        "--output",
        str(options.output),
    ]


def test_command_keeps_unicode_path_as_one_argument(tmp_path: Path) -> None:
    scene_dir = tmp_path / "空 格"
    scene_dir.mkdir()
    options = make_options(
        scene_dir,
        output=scene_dir / "渲染 结果.exr",
    )
    _, arguments = build_command(options)
    assert arguments[3] == str(options.scene)
    assert arguments[-1] == str(options.output)


@pytest.mark.parametrize(
    "overrides",
    [
        {"spp": 0},
        {"spp": UINT32_MAX + 1},
        {"seed": -1},
        {"seed": UINT32_MAX + 1},
        {"width": 0},
        {"height": 0},
        {"width": 65536, "height": 65536},
        {"output": Path("result.png")},
    ],
)
def test_validation_rejects_invalid_values(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    options = make_options(tmp_path, **overrides)
    with pytest.raises(ValueError):
        validate_render_options(options)


def test_validation_rejects_missing_scene_and_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing PBRT"):
        validate_render_options(make_options(tmp_path, scene=tmp_path / "missing.pbrt"))
    with pytest.raises(ValueError, match="output directory"):
        validate_render_options(
            make_options(tmp_path, output=tmp_path / "missing" / "render.exr")
        )


def test_find_project_root() -> None:
    root = find_project_root(Path(__file__))
    assert (root / "pyproject.toml").is_file()
    assert (root / "xmake.lua").is_file()


def test_reads_resolution_from_pbrt_film(tmp_path: Path) -> None:
    scene = tmp_path / "scene.pbrt"
    scene.write_text(
        "\n".join(
            [
                'Film "rgb"',
                '    "integer yresolution" [ 720 ]',
                '    "integer xresolution" [ 1280 ]',
            ]
        ),
        encoding="utf-8",
    )
    assert read_pbrt_resolution(scene) == (1280, 720)


def test_pbrt_resolution_uses_default_for_missing_values(tmp_path: Path) -> None:
    scene = tmp_path / "scene.pbrt"
    scene.write_text(
        '# "integer xresolution" [ 640 ]\nFilm "rgb" "integer yresolution" [ 900 ]\n',
        encoding="utf-8",
    )
    assert read_pbrt_resolution(scene) == (DEFAULT_RESOLUTION[0], 900)
    assert read_pbrt_resolution(tmp_path / "missing.pbrt") == DEFAULT_RESOLUTION


def test_window_defaults_and_running_state(qtbot, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window = MainWindow(project_root=tmp_path, settings=settings)
    qtbot.addWidget(window)

    assert window.backend_combo.currentText() == "dx"
    assert window.mode_combo.currentData() == RenderMode.OFFLINE.value
    assert window.spp_edit.text() == "16"
    assert window.seed_edit.text() == "20120712"
    assert window.width_edit.text() == "1920"
    assert window.height_edit.text() == "1080"

    scene = tmp_path / "inherited.pbrt"
    scene.write_text(
        'Film "rgb"\n'
        '    "integer xresolution" [ 800 ]\n'
        '    "integer yresolution" [ 600 ]\n',
        encoding="utf-8",
    )
    window.scene_edit.setText(str(scene))
    window._inherit_scene_resolution()
    assert window.width_edit.text() == "800"
    assert window.height_edit.text() == "600"

    window._on_state_changed(RenderState.RUNNING.value)
    assert not window.scene_edit.isEnabled()
    assert not window.render_button.isEnabled()
    assert window.stop_button.isEnabled()

    window._on_state_changed(RenderState.SUCCEEDED.value)
    assert window.scene_edit.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.status_label.text() == RenderState.SUCCEEDED.value


def test_window_restores_settings(qtbot, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.IniFormat)
    first = MainWindow(project_root=tmp_path, settings=settings)
    qtbot.addWidget(first)
    first.backend_combo.setCurrentText("vk")
    first.mode_combo.setCurrentIndex(
        first.mode_combo.findData(RenderMode.HEADLESS.value)
    )
    first.spp_edit.setText("32")
    first._save_settings()

    restored = MainWindow(
        project_root=tmp_path,
        settings=QSettings(str(settings_path), QSettings.IniFormat),
    )
    qtbot.addWidget(restored)
    assert restored.backend_combo.currentText() == "vk"
    assert restored.mode_combo.currentData() == RenderMode.HEADLESS.value
    assert restored.spp_edit.text() == "32"


def test_controller_reports_missing_xmake(qtbot, monkeypatch, tmp_path: Path) -> None:
    controller = RenderProcessController()
    errors: list[str] = []
    states: list[str] = []
    controller.error_occurred.connect(errors.append)
    controller.state_changed.connect(states.append)
    monkeypatch.setattr("yutrel_studio.process.shutil.which", lambda _: None)

    assert not controller.start(make_options(tmp_path), tmp_path)
    assert errors == ["xmake was not found in PATH."]
    assert states[-1] == RenderState.FAILED.value


def test_controller_reports_nonzero_exit(qtbot, monkeypatch, tmp_path: Path) -> None:
    controller = RenderProcessController()
    monkeypatch.setattr("yutrel_studio.process.shutil.which", lambda _: sys.executable)

    assert controller.start(make_options(tmp_path), tmp_path)
    qtbot.waitUntil(
        lambda: controller.state == RenderState.FAILED,
        timeout=5000,
    )
    assert controller._process.exitStatus() == QProcess.NormalExit
    assert controller._process.exitCode() != 0


def test_controller_stops_running_process_tree(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    controller = RenderProcessController()
    real_which = shutil.which

    def find_program(name: str) -> str | None:
        if name == "xmake":
            return sys.executable
        return real_which(name)

    monkeypatch.setattr("yutrel_studio.process.shutil.which", find_program)
    monkeypatch.setattr(
        "yutrel_studio.process.build_command",
        lambda _: ("xmake", ["-c", "import time; time.sleep(30)"]),
    )

    assert controller.start(make_options(tmp_path), tmp_path)
    qtbot.waitUntil(
        lambda: controller._process.state() == QProcess.Running,
        timeout=5000,
    )
    controller.stop()
    qtbot.waitUntil(
        lambda: controller.state == RenderState.STOPPED,
        timeout=10000,
    )
    assert not controller.is_running
