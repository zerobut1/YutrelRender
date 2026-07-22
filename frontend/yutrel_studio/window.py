"""Qt Widgets main window for Yutrel Studio."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QRegularExpression, QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QRegularExpressionValidator, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .model import (
    DEFAULT_RESOLUTION,
    RenderMode,
    RenderOptions,
    RenderState,
    build_command,
    find_project_root,
    format_command,
    read_pbrt_resolution,
    validate_render_options,
)
from .process import RenderProcessController


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.project_root = project_root or find_project_root()
        self.settings = (
            settings if settings is not None else QSettings("Yutrel", "YutrelStudio")
        )
        self.controller = RenderProcessController(self)
        self._close_when_stopped = False
        self._parameter_widgets: list[QWidget] = []

        self.setWindowTitle("Yutrel Studio")
        self.resize(1000, 700)
        self._create_ui()
        self._connect_signals()
        self._restore_settings()
        self._update_command_preview()

    def _create_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)

        settings_box = QGroupBox("Render Settings")
        settings_layout = QFormLayout(settings_box)

        self.scene_edit = QLineEdit()
        self.scene_button = QPushButton("Browse...")
        scene_row = QWidget()
        scene_layout = QHBoxLayout(scene_row)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.addWidget(self.scene_edit)
        scene_layout.addWidget(self.scene_button)
        settings_layout.addRow("PBRT scene", scene_row)

        self.output_edit = QLineEdit()
        self.output_button = QPushButton("Browse...")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_button)
        settings_layout.addRow("EXR output", output_row)

        numeric_validator = QRegularExpressionValidator(
            QRegularExpression(r"\d{0,10}"), self
        )
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["dx", "cuda", "vk"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Offline Display", RenderMode.OFFLINE.value)
        self.mode_combo.addItem("Headless", RenderMode.HEADLESS.value)
        self.mode_combo.addItem("Interactive", RenderMode.INTERACTIVE.value)
        self.spp_edit = QLineEdit("16")
        self.seed_edit = QLineEdit("20120712")
        self.width_edit = QLineEdit(str(DEFAULT_RESOLUTION[0]))
        self.height_edit = QLineEdit(str(DEFAULT_RESOLUTION[1]))
        for edit in (
            self.spp_edit,
            self.seed_edit,
            self.width_edit,
            self.height_edit,
        ):
            edit.setValidator(numeric_validator)

        options_row = QWidget()
        options_layout = QGridLayout(options_row)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.addWidget(QLabel("Backend"), 0, 0)
        options_layout.addWidget(self.backend_combo, 0, 1)
        options_layout.addWidget(QLabel("Mode"), 0, 2)
        options_layout.addWidget(self.mode_combo, 0, 3)
        options_layout.addWidget(QLabel("SPP"), 1, 0)
        options_layout.addWidget(self.spp_edit, 1, 1)
        options_layout.addWidget(QLabel("Seed"), 1, 2)
        options_layout.addWidget(self.seed_edit, 1, 3)
        options_layout.addWidget(QLabel("Width"), 2, 0)
        options_layout.addWidget(self.width_edit, 2, 1)
        options_layout.addWidget(QLabel("Height"), 2, 2)
        options_layout.addWidget(self.height_edit, 2, 3)
        settings_layout.addRow(options_row)

        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setPlaceholderText("Select a valid PBRT scene.")
        settings_layout.addRow("Command", self.command_edit)

        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: #c45c5c;")
        settings_layout.addRow("", self.validation_label)

        button_row = QHBoxLayout()
        self.render_button = QPushButton("Render")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.status_label = QLabel(RenderState.IDLE.value)
        button_row.addWidget(self.render_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch()
        button_row.addWidget(QLabel("Status:"))
        button_row.addWidget(self.status_label)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.NoWrap)

        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(settings_box)
        top_layout.addLayout(button_row)
        splitter.addWidget(top)
        splitter.addWidget(self.log_edit)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self._parameter_widgets = [
            self.scene_edit,
            self.scene_button,
            self.output_edit,
            self.output_button,
            self.backend_combo,
            self.mode_combo,
            self.spp_edit,
            self.seed_edit,
            self.width_edit,
            self.height_edit,
        ]

    def _connect_signals(self) -> None:
        self.scene_button.clicked.connect(self._browse_scene)
        self.scene_edit.editingFinished.connect(self._inherit_scene_resolution)
        self.output_button.clicked.connect(self._browse_output)
        self.render_button.clicked.connect(self._start_render)
        self.stop_button.clicked.connect(self.controller.stop)
        self.controller.log_received.connect(self._append_log)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.error_occurred.connect(self._on_process_error)

        for edit in (
            self.scene_edit,
            self.output_edit,
            self.spp_edit,
            self.seed_edit,
            self.width_edit,
            self.height_edit,
        ):
            edit.textChanged.connect(self._update_command_preview)
        self.backend_combo.currentIndexChanged.connect(self._update_command_preview)
        self.mode_combo.currentIndexChanged.connect(self._update_command_preview)

    def _resolve_path(self, text: str) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(text.strip()))
        path = Path(expanded)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @staticmethod
    def _parse_uint(text: str, label: str) -> int:
        if not text:
            raise ValueError(f"{label} is required.")
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(f"{label} must be an integer.") from error

    def _read_options(self) -> RenderOptions:
        scene_text = self.scene_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not scene_text:
            raise ValueError("Select a PBRT scene.")
        if not output_text:
            raise ValueError("Select an EXR output path.")
        options = RenderOptions(
            scene=self._resolve_path(scene_text),
            output=self._resolve_path(output_text),
            backend=self.backend_combo.currentText(),
            mode=RenderMode(self.mode_combo.currentData()),
            spp=self._parse_uint(self.spp_edit.text(), "SPP"),
            seed=self._parse_uint(self.seed_edit.text(), "Seed"),
            width=self._parse_uint(self.width_edit.text(), "Width"),
            height=self._parse_uint(self.height_edit.text(), "Height"),
        )
        validate_render_options(options)
        return options

    def _update_command_preview(self) -> None:
        try:
            program, arguments = build_command(self._read_options())
        except (ValueError, OSError) as error:
            self.command_edit.clear()
            self.validation_label.setText(str(error))
            valid = False
        else:
            self.command_edit.setText(format_command(program, arguments))
            self.validation_label.clear()
            valid = True
        self.render_button.setEnabled(valid and not self.controller.is_running)

    def _browse_scene(self) -> None:
        initial = self.scene_edit.text().strip()
        if not initial:
            initial = str(self.project_root / "scene")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select PBRT Scene",
            initial,
            "PBRT Scenes (*.pbrt);;All Files (*)",
        )
        if filename:
            scene = Path(filename).resolve()
            self.scene_edit.setText(str(scene))
            self.output_edit.setText(str(scene.with_suffix(".exr")))
            self._inherit_scene_resolution()

    def _inherit_scene_resolution(self) -> None:
        scene_text = self.scene_edit.text().strip()
        resolution = DEFAULT_RESOLUTION
        if scene_text:
            resolution = read_pbrt_resolution(self._resolve_path(scene_text))
        self.width_edit.setText(str(resolution[0]))
        self.height_edit.setText(str(resolution[1]))

    def _browse_output(self) -> None:
        initial = self.output_edit.text().strip()
        if not initial and self.scene_edit.text().strip():
            initial = str(
                self._resolve_path(self.scene_edit.text()).with_suffix(".exr")
            )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select EXR Output",
            initial,
            "OpenEXR Images (*.exr)",
        )
        if filename:
            output = Path(filename)
            if not output.suffix:
                output = output.with_suffix(".exr")
            self.output_edit.setText(str(output.resolve()))

    def _start_render(self) -> None:
        try:
            options = self._read_options()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Invalid Render Settings", str(error))
            return
        if options.output.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite Output",
                f"The output file already exists:\n{options.output}\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.log_edit.clear()
        self.controller.start(options, self.project_root)

    def _append_log(self, text: str) -> None:
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def _on_state_changed(self, state: str) -> None:
        self.status_label.setText(state)
        running = state == RenderState.RUNNING.value
        for widget in self._parameter_widgets:
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self._update_command_preview()
        if not running and self._close_when_stopped:
            self._close_when_stopped = False
            QTimer.singleShot(0, self.close)

    def _on_process_error(self, message: str) -> None:
        self._append_log(f"Error: {message}\n")
        QMessageBox.critical(self, "Yutrel Studio", message)

    def _restore_settings(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.scene_edit.setText(str(self.settings.value("render/scene", "")))
        self.output_edit.setText(str(self.settings.value("render/output", "")))
        self._set_combo_value(
            self.backend_combo, str(self.settings.value("render/backend", "dx"))
        )
        self._set_combo_value(
            self.mode_combo,
            str(self.settings.value("render/mode", RenderMode.OFFLINE.value)),
            use_data=True,
        )
        self.spp_edit.setText(str(self.settings.value("render/spp", 16)))
        self.seed_edit.setText(str(self.settings.value("render/seed", 20120712)))
        self.width_edit.setText(
            str(self.settings.value("render/width", DEFAULT_RESOLUTION[0]))
        )
        self.height_edit.setText(
            str(self.settings.value("render/height", DEFAULT_RESOLUTION[1]))
        )
        self._inherit_scene_resolution()

    @staticmethod
    def _set_combo_value(
        combo: QComboBox, value: str, *, use_data: bool = False
    ) -> None:
        index = combo.findData(value) if use_data else combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _save_settings(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("render/scene", self.scene_edit.text())
        self.settings.setValue("render/output", self.output_edit.text())
        self.settings.setValue("render/backend", self.backend_combo.currentText())
        self.settings.setValue("render/mode", self.mode_combo.currentData())
        self.settings.setValue("render/spp", self.spp_edit.text())
        self.settings.setValue("render/seed", self.seed_edit.text())
        self.settings.setValue("render/width", self.width_edit.text())
        self.settings.setValue("render/height", self.height_edit.text())
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.is_running:
            answer = QMessageBox.question(
                self,
                "Stop Rendering",
                "A render is still running. Stop it and close Yutrel Studio?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._close_when_stopped = True
            self.controller.stop()
            event.ignore()
            return
        self._save_settings()
        event.accept()
