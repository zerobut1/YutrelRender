"""Asynchronous xmake/Yutrel process management."""

from __future__ import annotations

import codecs
import os
import re
import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from .model import RenderOptions, RenderState, build_command, format_command

_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class RenderProcessController(QObject):
    log_received = Signal(str)
    state_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(self._on_finished)
        self._state = RenderState.IDLE
        self._stopping = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._tree_killer: QProcess | None = None

    @property
    def state(self) -> RenderState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    def start(self, options: RenderOptions, project_root: Path) -> bool:
        if self.is_running:
            self.error_occurred.emit("A render process is already running.")
            return False

        xmake = shutil.which("xmake")
        if xmake is None:
            self._set_state(RenderState.FAILED)
            self.error_occurred.emit("xmake was not found in PATH.")
            return False

        _, arguments = build_command(options)
        self._stopping = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._process.setWorkingDirectory(str(project_root))
        self.log_received.emit(f"$ {format_command('xmake', arguments)}\n")
        self._process.start(xmake, arguments)
        self._set_state(RenderState.RUNNING)
        return True

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stopping = True
        pid = int(self._process.processId())
        self.log_received.emit("Stopping render process...\n")

        if os.name == "nt" and pid > 0:
            taskkill = shutil.which("taskkill")
            if taskkill is not None:
                self._tree_killer = QProcess(self)
                self._tree_killer.setProcessChannelMode(QProcess.MergedChannels)
                self._tree_killer.readyReadStandardOutput.connect(
                    self._read_tree_killer_output
                )
                self._tree_killer.finished.connect(self._on_tree_killer_finished)
                self._tree_killer.start(taskkill, ["/PID", str(pid), "/T", "/F"])
                return

        self._process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _set_state(self, state: RenderState) -> None:
        if self._state == state:
            return
        self._state = state
        self.state_changed.emit(state.value)

    def _normalise_output(self, data: bytes, *, final: bool = False) -> str:
        text = self._decoder.decode(data, final=final)
        text = _ANSI_ESCAPE.sub("", text)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _read_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput())
        if text := self._normalise_output(data):
            self.log_received.emit(text)

    def _read_tree_killer_output(self) -> None:
        if self._tree_killer is None:
            return
        data = bytes(self._tree_killer.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if text:
            self.log_received.emit(_ANSI_ESCAPE.sub("", text))

    def _on_tree_killer_finished(self) -> None:
        self._tree_killer = None
        self._kill_if_running()

    def _kill_if_running(self) -> None:
        if self.is_running:
            self._process.kill()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.FailedToStart:
            self._set_state(RenderState.FAILED)
            self.error_occurred.emit(
                f"Failed to start xmake: {self._process.errorString()}"
            )
        elif not self._stopping:
            self.log_received.emit(f"Process error: {self._process.errorString()}\n")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if text := self._normalise_output(b"", final=True):
            self.log_received.emit(text)
        if self._stopping:
            self._set_state(RenderState.STOPPED)
        elif exit_status == QProcess.NormalExit and exit_code == 0:
            self._set_state(RenderState.SUCCEEDED)
        else:
            self._set_state(RenderState.FAILED)
            self.log_received.emit(f"Process exited with code {exit_code}.\n")
        self._stopping = False
