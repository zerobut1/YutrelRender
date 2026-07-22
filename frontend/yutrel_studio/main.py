"""Yutrel Studio application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .window import MainWindow


def main() -> int:
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv)
    app.setOrganizationName("Yutrel")
    app.setApplicationName("YutrelStudio")

    window = MainWindow()
    window.show()
    if owns_application:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
