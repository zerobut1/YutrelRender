"""Render options, validation, and command construction."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

UINT32_MAX = 2**32 - 1
SUPPORTED_BACKENDS = ("dx", "cuda", "vk")
SUPPORTED_SCENE_EXTENSIONS = (".pbrt", ".usd", ".usda", ".usdc", ".usdz")
DEFAULT_RESOLUTION = (1920, 1080)
_RESOLUTION_PARAMETER = re.compile(
    r'"integer\s+(xresolution|yresolution)"\s*\[?\s*(\d+)',
    re.IGNORECASE,
)


class RenderMode(StrEnum):
    OFFLINE = "offline"
    HEADLESS = "headless"
    INTERACTIVE = "interactive"


class RenderState(StrEnum):
    IDLE = "Idle"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    STOPPED = "Stopped"


@dataclass(frozen=True)
class RenderOptions:
    scene: Path
    output: Path
    backend: str = "dx"
    mode: RenderMode = RenderMode.OFFLINE
    spp: int = 16
    seed: int = 20120712
    width: int = DEFAULT_RESOLUTION[0]
    height: int = DEFAULT_RESOLUTION[1]


def read_pbrt_resolution(
    path: Path, fallback: tuple[int, int] = DEFAULT_RESOLUTION
) -> tuple[int, int]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return fallback
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    values = {
        name.lower(): int(value) for name, value in _RESOLUTION_PARAMETER.findall(text)
    }
    return (
        values.get("xresolution", fallback[0]),
        values.get("yresolution", fallback[1]),
    )


def read_scene_resolution(
    path: Path, fallback: tuple[int, int] = DEFAULT_RESOLUTION
) -> tuple[int, int]:
    if path.suffix.lower() == ".pbrt":
        return read_pbrt_resolution(path, fallback)
    return fallback


def validate_render_options(options: RenderOptions) -> None:
    if options.backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {options.backend}")
    if not isinstance(options.mode, RenderMode):
        raise ValueError(f"Unsupported render mode: {options.mode}")
    if not options.scene.is_file():
        raise ValueError("Select an existing PBRT or USD scene.")
    if options.scene.suffix.lower() not in SUPPORTED_SCENE_EXTENSIONS:
        raise ValueError("The scene must use a PBRT or USD extension.")
    if not 0 < options.spp <= UINT32_MAX:
        raise ValueError("SPP must be between 1 and UINT32_MAX.")
    if not 0 <= options.seed <= UINT32_MAX:
        raise ValueError("Seed must be between 0 and UINT32_MAX.")
    if options.width <= 0 or options.height <= 0:
        raise ValueError("Resolution dimensions must be positive.")
    if options.width > UINT32_MAX or options.height > UINT32_MAX:
        raise ValueError("Resolution dimensions must fit in uint32.")
    if options.width * options.height > UINT32_MAX:
        raise ValueError("Resolution exceeds the supported 32-bit pixel count.")
    if options.output.suffix.lower() != ".exr":
        raise ValueError("The output path must use the .exr extension.")
    if not options.output.parent.is_dir():
        raise ValueError("The output directory does not exist.")
    if options.output.is_dir():
        raise ValueError("The output path points to a directory.")


def build_command(options: RenderOptions) -> tuple[str, list[str]]:
    validate_render_options(options)
    arguments = ["run", "Yutrel", options.backend, str(options.scene)]
    if options.mode == RenderMode.HEADLESS:
        arguments.append("--headless")
    elif options.mode == RenderMode.INTERACTIVE:
        arguments.append("--interactive")
    arguments.extend(
        [
            "--spp",
            str(options.spp),
            "--seed",
            str(options.seed),
            "--resolution",
            f"{options.width}x{options.height}",
            "--output",
            str(options.output),
        ]
    )
    return "xmake", arguments


def format_command(program: str, arguments: list[str]) -> str:
    command = [program, *arguments]
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "xmake.lua"
        ).is_file():
            return candidate
    raise FileNotFoundError("Could not locate the Yutrel project root.")
