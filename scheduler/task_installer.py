"""Register the pipeline scheduler as a Windows Task Scheduler job (survives reboots).

The BlockingScheduler dies on logout/reboot; this registers an ONLOGON task that
relaunches it. Pure builders are unit-tested; install() performs the side effects.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

TASK_NAME = "ResearchAgentScheduler"


def build_bat_content(project_root: Path, log_path: Path) -> str:
    """Batch script that runs the scheduler daemon from the project root."""
    return (
        "@echo off\r\n"
        f'cd /d "{project_root}"\r\n'
        f'uv run python main.py scheduler >> "{log_path}" 2>&1\r\n'
    )


def build_schtasks_create_args(bat_path: Path, task_name: str = TASK_NAME) -> list[str]:
    """schtasks arguments to (re)create the ONLOGON task."""
    return [
        "schtasks", "/Create", "/TN", task_name,
        "/TR", f'"{bat_path}"', "/SC", "ONLOGON", "/RL", "LIMITED", "/F",
    ]


def build_schtasks_delete_args(task_name: str = TASK_NAME) -> list[str]:
    return ["schtasks", "/Delete", "/TN", task_name, "/F"]


def install(project_root: Path, state_dir: Path) -> tuple[Path, str]:
    """Write the launcher .bat and register the task. Returns (bat_path, removal_cmd)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    bat_path = state_dir / "run_scheduler.bat"
    log_path = state_dir / "scheduler.log"
    bat_path.write_text(build_bat_content(project_root, log_path), encoding="utf-8")

    subprocess.run(build_schtasks_create_args(bat_path), check=True,
                   capture_output=True, text=True)
    removal_cmd = " ".join(build_schtasks_delete_args())
    return bat_path, removal_cmd
