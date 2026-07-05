"""Tests for scheduler/task_installer.py command builders."""
from __future__ import annotations

from pathlib import Path

from scheduler import task_installer


def test_bat_content_cds_and_runs_scheduler():
    bat = task_installer.build_bat_content(Path(r"D:\Research Agent"), Path(r"D:\Research Agent\data\state\scheduler.log"))
    assert 'cd /d "D:\\Research Agent"' in bat
    assert "main.py scheduler" in bat
    assert "scheduler.log" in bat


def test_create_args_are_onlogon_and_forced():
    args = task_installer.build_schtasks_create_args(Path(r"C:\x\run_scheduler.bat"))
    assert args[:3] == ["schtasks", "/Create", "/TN"]
    assert "ONLOGON" in args
    assert "/F" in args
    assert task_installer.TASK_NAME in args


def test_delete_args():
    args = task_installer.build_schtasks_delete_args()
    assert args == ["schtasks", "/Delete", "/TN", task_installer.TASK_NAME, "/F"]
