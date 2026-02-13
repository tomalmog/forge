"""Unit tests for hardware-profile CLI command."""

from __future__ import annotations

from cli.main import main


def test_cli_hardware_profile_prints_sorted_key_value_rows(
    monkeypatch,
    capsys,
) -> None:
    """Hardware-profile command should print deterministic key=value output."""

    class _FakeProfile:
        def to_dict(self) -> dict[str, object]:
            return {"gpu_count": 0, "accelerator": "cpu"}

    monkeypatch.setattr(
        "cli.hardware_profile_command.detect_hardware_profile", lambda: _FakeProfile()
    )
    exit_code = main(["hardware-profile"])
    output = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0 and output == ["accelerator=cpu", "gpu_count=0"]
