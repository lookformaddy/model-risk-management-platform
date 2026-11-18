"""Tests for the CLI."""

from mrmp.cli import build_parser, main


def test_tier_subcommand(capsys):
    assert main(["tier", "--impact", "3", "--sensitivity", "3",
                 "--autonomy", "3", "--exposure", "3"]) == 0
    out = capsys.readouterr().out
    assert "tier: CRITICAL" in out
    assert "risk_officer" in out


def test_tier_low(capsys):
    assert main(["tier", "--impact", "0", "--sensitivity", "0",
                 "--autonomy", "0", "--exposure", "0"]) == 0
    assert "tier: LOW" in capsys.readouterr().out


def test_parser_requires_command():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
