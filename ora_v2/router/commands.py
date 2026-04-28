"""Command parser for slash commands."""
from __future__ import annotations
import re
import shlex
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    name: str        # e.g. "task", "stop", "memory"
    args: str = ""   # everything after the command name


def parse(text: str) -> ParsedCommand | None:
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text.split(None, 1)
    cmd = parts[0].lstrip("/").lower()
    # strip @bot suffix
    cmd = cmd.split("@")[0]
    args = parts[1].strip() if len(parts) > 1 else ""
    return ParsedCommand(name=cmd, args=args)
