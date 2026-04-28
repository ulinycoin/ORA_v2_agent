"""Unit tests: SkillCodeAuditor."""
import pytest
from ora_v2.skills.sandbox import SkillCodeAuditor

auditor = SkillCodeAuditor()


def test_blocks_import_os():
    result = auditor.audit("import os\ndef run(): return os.getcwd()")
    assert not result.safe
    assert any("os" in v for v in result.violations)


def test_blocks_import_subprocess():
    result = auditor.audit("import subprocess\ndef run(): return subprocess.check_output(['ls'])")
    assert not result.safe


def test_blocks_exec():
    result = auditor.audit("def run(): exec('print(1)')")
    assert not result.safe
    assert any("exec" in v for v in result.violations)


def test_blocks_eval():
    result = auditor.audit("def run(x): return eval(x)")
    assert not result.safe


def test_blocks_open():
    result = auditor.audit("def run(): return open('/etc/passwd').read()")
    assert not result.safe


def test_blocks_os_environ_attr():
    result = auditor.audit("import os\ndef run(): return os.environ.get('KEY')")
    assert not result.safe


def test_allows_safe_code():
    code = """
import json
import re

DESCRIPTION = "parse json"
ARGS = {"text": "json string"}

def run(text: str) -> str:
    data = json.loads(text)
    return str(data.get("key", ""))
"""
    result = auditor.audit(code)
    assert result.safe
    assert result.violations == []


def test_blocks_from_os_import():
    result = auditor.audit("from os import path\ndef run(): return path.exists('/tmp')")
    assert not result.safe


def test_syntax_error():
    result = auditor.audit("def run(: pass")
    assert not result.safe
    assert any("Syntax" in v for v in result.violations)
