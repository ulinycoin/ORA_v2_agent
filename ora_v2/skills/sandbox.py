"""AST-based static auditor for LLM-generated skill code."""
from __future__ import annotations

import ast
import ipaddress
import logging
import re
import urllib.parse
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Private/loopback ranges blocked in generated skills (mirrors validate_url SSRF rules)
_PRIVATE_PATTERNS = re.compile(
    r"^(localhost"
    r"|127\.\d+\.\d+\.\d+"
    r"|0\.0\.0\.0"
    r"|10\.\d+\.\d+\.\d+"
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+"
    r"|192\.168\.\d+\.\d+"
    r"|169\.254\.\d+\.\d+"
    r"|::1|fc[0-9a-f]{2}:|fd[0-9a-f]{2}:)$",
    re.IGNORECASE,
)

_BANNED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "importlib",
    "ctypes", "shutil", "pickle", "marshal", "pty",
    "multiprocessing", "signal", "resource", "fcntl",
    "builtins", "runpy", "code", "codeop", "zipimport",
    "distutils", "setuptools", "pip", "pkg_resources",
})

_BANNED_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "breakpoint", "input", "globals", "locals", "vars",
    "dir", "getattr", "setattr", "delattr", "hasattr",
})

_BANNED_ATTRS = frozenset({
    "environ", "system", "popen", "unlink", "rmdir",
    "remove", "chmod", "chown", "kill", "fork",
    "__builtins__", "__globals__", "__class__", "__bases__",
    "__subclasses__", "__reduce__", "__reduce_ex__",
    "mro", "__code__", "__closure__",
})

_BANNED_STRINGS = frozenset({
    "__import__", "importlib", "subprocess", "os.system",
    "os.popen", "builtins",
})


@dataclass
class AuditResult:
    safe: bool
    violations: list[str] = field(default_factory=list)


def _host_allowed(host: str, allowed_hosts: list[str]) -> bool:
    """Return True if host matches any entry in allowed_hosts (exact or suffix)."""
    host = host.lower()
    for entry in allowed_hosts:
        if host == entry or host.endswith("." + entry):
            return True
    return False


def _check_url_literal(url: str, allowed_hosts: list[str]) -> str | None:
    """Return a violation string if the URL targets a private range or a non-allowed host."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https", ""):
        return None
    host = parsed.hostname or ""
    if not host:
        return None
    # Block private/loopback regardless of allowlist
    if _PRIVATE_PATTERNS.match(host):
        return f"SSRF: URL targets private/loopback host: {host!r}"
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return f"SSRF: URL targets private IP: {host!r}"
    except ValueError:
        pass  # not a bare IP — hostname
    # Enforce allowlist when configured
    if allowed_hosts and not _host_allowed(host, allowed_hosts):
        return f"Host not in allowed_hosts: {host!r}"
    return None


class SkillCodeAuditor:
    def audit(self, code: str, allowed_hosts: list[str] | None = None) -> AuditResult:
        allowed_hosts = allowed_hosts or []
        violations: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AuditResult(safe=False, violations=[f"SyntaxError: {e}"])

        for node in ast.walk(tree):
            # import os / import subprocess
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _BANNED_IMPORTS:
                        violations.append(f"Banned import: {alias.name}")

            # from os import system
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in _BANNED_IMPORTS:
                    violations.append(f"Banned import from: {node.module}")

            # eval(...) / exec(...) / globals() etc.
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _BANNED_BUILTINS:
                    violations.append(f"Banned builtin call: {node.func.id}()")
                # __builtins__['eval'](...) or dict['key'](...) bypass
                if isinstance(node.func, ast.Subscript):
                    violations.append("Banned subscript call (possible builtin bypass)")
                # ().__class__.__bases__[0].__subclasses__() chain
                if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_ATTRS:
                    violations.append(f"Banned attribute call: .{node.func.attr}()")

            # x.environ / obj.__builtins__ / dunder chain via attribute
            elif isinstance(node, ast.Attribute):
                if node.attr in _BANNED_ATTRS:
                    violations.append(f"Banned attribute access: .{node.attr}")
                # catch any __dunder__ not already listed
                if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in (
                    "__init__", "__str__", "__repr__", "__len__", "__iter__",
                    "__next__", "__enter__", "__exit__", "__call__", "__name__",
                    "__doc__", "__all__", "__slots__", "__annotations__",
                ):
                    violations.append(f"Suspicious dunder attribute: .{node.attr}")

            # string literals that name banned modules (e.g. __import__('os'))
            # and URL literals that point at private ranges or non-allowed hosts
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for banned in _BANNED_STRINGS:
                    if banned in node.value:
                        violations.append(f"Banned string literal containing: {banned!r}")
                        break
                if node.value.startswith(("http://", "https://")):
                    err = _check_url_literal(node.value, allowed_hosts)
                    if err:
                        violations.append(err)

        return AuditResult(safe=len(violations) == 0, violations=violations)
