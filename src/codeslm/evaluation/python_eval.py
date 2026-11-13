"""Syntax checks and subprocess execution for Python functions.

Generated code runs in a child interpreter with a timeout, a CPU limit, an
address-space limit, and a clean environment. The child is not a container:
it can still read files the user can read. Docker is not installed on this
machine, so this is the isolation used until a container runtime is added.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_CHILD = r"""
import ast
import json
import resource
import sys

def _limit(resource_name, value):
    try:
        resource.setrlimit(resource_name, (value, value))
    except (ValueError, OSError):
        return

_limit(resource.RLIMIT_CPU, 2)
_limit(resource.RLIMIT_AS, 512 * 1024 * 1024)
_limit(resource.RLIMIT_FSIZE, 1_000_000)
_limit(resource.RLIMIT_NOFILE, 32)

payload = json.loads(sys.stdin.read())
source = payload["source"]
try:
    ast.parse(source)
except SyntaxError as exc:
    json.dump({"status": "syntax_error", "error": f"{exc.msg} line {exc.lineno}"}, sys.stdout)
    raise SystemExit(0)

namespace = {"__builtins__": __builtins__}
try:
    exec(compile(source, "<candidate>", "exec"), namespace, namespace)
except Exception as exc:
    json.dump({"status": "runtime_error", "error": type(exc).__name__}, sys.stdout)
    raise SystemExit(0)

function = namespace.get(payload["function"])
if not callable(function):
    json.dump({"status": "runtime_error", "error": "missing_function"}, sys.stdout)
    raise SystemExit(0)

results = []
for test in payload["tests"]:
    try:
        got = function(*test["args"])
        results.append({"ok": got == test["expected"], "error": ""})
    except Exception as exc:
        results.append({"ok": False, "error": type(exc).__name__})
json.dump({"status": "completed", "results": results}, sys.stdout)
"""


@dataclass(frozen=True)
class SyntaxResult:
    ok: bool
    error: str


@dataclass(frozen=True)
class FunctionTest:
    args: tuple[object, ...]
    expected: object


@dataclass(frozen=True)
class FunctionCase:
    name: str
    source: str
    function: str
    tests: tuple[FunctionTest, ...]


@dataclass(frozen=True)
class ExecutionResult:
    name: str
    status: str
    tests: int
    tests_passed: int
    error: str


@dataclass(frozen=True)
class ExecutionReport:
    results: tuple[ExecutionResult, ...]
    pass_rate: float
    syntax_error_rate: float
    runtime_error_rate: float
    test_pass_rate: float


def check_syntax(source: str) -> SyntaxResult:
    """Parse Python source. A syntax error is returned, not raised."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return SyntaxResult(ok=False, error=f"{exc.msg} line {exc.lineno}")
    return SyntaxResult(ok=True, error="")


def execute_function(
    case: FunctionCase,
    *,
    timeout: float,
) -> ExecutionResult:
    """Run one function in a child process. Syntax errors are not executed."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    syntax = check_syntax(case.source)
    if not syntax.ok:
        return ExecutionResult(
            name=case.name,
            status="syntax_error",
            tests=len(case.tests),
            tests_passed=0,
            error=syntax.error,
        )
    payload = json.dumps(
        {
            "source": case.source,
            "function": case.function,
            "tests": [{"args": list(test.args), "expected": test.expected} for test in case.tests],
        }
    )
    with tempfile.TemporaryDirectory(prefix="codeslm-eval-") as directory:
        try:
            completed = _run_child(payload, timeout=timeout, cwd=Path(directory))
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                name=case.name,
                status="timeout",
                tests=len(case.tests),
                tests_passed=0,
                error="timeout",
            )
    return _interpret(case, completed)


def execute_functions(cases: Sequence[FunctionCase], *, timeout: float) -> ExecutionReport:
    """Score several functions and summarize pass, syntax, and runtime rates."""
    if not cases:
        raise ValueError("no functions to execute")
    results = tuple(execute_function(case, timeout=timeout) for case in cases)
    return summarize(results)


def summarize(results: Sequence[ExecutionResult]) -> ExecutionReport:
    count = len(results)
    if count == 0:
        raise ValueError("no execution results")
    syntax_errors = sum(result.status == "syntax_error" for result in results)
    runtime_errors = sum(result.status in {"runtime_error", "timeout"} for result in results)
    passed = sum(result.status == "passed" for result in results)
    tests = sum(result.tests for result in results)
    tests_passed = sum(result.tests_passed for result in results)
    return ExecutionReport(
        results=tuple(results),
        pass_rate=passed / count,
        syntax_error_rate=syntax_errors / count,
        runtime_error_rate=runtime_errors / count,
        test_pass_rate=(tests_passed / tests) if tests else 0.0,
    )


def _run_child(payload: str, *, timeout: float, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.kill()
    process.communicate()


def _interpret(case: FunctionCase, completed: subprocess.CompletedProcess[str]) -> ExecutionResult:
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = (completed.stderr or "").strip().splitlines()
        message = detail[-1][:200] if detail else f"exit {completed.returncode}"
        status = "timeout" if completed.returncode < 0 else "runtime_error"
        return ExecutionResult(case.name, status, len(case.tests), 0, message)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ExecutionResult(case.name, "runtime_error", len(case.tests), 0, "child output was not JSON")
    status = payload.get("status")
    if status == "syntax_error":
        return ExecutionResult(case.name, "syntax_error", len(case.tests), 0, str(payload.get("error", "")))
    if status == "runtime_error":
        return ExecutionResult(case.name, "runtime_error", len(case.tests), 0, str(payload.get("error", "")))
    checks = payload.get("results", [])
    passed = sum(1 for check in checks if check.get("ok"))
    raised = any(check.get("error") for check in checks)
    if raised:
        error = next(check.get("error", "") for check in checks if check.get("error"))
        return ExecutionResult(case.name, "runtime_error", len(checks), passed, str(error))
    if passed == len(checks) and checks:
        return ExecutionResult(case.name, "passed", len(checks), passed, "")
    return ExecutionResult(case.name, "failed", len(checks), passed, "expected value mismatch")
