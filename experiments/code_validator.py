"""AST + bandit static safety checks for generated experiment code."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger()

_FORBIDDEN_NODES = {
    "subprocess",
    "os.system",
    "os.popen",
    "shutil.rmtree",
    "shutil.rmdir",
    "eval",
    "exec",
    "__import__",
}

_FORBIDDEN_CALLS = {
    "system",
    "popen",
    "rmtree",
    "rmdir",
}


class ValidationError(Exception):
    pass


def _check_ast(code: str) -> None:
    """Reject forbidden AST patterns."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        # Detect import of forbidden modules
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name in ("subprocess", "pty", "ctypes"):
                    raise ValidationError(f"Forbidden import: {name}")

        # Detect attribute access like os.system
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_CALLS:
                raise ValidationError(f"Forbidden attribute call: .{node.attr}")

        # Detect eval/exec calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "__import__"):
                    raise ValidationError(f"Forbidden call: {node.func.id}()")


def _check_bandit(code: str) -> None:
    """Run bandit and reject HIGH severity findings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", tmp_path, "-f", "json", "-ll"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # bandit exits 1 if issues found
        if result.stdout:
            report = json.loads(result.stdout)
            high_issues = [
                i for i in report.get("results", [])
                if i.get("issue_severity") == "HIGH"
            ]
            if high_issues:
                details = "; ".join(i.get("issue_text", "") for i in high_issues[:3])
                raise ValidationError(f"Bandit HIGH severity: {details}")
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        pass  # bandit not available or timeout — skip
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def validate(code: str) -> None:
    """Raise ValidationError if code fails safety checks."""
    _check_ast(code)
    _check_bandit(code)


def validate_with_retry(code: str, paper_id: str, context: str = "") -> tuple[str, bool]:
    """
    Try to validate code, ask Claude to fix if it fails (up to 3 iterations).
    Returns (validated_code, success).
    """
    import anthropic
    from config import settings

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    current_code = code
    for attempt in range(3):
        try:
            validate(current_code)
            return current_code, True
        except ValidationError as e:
            log.warning("code_validator.failed", attempt=attempt + 1, error=str(e))
            if attempt == 2:
                break
            # Ask Claude to fix
            try:
                fix_resp = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{
                        "role": "user",
                        "content": f"""The following Python experiment code failed safety validation:

Error: {e}

Code:
```python
{current_code}
```

Please fix the code to remove the security issue while preserving the experiment logic.
Return ONLY the fixed Python code, no explanation."""
                    }],
                )
                text = fix_resp.content[0].text if fix_resp.content else ""
                # Extract code block if present
                if "```python" in text:
                    current_code = text.split("```python")[1].split("```")[0].strip()
                elif "```" in text:
                    current_code = text.split("```")[1].split("```")[0].strip()
                else:
                    current_code = text.strip()
            except Exception as fix_err:
                log.error("code_validator.fix_failed", error=str(fix_err))
                break

    return current_code, False
