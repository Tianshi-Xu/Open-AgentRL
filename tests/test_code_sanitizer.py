import re

from recipe.demystify.reward import _sanitize_code


CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)


def sanitize(snippet: str) -> str:
    return _sanitize_code(snippet, CODE_PATTERN)


def test_basic_expression_wrapped():
    result = sanitize("1 + 1")
    assert "print(1 + 1)" in result
    assert result.startswith("import os\nos.environ.setdefault")


def test_existing_print_not_wrapped():
    code = "print('hello')"
    result = sanitize(code)
    assert result.count("print('hello')") == 1
    assert "print(print" not in result


def test_indented_print_not_modified():
    code = "for i in range(3):\n    print(i)"
    result = sanitize(code)
    assert "for i in range(3):" in result
    assert "    print(i)" in result
    assert "print(for" not in result


def test_assignment_not_wrapped():
    code = "x = 42"
    result = sanitize(code)
    assert "print(x = 42)" not in result


def test_code_block_stripped():
    fenced = """```python\nprint('hi')\n```"""
    result = sanitize(fenced)
    assert "print('hi')" in result
    assert result.count("print('hi')") == 1


def test_multi_line_expression_wraps_last_value():
    code = "x = 1\ny = 2\nx + y"
    result = sanitize(code)
    assert "print(x + y)" in result


def test_control_flow_not_wrapped():
    code = "if True:\n    x = 1\nprint(x)"
    result = sanitize(code)
    # ensure trailing print preserved
    assert result.strip().endswith("print(x)")

