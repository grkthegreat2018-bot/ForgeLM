"""Test case registry mapping function names to (args, expected_output) test pairs.

NOTE: In the goal-oriented self-play system (research.goal_tasks), the I/O pair
format here is directly compatible with GoalTask.test_cases. The goal generator
in goal_tasks.py produces its own I/O pairs via trusted references, but this
registry remains useful as an additional source of verified test cases.

Exports:
    TESTS: dict mapping func_name -> list of test dicts.
    get_tests(func_name): returns list of (args_tuple, expected_result) pairs.
    has_tests(func_name): returns True if tests are registered for func_name.
    run_tests(code, func_name): execs code, runs tests, returns (all_passed, error_message).
"""

from typing import Any, List, Tuple, Dict
import math


TESTS: Dict[str, List[Dict[str, Any]]] = {
    # ----------------------------------------------------------------- #
    # Python General (easy)
    # ----------------------------------------------------------------- #
    "add": [
        {"args": (1, 2), "expected": 3},
        {"args": (0, 0), "expected": 0},
        {"args": (-1, 1), "expected": 0},
        {"args": (100, 200), "expected": 300},
    ],
    "subtract": [
        {"args": (5, 3), "expected": 2},
        {"args": (0, 0), "expected": 0},
        {"args": (10, 20), "expected": -10},
    ],
    "multiply": [
        {"args": (3, 4), "expected": 12},
        {"args": (0, 5), "expected": 0},
        {"args": (-2, 3), "expected": -6},
    ],
    "divide": [
        {"args": (10, 2), "expected": 5.0},
        {"args": (0, 5), "expected": 0},
        {"args": (7, 1), "expected": 7.0},
    ],
    "max_of_two": [
        {"args": (1, 2), "expected": 2},
        {"args": (5, 3), "expected": 5},
        {"args": (-1, -2), "expected": -1},
    ],
    "min_of_two": [
        {"args": (1, 2), "expected": 1},
        {"args": (5, 3), "expected": 3},
        {"args": (-1, -2), "expected": -2},
    ],
    "is_even": [
        {"args": (4,), "expected": True},
        {"args": (3,), "expected": False},
        {"args": (0,), "expected": True},
        {"args": (-2,), "expected": True},
    ],
    "is_odd": [
        {"args": (3,), "expected": True},
        {"args": (4,), "expected": False},
        {"args": (0,), "expected": False},
    ],
    "is_positive": [
        {"args": (5,), "expected": True},
        {"args": (0,), "expected": False},
        {"args": (-3,), "expected": False},
    ],
    "is_negative": [
        {"args": (-3,), "expected": True},
        {"args": (0,), "expected": False},
        {"args": (5,), "expected": False},
    ],
    "is_zero": [
        {"args": (0,), "expected": True},
        {"args": (1,), "expected": False},
        {"args": (-1,), "expected": False},
    ],
    "square": [
        {"args": (3,), "expected": 9},
        {"args": (0,), "expected": 0},
        {"args": (-4,), "expected": 16},
    ],
    "cube": [
        {"args": (2,), "expected": 8},
        {"args": (0,), "expected": 0},
        {"args": (-3,), "expected": -27},
    ],
    "double": [
        {"args": (5,), "expected": 10},
        {"args": (0,), "expected": 0},
        {"args": (-3,), "expected": -6},
    ],
    "triple": [
        {"args": (2,), "expected": 6},
        {"args": (0,), "expected": 0},
        {"args": (-3,), "expected": -9},
    ],
    "absolute": [
        {"args": (-5,), "expected": 5},
        {"args": (5,), "expected": 5},
        {"args": (0,), "expected": 0},
    ],
    "sign": [
        {"args": (5,), "expected": 1},
        {"args": (-3,), "expected": -1},
        {"args": (0,), "expected": 0},
    ],
    "greet": [
        {"args": ("World",), "expected": "Hello, World!"},
        {"args": ("Bob",), "expected": "Hello, Bob!"},
    ],
    "length": [
        {"args": ("hello",), "expected": 5},
        {"args": ("",), "expected": 0},
        {"args": ("a",), "expected": 1},
    ],
    "to_upper": [
        {"args": ("hello",), "expected": "HELLO"},
        {"args": ("",), "expected": ""},
    ],
    "to_lower": [
        {"args": ("HELLO",), "expected": "hello"},
        {"args": ("",), "expected": ""},
    ],
    "count_vowels": [
        {"args": ("hello",), "expected": 2},
        {"args": ("xyz",), "expected": 0},
        {"args": ("aeiou",), "expected": 5},
    ],
    "count_consonants": [
        {"args": ("hello",), "expected": 3},
        {"args": ("aeiou",), "expected": 0},
    ],
    "is_equal": [
        {"args": (1, 1), "expected": True},
        {"args": (1, 2), "expected": False},
    ],
    "is_greater": [
        {"args": (2, 1), "expected": True},
        {"args": (1, 2), "expected": False},
    ],
    "average": [
        {"args": (4, 6), "expected": 5.0},
        {"args": (0, 0), "expected": 0.0},
    ],
    "midpoint": [
        {"args": (0, 10), "expected": 5.0},
        {"args": (2, 4), "expected": 3.0},
    ],
    "negate": [
        {"args": (5,), "expected": -5},
        {"args": (-3,), "expected": 3},
        {"args": (0,), "expected": 0},
    ],
    "swap": [
        {"args": (1, 2), "expected": (2, 1)},
        {"args": ("a", "b"), "expected": ("b", "a")},
    ],
    "identity": [
        {"args": (42,), "expected": 42},
        {"args": ("hello",), "expected": "hello"},
    ],
    "factorial": [
        {"args": (0,), "expected": 1},
        {"args": (1,), "expected": 1},
        {"args": (5,), "expected": 120},
        {"args": (3,), "expected": 6},
    ],
    "fib": [
        {"args": (0,), "expected": 0},
        {"args": (1,), "expected": 1},
        {"args": (5,), "expected": 5},
        {"args": (10,), "expected": 55},
    ],
    "is_leap_year": [
        {"args": (2000,), "expected": True},
        {"args": (1900,), "expected": False},
        {"args": (2024,), "expected": True},
        {"args": (2023,), "expected": False},
    ],

    # ----------------------------------------------------------------- #
    # Python Math
    # ----------------------------------------------------------------- #
    "is_prime": [
        {"args": (2,), "expected": True},
        {"args": (3,), "expected": True},
        {"args": (4,), "expected": False},
        {"args": (1,), "expected": False},
        {"args": (0,), "expected": False},
        {"args": (17,), "expected": True},
        {"args": (15,), "expected": False},
    ],
    "gcd": [
        {"args": (12, 8), "expected": 4},
        {"args": (7, 13), "expected": 1},
        {"args": (0, 5), "expected": 5},
    ],
    "lcm": [
        {"args": (4, 6), "expected": 12},
        {"args": (3, 7), "expected": 21},
    ],
    "power": [
        {"args": (2, 3), "expected": 8},
        {"args": (5, 0), "expected": 1},
        {"args": (3, 2), "expected": 9},
    ],
    "sum_digits": [
        {"args": (123,), "expected": 6},
        {"args": (0,), "expected": 0},
        {"args": (999,), "expected": 27},
    ],
    "reverse_number": [
        {"args": (123,), "expected": 321},
        {"args": (100,), "expected": 1},
        {"args": (0,), "expected": 0},
    ],
    "count_digits": [
        {"args": (123,), "expected": 3},
        {"args": (0,), "expected": 1},
        {"args": (99999,), "expected": 5},
    ],
    "is_perfect_square": [
        {"args": (4,), "expected": True},
        {"args": (9,), "expected": True},
        {"args": (10,), "expected": False},
        {"args": (0,), "expected": True},
    ],
    "is_power_of_two": [
        {"args": (1,), "expected": True},
        {"args": (2,), "expected": True},
        {"args": (4,), "expected": True},
        {"args": (3,), "expected": False},
        {"args": (0,), "expected": False},
    ],
    "digital_root": [
        {"args": (123,), "expected": 6},
        {"args": (0,), "expected": 0},
        {"args": (999,), "expected": 9},
    ],
    "to_binary": [
        {"args": (5,), "expected": "101"},
        {"args": (0,), "expected": "0"},
        {"args": (10,), "expected": "1010"},
    ],
    "circle_area": [
        {"args": (1,), "expected": math.pi, "approx": True},
        {"args": (2,), "expected": math.pi * 4, "approx": True},
        {"args": (0,), "expected": 0.0, "approx": True},
    ],
    "triangle_area": [
        {"args": (4, 3), "expected": 6.0},
        {"args": (6, 2), "expected": 6.0},
    ],
    "pythagorean": [
        {"args": (3, 4), "expected": 5.0},
        {"args": (5, 12), "expected": 13.0},
    ],
    "nCr": [
        {"args": (5, 2), "expected": 10},
        {"args": (10, 3), "expected": 120},
        {"args": (5, 0), "expected": 1},
    ],
    "nPr": [
        {"args": (5, 2), "expected": 20},
        {"args": (3, 3), "expected": 6},
    ],

    # ----------------------------------------------------------------- #
    # Python Strings
    # ----------------------------------------------------------------- #
    "reverse": [
        {"args": ("hello",), "expected": "olleh"},
        {"args": ("",), "expected": ""},
        {"args": ("a",), "expected": "a"},
    ],
    "is_palindrome": [
        {"args": ("racecar",), "expected": True},
        {"args": ("hello",), "expected": False},
        {"args": ("",), "expected": True},
        {"args": ("a",), "expected": True},
    ],
    "is_anagram": [
        {"args": ("listen", "silent"), "expected": True},
        {"args": ("hello", "world"), "expected": False},
    ],
    "count_char": [
        {"args": ("hello", "l"), "expected": 2},
        {"args": ("", "a"), "expected": 0},
    ],
    "remove_spaces": [
        {"args": ("a b c",), "expected": "abc"},
        {"args": ("",), "expected": ""},
    ],
    "replace_vowels": [
        {"args": ("hello", "*"), "expected": "h*ll*"},
        {"args": ("xyz", "*"), "expected": "xyz"},
    ],
    "capitalize_words": [
        {"args": ("hello world",), "expected": "Hello World"},
        {"args": ("",), "expected": ""},
    ],
    "levenshtein": [
        {"args": ("kitten", "sitting"), "expected": 3},
        {"args": ("", ""), "expected": 0},
        {"args": ("abc", "abc"), "expected": 0},
    ],
    "hamming_distance": [
        {"args": ("abc", "abc"), "expected": 0},
        {"args": ("abc", "abd"), "expected": 1},
    ],
    "caesar_cipher": [
        {"args": ("abc", 1), "expected": "bcd"},
        {"args": ("xyz", 3), "expected": "abc"},
    ],
    "rot13": [
        {"args": ("hello",), "expected": "uryyb"},
        {"args": ("abc",), "expected": "nop"},
    ],
    "run_length_encode": [
        {"args": ("aaabb",), "expected": "3a2b"},
        {"args": ("",), "expected": ""},
    ],
    "camel_to_snake": [
        {"args": ("camelCase",), "expected": "camel_case"},
        {"args": ("SnakeCase",), "expected": "snake_case"},
    ],

    # ----------------------------------------------------------------- #
    # Python Algorithms
    # ----------------------------------------------------------------- #
    "bubble_sort": [
        {"args": ([3, 1, 2],), "expected": [1, 2, 3]},
        {"args": ([],), "expected": []},
        {"args": ([1],), "expected": [1]},
    ],
    "binary_search": [
        {"args": ([1, 2, 3, 4, 5], 3), "expected": 2},
        {"args": ([1, 2, 3], 5), "expected": -1},
    ],
    "linear_search": [
        {"args": ([3, 1, 2], 1), "expected": 1},
        {"args": ([1, 2, 3], 5), "expected": -1},
    ],
    "find_max": [
        {"args": ([1, 5, 3],), "expected": 5},
        {"args": ([-1, -5],), "expected": -1},
    ],
    "find_min": [
        {"args": ([1, 5, 3],), "expected": 1},
        {"args": ([-1, -5],), "expected": -5},
    ],
    "remove_duplicates": [
        {"args": ([1, 1, 2, 3, 3],), "expected": [1, 2, 3]},
        {"args": ([],), "expected": []},
    ],
    "reverse_list": [
        {"args": ([1, 2, 3],), "expected": [3, 2, 1]},
        {"args": ([],), "expected": []},
    ],
    "merge_sorted": [
        {"args": ([1, 3], [2, 4]), "expected": [1, 2, 3, 4]},
        {"args": ([], [1]), "expected": [1]},
    ],
    "is_sorted": [
        {"args": ([1, 2, 3],), "expected": True},
        {"args": ([3, 2, 1],), "expected": False},
        {"args": ([],), "expected": True},
    ],
    "sum_list": [
        {"args": ([1, 2, 3],), "expected": 6},
        {"args": ([],), "expected": 0},
    ],
    "two_sum": [
        {"args": ([2, 7, 11, 15], 9), "expected": [0, 1]},
    ],
    "max_subarray": [
        {"args": ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), "expected": 6},
    ],
    "climb_stairs": [
        {"args": (1,), "expected": 1},
        {"args": (2,), "expected": 2},
        {"args": (3,), "expected": 3},
        {"args": (5,), "expected": 8},
    ],
    "coin_change": [
        {"args": ([1, 5, 10], 11), "expected": 2},
    ],
    "valid_parens": [
        {"args": ("()",), "expected": True},
        {"args": ("()[]{}",), "expected": True},
        {"args": ("(]",), "expected": False},
    ],
    "pascal_row": [
        {"args": (0,), "expected": [1]},
        {"args": (1,), "expected": [1, 1]},
        {"args": (2,), "expected": [1, 2, 1]},
    ],

    # ----------------------------------------------------------------- #
    # Python OOP (class tests)
    # ----------------------------------------------------------------- #
    "Dog": [
        {"type": "class", "init_args": ("Rex",), "method": "name", "method_args": (), "expected": "Rex"},
    ],
    "Counter": [
        {"type": "class", "init_args": (), "method": "count", "method_args": (), "expected": 0},
        {"type": "class", "init_args": (), "method": "increment", "method_args": (), "expected_chain": True,
         "chain_method": "count", "chain_args": (), "expected": 1},
    ],
    "Rectangle": [
        {"type": "class", "init_args": (3, 4), "method": "area", "method_args": (), "expected": 12},
    ],
    "Stack": [
        {"type": "class", "init_args": (), "method": "push", "method_args": (1,), "expected_chain": True,
         "chain_method": "pop", "chain_args": (), "expected": 1},
    ],
}


# ── Merge auto-generated tests ─────────────────────────────────────
try:
    from research.evaluation.prompt_tests_auto import TESTS_AUTO
    for _name, _tests in TESTS_AUTO.items():
        if _name not in TESTS:
            TESTS[_name] = _tests
except ImportError:
    pass

# ── Merge g2 tests (subagent-generated) ────────────────────────────
try:
    from research.evaluation.prompt_tests_g2 import TESTS_G2
    for _name, _tests in TESTS_G2.items():
        if _name not in TESTS:
            TESTS[_name] = _tests
except ImportError:
    pass


def get_tests(func_name: str) -> List[Tuple[tuple, Any]]:
    """Return a list of (args_tuple, expected_result) pairs for a function name.

    For class tests, returns the raw test dicts (with type='class') so callers
    can distinguish them. Returns an empty list if no tests are registered.
    """
    tests = TESTS.get(func_name, [])
    result: List[Tuple[tuple, Any]] = []
    for t in tests:
        if t.get("type") == "class":
            # Return the dict itself wrapped so callers can detect class tests.
            result.append((t,))  # type: ignore[arg-type]
        else:
            result.append((t["args"], t["expected"]))
    return result


def has_tests(func_name: str) -> bool:
    """Return True if tests are registered for the given function name."""
    return func_name in TESTS and len(TESTS[func_name]) > 0


def _values_equal(actual: Any, expected: Any, approx: bool = False) -> bool:
    """Compare actual vs expected, with optional float approximation."""
    if approx:
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def run_tests(code: str, func_name: str) -> Tuple[bool, str]:
    """Execute the given code, run all registered tests for func_name.

    Returns (True, "") if all tests pass.
    Returns (False, message) if any test fails or an error occurs.
    """
    tests = TESTS.get(func_name)
    if not tests:
        return False, f"no tests registered for '{func_name}'"

    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as e:  # noqa: BLE001
        return False, f"code execution failed: {type(e).__name__}: {e}"

    if func_name not in namespace:
        return False, f"'{func_name}' not found in executed code"

    obj = namespace[func_name]

    for i, test in enumerate(tests):
        try:
            if test.get("type") == "class":
                # Build instance
                instance = obj(*test["init_args"])
                # Invoke method (or attribute access)
                method = getattr(instance, test["method"])
                if test.get("expected_chain"):
                    # method returns self (or instance) for chaining
                    chained = method(*test["method_args"])
                    chain_attr = getattr(chained, test["chain_method"])
                    if callable(chain_attr):
                        actual = chain_attr(*test["chain_args"])
                    else:
                        actual = chain_attr  # attribute access
                else:
                    if callable(method):
                        actual = method(*test["method_args"])
                    else:
                        # attribute access
                        actual = method
                expected = test["expected"]
            else:
                args = test["args"]
                expected = test["expected"]
                actual = obj(*args)
        except Exception as e:  # noqa: BLE001
            return False, f"test {i} raised: {type(e).__name__}: {e}"

        if not _values_equal(actual, expected, approx=test.get("approx", False)):
            return False, f"test {i} failed: expected {test['expected']!r} got {actual!r}"

    return True, ""


if __name__ == "__main__":
    # Quick self-verification of the registry and helpers.
    print("Registered functions:", len(TESTS))
    print("Total test cases:", sum(len(v) for v in TESTS.values()))

    # Verify get_tests / has_tests
    assert has_tests("add"), "add should have tests"
    assert not has_tests("nonexistent_func"), "nonexistent_func should not have tests"
    add_tests = get_tests("add")
    assert add_tests[0] == ((1, 2), 3), f"unexpected: {add_tests[0]}"
    assert get_tests("nonexistent") == [], "expected empty list for unknown func"

    # Verify run_tests with a simple passing implementation
    add_code = "def add(a, b):\n    return a + b\n"
    ok, msg = run_tests(add_code, "add")
    assert ok, f"add tests should pass: {msg}"

    # Verify run_tests with a failing implementation
    bad_add_code = "def add(a, b):\n    return a - b\n"
    ok, msg = run_tests(bad_add_code, "add")
    assert not ok, "bad add should fail"
    assert "test 0 failed" in msg, f"unexpected message: {msg}"

    # Verify a class test (Counter)
    counter_code = """
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self
"""
    ok, msg = run_tests(counter_code, "Counter")
    assert ok, f"Counter tests should pass: {msg}"

    # Verify factorial (appears in both general and algorithms sections; registry has one entry)
    fact_code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n"
    ok, msg = run_tests(fact_code, "factorial")
    assert ok, f"factorial tests should pass: {msg}"

    # Verify circle_area with approx
    circle_code = "import math\ndef circle_area(r):\n    return math.pi * r * r\n"
    ok, msg = run_tests(circle_code, "circle_area")
    assert ok, f"circle_area tests should pass: {msg}"

    # Verify missing function detection
    ok, msg = run_tests("x = 1\n", "add")
    assert not ok and "not found" in msg, f"unexpected: {ok}, {msg}"

    print("All self-verification checks passed.")
