"""Test case registry (Group 2) mapping function names to test pairs.

Exports:
    TESTS_G2: dict mapping func_name -> list of test dicts.
"""

import math
from typing import Any, Dict, List

TESTS_G2: dict[str, list[dict[str, Any]]] = {
    # ----------------------------------------------------------------- #
    # find_* functions
    # ----------------------------------------------------------------- #
    "find_in_file": [
        {"args": ("hello\nworld\n", "world"), "expected": True},
        {"args": ("hello\nworld\n", "xyz"), "expected": False},
        {"args": ("", "anything"), "expected": False},
    ],
    "find_in_matrix": [
        {"args": ([[1, 2], [3, 4]], 3), "expected": True},
        {"args": ([[1, 2], [3, 4]], 5), "expected": False},
        {"args": ([[], []], 1), "expected": False},
    ],
    "find_insert_position": [
        {"args": ([1, 3, 5, 6], 5), "expected": 2},
        {"args": ([1, 3, 5, 6], 2), "expected": 1},
        {"args": ([1, 3, 5, 6], 7), "expected": 4},
        {"args": ([], 1), "expected": 0},
    ],
    "find_kth_largest": [
        {"args": ([3, 1, 4, 1, 5], 2), "expected": 4},
        {"args": ([1], 1), "expected": 1},
        {"args": ([5, 5, 5], 1), "expected": 5},
    ],
    "find_kth_smallest": [
        {"args": ([3, 1, 4, 1, 5], 2), "expected": 1},
        {"args": ([1], 1), "expected": 1},
        {"args": ([5, 3, 1], 1), "expected": 1},
    ],
    "find_last_sorted": [
        {"args": ([1, 2, 3, 3, 4], 3), "expected": 3},
        {"args": ([1, 2, 3], 5), "expected": -1},
        {"args": ([], 1), "expected": -1},
    ],
    "find_majority": [
        {"args": ([1, 1, 1, 2],), "expected": 1},
        {"args": ([1, 2, 3],), "expected": None},
        {"args": ([1],), "expected": 1},
    ],
    "find_median": [
        {"args": ([1, 2, 3],), "expected": 2},
        {"args": ([1, 2, 3, 4],), "expected": 2.5, "approx": True},
        {"args": ([5],), "expected": 5},
    ],
    "find_median_sorted_arrays": [
        {"args": ([1, 3], [2]), "expected": 2.0, "approx": True},
        {"args": ([1, 2], [3, 4]), "expected": 2.5, "approx": True},
        {"args": ([], [1]), "expected": 1.0, "approx": True},
    ],
    "find_mode_quickselect": [
        {"args": ([1, 1, 2, 3],), "expected": 1},
        {"args": ([5, 5, 5, 5],), "expected": 5},
        {"args": ([1],), "expected": 1},
    ],
    "find_peak": [
        {"args": ([1, 2, 3, 1],), "expected": 3},
        {"args": ([1, 2, 3, 4],), "expected": 4},
        {"args": ([1],), "expected": 1},
    ],
    "find_peak_binary": [
        {"args": ([1, 2, 3, 1],), "expected": 3},
        {"args": ([1, 3, 2, 1],), "expected": 3},
        {"args": ([1],), "expected": 1},
    ],
    "find_second_max": [
        {"args": ([1, 3, 2, 5],), "expected": 3},
        {"args": ([5, 5, 3],), "expected": 3},
        {"args": ([1, 2],), "expected": 1},
    ],
    "find_second_min": [
        {"args": ([5, 3, 2, 1],), "expected": 2},
        {"args": ([1, 1, 3],), "expected": 3},
        {"args": ([3, 1],), "expected": 3},
    ],
    "find_sublist": [
        {"args": ([1, 2, 3, 4], [2, 3]), "expected": True},
        {"args": ([1, 2, 3], [4, 5]), "expected": False},
        {"args": ([1, 2, 3], []), "expected": True},
    ],

    # ----------------------------------------------------------------- #
    # first_* functions
    # ----------------------------------------------------------------- #
    "first_char": [
        {"args": ("hello",), "expected": "h"},
        {"args": ("a",), "expected": "a"},
        {"args": ("",), "expected": ""},
    ],
    "first_element": [
        {"args": ([1, 2, 3],), "expected": 1},
        {"args": (["a"],), "expected": "a"},
        {"args": ([],), "expected": None},
    ],
    "first_greater_than": [
        {"args": ([1, 3, 5, 7], 4), "expected": 5},
        {"args": ([1, 2, 3], 10), "expected": None},
        {"args": ([], 1), "expected": None},
    ],
    "first_index": [
        {"args": ([1, 2, 3], 2), "expected": 1},
        {"args": ([1, 2, 3], 5), "expected": -1},
        {"args": ([], 1), "expected": -1},
    ],
    "first_less_than": [
        {"args": ([5, 3, 1, 7], 4), "expected": 3},
        {"args": ([1, 2, 3], 0), "expected": None},
        {"args": ([], 1), "expected": None},
    ],
    "first_missing_positive": [
        {"args": ([1, 2, 0],), "expected": 3},
        {"args": ([3, 4, -1, 1],), "expected": 2},
        {"args": ([1, 2, 3],), "expected": 4},
        {"args": ([],), "expected": 1},
    ],
    "first_unique": [
        {"args": ("aabbcde",), "expected": "c"},
        {"args": ("aabb",), "expected": None},
        {"args": ("x",), "expected": "x"},
    ],
    "first_word": [
        {"args": ("hello world",), "expected": "hello"},
        {"args": ("one",), "expected": "one"},
        {"args": ("  leading",), "expected": "leading"},
    ],

    # ----------------------------------------------------------------- #
    # flatten_* functions
    # ----------------------------------------------------------------- #
    "flatten_deep": [
        {"args": ([1, [2, [3, [4]]]],), "expected": [1, 2, 3, 4]},
        {"args": ([],), "expected": []},
        {"args": ([1, 2, 3],), "expected": [1, 2, 3]},
    ],
    "flatten_list": [
        {"args": ([1, [2, 3], [4]],), "expected": [1, 2, 3, 4]},
        {"args": ([],), "expected": []},
        {"args": ([1, 2],), "expected": [1, 2]},
    ],
    "flatten_once": [
        {"args": ([1, [2, 3], [4]],), "expected": [1, 2, 3, 4]},
        {"args": ([],), "expected": []},
        {"args": ([1, [2, [3]]],), "expected": [1, 2, [3]]},
    ],
    "flatten_one_level": [
        {"args": ([1, [2, 3], [4]],), "expected": [1, 2, 3, 4]},
        {"args": ([],), "expected": []},
        {"args": ([1, [2, [3]]],), "expected": [1, 2, [3]]},
    ],
    "flatten_pairs": [
        {"args": ([(1, 2), (3, 4)],), "expected": [1, 2, 3, 4]},
        {"args": ([],), "expected": []},
        {"args": ([(1, 2)],), "expected": [1, 2]},
    ],
    "flatten_triples": [
        {"args": ([(1, 2, 3), (4, 5, 6)],), "expected": [1, 2, 3, 4, 5, 6]},
        {"args": ([],), "expected": []},
        {"args": ([(1, 2, 3)],), "expected": [1, 2, 3]},
    ],

    # __INSERT_POINT__
}


if __name__ == "__main__":
    # Load the list of required function names
    import json
    import os

    _tmp_path = os.path.join(os.path.dirname(__file__), "..", ".devin", "tmp", "missing_funcs_g2.json")
    if os.path.exists(_tmp_path):
        with open(_tmp_path) as f:
            required = json.load(f)
    else:
        required = []

    missing = [name for name in required if name not in TESTS_G2 or len(TESTS_G2[name]) == 0]
    if missing:
        print(f"MISSING tests for {len(missing)} functions: {missing[:20]}...")
        raise SystemExit(1)

    print(f"TESTS_G2: {len(TESTS_G2)} functions registered")
    print(f"Total test cases: {sum(len(v) for v in TESTS_G2.values())}")
    print("All functions have at least 1 test case.")
