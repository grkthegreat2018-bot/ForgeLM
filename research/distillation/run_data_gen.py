"""Data generation loop: generates verified training data from free API providers.

Runs the distillation client against a set of coding tasks with test cases,
verifies solutions, and saves correct ones to a JSONL file for SFT training.

Usage:
    python -m research.distillation.run_data_gen [--n-samples N] [--output PATH]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Load .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

# Fix Windows encoding
os.environ.setdefault("PYTHONUTF8", "1")

from research.distillation.distill_client import DistillationClient
from research.distillation.agentic_distill import AgenticDistillClient


# ─── Coding Tasks with Test Cases ───────────────────────────────────────
# Each task has: goal (prompt), test_cases (input→output pairs for verification)
# Verification runs the solution as a Python function and checks outputs.

TASKS = [
    # ── Basic algorithms (20) ──
    {
        "goal": "Write a Python function `fib(n)` that returns the nth Fibonacci number. fib(0)=0, fib(1)=1.",
        "test_cases": [
            {"input": "0", "output": "0"},
            {"input": "1", "output": "1"},
            {"input": "5", "output": "5"},
            {"input": "10", "output": "55"},
            {"input": "20", "output": "6765"},
        ],
    },
    {
        "goal": "Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise.",
        "test_cases": [
            {"input": "2", "output": "True"},
            {"input": "7", "output": "True"},
            {"input": "10", "output": "False"},
            {"input": "1", "output": "False"},
            {"input": "13", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `factorial(n)` that returns n! (n factorial).",
        "test_cases": [
            {"input": "0", "output": "1"},
            {"input": "1", "output": "1"},
            {"input": "5", "output": "120"},
            {"input": "10", "output": "3628800"},
        ],
    },
    {
        "goal": "Write a Python function `gcd(a, b)` that returns the greatest common divisor using Euclid's algorithm.",
        "test_cases": [
            {"input": "12, 8", "output": "4"},
            {"input": "17, 5", "output": "1"},
            {"input": "100, 50", "output": "50"},
            {"input": "7, 13", "output": "1"},
        ],
    },
    {
        "goal": "Write a Python function `power(base, exp)` that returns base raised to exp without using ** or pow().",
        "test_cases": [
            {"input": "2, 3", "output": "8"},
            {"input": "5, 0", "output": "1"},
            {"input": "3, 4", "output": "81"},
            {"input": "10, 2", "output": "100"},
            {"input": "2, 10", "output": "1024"},
        ],
    },
    {
        "goal": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1 if not found.",
        "test_cases": [
            {"input": "[1,2,3,4,5], 3", "output": "2"},
            {"input": "[1,2,3,4,5], 6", "output": "-1"},
            {"input": "[1,2,3,4,5], 1", "output": "0"},
            {"input": "[1,2,3,4,5], 5", "output": "4"},
            {"input": "[], 1", "output": "-1"},
        ],
    },
    {
        "goal": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
        "test_cases": [
            {"input": "[1,3,5], [2,4,6]", "output": "[1, 2, 3, 4, 5, 6]"},
            {"input": "[], [1,2]", "output": "[1, 2]"},
            {"input": "[1,2], []", "output": "[1, 2]"},
            {"input": "[1,1,3], [2,2,4]", "output": "[1, 1, 2, 2, 3, 4]"},
        ],
    },
    {
        "goal": "Write a Python function `quick_sort(arr)` that returns a sorted list using quicksort.",
        "test_cases": [
            {"input": "[3,1,4,1,5,9,2,6]", "output": "[1, 1, 2, 3, 4, 5, 6, 9]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"},
            {"input": "[5,4,3,2,1]", "output": "[1, 2, 3, 4, 5]"},
        ],
    },
    {
        "goal": "Write a Python function `bubble_sort(arr)` that returns a sorted list using bubble sort.",
        "test_cases": [
            {"input": "[3,1,2]", "output": "[1, 2, 3]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"},
            {"input": "[5,4,3,2,1]", "output": "[1, 2, 3, 4, 5]"},
        ],
    },
    {
        "goal": "Write a Python function `insertion_sort(arr)` that returns a sorted list using insertion sort.",
        "test_cases": [
            {"input": "[3,1,4,1,5]", "output": "[1, 1, 3, 4, 5]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"},
            {"input": "[2,1]", "output": "[1, 2]"},
        ],
    },
    {
        "goal": "Write a Python function `lcm(a, b)` that returns the least common multiple of a and b.",
        "test_cases": [
            {"input": "4, 6", "output": "12"},
            {"input": "3, 4", "output": "12"},
            {"input": "12, 18", "output": "36"},
            {"input": "7, 13", "output": "91"},
        ],
    },
    {
        "goal": "Write a Python function `is_perfect(n)` that returns True if n is a perfect number (sum of proper divisors equals n).",
        "test_cases": [
            {"input": "6", "output": "True"},
            {"input": "28", "output": "True"},
            {"input": "12", "output": "False"},
            {"input": "496", "output": "True"},
            {"input": "1", "output": "False"},
        ],
    },
    {
        "goal": "Write a Python function `collatz(n)` that returns the Collatz sequence length starting from n.",
        "test_cases": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "2"},
            {"input": "3", "output": "8"},
            {"input": "6", "output": "9"},
            {"input": "27", "output": "112"},
        ],
    },
    {
        "goal": "Write a Python function `digit_sum(n)` that returns the sum of digits of a non-negative integer.",
        "test_cases": [
            {"input": "123", "output": "6"},
            {"input": "0", "output": "0"},
            {"input": "999", "output": "27"},
            {"input": "1001", "output": "2"},
        ],
    },
    {
        "goal": "Write a Python function `reverse_number(n)` that reverses the digits of a non-negative integer.",
        "test_cases": [
            {"input": "123", "output": "321"},
            {"input": "0", "output": "0"},
            {"input": "100", "output": "1"},
            {"input": "1200", "output": "21"},
        ],
    },
    {
        "goal": "Write a Python function `is_armstrong(n)` that returns True if n is an Armstrong number (sum of its digits raised to the power of digit count equals n).",
        "test_cases": [
            {"input": "153", "output": "True"},
            {"input": "370", "output": "True"},
            {"input": "9474", "output": "True"},
            {"input": "100", "output": "False"},
            {"input": "1", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `to_binary(n)` that returns the binary representation of a non-negative integer as a string.",
        "test_cases": [
            {"input": "0", "output": "'0'"},
            {"input": "1", "output": "'1'"},
            {"input": "10", "output": "'1010'"},
            {"input": "255", "output": "'11111111'"},
        ],
    },
    {
        "goal": "Write a Python function `from_binary(s)` that converts a binary string to an integer.",
        "test_cases": [
            {"input": "'0'", "output": "0"},
            {"input": "'1'", "output": "1"},
            {"input": "'1010'", "output": "10"},
            {"input": "'11111111'", "output": "255"},
        ],
    },
    {
        "goal": "Write a Python function `count_bits(n)` that returns the number of 1 bits in the binary representation of n.",
        "test_cases": [
            {"input": "0", "output": "0"},
            {"input": "1", "output": "1"},
            {"input": "7", "output": "3"},
            {"input": "255", "output": "8"},
            {"input": "1023", "output": "10"},
        ],
    },
    {
        "goal": "Write a Python function `tower_of_hanoi(n)` that returns the minimum number of moves to solve Tower of Hanoi with n disks.",
        "test_cases": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "3"},
            {"input": "3", "output": "7"},
            {"input": "4", "output": "15"},
            {"input": "5", "output": "31"},
        ],
    },
    # ── String manipulation (12) ──
    {
        "goal": "Write a Python function `reverse_string(s)` that returns the reversed string.",
        "test_cases": [
            {"input": "'hello'", "output": "'olleh'"},
            {"input": "'abc'", "output": "'cba'"},
            {"input": "''", "output": "''"},
            {"input": "'a'", "output": "'a'"},
        ],
    },
    {
        "goal": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
        "test_cases": [
            {"input": "'racecar'", "output": "True"},
            {"input": "'hello'", "output": "False"},
            {"input": "'aba'", "output": "True"},
            {"input": "''", "output": "True"},
            {"input": "'a'", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `count_words(s)` that returns the number of words in string s (split by spaces).",
        "test_cases": [
            {"input": "'hello world'", "output": "2"},
            {"input": "''", "output": "0"},
            {"input": "'one'", "output": "1"},
            {"input": "'  a  b  c  '", "output": "3"},
        ],
    },
    {
        "goal": "Write a Python function `capitalize_words(s)` that capitalizes the first letter of each word in s.",
        "test_cases": [
            {"input": "'hello world'", "output": "'Hello World'"},
            {"input": "'python programming'", "output": "'Python Programming'"},
            {"input": "''", "output": "''"},
            {"input": "'a'", "output": "'A'"},
        ],
    },
    {
        "goal": "Write a Python function `is_anagram(a, b)` that returns True if strings a and b are anagrams.",
        "test_cases": [
            {"input": "'listen', 'silent'", "output": "True"},
            {"input": "'hello', 'world'", "output": "False"},
            {"input": "'', ''", "output": "True"},
            {"input": "'a', 'a'", "output": "True"},
            {"input": "'ab', 'ba'", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `compress(s)` that does basic string compression (e.g. 'aabbb' -> 'a2b3').",
        "test_cases": [
            {"input": "'aabbb'", "output": "'a2b3'"},
            {"input": "'abc'", "output": "'a1b1c1'"},
            {"input": "''", "output": "''"},
            {"input": "'aaaa'", "output": "'a4'"},
        ],
    },
    {
        "goal": "Write a Python function `decompress(s)` that decompresses a string like 'a2b3' to 'aabbb'.",
        "test_cases": [
            {"input": "'a2b3'", "output": "'aabbb'"},
            {"input": "'a1b1c1'", "output": "'abc'"},
            {"input": "''", "output": "''"},
            {"input": "'a4'", "output": "'aaaa'"},
        ],
    },
    {
        "goal": "Write a Python function `count_vowels(s)` that returns the number of vowels (a,e,i,o,u) in s (case-insensitive).",
        "test_cases": [
            {"input": "'hello'", "output": "2"},
            {"input": "'AEIOU'", "output": "5"},
            {"input": "''", "output": "0"},
            {"input": "'xyz'", "output": "0"},
            {"input": "'aAeE'", "output": "4"},
        ],
    },
    {
        "goal": "Write a Python function `count_consonants(s)` that returns the number of consonants in s (case-insensitive, alpha only).",
        "test_cases": [
            {"input": "'hello'", "output": "3"},
            {"input": "'aeiou'", "output": "0"},
            {"input": "''", "output": "0"},
            {"input": "'XYZ'", "output": "3"},
        ],
    },
    {
        "goal": "Write a Python function `replace_vowels(s, ch)` that replaces all vowels in s with character ch.",
        "test_cases": [
            {"input": "'hello', '*'", "output": "'h*ll*'"},
            {"input": "'AEIOU', '-'", "output": "'-----'"},
            {"input": "'xyz', '_'", "output": "'xyz'"},
            {"input": "'', '*'", "output": "''"},
        ],
    },
    {
        "goal": "Write a Python function `camel_to_snake(s)` that converts a camelCase string to snake_case.",
        "test_cases": [
            {"input": "'helloWorld'", "output": "'hello_world'"},
            {"input": "'getHTTPResponse'", "output": "'get_http_response'"},
            {"input": "'simple'", "output": "'simple'"},
            {"input": "'aB'", "output": "'a_b'"},
        ],
    },
    {
        "goal": "Write a Python function `snake_to_camel(s)` that converts a snake_case string to camelCase.",
        "test_cases": [
            {"input": "'hello_world'", "output": "'helloWorld'"},
            {"input": "'get_http_response'", "output": "'getHttpResponse'"},
            {"input": "'simple'", "output": "'simple'"},
            {"input": "'a_b'", "output": "'aB'"},
        ],
    },
    # ── Data structures (10) ──
    {
        "goal": "Write a Python function `sum_list(lst)` that returns the sum of all elements in a list.",
        "test_cases": [
            {"input": "[1,2,3,4,5]", "output": "15"},
            {"input": "[]", "output": "0"},
            {"input": "[10]", "output": "10"},
            {"input": "[-1,1]", "output": "0"},
        ],
    },
    {
        "goal": "Write a Python function `max_element(lst)` that returns the maximum element in a non-empty list.",
        "test_cases": [
            {"input": "[1,3,2,5,4]", "output": "5"},
            {"input": "[10]", "output": "10"},
            {"input": "[-1,-5,-3]", "output": "-1"},
            {"input": "[0,0,0]", "output": "0"},
        ],
    },
    {
        "goal": "Write a Python function `min_element(lst)` that returns the minimum element in a non-empty list.",
        "test_cases": [
            {"input": "[1,3,2,5,4]", "output": "1"},
            {"input": "[10]", "output": "10"},
            {"input": "[-1,-5,-3]", "output": "-5"},
            {"input": "[0,0,0]", "output": "0"},
        ],
    },
    {
        "goal": "Write a Python function `remove_duplicates(lst)` that returns a new list with duplicates removed, preserving order.",
        "test_cases": [
            {"input": "[1,2,2,3,3,3,4]", "output": "[1, 2, 3, 4]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"},
            {"input": "[5,5,5,5]", "output": "[5]"},
        ],
    },
    {
        "goal": "Write a Python function `flatten(nested)` that flattens a nested list one level deep.",
        "test_cases": [
            {"input": "[[1,2],[3,4]]", "output": "[1, 2, 3, 4]"},
            {"input": "[[1],[2],[3]]", "output": "[1, 2, 3]"},
            {"input": "[]", "output": "[]"},
            {"input": "[[],[1]]", "output": "[1]"},
        ],
    },
    {
        "goal": "Write a Python function `matrix_transpose(m)` that returns the transpose of a 2D matrix.",
        "test_cases": [
            {"input": "[[1,2],[3,4]]", "output": "[[1, 3], [2, 4]]"},
            {"input": "[[1,2,3]]", "output": "[[1], [2], [3]]"},
            {"input": "[[1],[2],[3]]", "output": "[[1, 2, 3]]"},
        ],
    },
    {
        "goal": "Write a Python function `matrix_multiply(a, b)` that multiplies two 2D matrices.",
        "test_cases": [
            {"input": "[[1,2],[3,4]], [[5,6],[7,8]]", "output": "[[19, 22], [43, 50]]"},
            {"input": "[[1,2,3]], [[4],[5],[6]]", "output": "[[32]]"},
            {"input": "[[1]], [[2]]", "output": "[[2]]"},
        ],
    },
    {
        "goal": "Write a Python function `rotate_list(lst, k)` that rotates list right by k positions.",
        "test_cases": [
            {"input": "[1,2,3,4,5], 2", "output": "[4, 5, 1, 2, 3]"},
            {"input": "[1,2,3], 0", "output": "[1, 2, 3]"},
            {"input": "[1,2,3], 3", "output": "[1, 2, 3]"},
            {"input": "[1,2,3], 1", "output": "[3, 1, 2]"},
        ],
    },
    {
        "goal": "Write a Python function `chunk(lst, size)` that splits a list into chunks of the given size.",
        "test_cases": [
            {"input": "[1,2,3,4,5,6], 2", "output": "[[1, 2], [3, 4], [5, 6]]"},
            {"input": "[1,2,3,4,5], 2", "output": "[[1, 2], [3, 4], [5]]"},
            {"input": "[], 3", "output": "[]"},
            {"input": "[1], 1", "output": "[[1]]"},
        ],
    },
    {
        "goal": "Write a Python function `stack_sort(stack)` that sorts a list treated as a stack (only using pop and append) and returns the sorted list.",
        "test_cases": [
            {"input": "[3,1,4,1,5,9,2,6]", "output": "[1, 1, 2, 3, 4, 5, 6, 9]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"},
            {"input": "[5,4,3,2,1]", "output": "[1, 2, 3, 4, 5]"},
        ],
    },
    # ── Validation & parsing (8) ──
    {
        "goal": "Write a Python function `valid_parens(s)` that returns True if parentheses in s are balanced.",
        "test_cases": [
            {"input": "'()'", "output": "True"},
            {"input": "'()[]{}'", "output": "True"},
            {"input": "'(]'", "output": "False"},
            {"input": "'([)]'", "output": "False"},
            {"input": "''", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `is_valid_email(s)` that returns True if s is a valid email address (contains @ with text before and after, and a domain with a dot).",
        "test_cases": [
            {"input": "'user@example.com'", "output": "True"},
            {"input": "'invalid'", "output": "False"},
            {"input": "'@example.com'", "output": "False"},
            {"input": "'user@'", "output": "False"},
            {"input": "'a@b.co'", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `is_valid_ip(s)` that returns True if s is a valid IPv4 address (4 octets 0-255 separated by dots).",
        "test_cases": [
            {"input": "'192.168.1.1'", "output": "True"},
            {"input": "'255.255.255.255'", "output": "True"},
            {"input": "'0.0.0.0'", "output": "True"},
            {"input": "'256.1.1.1'", "output": "False"},
            {"input": "'1.2.3'", "output": "False"},
        ],
    },
    {
        "goal": "Write a Python function `parse_csv_line(s)` that parses a CSV line into a list of strings (handle quoted fields with commas).",
        "test_cases": [
            {"input": "'a,b,c'", "output": "['a', 'b', 'c']"},
            {"input": "'hello'", "output": "['hello']"},
            {"input": "'\"a,b\",c'", "output": "['a,b', 'c']"},
            {"input": "''", "output": "['']"},
        ],
    },
    {
        "goal": "Write a Python function `validate_password(s)` that returns True if password is at least 8 chars, has uppercase, lowercase, and a digit.",
        "test_cases": [
            {"input": "'Abcdef12'", "output": "True"},
            {"input": "'password'", "output": "False"},
            {"input": "'PASSWORD1'", "output": "False"},
            {"input": "'Ab1'", "output": "False"},
            {"input": "'Abcdefgh'", "output": "False"},
        ],
    },
    {
        "goal": "Write a Python function `is_leap_year(year)` that returns True if year is a leap year.",
        "test_cases": [
            {"input": "2000", "output": "True"},
            {"input": "1900", "output": "False"},
            {"input": "2024", "output": "True"},
            {"input": "2023", "output": "False"},
            {"input": "1600", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `days_in_month(month, year)` that returns the number of days in the given month (1-12), accounting for leap years.",
        "test_cases": [
            {"input": "2, 2024", "output": "29"},
            {"input": "2, 2023", "output": "28"},
            {"input": "1, 2023", "output": "31"},
            {"input": "4, 2023", "output": "30"},
            {"input": "12, 2023", "output": "31"},
        ],
    },
    {
        "goal": "Write a Python function `day_of_week(year, month, day)` that returns the day name (Monday, Tuesday, etc.) for a given date using Zeller's formula or equivalent.",
        "test_cases": [
            {"input": "2024, 1, 1", "output": "'Monday'"},
            {"input": "2023, 12, 25", "output": "'Monday'"},
            {"input": "2000, 1, 1", "output": "'Saturday'"},
            {"input": "2024, 7, 4", "output": "'Thursday'"},
        ],
    },
    # ── Math & number theory (8) ──
    {
        "goal": "Write a Python function `is_even(n)` that returns True if n is even, False if odd.",
        "test_cases": [
            {"input": "0", "output": "True"},
            {"input": "1", "output": "False"},
            {"input": "2", "output": "True"},
            {"input": "-1", "output": "False"},
            {"input": "-2", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `is_odd(n)` that returns True if n is odd, False if even.",
        "test_cases": [
            {"input": "0", "output": "False"},
            {"input": "1", "output": "True"},
            {"input": "2", "output": "False"},
            {"input": "-1", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `is_perfect_square(n)` that returns True if n is a perfect square.",
        "test_cases": [
            {"input": "0", "output": "True"},
            {"input": "1", "output": "True"},
            {"input": "4", "output": "True"},
            {"input": "16", "output": "True"},
            {"input": "15", "output": "False"},
            {"input": "144", "output": "True"},
        ],
    },
    {
        "goal": "Write a Python function `nth_prime(n)` that returns the nth prime number (1-indexed, so nth_prime(1)=2).",
        "test_cases": [
            {"input": "1", "output": "2"},
            {"input": "2", "output": "3"},
            {"input": "5", "output": "11"},
            {"input": "10", "output": "29"},
            {"input": "25", "output": "97"},
        ],
    },
    {
        "goal": "Write a Python function `prime_factors(n)` that returns a list of prime factors of n in ascending order.",
        "test_cases": [
            {"input": "12", "output": "[2, 2, 3]"},
            {"input": "17", "output": "[17]"},
            {"input": "100", "output": "[2, 2, 5, 5]"},
            {"input": "1", "output": "[]"},
        ],
    },
    {
        "goal": "Write a Python function `tribonacci(n)` that returns the nth Tribonacci number. trib(0)=0, trib(1)=0, trib(2)=1, trib(n)=trib(n-1)+trib(n-2)+trib(n-3).",
        "test_cases": [
            {"input": "0", "output": "0"},
            {"input": "1", "output": "0"},
            {"input": "2", "output": "1"},
            {"input": "5", "output": "7"},
            {"input": "10", "output": "149"},
        ],
    },
    {
        "goal": "Write a Python function `modular_pow(base, exp, mod)` that returns (base^exp) % mod efficiently using modular exponentiation.",
        "test_cases": [
            {"input": "2, 10, 1000", "output": "24"},
            {"input": "3, 4, 5", "output": "1"},
            {"input": "7, 13, 19", "output": "7"},
            {"input": "2, 0, 100", "output": "1"},
        ],
    },
    {
        "goal": "Write a Python function `pascals_triangle(n)` that returns the nth row (0-indexed) of Pascal's triangle as a list.",
        "test_cases": [
            {"input": "0", "output": "[1]"},
            {"input": "1", "output": "[1, 1]"},
            {"input": "2", "output": "[1, 2, 1]"},
            {"input": "4", "output": "[1, 4, 6, 4, 1]"},
            {"input": "5", "output": "[1, 5, 10, 10, 5, 1]"},
        ],
    },
    # ── Recursion & DP (6) ──
    {
        "goal": "Write a Python function `climb_stairs(n)` that returns the number of distinct ways to climb n stairs taking 1 or 2 steps at a time.",
        "test_cases": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "2"},
            {"input": "3", "output": "3"},
            {"input": "5", "output": "8"},
            {"input": "10", "output": "89"},
        ],
    },
    {
        "goal": "Write a Python function `coin_change(coins, amount)` that returns the minimum number of coins needed to make amount, or -1 if impossible.",
        "test_cases": [
            {"input": "[1,5,10], 11", "output": "2"},
            {"input": "[2], 3", "output": "-1"},
            {"input": "[1], 0", "output": "0"},
            {"input": "[1,2,5], 11", "output": "3"},
        ],
    },
    {
        "goal": "Write a Python function `lcs(a, b)` that returns the length of the longest common subsequence of strings a and b.",
        "test_cases": [
            {"input": "'abc', 'abc'", "output": "3"},
            {"input": "'abc', 'def'", "output": "0"},
            {"input": "'abcde', 'ace'", "output": "3"},
            {"input": "'', ''", "output": "0"},
        ],
    },
    {
        "goal": "Write a Python function `lis(arr)` that returns the length of the longest increasing subsequence.",
        "test_cases": [
            {"input": "[10,9,2,5,3,7,101,18]", "output": "4"},
            {"input": "[1,2,3,4,5]", "output": "5"},
            {"input": "[5,4,3,2,1]", "output": "1"},
            {"input": "[]", "output": "0"},
            {"input": "[1]", "output": "1"},
        ],
    },
    {
        "goal": "Write a Python function `max_subarray(arr)` that returns the maximum subarray sum (Kadane's algorithm).",
        "test_cases": [
            {"input": "[-2,1,-3,4,-1,2,1,-5,4]", "output": "6"},
            {"input": "[1]", "output": "1"},
            {"input": "[-1]", "output": "-1"},
            {"input": "[5,4,-1,7,8]", "output": "23"},
        ],
    },
    {
        "goal": "Write a Python function `knapsack(weights, values, capacity)` that returns the maximum value for the 0/1 knapsack problem.",
        "test_cases": [
            {"input": "[1,2,3], [6,10,12], 5", "output": "22"},
            {"input": "[1], [1], 0", "output": "0"},
            {"input": "[2,3,4], [3,4,5], 5", "output": "7"},
            {"input": "[], [], 10", "output": "0"},
        ],
    },
    # ── Utility & I/O (6) ──
    {
        "goal": "Write a Python function `range_intersect(a_start, a_end, b_start, b_end)` that returns the intersection of two integer ranges [a_start,a_end] and [b_start,b_end] as a tuple, or None if no overlap.",
        "test_cases": [
            {"input": "1, 5, 3, 7", "output": "(3, 5)"},
            {"input": "1, 3, 5, 7", "output": "None"},
            {"input": "1, 10, 3, 5", "output": "(3, 5)"},
            {"input": "5, 5, 5, 5", "output": "(5, 5)"},
        ],
    },
    {
        "goal": "Write a Python function `format_number(n, decimals)` that formats a number with the given number of decimal places and thousands separators (commas).",
        "test_cases": [
            {"input": "1234567, 2", "output": "'1,234,567.00'"},
            {"input": "1000, 0", "output": "'1,000'"},
            {"input": "0, 2", "output": "'0.00'"},
            {"input": "123.456, 1", "output": "'123.5'"},
        ],
    },
    {
        "goal": "Write a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer.",
        "test_cases": [
            {"input": "'III'", "output": "3"},
            {"input": "'IV'", "output": "4"},
            {"input": "'IX'", "output": "9"},
            {"input": "'LVIII'", "output": "58"},
            {"input": "'MCMXCIV'", "output": "1994"},
        ],
    },
    {
        "goal": "Write a Python function `int_to_roman(n)` that converts an integer (1-3999) to a Roman numeral string.",
        "test_cases": [
            {"input": "3", "output": "'III'"},
            {"input": "4", "output": "'IV'"},
            {"input": "9", "output": "'IX'"},
            {"input": "58", "output": "'LVIII'"},
            {"input": "1994", "output": "'MCMXCIV'"},
        ],
    },
    {
        "goal": "Write a Python function `encode_rle(s)` that encodes a string using run-length encoding (e.g. 'aaabbc' -> '3a2b1c').",
        "test_cases": [
            {"input": "'aaabbc'", "output": "'3a2b1c'"},
            {"input": "''", "output": "''"},
            {"input": "'a'", "output": "'1a'"},
            {"input": "'aaaa'", "output": "'4a'"},
        ],
    },
    {
        "goal": "Write a Python function `decode_rle(s)` that decodes a run-length encoded string (e.g. '3a2b1c' -> 'aaabbc').",
        "test_cases": [
            {"input": "'3a2b1c'", "output": "'aaabbc'"},
            {"input": "''", "output": "''"},
            {"input": "'1a'", "output": "'a'"},
            {"input": "'4a'", "output": "'aaaa'"},
        ],
    },
]


# Sandboxed verification timeout (seconds). Generated code that runs longer
# than this is treated as a failure (infinite-loop / DoS protection, NS3).
_VERIFY_TIMEOUT_S = 10

# Function names searched for in generated solutions (shared with the sandbox
# runner script below).
_VERIFY_FUNC_NAMES = [
    # Basic algorithms
    "fib", "is_prime", "factorial", "gcd", "power", "binary_search",
    "merge_sorted", "quick_sort", "bubble_sort", "insertion_sort",
    "lcm", "is_perfect", "collatz", "digit_sum", "reverse_number",
    "is_armstrong", "to_binary", "from_binary", "count_bits",
    "tower_of_hanoi",
    # String manipulation
    "reverse_string", "is_palindrome", "count_words",
    "capitalize_words", "is_anagram", "compress", "decompress",
    "count_vowels", "count_consonants", "replace_vowels",
    "camel_to_snake", "snake_to_camel",
    # Data structures
    "sum_list", "max_element", "min_element", "remove_duplicates",
    "flatten", "matrix_transpose", "matrix_multiply", "rotate_list",
    "chunk", "stack_sort",
    # Validation & parsing
    "valid_parens", "is_valid_email", "is_valid_ip", "parse_csv_line",
    "validate_password", "is_leap_year", "days_in_month",
    "day_of_week",
    # Math & number theory
    "is_even", "is_odd", "is_perfect_square", "nth_prime",
    "prime_factors", "tribonacci", "modular_pow", "pascals_triangle",
    # Recursion & DP
    "climb_stairs", "coin_change", "lcs", "lis", "max_subarray",
    "knapsack",
    # Utility & I/O
    "range_intersect", "format_number", "roman_to_int",
    "int_to_roman", "encode_rle", "decode_rle",
]

# Sandboxed runner executed in a *separate* Python subprocess. It restricts
# dangerous imports, executes the (model-generated) solution, locates the
# target function, and runs the test cases using ast.literal_eval (NS2: no
# arbitrary eval on test data). Prints "PASS" on success or "FAIL:..." on
# failure. The parent treats any non-"PASS" outcome (including timeout) as a
# verification failure.
_VERIFY_RUNNER = '''\
import ast as _ast
import json as _json
import sys as _sys

# --- Restricted import hook (NS1) -----------------------------------------
_BLOCKED = {
    "os", "subprocess", "socket", "urllib", "requests", "http", "ctypes",
    "multiprocessing", "importlib", "shutil", "pathlib", "webbrowser",
    "antigravity", "pickle", "marshal",
}
_builtins = _sys.modules["builtins"]
_orig_import = _builtins.__import__


def _restricted_import(name, *args, **kwargs):
    if name.split(".")[0] in _BLOCKED:
        raise ImportError("Module '%%s' is blocked in sandbox" %% name)
    return _orig_import(name, *args, **kwargs)


_builtins.__import__ = _restricted_import

# --- Load solution + test cases from files --------------------------------
_sol_path = _sys.argv[1]
_tests_path = _sys.argv[2]
with open(_sol_path, "r", encoding="utf-8") as _f:
    _solution = _f.read()
with open(_tests_path, "r", encoding="utf-8") as _f:
    _test_cases = _json.load(_f)

_FUNC_NAMES = %r

try:
    _ns = {"__builtins__": _builtins}
    exec(_solution, _ns)
except Exception as _e:
    print("FAIL:exec error:%%s" %% _e)
    _sys.exit(0)

_func = None
for _name in _FUNC_NAMES:
    if _name in _ns and callable(_ns[_name]):
        _func = _ns[_name]
        break
if _func is None:
    for _name, _obj in _ns.items():
        if callable(_obj) and not _name.startswith("__"):
            _func = _obj
            break
if _func is None:
    print("FAIL:no function")
    _sys.exit(0)

for _tc in _test_cases:
    try:
        _inp = _ast.literal_eval(_tc["input"])
        _expected = _ast.literal_eval(_tc["output"])
    except Exception:
        print("FAIL:bad test case")
        _sys.exit(0)
    try:
        if isinstance(_inp, tuple):
            _result = _func(*_inp)
        else:
            _result = _func(_inp)
    except Exception as _e:
        print("FAIL:runtime error:%%s" %% _e)
        _sys.exit(0)
    if _result != _expected:
        print("FAIL:wrong result")
        _sys.exit(0)

print("PASS")
''' % (_VERIFY_FUNC_NAMES,)


def verify_solution(solution: str, test_cases: list[dict]) -> bool:
    """Verify a solution against test cases by executing it.

    Security (NS1/NS2/NS3): the solution is executed in a *separate* Python
    subprocess with a restricted import hook (no os/subprocess/socket/etc.),
    a hard timeout (``_VERIFY_TIMEOUT_S``), and test-case inputs/outputs are
    parsed with ``ast.literal_eval`` (literals only — no arbitrary code
    execution on potentially model-generated data). On timeout or any error
    the solution is treated as incorrect.
    """
    import ast
    import re
    import subprocess
    import tempfile

    # Extract code from markdown code blocks if present
    code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
    if code_match:
        solution = code_match.group(1)
    else:
        code_match = re.search(r'```\n(.*?)```', solution, re.DOTALL)
        if code_match:
            solution = code_match.group(1)

    # Reject obviously non-literal test-case data early (defence in depth for
    # NS2): ast.literal_eval will also reject it inside the sandbox, but
    # checking here avoids spawning a process for malformed data.
    for tc in test_cases:
        try:
            ast.literal_eval(tc["input"])
            ast.literal_eval(tc["output"])
        except (ValueError, SyntaxError):
            return False

    tmp_dir = tempfile.gettempdir()
    sol_fd, sol_path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
    tests_fd, tests_path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
    runner_fd, runner_path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
    try:
        import os as _os
        with _os.fdopen(sol_fd, "w", encoding="utf-8") as f:
            f.write(solution)
        with _os.fdopen(tests_fd, "w", encoding="utf-8") as f:
            json.dump(test_cases, f)
        with _os.fdopen(runner_fd, "w", encoding="utf-8") as f:
            f.write(_VERIFY_RUNNER)

        try:
            proc = subprocess.run(
                [sys.executable, runner_path, sol_path, tests_path],
                capture_output=True, text=True,
                timeout=_VERIFY_TIMEOUT_S,
                cwd=tmp_dir,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False  # NS3: infinite loop / hang -> failure
        except Exception:
            return False

        return proc.stdout.strip() == "PASS"
    finally:
        import os as _os
        for _p in (sol_path, tests_path, runner_path):
            try:
                _os.unlink(_p)
            except OSError:
                pass


def run_agentic_mode(args):
    """Run agentic distillation: teacher models call tools to solve tasks."""
    print("=" * 70)
    print("  ForgeAI Agentic Distillation Loop")
    print("  (teacher models call tools, no fine-tuning)")
    print("=" * 70)

    client = AgenticDistillClient(
        max_turns=args.max_turns,
        max_gen_tokens=args.max_tokens,
        agentic_temperature=args.temperature,
    )
    print(f"\nModels available: {len(client.models)}")
    print(f"Providers: {sorted(set(m.provider for m in client.models))}")
    print(f"Max turns per task: {args.max_turns}")
    print(f"Max gen tokens/turn: {args.max_tokens}")

    # Step 1: Optionally generate new tasks
    tasks_to_run = []
    if args.gen_tasks > 0:
        print(f"\n{'─'*70}")
        print(f"  PHASE 1: Generating {args.gen_tasks} new tasks via teacher model")
        print(f"{'─'*70}")
        generated = client.generate_tasks(n_tasks=args.gen_tasks)
        print(f"  Generated {len(generated)} valid tasks (after filtering)")
        for t in generated[:5]:
            print(f"    → {t['task'][:70]}...")
        tasks_to_run = generated

    # Step 2: Add predefined tasks
    if not args.gen_only:
        for t in TASKS:
            tasks_to_run.append({
                "task": t["goal"],
                "test_cases": t["test_cases"],
            })

    # Deduplicate
    seen = set()
    unique_tasks = []
    for t in tasks_to_run:
        key = t["task"][:80]
        if key not in seen:
            seen.add(key)
            unique_tasks.append(t)

    print(f"\n{'─'*70}")
    print(f"  PHASE 2: Running {len(unique_tasks)} tasks agenticly")
    print(f"  ({args.n_samples} teacher models per task)")
    print(f"{'─'*70}")

    task_strs = [t["task"] for t in unique_tasks]
    test_cases = [t.get("test_cases", []) for t in unique_tasks]

    trajectories = client.run_agentic_batch(
        tasks=task_strs,
        test_cases_per_task=test_cases,
        n_samples_per_task=args.n_samples,
        delay_between=args.delay,
        min_reward=args.min_reward,
    )

    # Save trajectories
    output_path = Path(args.output)
    n_saved = client.save_trajectories(trajectories, output_path)

    print(f"\n{'='*70}")
    print(f"  AGENTIC DISTILLATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Tasks run:          {len(unique_tasks)}")
    print(f"  Trajectories kept:  {len(trajectories)} (reward >= {args.min_reward})")
    print(f"  Saved to:           {output_path}")
    print(f"  File size:          {output_path.stat().st_size / 1024:.1f} KB")

    # Stats
    if trajectories:
        rewards = [t.reward for t in trajectories]
        turns = [t.n_turns for t in trajectories]
        tools = [len(t.tool_calls) for t in trajectories]
        print(f"\n  Reward:  min={min(rewards):.2f} avg={sum(rewards)/len(rewards):.2f} max={max(rewards):.2f}")
        print(f"  Turns:   min={min(turns)} avg={sum(turns)/len(turns):.1f} max={max(turns)}")
        print(f"  Tools:   min={min(tools)} avg={sum(tools)/len(tools):.1f} max={max(tools)}")
        teachers = {}
        for t in trajectories:
            teachers[t.teacher_model] = teachers.get(t.teacher_model, 0) + 1
        print(f"\n  Teacher models used:")
        for m, c in sorted(teachers.items(), key=lambda x: -x[1]):
            print(f"    {m:45s} {c} trajectories")


def main():
    parser = argparse.ArgumentParser(description="Run distillation data generation loop")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of samples per goal (default: 4)")
    parser.add_argument("--output", type=str,
                        default="research/distillation/distill_data.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between goals in seconds (rate limiting)")
    # Agentic mode flags
    parser.add_argument("--agentic", action="store_true",
                        help="Run in agentic mode (teacher models call tools)")
    parser.add_argument("--gen-tasks", type=int, default=0,
                        help="Generate N new tasks via teacher model (agentic mode)")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only run generated tasks, skip predefined ones")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max tool-use turns per task (agentic mode)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per turn (agentic mode)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Temperature for agentic generation")
    parser.add_argument("--min-reward", type=float, default=0.3,
                        help="Min reward to keep trajectory (agentic mode)")
    args = parser.parse_args()

    if args.agentic:
        if args.gen_tasks > 0 or args.gen_only:
            args.output = "research/distillation/agentic_distill_data.jsonl"
        else:
            args.output = "research/distillation/agentic_distill_data.jsonl"
        run_agentic_mode(args)
        return

    print("=" * 70)
    print("  ForgeAI Distillation Data Generation Loop")
    print("=" * 70)

    # Initialize client with our verify function
    client = DistillationClient(verify_fn=verify_solution)
    print(f"\nModels available: {len(client.models)}")
    print(f"Providers: {sorted(set(m.provider for m in client.models))}")
    groups = client.canonical_groups()
    print(f"Canonical models: {len(groups)}")
    multi = {c: ps for c, ps in groups.items() if len(ps) > 1}
    print(f"Multi-provider models: {len(multi)}")
    for c, ps in sorted(multi.items(), key=lambda x: -len(x[1])):
        print(f"  {c}: {len(ps)} providers {ps}")

    print(f"\nTasks: {len(TASKS)}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Total requests: {len(TASKS) * args.n_samples}")
    print(f"Output: {args.output}")
    print()

    # Run the generation loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    stats = {
        "total_generated": 0,
        "total_correct": 0,
        "total_errors": 0,
        "per_provider": {},
        "per_canonical": {},
    }

    with open(output_path, "w") as f:
        for i, task in enumerate(TASKS):
            goal = task["goal"]
            test_cases = task["test_cases"]

            # Early exit: all providers exhausted
            if client.all_providers_exhausted():
                print(f"\n  ⚠ ALL PROVIDERS EXHAUSTED — ending early at task {i}/{len(TASKS)}")
                print(f"    Blacklisted: {sorted(client._blacklisted_providers)}")
                break

            print(f"\n[{i+1}/{len(TASKS)}] {goal[:70]}...")
            active = client.active_providers()
            if len(active) <= 2:
                print(f"  ⚠ Only {len(active)} active providers: {active}")

            results = client.generate_for_goal(
                goal=goal,
                test_cases=test_cases,
                n_samples=args.n_samples,
                verify=True,
            )

            task_correct = 0
            for r in results:
                stats["total_generated"] += 1
                provider = r.model.split("/")[0] if "/" in r.model else r.model
                stats["per_provider"].setdefault(provider, {"gen": 0, "ok": 0, "err": 0})
                stats["per_provider"][provider]["gen"] += 1

                if r.error:
                    stats["total_errors"] += 1
                    stats["per_provider"][provider]["err"] += 1
                    print(f"  ERROR  {r.model:40s} {r.error[:60]}")
                    continue

                if r.correct:
                    stats["total_correct"] += 1
                    task_correct += 1
                    stats["per_provider"][provider]["ok"] += 1
                    print(f"  OK     {r.model:40s} {r.latency_ms:.0f}ms {r.tokens_out}tok")

                    # Save to JSONL
                    pair = {
                        "prompt": goal,
                        "solution": r.solution,
                        "reasoning": r.reasoning,
                        "teacher_model": r.model,
                        "test_passed": True,
                        "latency_ms": r.latency_ms,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                    }
                    f.write(json.dumps(pair) + "\n")
                    f.flush()
                    all_pairs.append(pair)
                else:
                    print(f"  FAIL   {r.model:40s} {r.latency_ms:.0f}ms (verification failed)")

            print(f"  → {task_correct}/{len(results)} correct for this task")

            if args.delay > 0 and i < len(TASKS) - 1:
                time.sleep(args.delay)

    # Print final stats
    print("\n" + "=" * 70)
    print("  FINAL STATISTICS")
    print("=" * 70)
    print(f"  Total generated:  {stats['total_generated']}")
    print(f"  Total correct:    {stats['total_correct']}")
    print(f"  Total errors:     {stats['total_errors']}")
    print(f"  Accuracy:         {stats['total_correct'] / max(stats['total_generated'], 1) * 100:.1f}%")
    print(f"  Saved pairs:      {len(all_pairs)}")
    print(f"\n  Per-provider breakdown:")
    for prov, s in sorted(stats["per_provider"].items()):
        print(f"    {prov:15s}  gen={s['gen']:3d}  ok={s['ok']:3d}  err={s['err']:3d}")
    print(f"\n  Output saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
