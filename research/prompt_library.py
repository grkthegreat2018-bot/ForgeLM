"""DEPRECATED: Function-stub prompt library for self-play expert training.

Superseded by research.goal_tasks (Goal-Oriented Self-Play).
This module generates function STUBS that tell the model WHAT to implement,
which is counter-productive — it grades against a specific implementation
rather than whether the model achieves the goal.

Kept for reference only. Use GoalTaskGenerator from research.goal_tasks instead.

Original description:
Generates hundreds of prompts per topic across difficulty levels:
  - Easy: simple functions, basic operations
  - Medium: algorithms, data structures, multi-step logic
  - Hard: edge cases, optimizations, combined concepts

Topics:
  - python_general: basic Python functions
  - python_math: math-focused Python functions
  - python_strings: string manipulation
  - python_algorithms: sorting, searching, data structures
  - python_oop: classes and objects
  - python_file_io: file operations
  - math_*: math problems (text-based)
  - reasoning/logic/general: knowledge and reasoning
"""
import random
import itertools
from typing import Dict, List


def _f(name: str, docstring: str, args: str = "") -> str:
    """Build a function prompt stub."""
    if args:
        return f'def {name}({args}):\n    """{docstring}"""\n    '
    return f'def {name}():\n    """{docstring}"""\n    '


def _cls(name: str, docstring: str, init_args: str = "") -> str:
    """Build a class prompt stub."""
    return f'class {name}:\n    """{docstring}"""\n    def __init__(self{", " + init_args if init_args else ""}):\n        '


def _math(problem: str) -> str:
    """Build a math problem prompt."""
    return problem + " "


# ── Python General ─────────────────────────────────────────────────
def _python_general() -> List[str]:
    prompts = []

    # Basic arithmetic functions
    for a, b in [("a", "b"), ("x", "y"), ("num1", "num2"), ("a", "b"), ("x", "y")]:
        prompts.append(_f("add", "Return sum", f"{a}, {b}"))
        prompts.append(_f("subtract", "Return difference", f"{a}, {b}"))
        prompts.append(_f("multiply", "Return product", f"{a}, {b}"))
        prompts.append(_f("divide", "Return quotient", f"{a}, {b}"))
        prompts.append(_f("max_of_two", "Return larger number", f"{a}, {b}"))
        prompts.append(_f("min_of_two", "Return smaller number", f"{a}, {b}"))

    # Number properties
    for n in ["n", "num", "x"]:
        prompts.append(_f("is_even", "Check if even", n))
        prompts.append(_f("is_odd", "Check if odd", n))
        prompts.append(_f("is_positive", "Check if positive", n))
        prompts.append(_f("is_negative", "Check if negative", n))
        prompts.append(_f("is_zero", "Check if zero", n))
        prompts.append(_f("square", "Return square", n))
        prompts.append(_f("cube", "Return cube", n))
        prompts.append(_f("double", "Return doubled value", n))
        prompts.append(_f("triple", "Return tripled value", n))
        prompts.append(_f("absolute", "Return absolute value", n))
        prompts.append(_f("sign", "Return 1, -1, or 0", n))
        prompts.append(_f("is_integer", "Check if integer", n))

    # String basics
    for s in ["s", "text", "word"]:
        prompts.append(_f("length", "Return length", s))
        prompts.append(_f("to_upper", "Convert to uppercase", s))
        prompts.append(_f("to_lower", "Convert to lowercase", s))
        prompts.append(_f("first_char", "Return first character", s))
        prompts.append(_f("last_char", "Return last character", s))

    # Greeting / formatting
    prompts.append(_f("greet", "Greet someone by name", "name"))
    prompts.append(_f("farewell", "Say goodbye", "name"))
    prompts.append(_f("format_name", "Format as 'First Last'", "first, last"))
    prompts.append(_f("repeat_string", "Repeat string n times", "s, n"))
    prompts.append(_f("repeat_char", "Repeat char n times", "c, n"))

    # Boolean / logic
    prompts.append(_f("is_equal", "Check if two values equal", "a, b"))
    prompts.append(_f("is_greater", "Check if a > b", "a, b"))
    prompts.append(_f("is_less", "Check if a < b", "a, b"))
    prompts.append(_f("in_range", "Check if n in [low, high]", "n, low, high"))
    prompts.append(_f("clamp", "Clamp value to range", "n, low, high"))

    # Conversion
    prompts.append(_f("to_celsius", "Convert Fahrenheit to Celsius", "f"))
    prompts.append(_f("to_fahrenheit", "Convert Celsius to Fahrenheit", "c"))
    prompts.append(_f("miles_to_km", "Convert miles to km", "miles"))
    prompts.append(_f("kg_to_lbs", "Convert kg to pounds", "kg"))

    # List basics
    prompts.append(_f("first_element", "Return first element", "lst"))
    prompts.append(_f("last_element", "Return last element", "lst"))
    prompts.append(_f("list_length", "Return list length", "lst"))
    prompts.append(_f("is_empty", "Check if list is empty", "lst"))
    prompts.append(_f("sum_list", "Return sum of list", "lst"))
    prompts.append(_f("product_list", "Return product of list", "lst"))

    # Counting
    prompts.append(_f("count_vowels", "Count vowels in string", "s"))
    prompts.append(_f("count_consonants", "Count consonants", "s"))
    prompts.append(_f("count_words", "Count words in string", "s"))
    prompts.append(_f("count_char", "Count occurrences of char", "s, c"))

    # Random / misc
    prompts.append(_f("identity", "Return the input unchanged", "x"))
    prompts.append(_f("negate", "Return negation", "n"))
    prompts.append(_f("invert_bool", "Return not b", "b"))
    prompts.append(_f("swap", "Swap two values, return tuple", "a, b"))
    prompts.append(_f("average", "Return average of two numbers", "a, b"))
    prompts.append(_f("midpoint", "Return midpoint of two numbers", "a, b"))

    return list(dict.fromkeys(prompts))  # dedupe, preserve order


# ── Python Math ────────────────────────────────────────────────────
def _python_math() -> List[str]:
    prompts = []

    # Number theory
    prompts.append(_f("is_prime", "Check if prime", "n"))
    prompts.append(_f("fib", "Fibonacci number", "n"))
    prompts.append(_f("fib_iterative", "Fibonacci iteratively", "n"))
    prompts.append(_f("factorial", "Factorial of n", "n"))
    prompts.append(_f("factorial_iter", "Factorial iteratively", "n"))
    prompts.append(_f("gcd", "GCD of two numbers", "a, b"))
    prompts.append(_f("lcm", "LCM of two numbers", "a, b"))
    prompts.append(_f("power", "Base raised to exp", "base, exp"))
    prompts.append(_f("power_iter", "Power iteratively", "base, exp"))
    prompts.append(_f("is_perfect_square", "Check perfect square", "n"))
    prompts.append(_f("is_perfect_cube", "Check perfect cube", "n"))
    prompts.append(_f("is_power_of_two", "Check if power of 2", "n"))
    prompts.append(_f("is_power_of", "Check if n is power of base", "n, base"))
    prompts.append(_f("sum_digits", "Sum of digits", "n"))
    prompts.append(_f("reverse_number", "Reverse digits of number", "n"))
    prompts.append(_f("count_digits", "Count digits in number", "n"))
    prompts.append(_f("is_armstrong", "Check Armstrong number", "n"))
    prompts.append(_f("is_narcissistic", "Check narcissistic number", "n"))
    prompts.append(_f("digital_root", "Digital root of n", "n"))
    prompts.append(_f("collatz_steps", "Collatz conjecture steps", "n"))
    prompts.append(_f("next_prime", "Find next prime after n", "n"))
    prompts.append(_f("prev_prime", "Find previous prime before n", "n"))
    prompts.append(_f("prime_factors", "Return list of prime factors", "n"))
    prompts.append(_f("count_primes", "Count primes up to n", "n"))
    prompts.append(_f("is_fibonacci", "Check if n is a Fibonacci number", "n"))
    prompts.append(_f("tribonacci", "Tribonacci number", "n"))
    prompts.append(_f("is_triangular", "Check triangular number", "n"))
    prompts.append(_f("is_pentagonal", "Check pentagonal number", "n"))
    prompts.append(_f("euler_phi", "Euler's totient function", "n"))
    prompts.append(_f("is_coprime", "Check if coprime", "a, b"))

    # Sequences and series
    prompts.append(_f("arithmetic_sum", "Sum of arithmetic sequence", "a, d, n"))
    prompts.append(_f("geometric_sum", "Sum of geometric sequence", "a, r, n"))
    prompts.append(_f("triangular_number", "Nth triangular number", "n"))
    prompts.append(_f("square_number", "Nth square number", "n"))
    prompts.append(_f("pentagonal_number", "Nth pentagonal number", "n"))

    # Modular arithmetic
    prompts.append(_f("mod_pow", "Modular exponentiation", "base, exp, mod"))
    prompts.append(_f("mod_inverse", "Modular inverse", "a, mod"))
    prompts.append(_f("chinese_remainder", "Chinese Remainder Theorem", "remainders, moduli"))

    # Combinatorics
    prompts.append(_f("nCr", "Combinations n choose r", "n, r"))
    prompts.append(_f("nPr", "Permutations n permute r", "n, r"))
    prompts.append(_f("catalan", "Catalan number", "n"))
    prompts.append(_f("bell_number", "Bell number", "n"))
    prompts.append(_f("stirling_second", "Stirling number second kind", "n, k"))

    # Geometry math
    prompts.append(_f("circle_area", "Area of circle", "r"))
    prompts.append(_f("circle_circumference", "Circumference of circle", "r"))
    prompts.append(_f("triangle_area", "Area of triangle", "base, height"))
    prompts.append(_f("rectangle_area", "Area of rectangle", "w, h"))
    prompts.append(_f("rectangle_perimeter", "Perimeter of rectangle", "w, h"))
    prompts.append(_f("cube_volume", "Volume of cube", "side"))
    prompts.append(_f("sphere_volume", "Volume of sphere", "r"))
    prompts.append(_f("pythagorean", "Hypotenuse via Pythagorean theorem", "a, b"))
    prompts.append(_f("distance", "Euclidean distance between 2 points", "x1, y1, x2, y2"))

    # Number conversion
    prompts.append(_f("to_binary", "Convert to binary string", "n"))
    prompts.append(_f("to_octal", "Convert to octal string", "n"))
    prompts.append(_f("to_hex", "Convert to hex string", "n"))
    prompts.append(_f("from_binary", "Convert binary string to int", "s"))
    prompts.append(_f("from_hex", "Convert hex string to int", "s"))

    # Matrix
    prompts.append(_f("matrix_add", "Add two matrices", "a, b"))
    prompts.append(_f("matrix_multiply", "Multiply two matrices", "a, b"))
    prompts.append(_f("matrix_transpose", "Transpose matrix", "m"))
    prompts.append(_f("matrix_trace", "Trace of matrix", "m"))
    prompts.append(_f("matrix_determinant", "Determinant of 2x2 matrix", "m"))

    return list(dict.fromkeys(prompts))


# ── Python Strings ─────────────────────────────────────────────────
def _python_strings() -> List[str]:
    prompts = []

    # Basic transformations
    prompts.append(_f("reverse", "Reverse string", "s"))
    prompts.append(_f("to_upper", "Convert to uppercase", "s"))
    prompts.append(_f("to_lower", "Convert to lowercase", "s"))
    prompts.append(_f("to_title", "Convert to title case", "s"))
    prompts.append(_f("to_capitalize", "Capitalize first letter", "s"))
    prompts.append(_f("swap_case", "Swap case of string", "s"))

    # Checking
    prompts.append(_f("is_palindrome", "Check palindrome", "s"))
    prompts.append(_f("is_anagram", "Check if two strings are anagrams", "s1, s2"))
    prompts.append(_f("is_pangram", "Check if pangram", "s"))
    prompts.append(_f("is_digit", "Check if all digits", "s"))
    prompts.append(_f("is_alpha", "Check if all alphabetic", "s"))
    prompts.append(_f("is_alnum", "Check if alphanumeric", "s"))
    prompts.append(_f("is_upper", "Check if uppercase", "s"))
    prompts.append(_f("is_lower", "Check if lowercase", "s"))
    prompts.append(_f("is_space", "Check if all whitespace", "s"))
    prompts.append(_f("starts_with", "Check if s starts with prefix", "s, prefix"))
    prompts.append(_f("ends_with", "Check if s ends with suffix", "s, suffix"))
    prompts.append(_f("contains", "Check if s contains substring", "s, sub"))

    # Counting
    prompts.append(_f("count_char", "Count occurrences of char", "s, c"))
    prompts.append(_f("count_vowels", "Count vowels", "s"))
    prompts.append(_f("count_consonants", "Count consonants", "s"))
    prompts.append(_f("count_words", "Count words", "s"))
    prompts.append(_f("count_substring", "Count substring occurrences", "s, sub"))
    prompts.append(_f("count_upper", "Count uppercase letters", "s"))
    prompts.append(_f("count_lower", "Count lowercase letters", "s"))
    prompts.append(_f("count_digits", "Count digits in string", "s"))
    prompts.append(_f("count_special", "Count special characters", "s"))

    # Manipulation
    prompts.append(_f("remove_spaces", "Remove all spaces", "s"))
    prompts.append(_f("remove_vowels", "Remove all vowels", "s"))
    prompts.append(_f("remove_digits", "Remove all digits", "s"))
    prompts.append(_f("remove_punctuation", "Remove punctuation", "s"))
    prompts.append(_f("replace_vowels", "Replace vowels with char", "s, c"))
    prompts.append(_f("replace_char", "Replace char a with char b", "s, a, b"))
    prompts.append(_f("capitalize_words", "Capitalize each word", "s"))
    prompts.append(_f("reverse_words", "Reverse order of words", "s"))
    prompts.append(_f("reverse_each_word", "Reverse each word individually", "s"))
    prompts.append(_f("strip_extra_spaces", "Collapse multiple spaces to one", "s"))
    prompts.append(_f("pad_left", "Pad string on left with char", "s, n, c"))
    prompts.append(_f("pad_right", "Pad string on right with char", "s, n, c"))
    prompts.append(_f("center_string", "Center string in width", "s, width"))
    prompts.append(_f("truncate", "Truncate to max length with ellipsis", "s, max_len"))
    prompts.append(_f("wrap_text", "Wrap text at width", "s, width"))

    # Extraction
    prompts.append(_f("first_word", "Return first word", "s"))
    prompts.append(_f("last_word", "Return last word", "s"))
    prompts.append(_f("get_words", "Return list of words", "s"))
    prompts.append(_f("get_digits", "Extract all digits", "s"))
    prompts.append(_f("get_alpha", "Extract all alphabetic chars", "s"))
    prompts.append(_f("get_unique_chars", "Return unique characters", "s"))
    prompts.append(_f("get_char_freq", "Return char frequency dict", "s"))

    # Encoding / ciphers
    prompts.append(_f("caesar_cipher", "Caesar cipher shift by n", "s, n"))
    prompts.append(_f("caesar_decipher", "Caesar cipher decode", "s, n"))
    prompts.append(_f("rot13", "ROT13 encoding", "s"))
    prompts.append(_f("atbash", "Atbash cipher", "s"))
    prompts.append(_f("vigenere_cipher", "Vigenere cipher", "s, key"))
    prompts.append(_f("morse_encode", "Encode to Morse code", "s"))
    prompts.append(_f("morse_decode", "Decode from Morse code", "s"))
    prompts.append(_f("run_length_encode", "Run-length encoding", "s"))
    prompts.append(_f("run_length_decode", "Run-length decoding", "s"))

    # Splitting / joining
    prompts.append(_f("split_lines", "Split into lines", "s"))
    prompts.append(_f("join_words", "Join words with separator", "words, sep"))
    prompts.append(_f("split_csv", "Parse CSV line into list", "s"))
    prompts.append(_f("camel_to_snake", "Convert camelCase to snake_case", "s"))
    prompts.append(_f("snake_to_camel", "Convert snake_case to camelCase", "s"))

    # Comparison
    prompts.append(_f("compare_strings", "Compare two strings, return -1/0/1", "s1, s2"))
    prompts.append(_f("levenshtein", "Levenshtein edit distance", "s1, s2"))
    prompts.append(_f("hamming_distance", "Hamming distance", "s1, s2"))
    prompts.append(_f("longest_common_prefix", "Longest common prefix", "s1, s2"))
    prompts.append(_f("longest_common_substring", "Longest common substring", "s1, s2"))

    return list(dict.fromkeys(prompts))


# ── Python Algorithms ──────────────────────────────────────────────
def _python_algorithms() -> List[str]:
    prompts = []

    # Sorting
    prompts.append(_f("bubble_sort", "Bubble sort ascending", "arr"))
    prompts.append(_f("selection_sort", "Selection sort", "arr"))
    prompts.append(_f("insertion_sort", "Insertion sort", "arr"))
    prompts.append(_f("merge_sort", "Merge sort", "arr"))
    prompts.append(_f("quick_sort", "Quick sort", "arr"))
    prompts.append(_f("heap_sort", "Heap sort", "arr"))
    prompts.append(_f("counting_sort", "Counting sort", "arr"))
    prompts.append(_f("radix_sort", "Radix sort", "arr"))
    prompts.append(_f("tim_sort", "Tim sort (simplified)", "arr"))
    prompts.append(_f("cocktail_sort", "Cocktail shaker sort", "arr"))
    prompts.append(_f("shell_sort", "Shell sort", "arr"))
    prompts.append(_f("gnome_sort", "Gnome sort", "arr"))
    prompts.append(_f("is_sorted", "Check if list is sorted", "arr"))
    prompts.append(_f("sort_descending", "Sort in descending order", "arr"))

    # Searching
    prompts.append(_f("linear_search", "Linear search", "arr, target"))
    prompts.append(_f("binary_search", "Binary search (sorted list)", "arr, target"))
    prompts.append(_f("binary_search_recursive", "Binary search recursive", "arr, target"))
    prompts.append(_f("interpolation_search", "Interpolation search", "arr, target"))
    prompts.append(_f("jump_search", "Jump search", "arr, target"))
    prompts.append(_f("exponential_search", "Exponential search", "arr, target"))
    prompts.append(_f("find_min", "Find minimum element", "arr"))
    prompts.append(_f("find_max", "Find maximum element", "arr"))
    prompts.append(_f("find_second_min", "Find second smallest", "arr"))
    prompts.append(_f("find_second_max", "Find second largest", "arr"))
    prompts.append(_f("find_kth_smallest", "Find kth smallest element", "arr, k"))
    prompts.append(_f("find_kth_largest", "Find kth largest element", "arr, k"))
    prompts.append(_f("find_peak", "Find a peak element", "arr"))
    prompts.append(_f("find_majority", "Find majority element (Boyer-Moore)", "arr"))

    # Data structures
    prompts.append(_f("stack_push", "Push to stack", "stack, item"))
    prompts.append(_f("stack_pop", "Pop from stack", "stack"))
    prompts.append(_f("queue_enqueue", "Enqueue to queue", "queue, item"))
    prompts.append(_f("queue_dequeue", "Dequeue from queue", "queue"))
    prompts.append(_f("linked_list_insert", "Insert into linked list", "head, value"))
    prompts.append(_f("linked_list_delete", "Delete from linked list", "head, value"))
    prompts.append(_f("linked_list_reverse", "Reverse linked list", "head"))
    prompts.append(_f("hash_table_get", "Get from hash table", "table, key"))
    prompts.append(_f("hash_table_put", "Put into hash table", "table, key, value"))

    # List operations
    prompts.append(_f("remove_duplicates", "Remove duplicates from list", "arr"))
    prompts.append(_f("rotate_list", "Rotate list by k positions", "arr, k"))
    prompts.append(_f("reverse_list", "Reverse a list in place", "arr"))
    prompts.append(_f("reverse_list_copy", "Return reversed copy", "arr"))
    prompts.append(_f("flatten_list", "Flatten nested list", "nested"))
    prompts.append(_f("chunk_list", "Split list into chunks of size n", "arr, n"))
    prompts.append(_f("zip_lists", "Zip two lists together", "a, b"))
    prompts.append(_f("interleave", "Interleave two lists", "a, b"))
    prompts.append(_f("merge_sorted", "Merge two sorted lists", "a, b"))
    prompts.append(_f("partition", "Partition list around pivot", "arr, pivot"))
    prompts.append(_f("shuffle_list", "Shuffle list (Fisher-Yates)", "arr"))
    prompts.append(_f("sample_list", "Random sample of k elements", "arr, k"))
    prompts.append(_f("cumulative_sum", "Cumulative sum", "arr"))
    prompts.append(_f("running_average", "Running average", "arr"))
    prompts.append(_f("moving_average", "Moving average with window", "arr, window"))
    prompts.append(_f("diff_list", "Consecutive differences", "arr"))

    # Set operations
    prompts.append(_f("union", "Union of two lists", "a, b"))
    prompts.append(_f("intersection", "Intersection of two lists", "a, b"))
    prompts.append(_f("difference", "Set difference a - b", "a, b"))
    prompts.append(_f("symmetric_diff", "Symmetric difference", "a, b"))
    prompts.append(_f("is_subset", "Check if a is subset of b", "a, b"))
    prompts.append(_f("is_disjoint", "Check if disjoint", "a, b"))

    # Recursion
    prompts.append(_f("power_recursive", "Power recursively", "base, exp"))
    prompts.append(_f("sum_recursive", "Sum list recursively", "arr"))
    prompts.append(_f("reverse_recursive", "Reverse string recursively", "s"))
    prompts.append(_f("count_recursive", "Count occurrences recursively", "arr, target"))
    prompts.append(_f("is_palindrome_rec", "Check palindrome recursively", "s"))
    prompts.append(_f("tree_depth", "Depth of binary tree", "root"))
    prompts.append(_f("tree_inorder", "Inorder traversal", "root"))
    prompts.append(_f("tree_preorder", "Preorder traversal", "root"))
    prompts.append(_f("tree_postorder", "Postorder traversal", "root"))
    prompts.append(_f("tree_bfs", "BFS traversal of tree", "root"))
    prompts.append(_f("tree_dfs", "DFS traversal of tree", "root"))
    prompts.append(_f("count_leaves", "Count leaf nodes in tree", "root"))
    prompts.append(_f("tree_insert", "Insert into BST", "root, value"))
    prompts.append(_f("tree_search", "Search in BST", "root, value"))

    # Graph algorithms
    prompts.append(_f("graph_bfs", "BFS on graph from start node", "graph, start"))
    prompts.append(_f("graph_dfs", "DFS on graph from start node", "graph, start"))
    prompts.append(_f("has_cycle", "Detect cycle in graph", "graph"))
    prompts.append(_f("topological_sort", "Topological sort of DAG", "graph"))
    prompts.append(_f("shortest_path", "Shortest path BFS (unweighted)", "graph, start, end"))
    prompts.append(_f("connected_components", "Count connected components", "graph"))

    # Dynamic programming
    prompts.append(_f("fib_dp", "Fibonacci with DP", "n"))
    prompts.append(_f("climb_stairs", "Ways to climb n stairs (1 or 2)", "n"))
    prompts.append(_f("coin_change", "Min coins to make amount", "coins, amount"))
    prompts.append(_f("knapsack", "0/1 knapsack", "weights, values, capacity"))
    prompts.append(_f("longest_increasing", "Longest increasing subsequence", "arr"))
    prompts.append(_f("longest_common_seq", "Longest common subsequence", "a, b"))
    prompts.append(_f("edit_distance", "Edit distance (Levenshtein)", "s1, s2"))
    prompts.append(_f("max_subarray", "Max subarray sum (Kadane)", "arr"))
    prompts.append(_f("house_robber", "House robber max", "nums"))
    prompts.append(_f("word_break", "Word break problem", "s, word_dict"))

    # Misc algorithms
    prompts.append(_f("two_sum", "Find indices that sum to target", "arr, target"))
    prompts.append(_f("three_sum", "Find triplets that sum to 0", "arr"))
    prompts.append(_f("move_zeros", "Move zeros to end", "arr"))
    prompts.append(_f("contains_duplicate", "Check for duplicates", "arr"))
    prompts.append(_f("missing_number", "Find missing number 0..n", "arr"))
    prompts.append(_f("first_unique", "First unique character index", "s"))
    prompts.append(_f("group_anagrams", "Group anagrams together", "words"))
    prompts.append(_f("valid_parens", "Check valid parentheses", "s"))
    prompts.append(_f("valid_sudoku", "Check valid sudoku board", "board"))
    prompts.append(_f("rotate_matrix", "Rotate matrix 90 degrees", "matrix"))
    prompts.append(_f("spiral_order", "Spiral order of matrix", "matrix"))
    prompts.append(_f("set_zeroes", "Set matrix zeroes", "matrix"))
    prompts.append(_f("pascal_row", "Nth row of Pascal's triangle", "n"))
    prompts.append(_f("pascal_triangle", "Pascal's triangle up to n rows", "n"))

    return list(dict.fromkeys(prompts))


# ── Python OOP ─────────────────────────────────────────────────────
def _python_oop() -> List[str]:
    prompts = []

    # Basic classes
    prompts.append(_cls("Dog", "A simple dog class", "name"))
    prompts.append(_cls("Cat", "A simple cat class", "name"))
    prompts.append(_cls("Counter", "Counting class"))
    prompts.append(_cls("Stack", "Stack data structure"))
    prompts.append(_cls("Queue", "Queue data structure"))
    prompts.append(_cls("Rectangle", "Rectangle with area", "width, height"))
    prompts.append(_cls("Circle", "Circle with area and circumference", "radius"))
    prompts.append(_cls("BankAccount", "Simple bank account", "balance=0"))
    prompts.append(_cls("Person", "Person with name and age", "name, age"))
    prompts.append(_cls("Student", "Student extends Person", "name, age, grade"))
    prompts.append(_cls("Book", "Book with title and author", "title, author"))
    prompts.append(_cls("Library", "Library holding books"))
    prompts.append(_cls("Calculator", "Simple calculator"))
    prompts.append(_cls("Temperature", "Temperature converter", "celsius=0"))
    prompts.append(_cls("Vector2D", "2D vector with operations", "x, y"))
    prompts.append(_cls("Matrix", "2D matrix", "rows, cols"))
    prompts.append(_cls("LinkedList", "Singly linked list"))
    prompts.append(_cls("TreeNode", "Binary tree node", "val"))
    prompts.append(_cls("HashMap", "Simple hash map"))
    prompts.append(_cls("Set", "Simple set implementation"))
    prompts.append(_cls("PriorityQueue", "Priority queue"))
    prompts.append(_cls("Deck", "Deck of cards"))
    prompts.append(_cls("Card", "Playing card", "suit, rank"))
    prompts.append(_cls("Dice", "Dice roller", "sides=6"))
    prompts.append(_cls("Timer", "Simple timer"))
    prompts.append(_cls("Logger", "Simple logger"))
    prompts.append(_cls("Config", "Configuration manager"))
    prompts.append(_cls("Observable", "Observable pattern"))
    prompts.append(_cls("Iterator", "Custom iterator for list", "data"))

    # Class with methods (more complex stubs)
    prompts.append('class Stack:\n    """Stack with push, pop, peek"""\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        ')
    prompts.append('class Queue:\n    """Queue with enqueue, dequeue"""\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        ')
    prompts.append('class BankAccount:\n    """Account with deposit and withdraw"""\n    def __init__(self, balance=0):\n        self.balance = balance\n    def deposit(self, amount):\n        ')
    prompts.append('class Rectangle:\n    """Rectangle with area and perimeter"""\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    def area(self):\n        ')
    prompts.append('class Vector2D:\n    """2D vector with add and magnitude"""\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def add(self, other):\n        ')
    prompts.append('class Counter:\n    """Counter with increment and reset"""\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        ')
    prompts.append('class Temperature:\n    """Temperature with Celsius and Fahrenheit"""\n    def __init__(self, celsius=0):\n        self.celsius = celsius\n    def to_fahrenheit(self):\n        ')
    prompts.append('class Dice:\n    """Dice with roll"""\n    def __init__(self, sides=6):\n        self.sides = sides\n    def roll(self):\n        ')
    prompts.append('class Logger:\n    """Logger with log method"""\n    def __init__(self, name):\n        self.name = name\n        self.logs = []\n    def log(self, message):\n        ')
    prompts.append('class Deck:\n    """Deck of cards with shuffle and deal"""\n    def __init__(self):\n        self.cards = []\n    def shuffle(self):\n        ')

    return list(dict.fromkeys(prompts))


# ── Python File I/O ────────────────────────────────────────────────
def _python_file_io() -> List[str]:
    prompts = []

    prompts.append(_f("read_file", "Read file contents", "path"))
    prompts.append(_f("write_file", "Write to file", "path, content"))
    prompts.append(_f("append_file", "Append to file", "path, content"))
    prompts.append(_f("count_lines", "Count lines in file", "path"))
    prompts.append(_f("read_lines", "Read file as list of lines", "path"))
    prompts.append(_f("read_csv", "Parse CSV file", "path"))
    prompts.append(_f("write_csv", "Write list of rows to CSV", "path, rows"))
    prompts.append(_f("file_exists", "Check if file exists", "path"))
    prompts.append(_f("file_size", "Get file size in bytes", "path"))
    prompts.append(_f("delete_file", "Delete a file", "path"))
    prompts.append(_f("copy_file", "Copy file from src to dst", "src, dst"))
    prompts.append(_f("move_file", "Move/rename a file", "src, dst"))
    prompts.append(_f("list_dir", "List files in directory", "path"))
    prompts.append(_f("list_files_recursive", "List files recursively", "path"))
    prompts.append(_f("make_dir", "Create directory", "path"))
    prompts.append(_f("remove_dir", "Remove directory", "path"))
    prompts.append(_f("get_extension", "Get file extension", "filename"))
    prompts.append(_f("get_filename", "Get filename without path", "path"))
    prompts.append(_f("get_basename", "Get filename without extension", "path"))
    prompts.append(_f("join_path", "Join path components", "*parts"))
    prompts.append(_f("count_words_file", "Count words in file", "path"))
    prompts.append(_f("count_chars_file", "Count characters in file", "path"))
    prompts.append(_f("find_in_file", "Search for pattern in file", "path, pattern"))
    prompts.append(_f("replace_in_file", "Replace text in file", "path, old, new"))
    prompts.append(_f("read_json", "Read JSON file", "path"))
    prompts.append(_f("write_json", "Write dict to JSON file", "path, data"))
    prompts.append(_f("tail_file", "Return last n lines of file", "path, n"))
    prompts.append(_f("head_file", "Return first n lines of file", "path, n"))
    prompts.append(_f("read_binary", "Read binary file", "path"))
    prompts.append(_f("write_binary", "Write binary data to file", "path, data"))

    return list(dict.fromkeys(prompts))


# ── Math (text-based) ──────────────────────────────────────────────
def _math_arithmetic() -> List[str]:
    prompts = []
    random.seed(42)
    # Generate varied arithmetic problems
    for _ in range(60):
        a, b = random.randint(1, 99), random.randint(1, 99)
        op = random.choice(["+", "-", "*", "/"])
        if op == "/" and b != 0:
            # Make sure it divides evenly
            a = a * b
        prompts.append(_math(f"Compute: {a} {op} {b} ="))
    # Order of operations
    for _ in range(20):
        a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)
        prompts.append(_math(f"Compute: {a} + {b} * {c} ="))
        prompts.append(_math(f"Compute: ({a} + {b}) * {c} ="))
        prompts.append(_math(f"Compute: {a} * {b} + {c} ="))
    # Powers and mods
    for _ in range(20):
        a, b = random.randint(2, 12), random.randint(2, 6)
        prompts.append(_math(f"Compute: {a}^{b} ="))
    for _ in range(20):
        a, b = random.randint(10, 99), random.randint(2, 20)
        prompts.append(_math(f"Compute: {a} % {b} ="))
    return list(dict.fromkeys(prompts))


def _math_algebra() -> List[str]:
    prompts = []
    random.seed(42)
    # Linear equations
    for _ in range(30):
        x = random.randint(1, 20)
        a = random.randint(2, 9)
        b = random.randint(1, 50)
        prompts.append(_math(f"Solve: {a}x + {b} = {a*x+b}, x ="))
        prompts.append(_math(f"Solve: {a}x - {b} = {a*x-b}, x ="))
        prompts.append(_math(f"Solve: x/{a} = {x}, x ="))
        prompts.append(_math(f"Solve: {a}(x + {b}) = {a*(x+b)}, x ="))
    # Quadratic
    for r1, r2 in [(1, 6), (2, 3), (-1, -6), (-2, -3), (1, -6), (-1, 6), (2, -3), (-2, 3)]:
        b = -(r1 + r2)
        c = r1 * r2
        prompts.append(_math(f"Solve: x^2 + {b}x + {c} = 0, x ="))
    # Simplify
    for _ in range(20):
        a, b, c = random.randint(2, 9), random.randint(1, 10), random.randint(1, 10)
        prompts.append(_math(f"Simplify: {a}(x + {b}) - {c}x ="))
    return list(dict.fromkeys(prompts))


def _math_geometry() -> List[str]:
    prompts = []
    random.seed(42)
    for _ in range(30):
        r = random.randint(1, 20)
        prompts.append(_math(f"Area of circle with r={r}:"))
        prompts.append(_math(f"Circumference of circle with r={r}:"))
    for _ in range(30):
        b, h = random.randint(2, 15), random.randint(2, 15)
        prompts.append(_math(f"Area of triangle with base={b}, height={h}:"))
        prompts.append(_math(f"Area of rectangle {b}x{h}:"))
        prompts.append(_math(f"Perimeter of rectangle {b}x{h}:"))
    for _ in range(20):
        s = random.randint(2, 15)
        prompts.append(_math(f"Area of square with side={s}:"))
        prompts.append(_math(f"Volume of cube with side={s}:"))
    for _ in range(20):
        a, b = random.randint(3, 12), random.randint(3, 12)
        prompts.append(_math(f"Hypotenuse of right triangle with legs {a}, {b}:"))
    return list(dict.fromkeys(prompts))


def _math_probability() -> List[str]:
    prompts = [
        _math("P(heads) in coin flip ="),
        _math("P(tails) in coin flip ="),
        _math("P(rolling 6 on die) ="),
        _math("P(rolling even on die) ="),
        _math("P(rolling odd on die) ="),
        _math("P(rolling prime on die) ="),
        _math("P(red card from deck) ="),
        _math("P(black card from deck) ="),
        _math("P(ace from deck) ="),
        _math("P(king from deck) ="),
        _math("P(face card from deck) ="),
        _math("P(heart from deck) ="),
        _math("P(two heads in 2 flips) ="),
        _math("P(at least one head in 2 flips) ="),
        _math("P(all heads in 3 flips) ="),
        _math("P(no heads in 3 flips) ="),
        _math("Ways to arrange 3 items ="),
        _math("Ways to arrange 5 items ="),
        _math("Ways to choose 2 from 5 ="),
        _math("Ways to choose 3 from 7 ="),
        _math("Ways to choose 4 from 10 ="),
        _math("P(sum of 7 with two dice) ="),
        _math("P(sum of 8 with two dice) ="),
        _math("P(double with two dice) ="),
        _math("P(sum > 10 with two dice) ="),
        _math("P(first card is ace) ="),
        _math("P(second card is ace, first was ace) ="),
        _math("P(drawing 2 aces in a row) ="),
        _math("Expected value of die roll ="),
        _math("Expected value of coin flip (heads=1, tails=0) ="),
        _math("Variance of die roll ="),
    ]
    return prompts


def _math_theory() -> List[str]:
    prompts = [
        _math("State the Pythagorean theorem:"),
        _math("Define a prime number:"),
        _math("Define a composite number:"),
        _math("State the commutative property of addition:"),
        _math("State the associative property of addition:"),
        _math("State the distributive property:"),
        _math("Define an even number:"),
        _math("Define an odd number:"),
        _math("State the triangle inequality:"),
        _math("Define a rational number:"),
        _math("Define an irrational number:"),
        _math("State the fundamental theorem of arithmetic:"),
        _math("Define the greatest common divisor:"),
        _math("Define the least common multiple:"),
        _math("State Euclid's algorithm:"),
        _math("Define a perfect number:"),
        _math("State the quadratic formula:"),
        _math("Define a function in mathematics:"),
        _math("State the binomial theorem:"),
        _math("Define a factorial:"),
        _math("State the law of large numbers:"),
        _math("Define a permutation:"),
        _math("Define a combination:"),
        _math("State the Pigeonhole principle:"),
        _math("Define mathematical induction:"),
        _math("State the mean value theorem:"),
        _math("Define a limit:"),
        _math("Define a derivative:"),
        _math("Define an integral:"),
        _math("State the chain rule:"),
    ]
    return prompts


# ── Reasoning / Logic / General ────────────────────────────────────
def _reasoning_general() -> List[str]:
    return [
        _math("Why is the sky blue?"),
        _math("Explain why ice floats on water."),
        _math("Why do we need sleep?"),
        _math("Explain cause and effect."),
        _math("Why does rain happen?"),
        _math("Why do leaves change color in fall?"),
        _math("Explain why the earth has seasons."),
        _math("Why do we dream?"),
        _math("How does a vaccine work?"),
        _math("Why is exercise good for you?"),
        _math("Explain how plants make food."),
        _math("Why do things fall down?"),
        _math("How does a rainbow form?"),
        _math("Why is the ocean salty?"),
        _math("Explain how a battery works."),
        _math("Why do we sneeze?"),
        _math("How does the heart pump blood?"),
        _math("Why do birds migrate?"),
        _math("Explain why fire burns."),
        _math("How does a thermometer work?"),
        _math("Why do stars twinkle?"),
        _math("Explain how a mirror works."),
        _math("Why do we yawn?"),
        _math("How does a microwave heat food?"),
        _math("Why is water wet?"),
        _math("Explain how a compass works."),
        _math("Why do ships float?"),
        _math("How does a telescope work?"),
        _math("Why do volcanoes erupt?"),
        _math("Explain how earthquakes happen."),
    ]


def _logic() -> List[str]:
    return [
        _math("If all A are B, and all B are C, then all A are"),
        _math("If P implies Q, and P is true, then Q is"),
        _math("Contrapositive of 'If P then Q' is 'If not"),
        _math("Modus ponens: P, P→Q, therefore"),
        _math("Modus tollens: not Q, P→Q, therefore"),
        _math("If all cats are mammals, and all mammals are animals, then all cats are"),
        _math("If no birds can swim, and penguins can swim, then penguins are not"),
        _math("If A or B is true, and A is false, then"),
        _math("If A and B are both true, then A is"),
        _math("Inverse of 'If P then Q' is 'If not"),
        _math("Converse of 'If P then Q' is 'If"),
        _math("If all squares are rectangles, and X is a square, then X is a"),
        _math("If some birds can fly, and Tweety is a bird, then Tweety"),
        _math("De Morgan's law: not(A and B) ="),
        _math("De Morgan's law: not(A or B) ="),
        _math("If P is true and Q is false, then P and Q is"),
        _math("If P is true and Q is false, then P or Q is"),
        _math("If P is true and Q is false, then P implies Q is"),
        _math("A tautology is a statement that is always"),
        _math("A contradiction is a statement that is always"),
        _math("If A=B and B=C, then"),
        _math("If all dogs bark, and Rex is a dog, then Rex"),
        _math("If it rains, the ground gets wet. The ground is wet. It"),
        _math("Either the coin is heads or tails. It's not heads. It's"),
        _math("If x > 5, then x > 3. x = 4. Therefore x > 5 is"),
    ]


def _general() -> List[str]:
    return [
        _math("The capital of France is"),
        _math("The capital of Japan is"),
        _math("The capital of USA is"),
        _math("The capital of UK is"),
        _math("The capital of Germany is"),
        _math("The capital of Italy is"),
        _math("The capital of China is"),
        _math("The capital of Russia is"),
        _math("The capital of Brazil is"),
        _math("The capital of India is"),
        _math("The capital of Canada is"),
        _math("The capital of Australia is"),
        _math("The largest planet is"),
        _math("The smallest planet is"),
        _math("The closest planet to the Sun is"),
        _math("The farthest planet from the Sun is"),
        _math("Water boils at"),
        _math("Water freezes at"),
        _math("The speed of light is approximately"),
        _math("The speed of sound is approximately"),
        _math("DNA stands for"),
        _math("RNA stands for"),
        _math("HTTP stands for"),
        _math("HTML stands for"),
        _math("CPU stands for"),
        _math("GPU stands for"),
        _math("RAM stands for"),
        _math("The human body has approximately how many bones?"),
        _math("The Great Wall is located in"),
        _math("The Nile River is in"),
        _math("The Amazon River is in"),
        _math("The currency of Japan is"),
        _math("The currency of UK is"),
        _math("The largest ocean is"),
        _math("The smallest ocean is"),
        _math("The tallest mountain is"),
        _math("The largest country by area is"),
        _math("The most populous country is"),
        _math("Python was created by"),
        _math("The first computer programmer was"),
        _math("The theory of relativity was proposed by"),
    ]


# ── Build the full library ─────────────────────────────────────────
def build_topic_prompts() -> Dict[str, List[str]]:
    """Build the full prompt library for all topics.

    Merges the base library with the extended difficulty-tiered library.
    Extended prompts are (prompt, difficulty) tuples — we extract just the prompt.
    """
    # Base prompts (no difficulty tiers)
    base = {
        "python_general": _python_general(),
        "python_math": _python_math(),
        "python_strings": _python_strings(),
        "python_algorithms": _python_algorithms(),
        "python_oop": _python_oop(),
        "python_file_io": _python_file_io(),
        "math_arithmetic": _math_arithmetic(),
        "math_algebra": _math_algebra(),
        "math_geometry": _math_geometry(),
        "math_probability": _math_probability(),
        "math_theory": _math_theory(),
        "reasoning_general": _reasoning_general(),
        "logic": _logic(),
        "general": _general(),
    }

    # Extended prompts with difficulty tiers (merge in)
    try:
        from research.prompt_library_extended import (
            expanded_algorithms, expanded_general)
        ext_alg = expanded_algorithms()  # List[(prompt, difficulty)]
        ext_gen = expanded_general()
        base["python_algorithms"] = list(dict.fromkeys(
            base["python_algorithms"] + [p for p, _ in ext_alg]))
        base["python_general"] = list(dict.fromkeys(
            base["python_general"] + [p for p, _ in ext_gen]))
    except ImportError:
        pass

    try:
        from research.prompt_library_extended2 import (
            expanded_math, expanded_strings)
        ext_math = expanded_math()
        ext_str = expanded_strings()
        base["python_math"] = list(dict.fromkeys(
            base["python_math"] + [p for p, _ in ext_math]))
        base["python_strings"] = list(dict.fromkeys(
            base["python_strings"] + [p for p, _ in ext_str]))
    except ImportError:
        pass

    return base


if __name__ == "__main__":
    prompts = build_topic_prompts()
    total = 0
    print("=" * 50)
    print("Prompt Library Summary")
    print("=" * 50)
    for topic, plist in sorted(prompts.items()):
        print(f"  {topic:25s} {len(plist):4d} prompts")
        total += len(plist)
    print(f"\n  Total: {total} prompts across {len(prompts)} topics")
