"""DEPRECATED: Extended function-stub prompt library for math/strings.

Superseded by research.goal_tasks (Goal-Oriented Self-Play).
Kept for reference only. Use GoalTaskGenerator instead.

Original description:
Extended training prompt library for python_math and python_strings.

Adds 300 prompts per topic (120 easy, 120 medium, 60 hard) as
(prompt_string, difficulty) tuples.  Difficulty is one of
"easy", "medium", "hard".
"""
from typing import List, Tuple


def _f(name: str, docstring: str, args: str = "", difficulty: str = "easy") -> tuple[str, str]:
    """Build a (function_prompt, difficulty) tuple."""
    if args:
        return (f'def {name}({args}):\n    """{docstring}"""\n    ', difficulty)
    return (f'def {name}():\n    """{docstring}"""\n    ', difficulty)


# Rotating argument-name banks used to produce variant prompts.
_SUFFIX_BANKS = [
    ["a", "b", "c", "d"],
    ["x", "y", "z", "w"],
    ["m", "n", "k", "l"],
    ["p", "q", "r", "s"],
]


def _arg_variants(args: str, count: int) -> list[str]:
    """Return `count` renamed variants of a comma-separated arg string."""
    if not args:
        return [""] * count
    parts = [p.strip() for p in args.split(",")]
    n = len(parts)
    # Flatten all bank letters into one pool, cycle as needed
    pool = [c for bank in _SUFFIX_BANKS for c in bank]
    variants = []
    for vi in range(count):
        offset = vi * n
        names = [pool[(offset + i) % len(pool)] for i in range(n)]
        variants.append(", ".join(names))
    return variants


# ── python_math specs ───────────────────────────────────────────────
# Each spec: (function_name, docstring, args)

_EASY_MATH = [
    ("is_prime", "Return True if n is prime", "n"),
    ("is_even", "Return True if n is even", "n"),
    ("is_odd", "Return True if n is odd", "n"),
    ("factorial", "Return n factorial", "n"),
    ("gcd", "Return greatest common divisor of a and b", "a, b"),
    ("lcm", "Return least common multiple of a and b", "a, b"),
    ("power", "Return base raised to exp", "base, exp"),
    ("square", "Return the square of n", "n"),
    ("cube", "Return the cube of n", "n"),
    ("sum_digits", "Return the sum of digits of n", "n"),
    ("reverse_number", "Return the digits of n reversed", "n"),
    ("count_digits", "Return the number of digits in n", "n"),
    ("is_perfect_square", "Return True if n is a perfect square", "n"),
    ("is_perfect_cube", "Return True if n is a perfect cube", "n"),
    ("is_power_of_two", "Return True if n is a power of two", "n"),
    ("digital_root", "Return the digital root of n", "n"),
    ("next_prime", "Return the smallest prime greater than n", "n"),
    ("prev_prime", "Return the largest prime smaller than n", "n"),
    ("is_fibonacci", "Return True if n is a Fibonacci number", "n"),
    ("is_triangular", "Return True if n is a triangular number", "n"),
    ("is_coprime", "Return True if a and b are coprime", "a, b"),
    ("is_armstrong", "Return True if n is an Armstrong number", "n"),
    ("is_narcissistic", "Return True if n is a narcissistic number", "n"),
    ("collatz_steps", "Return number of Collatz steps to reach 1", "n"),
    ("prime_factors", "Return list of prime factors of n", "n"),
    ("count_primes", "Return count of primes up to n", "n"),
    ("tribonacci", "Return the n-th tribonacci number", "n"),
    ("euler_phi", "Return Euler's totient of n", "n"),
    ("mod_pow", "Return (base**exp) % mod", "base, exp, mod"),
    ("mod_inverse", "Return modular inverse of a mod m", "a, m"),
]

_MEDIUM_MATH = [
    # sequences
    ("arithmetic_sum", "Return sum of arithmetic sequence", "a, d, n"),
    ("geometric_sum", "Return sum of geometric sequence", "a, r, n"),
    ("triangular_number", "Return the n-th triangular number", "n"),
    ("square_number", "Return the n-th square number", "n"),
    ("pentagonal_number", "Return the n-th pentagonal number", "n"),
    ("catalan", "Return the n-th Catalan number", "n"),
    ("bell_number", "Return the n-th Bell number", "n"),
    ("stirling_second", "Return Stirling number of the second kind S(n,k)", "n, k"),
    ("fibonacci_golden", "Return n-th Fibonacci via golden ratio", "n"),
    ("lucas_numbers", "Return the n-th Lucas number", "n"),
    ("padovan_sequence", "Return the n-th Padovan number", "n"),
    # combinatorics
    ("nCr", "Return binomial coefficient n choose r", "n, r"),
    ("nPr", "Return number of permutations n P r", "n, r"),
    ("perm_with_repetition", "Return permutations of n items taken r with repetition", "n, r"),
    ("comb_with_repetition", "Return combinations of n items taken r with repetition", "n, r"),
    ("derangements", "Return number of derangements of n items", "n"),
    ("multinomial", "Return multinomial coefficient for given counts", "n, counts"),
    # modular arithmetic
    ("chinese_remainder", "Solve system via Chinese Remainder Theorem", "remainders, moduli"),
    ("extended_gcd", "Return (g, x, y) with a*x+b*y=gcd(a,b)", "a, b"),
    ("discrete_log", "Return x with base^x == target mod mod", "base, target, mod"),
    ("baby_step_giant_step", "Solve discrete log via baby-step giant-step", "base, target, mod"),
    # number conversions
    ("to_binary", "Return binary string of n", "n"),
    ("to_octal", "Return octal string of n", "n"),
    ("to_hex", "Return hexadecimal string of n", "n"),
    ("from_binary", "Return integer value of binary string s", "s"),
    ("from_hex", "Return integer value of hex string s", "s"),
    ("base_convert", "Convert n from base b1 to base b2", "n, b1, b2"),
    # geometry
    ("circle_area", "Return area of circle with radius r", "r"),
    ("circle_circumference", "Return circumference of circle with radius r", "r"),
    ("triangle_area", "Return area of triangle given base and height", "base, height"),
    ("rectangle_area", "Return area of rectangle", "length, width"),
    ("rectangle_perimeter", "Return perimeter of rectangle", "length, width"),
    ("cube_volume", "Return volume of cube with side s", "s"),
    ("sphere_volume", "Return volume of sphere with radius r", "r"),
    ("pythagorean", "Return hypotenuse given legs a and b", "a, b"),
    ("distance", "Return Euclidean distance between two points", "x1, y1, x2, y2"),
    ("heron_formula", "Return triangle area via Heron's formula", "a, b, c"),
    ("trapezoid_area", "Return area of trapezoid", "a, b, h"),
    ("sector_area", "Return area of a circular sector", "r, theta"),
    ("arc_length", "Return arc length of a circular sector", "r, theta"),
]

_HARD_MATH = [
    # matrix operations
    ("matrix_add", "Return sum of two matrices", "A, B"),
    ("matrix_multiply", "Return product of two matrices", "A, B"),
    ("matrix_transpose", "Return transpose of matrix", "M"),
    ("matrix_trace", "Return trace of square matrix", "M"),
    ("matrix_determinant_2x2", "Return determinant of 2x2 matrix", "M"),
    ("matrix_determinant_3x3", "Return determinant of 3x3 matrix", "M"),
    ("matrix_inverse_2x2", "Return inverse of 2x2 matrix", "M"),
    ("matrix_power", "Return matrix M raised to integer power n", "M, n"),
    ("eigenvalues_2x2", "Return eigenvalues of 2x2 matrix", "M"),
    ("solve_linear_2var", "Solve a*x+b*y=e and c*x+d*y=f", "a, b, e, c, d, f"),
    ("cramers_rule", "Solve linear system using Cramer's rule", "A, b"),
    ("gaussian_elimination", "Solve linear system via Gaussian elimination", "A, b"),
    # advanced number theory / algorithms
    ("sieve_of_eratosthenes", "Return list of primes up to n", "n"),
    ("miller_rabin", "Probabilistic primality test of n", "n"),
    ("pollard_rho", "Return a non-trivial factor of n via Pollard rho", "n"),
    ("discrete_sqrt", "Return square root of a modulo prime p", "a, p"),
    ("continued_fraction", "Return continued fraction expansion of sqrt(n)", "n"),
    ("pell_equation", "Return minimal solution to x^2 - d*y^2 = 1", "d"),
    ("fermat_little", "Verify Fermat's little theorem for a and prime p", "a, p"),
    ("wilson_theorem", "Verify Wilson's theorem for prime p", "p"),
    ("goldbach_check", "Return two primes summing to even n", "n"),
    ("twin_primes", "Return list of twin prime pairs up to n", "n"),
    ("mersenne_prime", "Return True if 2^p-1 is a Mersenne prime", "p"),
    ("perfect_number", "Return True if n is a perfect number", "n"),
    ("abundant_number", "Return True if n is an abundant number", "n"),
    ("deficient_number", "Return True if n is a deficient number", "n"),
    ("happy_number", "Return True if n is a happy number", "n"),
    ("kaprekar_number", "Return True if n is a Kaprekar number", "n"),
    ("harshad_number", "Return True if n is a Harshad number", "n"),
    ("automorphic_number", "Return True if n is an automorphic number", "n"),
    ("ugly_number", "Return True if n is an ugly number", "n"),
    ("hamming_distance_nums", "Return Hamming distance between two integers", "a, b"),
    ("gray_code", "Return Gray code of integer n", "n"),
    ("josephus", "Return survivor position in Josephus problem", "n, k"),
    ("ackermann", "Return Ackermann(m, n)", "m, n"),
    ("super_digit", "Return super digit of n repeated k times", "n, k"),
    ("look_and_say", "Return the n-th look-and-say sequence term", "n"),
    ("conway_constant", "Estimate Conway's constant from look-and-say growth", "n"),
    ("rsa_encrypt", "Return RSA ciphertext of message m", "m, e, n"),
    ("rsa_decrypt", "Return RSA plaintext of ciphertext c", "c, d, n"),
    ("sha256_simple", "Return SHA-256 hex digest of string s", "s"),
    ("crc32", "Return CRC-32 checksum of bytes data", "data"),
    ("lcg_random", "Return next value of a linear congruential generator", "seed, a, c, m"),
    ("mersenne_twister_seed", "Seed a Mersenne-Twister-like state from seed", "seed"),
    ("birthday_paradox", "Return probability of shared birthday for n people", "n"),
    ("monty_hall", "Return win probability of switching in Monty Hall with n doors", "n"),
    ("st_petersburg_paradox", "Return expected value of St. Petersburg game for n rounds", "n"),
]


# ── python_strings specs ────────────────────────────────────────────

_EASY_STRINGS = [
    # basic transforms
    ("reverse", "Return the reversed string", "s"),
    ("to_upper", "Return s converted to uppercase", "s"),
    ("to_lower", "Return s converted to lowercase", "s"),
    ("to_title", "Return s in title case", "s"),
    ("to_capitalize", "Return s with first character capitalized", "s"),
    ("swap_case", "Return s with case swapped", "s"),
    # checking
    ("is_palindrome", "Return True if s is a palindrome", "s"),
    ("is_anagram", "Return True if s1 and s2 are anagrams", "s1, s2"),
    ("is_pangram", "Return True if s is a pangram", "s"),
    ("is_digit", "Return True if s contains only digits", "s"),
    ("is_alpha", "Return True if s contains only letters", "s"),
    ("is_alnum", "Return True if s is alphanumeric", "s"),
    ("is_upper", "Return True if s is fully uppercase", "s"),
    ("is_lower", "Return True if s is fully lowercase", "s"),
    ("is_space", "Return True if s contains only whitespace", "s"),
    ("starts_with", "Return True if s starts with prefix", "s, prefix"),
    ("ends_with", "Return True if s ends with suffix", "s, suffix"),
    ("contains", "Return True if s contains substring", "s, sub"),
    # counting
    ("count_char", "Return count of char c in s", "s, c"),
    ("count_vowels", "Return number of vowels in s", "s"),
    ("count_consonants", "Return number of consonants in s", "s"),
    ("count_words", "Return number of words in s", "s"),
    ("count_substring", "Return number of occurrences of sub in s", "s, sub"),
    ("count_upper", "Return number of uppercase letters in s", "s"),
    ("count_lower", "Return number of lowercase letters in s", "s"),
    ("count_digits", "Return number of digits in s", "s"),
    ("count_special", "Return number of special characters in s", "s"),
    # manipulation
    ("remove_spaces", "Return s with all spaces removed", "s"),
    ("remove_vowels", "Return s with all vowels removed", "s"),
    ("remove_digits", "Return s with all digits removed", "s"),
    ("remove_punctuation", "Return s with punctuation removed", "s"),
    ("replace_vowels", "Replace every vowel in s with char c", "s, c"),
    ("replace_char", "Replace every occurrence of old with new in s", "s, old, new"),
    ("capitalize_words", "Capitalize first letter of each word in s", "s"),
    ("reverse_words", "Reverse the order of words in s", "s"),
    ("reverse_each_word", "Reverse each word in s in place", "s"),
    ("strip_extra_spaces", "Collapse repeated whitespace in s to single spaces", "s"),
    ("pad_left", "Left-pad s with char c to width w", "s, w, c"),
    ("pad_right", "Right-pad s with char c to width w", "s, w, c"),
    ("center_string", "Center s in field of width w with char c", "s, w, c"),
    ("truncate", "Truncate s to at most n characters with ellipsis", "s, n"),
]

_MEDIUM_STRINGS = [
    # extraction
    ("first_word", "Return the first word of s", "s"),
    ("last_word", "Return the last word of s", "s"),
    ("get_words", "Return list of words in s", "s"),
    ("get_digits", "Return all digits found in s", "s"),
    ("get_alpha", "Return only alphabetic characters of s", "s"),
    ("get_unique_chars", "Return sorted unique characters of s", "s"),
    ("get_char_freq", "Return dict of character frequencies in s", "s"),
    # splitting / joining
    ("split_lines", "Return list of lines in s", "s"),
    ("join_words", "Join list of words with delimiter d", "words, d"),
    ("split_csv", "Return list of fields from CSV line s", "s"),
    ("camel_to_snake", "Convert camelCase string to snake_case", "s"),
    ("snake_to_camel", "Convert snake_case string to camelCase", "s"),
    ("kebab_to_camel", "Convert kebab-case string to camelCase", "s"),
    ("camel_to_kebab", "Convert camelCase string to kebab-case", "s"),
    # comparison
    ("compare_strings", "Return -1, 0, or 1 comparing s1 and s2", "s1, s2"),
    ("levenshtein", "Return Levenshtein edit distance of s1 and s2", "s1, s2"),
    ("hamming_distance", "Return Hamming distance of equal-length s1 and s2", "s1, s2"),
    ("longest_common_prefix", "Return longest common prefix of s1 and s2", "s1, s2"),
    ("longest_common_substring", "Return longest common substring of s1 and s2", "s1, s2"),
    ("jaro_similarity", "Return Jaro similarity of s1 and s2", "s1, s2"),
    ("soundex", "Return Soundex code of s", "s"),
    # encoding / ciphers
    ("caesar_cipher", "Return Caesar-encrypted s with shift k", "s, k"),
    ("caesar_decipher", "Return Caesar-decrypted s with shift k", "s, k"),
    ("rot13", "Return ROT13 of s", "s"),
    ("atbash", "Return Atbash cipher of s", "s"),
    ("vigenere_cipher", "Return Vigenere-encrypted s with key", "s, key"),
    ("vigenere_decipher", "Return Vigenere-decrypted s with key", "s, key"),
    ("morse_encode", "Return Morse code encoding of s", "s"),
    ("morse_decode", "Return decoded string from Morse code s", "s"),
    ("run_length_encode", "Return run-length encoding of s", "s"),
    ("run_length_decode", "Return decoded string from RLE s", "s"),
    ("base64_encode_simple", "Return simple base64 encoding of s", "s"),
    ("url_encode", "Return URL-encoded form of s", "s"),
    ("html_escape", "Return HTML-escaped form of s", "s"),
    ("html_unescape", "Return HTML-unescaped form of s", "s"),
    # formatting
    ("format_phone", "Return formatted phone number from digits s", "s"),
    ("format_ssn", "Return formatted SSN from digits s", "s"),
    ("format_date", "Return formatted date string from s", "s"),
    ("format_currency_string", "Return currency-formatted string of amount", "amount"),
    ("indent_text", "Indent every line of s by n spaces", "s, n"),
    ("dedent_text", "Remove common leading whitespace from s", "s"),
    ("align_columns", "Align columns of text block s into a table", "s"),
    ("wrap_text", "Wrap text s to lines of width w", "s, w"),
    ("tab_to_spaces", "Convert tabs in s to n spaces", "s, n"),
    ("spaces_to_tab", "Convert every n spaces in s to a tab", "s, n"),
]

_HARD_STRINGS = [
    ("lcs_string", "Return longest common subsequence string of s1 and s2", "s1, s2"),
    ("scs_shortest_common_supersequence", "Return shortest common supersequence of s1 and s2", "s1, s2"),
    ("z_function", "Return Z-array of s", "s"),
    ("kmp_search", "Return list of indices of pattern p in text t via KMP", "t, p"),
    ("rabin_karp_search", "Return list of indices of p in t via Rabin-Karp", "t, p"),
    ("boyer_moore_search", "Return list of indices of p in t via Boyer-Moore", "t, p"),
    ("aho_corasick", "Return matches of any pattern in patterns against t", "t, patterns"),
    ("suffix_array", "Return suffix array of s", "s"),
    ("lcp_array", "Return LCP array given string s and its suffix array", "s, sa"),
    ("manacher_palindrome", "Return longest palindromic substring via Manacher", "s"),
    ("minimal_rotation", "Return lexicographically minimal rotation of s", "s"),
    ("duval_lyndon_factorization", "Return Lyndon factorization of s via Duval's algorithm", "s"),
    ("huffman_encoding", "Return Huffman-encoded bits of s", "s"),
    ("lzw_compression", "Return LZW-compressed codes of s", "s"),
    ("burrows_wheeler_transform", "Return Burrows-Wheeler transform of s", "s"),
    ("string_hashing", "Return polynomial rolling hash of s", "s"),
    ("rolling_hash", "Return rolling hash of s with window size w", "s, w"),
    ("double_hashing", "Return double hash pair of s", "s"),
    ("fnv_hash", "Return FNV-1a hash of s", "s"),
    ("murmurhash", "Return MurmurHash3 of s with seed", "s, seed"),
    ("find_all_anagrams", "Return start indices of anagrams of p in s", "s, p"),
    ("group_anagrams", "Return list of grouped anagrams from words", "words"),
    ("find_anagram_indices", "Return all anagram index pairs in s", "s"),
    ("longest_palindrome_substring", "Return longest palindromic substring of s", "s"),
    ("longest_palindrome_subseq", "Return longest palindromic subsequence of s", "s"),
    ("palindrome_partitioning_min", "Return minimum cuts for palindrome partitioning of s", "s"),
    ("word_break_ii", "Return all segmentations of s using dictionary words", "s, word_dict"),
    ("regex_match_simple", "Return True if s matches simple regex pattern p", "s, p"),
    ("wildcard_match", "Return True if s matches wildcard pattern p", "s, p"),
    ("validate_ipv4", "Return True if s is a valid IPv4 address", "s"),
    ("validate_ipv6", "Return True if s is a valid IPv6 address", "s"),
    ("validate_mac_address", "Return True if s is a valid MAC address", "s"),
    ("validate_credit_card", "Return True if s is a valid credit card number", "s"),
    ("validate_isbn", "Return True if s is a valid ISBN", "s"),
    ("validate_uuid", "Return True if s is a valid UUID", "s"),
    ("json_parse_simple", "Return parsed object from simple JSON string s", "s"),
    ("xml_parse_simple", "Return parsed structure from simple XML string s", "s"),
    ("csv_to_json", "Return JSON string converted from CSV string s", "s"),
    ("json_to_csv", "Return CSV string converted from JSON string s", "s"),
    ("template_render", "Render template s with variables dict", "s, variables"),
    ("sprintf_simple", "Return formatted string from template s and args", "s, args"),
    ("markdown_to_text", "Return plain text from Markdown string s", "s"),
    ("slugify", "Return URL-safe slug of s", "s"),
    ("transliterate", "Return transliterated form of s to target script", "s"),
    ("unicode_normalize", "Return Unicode-normalized form of s", "s, form"),
    ("count_graphemes", "Return number of grapheme clusters in s", "s"),
    ("emoji_strip", "Return s with emoji characters removed", "s"),
    ("zwsp_remove", "Return s with zero-width spaces removed", "s"),
    ("homoglyph_detect", "Return list of homoglyph-confusable characters in s", "s"),
]


def expanded_math() -> list[tuple[str, str]]:
    """Return 300 (prompt, difficulty) tuples for python_math."""
    easy: list[tuple[str, str]] = []
    for name, doc, args in _EASY_MATH:
        for va in _arg_variants(args, 4):
            easy.append(_f(name, doc, va, "easy"))

    medium: list[tuple[str, str]] = []
    for name, doc, args in _MEDIUM_MATH:
        for va in _arg_variants(args, 3):
            medium.append(_f(name, doc, va, "medium"))

    hard: list[tuple[str, str]] = []
    for name, doc, args in _HARD_MATH:
        for va in _arg_variants(args, 2):
            hard.append(_f(name, doc, va, "hard"))

    return easy[:120] + medium[:120] + hard[:60]


def expanded_strings() -> list[tuple[str, str]]:
    """Return 300 (prompt, difficulty) tuples for python_strings."""
    easy: list[tuple[str, str]] = []
    for name, doc, args in _EASY_STRINGS:
        for va in _arg_variants(args, 3):
            easy.append(_f(name, doc, va, "easy"))

    medium: list[tuple[str, str]] = []
    for name, doc, args in _MEDIUM_STRINGS:
        for va in _arg_variants(args, 3):
            medium.append(_f(name, doc, va, "medium"))

    hard: list[tuple[str, str]] = []
    for name, doc, args in _HARD_STRINGS:
        for va in _arg_variants(args, 2):
            hard.append(_f(name, doc, va, "hard"))

    return easy[:120] + medium[:120] + hard[:60]


if __name__ == "__main__":
    _m = expanded_math()
    _s = expanded_strings()
    from collections import Counter
    print("math:", len(_m), Counter(d for _, d in _m))
    print("strings:", len(_s), Counter(d for _, d in _s))
    assert len(_m) == 300 and len(_s) == 300
    assert sum(1 for _, d in _m if d == "easy") == 120
    assert sum(1 for _, d in _m if d == "medium") == 120
    assert sum(1 for _, d in _m if d == "hard") == 60
    assert sum(1 for _, d in _s if d == "easy") == 120
    assert sum(1 for _, d in _s if d == "medium") == 120
    assert sum(1 for _, d in _s if d == "hard") == 60
    print("OK")
