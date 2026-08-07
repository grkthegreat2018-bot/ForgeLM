"""Auto-generate test cases using stdlib reference implementations.

For each function name, we try to map it to a known Python stdlib function
or a simple implementation, then run it on edge cases to generate (args, expected) pairs.
"""
import re
import sys
sys.path.insert(0, '.')

from research.prompt_library import build_topic_prompts
from research.prompt_tests import TESTS as EXISTING_TESTS


# ── Reference implementations ─────────────────────────────────────
# Map function name patterns to reference code generators

def make_ref(name, args, doc):
    """Generate reference code for a function based on name + docstring."""
    n = name.lower()
    d = doc.lower()
    parts = [a.strip() for a in args.split(",") if a.strip()]
    nargs = len(parts)

    # ── Direct name matches ──────────────────────────────────────
    simple_refs = {
        # Arithmetic
        "add": lambda: f"def {name}({args}):\n    return {' + '.join(parts)}",
        "sum": lambda: f"def {name}({args}):\n    return {' + '.join(parts)}" if nargs >= 2 else f"def {name}({args}):\n    return sum({parts[0]})",
        "subtract": lambda: f"def {name}({args}):\n    return {parts[0]} - {parts[1]}",
        "multiply": lambda: f"def {name}({args}):\n    return {' * '.join(parts)}",
        "divide": lambda: f"def {name}({args}):\n    return {parts[0]} / {parts[1]}",
        "mod": lambda: f"def {name}({args}):\n    return {parts[0]} % {parts[1]}",
        "negate": lambda: f"def {name}(n):\n    return -n",
        "absolute": lambda: f"def {name}(n):\n    return abs(n)",
        "abs": lambda: f"def {name}(n):\n    return abs(n)",
        "square": lambda: f"def {name}(n):\n    return n * n",
        "cube": lambda: f"def {name}(n):\n    return n ** 3",
        "double": lambda: f"def {name}(n):\n    return n * 2",
        "triple": lambda: f"def {name}(n):\n    return n * 3",
        "sign": lambda: f"def {name}(n):\n    return (n > 0) - (n < 0)",
        "identity": lambda: f"def {name}(x):\n    return x",
        "average": lambda: f"def {name}({args}):\n    return ({' + '.join(parts)}) / {nargs}" if nargs >= 2 else f"def {name}({args}):\n    return sum({parts[0]}) / len({parts[0]})" if nargs == 1 else None,
        "midpoint": lambda: f"def {name}(a, b):\n    return (a + b) / 2",
        "factorial": lambda: f"def {name}(n):\n    import math\n    return math.factorial(n)",
        "fib": lambda: f"def {name}(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "fibonacci": lambda: f"def {name}(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "gcd": lambda: f"def {name}(a, b):\n    import math\n    return math.gcd(a, b)",
        "lcm": lambda: f"def {name}(a, b):\n    import math\n    return abs(a*b)//math.gcd(a,b) if a and b else 0",
        "power": lambda: f"def {name}(base, exp):\n    return base ** exp",
        "pow": lambda: f"def {name}(base, exp):\n    return base ** exp",
        "sum_digits": lambda: f"def {name}(n):\n    return sum(int(d) for d in str(abs(n)))",
        "count_digits": lambda: f"def {name}(n):\n    return len(str(abs(n)))",
        "reverse_number": lambda: f"def {name}(n):\n    return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)",
        "digital_root": lambda: f"def {name}(n):\n    while n > 9: n = sum(int(d) for d in str(n))\n    return n",
        "is_prime": lambda: f"def {name}(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "is_even": lambda: f"def {name}(n):\n    return n % 2 == 0",
        "is_odd": lambda: f"def {name}(n):\n    return n % 2 != 0",
        "is_positive": lambda: f"def {name}(n):\n    return n > 0",
        "is_negative": lambda: f"def {name}(n):\n    return n < 0",
        "is_zero": lambda: f"def {name}(n):\n    return n == 0",
        "is_power_of_two": lambda: f"def {name}(n):\n    return n > 0 and (n & (n-1)) == 0",
        "is_perfect_square": lambda: f"def {name}(n):\n    import math\n    r = math.isqrt(n)\n    return r * r == n",
        "is_palindrome": lambda: f"def {name}(s):\n    return s == s[::-1]",
        "is_anagram": lambda: f"def {name}(s1, s2):\n    return sorted(s1) == sorted(s2)",
        "is_sorted": lambda: f"def {name}(arr):\n    return arr == sorted(arr)",
        "is_empty": lambda: f"def {name}(s):\n    return len(s) == 0",
        "is_unique": lambda: f"def {name}(arr):\n    return len(arr) == len(set(arr))",
        "is_leap_year": lambda: f"def {name}(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
        "is_upper": lambda: f"def {name}(s):\n    return s.isupper()",
        "is_lower": lambda: f"def {name}(s):\n    return s.islower()",
        "is_alpha": lambda: f"def {name}(s):\n    return s.isalpha()",
        "is_digit": lambda: f"def {name}(s):\n    return s.isdigit()",
        "is_coprime": lambda: f"def {name}(a, b):\n    import math\n    return math.gcd(a, b) == 1",
        "is_armstrong": lambda: f"def {name}(n):\n    s = str(n)\n    return sum(int(d)**len(s) for d in s) == n",
        "is_narcissistic": lambda: f"def {name}(n):\n    s = str(n)\n    return sum(int(d)**len(s) for d in s) == n",
        "is_triangular": lambda: f"def {name}(n):\n    import math\n    r = int((math.sqrt(8*n+1)-1)/2)\n    return r*(r+1)//2 == n",
        "is_fibonacci": lambda: f"def {name}(n):\n    a, b = 0, 1\n    while a < n: a, b = b, a+b\n    return a == n",
        "is_subset": lambda: f"def {name}(a, b):\n    return set(a) <= set(b)",
        "is_disjoint": lambda: f"def {name}(a, b):\n    return set(a).isdisjoint(set(b))",
        "is_equal": lambda: f"def {name}(a, b):\n    return a == b",
        "is_greater": lambda: f"def {name}(a, b):\n    return a > b",
        "is_less": lambda: f"def {name}(a, b):\n    return a < b",
        "is_greater_than": lambda: f"def {name}(a, b):\n    return a > b",
        "is_less_than": lambda: f"def {name}(a, b):\n    return a < b",
        # String
        "reverse": lambda: f"def {name}(s):\n    return s[::-1]",
        "to_upper": lambda: f"def {name}(s):\n    return s.upper()",
        "to_lower": lambda: f"def {name}(s):\n    return s.lower()",
        "to_title": lambda: f"def {name}(s):\n    return s.title()",
        "to_capitalize": lambda: f"def {name}(s):\n    return s.capitalize()",
        "swap_case": lambda: f"def {name}(s):\n    return s.swapcase()",
        "length": lambda: f"def {name}(s):\n    return len(s)",
        "count_vowels": lambda: f"def {name}(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "count_consonants": lambda: f"def {name}(s):\n    return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')",
        "count_words": lambda: f"def {name}(s):\n    return len(s.split())",
        "remove_spaces": lambda: f"def {name}(s):\n    return s.replace(' ', '')",
        "remove_vowels": lambda: f"def {name}(s):\n    return ''.join(c for c in s if c.lower() not in 'aeiou')",
        "remove_digits": lambda: f"def {name}(s):\n    return ''.join(c for c in s if not c.isdigit())",
        "remove_punctuation": lambda: f"def {name}(s):\n    import string\n    return ''.join(c for c in s if c not in string.punctuation)",
        "first_char": lambda: f"def {name}(s):\n    return s[0] if s else ''",
        "last_char": lambda: f"def {name}(s):\n    return s[-1] if s else ''",
        "first_word": lambda: f"def {name}(s):\n    return s.split()[0] if s.split() else ''",
        "last_word": lambda: f"def {name}(s):\n    words = s.split()\n    return words[-1] if words else ''",
        "get_words": lambda: f"def {name}(s):\n    return s.split()",
        "get_digits": lambda: f"def {name}(s):\n    return ''.join(c for c in s if c.isdigit())",
        "get_alpha": lambda: f"def {name}(s):\n    return ''.join(c for c in s if c.isalpha())",
        "get_unique_chars": lambda: f"def {name}(s):\n    return list(dict.fromkeys(s))",
        "capitalize_words": lambda: f"def {name}(s):\n    return ' '.join(w.capitalize() for w in s.split())",
        "reverse_words": lambda: f"def {name}(s):\n    return ' '.join(s.split()[::-1])",
        "reverse_each_word": lambda: f"def {name}(s):\n    return ' '.join(w[::-1] for w in s.split())",
        "strip_extra_spaces": lambda: f"def {name}(s):\n    return ' '.join(s.split())",
        "starts_with": lambda: f"def {name}(s, prefix):\n    return s.startswith(prefix)",
        "ends_with": lambda: f"def {name}(s, suffix):\n    return s.endswith(suffix)",
        "contains": lambda: f"def {name}(s, sub):\n    return sub in s",
        "repeat_string": lambda: f"def {name}(s, n):\n    return s * n",
        "repeat_char": lambda: f"def {name}(c, n):\n    return c * n",
        "repeat": lambda: f"def {name}(s, n):\n    return s * n",
        "truncate": lambda: f"def {name}(s, max_len):\n    return s[:max_len] + '...' if len(s) > max_len else s",
        "center_string": lambda: f"def {name}(s, width):\n    return s.center(width)",
        "pad_left": lambda: f"def {name}(s, n, c):\n    return s.rjust(n, c)",
        "pad_right": lambda: f"def {name}(s, n, c):\n    return s.ljust(n, c)",
        "compare_strings": lambda: f"def {name}(s1, s2):\n    return (s1 > s2) - (s1 < s2)",
        "longest_common_prefix": lambda: f"def {name}(s1, s2):\n    i = 0\n    while i < min(len(s1), len(s2)) and s1[i] == s2[i]: i += 1\n    return s1[:i]",
        "caesar_cipher": lambda: f"def {name}(s, n):\n    result = ''\n    for c in s:\n        if c.isalpha():\n            base = ord('a') if c.islower() else ord('A')\n            result += chr((ord(c) - base + n) % 26 + base)\n        else: result += c\n    return result",
        "rot13": lambda: f"def {name}(s):\n    result = ''\n    for c in s:\n        if c.isalpha():\n            base = ord('a') if c.islower() else ord('A')\n            result += chr((ord(c) - base + 13) % 26 + base)\n        else: result += c\n    return result",
        "atbash": lambda: f"def {name}(s):\n    result = ''\n    for c in s:\n        if c.islower(): result += chr(ord('z') - (ord(c) - ord('a')))\n        elif c.isupper(): result += chr(ord('Z') - (ord(c) - ord('A')))\n        else: result += c\n    return result",
        "run_length_encode": lambda: f"def {name}(s):\n    if not s: return ''\n    result = ''\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]: count += 1\n        else: result += str(count) + s[i-1]; count = 1\n    result += str(count) + s[-1]\n    return result",
        "camel_to_snake": lambda: f"def {name}(s):\n    result = ''\n    for c in s:\n        if c.isupper(): result += '_' + c.lower()\n        else: result += c\n    return result.lstrip('_')",
        "snake_to_camel": lambda: f"def {name}(s):\n    parts = s.split('_')\n    return parts[0] + ''.join(w.capitalize() for w in parts[1:])",
        "levenshtein": lambda: f"def {name}(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = list(range(n+1))\n    for i in range(1, m+1):\n        prev = dp[0]; dp[0] = i\n        for j in range(1, n+1):\n            tmp = dp[j]\n            dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (s1[i-1] != s2[j-1]))\n            prev = tmp\n    return dp[n]",
        "hamming_distance": lambda: f"def {name}(s1, s2):\n    return sum(a != b for a, b in zip(s1, s2))",
        # List
        "find_min": lambda: f"def {name}(arr):\n    return min(arr) if arr else None",
        "find_max": lambda: f"def {name}(arr):\n    return max(arr) if arr else None",
        "sum_list": lambda: f"def {name}(arr):\n    return sum(arr)",
        "product_list": lambda: f"def {name}(arr):\n    import math\n    return math.prod(arr) if arr else 1",
        "reverse_list": lambda: f"def {name}(arr):\n    return arr[::-1]",
        "sort_simple": lambda: f"def {name}(arr):\n    return sorted(arr)",
        "sort_ascending": lambda: f"def {name}(arr):\n    return sorted(arr)",
        "sort_descending": lambda: f"def {name}(arr):\n    return sorted(arr, reverse=True)",
        "bubble_sort": lambda: f"def {name}(arr):\n    arr = arr[:]\n    for i in range(len(arr)):\n        for j in range(len(arr)-1-i):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
        "linear_search": lambda: f"def {name}(arr, target):\n    for i, v in enumerate(arr):\n        if v == target: return i\n    return -1",
        "binary_search": lambda: f"def {name}(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "remove_duplicates": lambda: f"def {name}(arr):\n    seen = set()\n    return [x for x in arr if not (x in seen or seen.add(x))]",
        "is_sorted_asc": lambda: f"def {name}(arr):\n    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))",
        "is_sorted_desc": lambda: f"def {name}(arr):\n    return all(arr[i] >= arr[i+1] for i in range(len(arr)-1))",
        "all_unique": lambda: f"def {name}(arr):\n    return len(arr) == len(set(arr))",
        "all_positive": lambda: f"def {name}(arr):\n    return all(x > 0 for x in arr)",
        "any_negative": lambda: f"def {name}(arr):\n    return any(x < 0 for x in arr)",
        "all_true": lambda: f"def {name}(arr):\n    return all(arr)",
        "any_true": lambda: f"def {name}(arr):\n    return any(arr)",
        "count_true": lambda: f"def {name}(arr):\n    return sum(1 for x in arr if x)",
        "count_false": lambda: f"def {name}(arr):\n    return sum(1 for x in arr if not x)",
        "first_element": lambda: f"def {name}(arr):\n    return arr[0] if arr else None",
        "last_element": lambda: f"def {name}(arr):\n    return arr[-1] if arr else None",
        "list_length": lambda: f"def {name}(arr):\n    return len(arr)",
        "is_empty_list": lambda: f"def {name}(arr):\n    return len(arr) == 0",
        "cumulative_sum": lambda: f"def {name}(arr):\n    result = []\n    s = 0\n    for x in arr: s += x; result.append(s)\n    return result",
        "merge_sorted": lambda: f"def {name}(a, b):\n    return sorted(a + b)",
        "merge_sorted_simple": lambda: f"def {name}(a, b):\n    return sorted(a + b)",
        "set_union": lambda: f"def {name}(a, b):\n    return list(set(a) | set(b))",
        "set_intersection": lambda: f"def {name}(a, b):\n    return list(set(a) & set(b))",
        "set_difference": lambda: f"def {name}(a, b):\n    return list(set(a) - set(b))",
        "set_symmetric_difference": lambda: f"def {name}(a, b):\n    return list(set(a) ^ set(b))",
        "concat_lists": lambda: f"def {name}(a, b):\n    return a + b",
        "max_of_two": lambda: f"def {name}(a, b):\n    return max(a, b)",
        "min_of_two": lambda: f"def {name}(a, b):\n    return min(a, b)",
        "max_of_three": lambda: f"def {name}(a, b, c):\n    return max(a, b, c)",
        "min_of_three": lambda: f"def {name}(a, b, c):\n    return min(a, b, c)",
        "count_occurrences": lambda: f"def {name}(arr, value):\n    return arr.count(value)",
        "first_index": lambda: f"def {name}(arr, value):\n    return arr.index(value) if value in arr else -1",
        "last_index": lambda: f"def {name}(arr, value):\n    for i in range(len(arr)-1, -1, -1):\n        if arr[i] == value: return i\n    return -1",
        "mean_list": lambda: f"def {name}(arr):\n    return sum(arr) / len(arr) if arr else 0",
        "median_list": lambda: f"def {name}(arr):\n    s = sorted(arr)\n    n = len(s)\n    if n == 0: return 0\n    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2",
        "mode_list": lambda: f"def {name}(arr):\n    from collections import Counter\n    return Counter(arr).most_common(1)[0][0] if arr else None",
        "range_list": lambda: f"def {name}(arr):\n    return max(arr) - min(arr) if arr else 0",
        "variance_list": lambda: f"def {name}(arr):\n    if not arr: return 0\n    m = sum(arr) / len(arr)\n    return sum((x-m)**2 for x in arr) / len(arr)",
        "std_dev_list": lambda: f"def {name}(arr):\n    if not arr: return 0\n    m = sum(arr) / len(arr)\n    return (sum((x-m)**2 for x in arr) / len(arr)) ** 0.5",
        "argmin": lambda: f"def {name}(arr):\n    return arr.index(min(arr)) if arr else -1",
        "argmax": lambda: f"def {name}(arr):\n    return arr.index(max(arr)) if arr else -1",
        "second_smallest": lambda: f"def {name}(arr):\n    u = sorted(set(arr))\n    return u[1] if len(u) >= 2 else None",
        "second_largest": lambda: f"def {name}(arr):\n    u = sorted(set(arr), reverse=True)\n    return u[1] if len(u) >= 2 else None",
        "unique_sorted": lambda: f"def {name}(arr):\n    return sorted(set(arr))",
        "unique_count": lambda: f"def {name}(arr):\n    return len(set(arr))",
        "most_frequent": lambda: f"def {name}(arr):\n    from collections import Counter\n    return Counter(arr).most_common(1)[0][0] if arr else None",
        "least_frequent": lambda: f"def {name}(arr):\n    from collections import Counter\n    return Counter(arr).most_common()[-1][0] if arr else None",
        "frequency_map": lambda: f"def {name}(arr):\n    from collections import Counter\n    return dict(Counter(arr))",
        "to_binary": lambda: f"def {name}(n):\n    return bin(n)[2:]",
        "to_octal": lambda: f"def {name}(n):\n    return oct(n)[2:]",
        "to_hex": lambda: f"def {name}(n):\n    return hex(n)[2:]",
        "from_binary": lambda: f"def {name}(s):\n    return int(s, 2)",
        "from_hex": lambda: f"def {name}(s):\n    return int(s, 16)",
        "circle_area": lambda: f"def {name}(r):\n    import math\n    return math.pi * r * r",
        "circle_circumference": lambda: f"def {name}(r):\n    import math\n    return 2 * math.pi * r",
        "triangle_area": lambda: f"def {name}(base, height):\n    return 0.5 * base * height",
        "rectangle_area": lambda: f"def {name}(w, h):\n    return w * h",
        "rectangle_perimeter": lambda: f"def {name}(w, h):\n    return 2 * (w + h)",
        "cube_volume": lambda: f"def {name}(side):\n    return side ** 3",
        "sphere_volume": lambda: f"def {name}(r):\n    import math\n    return (4/3) * math.pi * r**3",
        "pythagorean": lambda: f"def {name}(a, b):\n    return (a*a + b*b) ** 0.5",
        "distance": lambda: f"def {name}(x1, y1, x2, y2):\n    return ((x2-x1)**2 + (y2-y1)**2) ** 0.5",
        "nCr": lambda: f"def {name}(n, r):\n    import math\n    return math.comb(n, r)",
        "nPr": lambda: f"def {name}(n, r):\n    import math\n    return math.perm(n, r)",
        "catalan": lambda: f"def {name}(n):\n    import math\n    return math.comb(2*n, n) // (n+1)",
        "to_celsius": lambda: f"def {name}(f):\n    return (f - 32) * 5/9",
        "to_fahrenheit": lambda: f"def {name}(c):\n    return c * 9/5 + 32",
        "miles_to_km": lambda: f"def {name}(miles):\n    return miles * 1.60934",
        "kg_to_lbs": lambda: f"def {name}(kg):\n    return kg * 2.20462",
        "greet": lambda: f"def {name}(name):\n    return 'Hello, ' + name + '!'",
        "farewell": lambda: f"def {name}(name):\n    return 'Goodbye, ' + name + '!'",
        "swap": lambda: f"def {name}(a, b):\n    return (b, a)",
        "invert_bool": lambda: f"def {name}(b):\n    return not b",
        "clamp": lambda: f"def {name}(n, low, high):\n    return max(low, min(n, high))",
        "in_range": lambda: f"def {name}(n, low, high):\n    return low <= n <= high",
        "between": lambda: f"def {name}(x, low, high):\n    return low <= x <= high",
        "max_subarray": lambda: f"def {name}(arr):\n    if not arr: return 0\n    best = cur = arr[0]\n    for x in arr[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
        "climb_stairs": lambda: f"def {name}(n):\n    a, b = 1, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "valid_parens": lambda: "def " + name + "(s):\n    stack = []\n    for c in s:\n        if c in '([{': stack.append(c)\n        elif not stack: return False\n        elif c == ')' and stack[-1] != '(': return False\n        elif c == ']' and stack[-1] != '[': return False\n        elif c == '}' and stack[-1] != '{': return False\n        else: stack.pop()\n    return not stack",
        "pascal_row": lambda: f"def {name}(n):\n    row = [1]\n    for _ in range(n):\n        row = [1] + [row[i]+row[i+1] for i in range(len(row)-1)] + [1]\n    return row",
        "two_sum": lambda: "def " + name + "(arr, target):\n    seen = {}\n    for i, v in enumerate(arr):\n        if target - v in seen: return [seen[target-v], i]\n        seen[v] = i\n    return []",
        "move_zeros": lambda: f"def {name}(arr):\n    return [x for x in arr if x != 0] + [x for x in arr if x == 0]",
        "contains_duplicate": lambda: f"def {name}(arr):\n    return len(arr) != len(set(arr))",
        "missing_number": lambda: f"def {name}(arr):\n    n = len(arr)\n    return n * (n + 1) // 2 - sum(arr)",
        "first_unique": lambda: f"def {name}(s):\n    from collections import Counter\n    c = Counter(s)\n    for i, ch in enumerate(s):\n        if c[ch] == 1: return i\n    return -1",
        "is_palindrome_list": lambda: f"def {name}(arr):\n    return arr == arr[::-1]",
        "mirror_list": lambda: f"def {name}(arr):\n    return arr + arr[::-1]",
        "chunk_list": lambda: f"def {name}(arr, n):\n    return [arr[i:i+n] for i in range(0, len(arr), n)]",
        "zip_lists": lambda: f"def {name}(a, b):\n    return list(zip(a, b))",
        "interleave": lambda: f"def {name}(a, b):\n    result = []\n    for i in range(max(len(a), len(b))):\n        if i < len(a): result.append(a[i])\n        if i < len(b): result.append(b[i])\n    return result",
        "pairwise_sum": lambda: f"def {name}(a, b):\n    return [x + y for x, y in zip(a, b)]",
        "pairwise_product": lambda: f"def {name}(a, b):\n    return [x * y for x, y in zip(a, b)]",
        "scale_list": lambda: f"def {name}(arr, factor):\n    return [x * factor for x in arr]",
        "offset_list": lambda: f"def {name}(arr, offset):\n    return [x + offset for x in arr]",
        "filter_positive": lambda: f"def {name}(arr):\n    return [x for x in arr if x > 0]",
        "filter_negative": lambda: f"def {name}(arr):\n    return [x for x in arr if x < 0]",
        "filter_even": lambda: f"def {name}(arr):\n    return [x for x in arr if x % 2 == 0]",
        "filter_odd": lambda: f"def {name}(arr):\n    return [x for x in arr if x % 2 != 0]",
        "diff_list": lambda: f"def {name}(arr):\n    return [arr[i+1] - arr[i] for i in range(len(arr)-1)]",
        "split_csv": lambda: f"def {name}(s):\n    return s.split(',')",
        "join_words": lambda: f"def {name}(words, sep):\n    return sep.join(words)",
        "split_lines": lambda: f"def {name}(s):\n    return s.split('\\n')",
        "count_substring": lambda: f"def {name}(s, sub):\n    return s.count(sub)",
        "replace_vowels": lambda: f"def {name}(s, c):\n    return ''.join(c if ch.lower() in 'aeiou' else ch for ch in s)",
        "replace_char": lambda: f"def {name}(s, old, new):\n    return s.replace(old, new)",
        "count_upper": lambda: f"def {name}(s):\n    return sum(1 for c in s if c.isupper())",
        "count_lower": lambda: f"def {name}(s):\n    return sum(1 for c in s if c.islower())",
        "count_digits_str": lambda: f"def {name}(s):\n    return sum(1 for c in s if c.isdigit())",
        "count_special": lambda: f"def {name}(s):\n    import string\n    return sum(1 for c in s if c in string.punctuation)",
    }

    if name in simple_refs:
        ref = simple_refs[name]()
        if ref:
            return ref

    # ── Pattern-based matches ────────────────────────────────────
    if n.startswith("is_"):
        if "even" in n: return f"def {name}(n):\n    return n % 2 == 0"
        if "odd" in n: return f"def {name}(n):\n    return n % 2 != 0"
        if "positive" in n: return f"def {name}(n):\n    return n > 0"
        if "negative" in n: return f"def {name}(n):\n    return n < 0"
        if "zero" in n: return f"def {name}(n):\n    return n == 0"
        if "empty" in n: return f"def {name}(s):\n    return len(s) == 0"
        if "prime" in n: return f"def {name}(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"
        if "palindrome" in n: return f"def {name}(s):\n    s = str(s)\n    return s == s[::-1]"
        if "sorted" in n: return f"def {name}(arr):\n    return arr == sorted(arr)"
        if "unique" in n: return f"def {name}(arr):\n    return len(arr) == len(set(arr))"
        if "power_of_two" in n: return f"def {name}(n):\n    return n > 0 and (n & (n-1)) == 0"
        if "anagram" in n: return f"def {name}(s1, s2):\n    return sorted(s1) == sorted(s2)"
        if "upper" in n: return f"def {name}(s):\n    return s.isupper()"
        if "lower" in n: return f"def {name}(s):\n    return s.islower()"
        if "digit" in n: return f"def {name}(s):\n    return s.isdigit()"
        if "alpha" in n: return f"def {name}(s):\n    return s.isalpha()"
        if "perfect_square" in n: return f"def {name}(n):\n    import math\n    r = math.isqrt(n)\n    return r * r == n"
        if "perfect_cube" in n: return f"def {name}(n):\n    r = round(n ** (1/3))\n    return r * r * r == n"
        if "leap" in n: return f"def {name}(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"
        if "coprime" in n: return f"def {name}(a, b):\n    import math\n    return math.gcd(a, b) == 1"
        if "armstrong" in n or "narcissistic" in n:
            return f"def {name}(n):\n    s = str(n)\n    return sum(int(d)**len(s) for d in s) == n"
        if "triangular" in n:
            return f"def {name}(n):\n    import math\n    r = int((math.sqrt(8*n+1)-1)/2)\n    return r*(r+1)//2 == n"
        if "fibonacci" in n:
            return f"def {name}(n):\n    a, b = 0, 1\n    while a < n: a, b = b, a+b\n    return a == n"
        if "subset" in n: return f"def {name}(a, b):\n    return set(a) <= set(b)"
        if "disjoint" in n: return f"def {name}(a, b):\n    return set(a).isdisjoint(set(b))"
        if "equal" in n: return f"def {name}(a, b):\n    return a == b"
        if "greater" in n: return f"def {name}(a, b):\n    return a > b"
        if "less" in n: return f"def {name}(a, b):\n    return a < b"

    if "count" in n and nargs >= 1:
        if "vowel" in n: return f"def {name}(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"
        if "consonant" in n: return f"def {name}(s):\n    return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')"
        if "word" in n: return f"def {name}(s):\n    return len(s.split())"
        if "upper" in n: return f"def {name}(s):\n    return sum(1 for c in s if c.isupper())"
        if "lower" in n: return f"def {name}(s):\n    return sum(1 for c in s if c.islower())"
        if "digit" in n: return f"def {name}(s):\n    return sum(1 for c in s if c.isdigit())"
        if "line" in n: return f"def {name}(s):\n    return s.count('\\n') + 1 if s else 0"

    if "reverse" in n:
        if "word" in n: return f"def {name}(s):\n    return ' '.join(s.split()[::-1])"
        return f"def {name}(s):\n    return s[::-1]"

    if "remove" in n:
        if "space" in n: return f"def {name}(s):\n    return s.replace(' ', '')"
        if "vowel" in n: return f"def {name}(s):\n    return ''.join(c for c in s if c.lower() not in 'aeiou')"
        if "digit" in n: return f"def {name}(s):\n    return ''.join(c for c in s if not c.isdigit())"
        if "punct" in n: return f"def {name}(s):\n    import string\n    return ''.join(c for c in s if c not in string.punctuation)"
        if "duplicate" in n: return f"def {name}(arr):\n    seen = set()\n    return [x for x in arr if not (x in seen or seen.add(x))]"

    if "sort" in n:
        if "desc" in n: return f"def {name}(arr):\n    return sorted(arr, reverse=True)"
        return f"def {name}(arr):\n    return sorted(arr)"

    if "max" in n and nargs <= 2:
        if nargs == 1: return f"def {name}(arr):\n    return max(arr) if arr else None"
        return f"def {name}(a, b):\n    return max(a, b)"
    if "min" in n and nargs <= 2:
        if nargs == 1: return f"def {name}(arr):\n    return min(arr) if arr else None"
        return f"def {name}(a, b):\n    return min(a, b)"

    if "sum" in n and nargs == 1:
        return f"def {name}(arr):\n    return sum(arr)"
    if "product" in n and nargs == 1:
        return f"def {name}(arr):\n    import math\n    return math.prod(arr) if arr else 1"

    # Default: None (can't generate)
    return None


def gen_tests(name, args, ref_code):
    """Run reference code on edge cases to generate test cases."""
    arg_names = [a.strip() for a in args.split(",") if a.strip()]
    n_args = len(arg_names)

    # Edge case inputs
    if n_args == 0:
        inputs = [()]
    elif n_args == 1:
        inputs = [(0,), (1,), (-1,), (5,), ("hello",), ("",), ([1,2,3],), ([],)]
    elif n_args == 2:
        inputs = [(0,0), (1,1), (2,3), (5,2), ("a","b"), ("hello","world"), ([1,2],[3,4]), ([],[])]
    elif n_args == 3:
        inputs = [(1,2,3), (0,0,0), (5,3,1), (10,2,8)]
    else:
        inputs = [(1,2,3,4), (0,0,0,0)]

    tests = []
    ns = {}
    try:
        exec(ref_code, ns)
    except Exception:
        return []

    func = ns.get(name)
    if not func:
        return []

    for inp in inputs:
        try:
            result = func(*inp)
            if isinstance(result, (int, float, str, bool, list, tuple, type(None))):
                # For floats, mark as approx
                test = {"args": inp, "expected": result}
                if isinstance(result, float):
                    test["approx"] = True
                tests.append(test)
        except Exception:
            pass

    return tests[:5]


def main():
    prompts = build_topic_prompts()
    all_funcs = {}
    for topic, plist in prompts.items():
        for p in plist:
            m = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', p)
            if m:
                name = m.group(1)
                args = m.group(2).strip()
                if name not in all_funcs and name not in EXISTING_TESTS:
                    all_funcs[name] = args

    print(f"Functions needing tests: {len(all_funcs)}")

    new_tests = {}
    skipped = []
    for name, args in all_funcs.items():
        ref = make_ref(name, args, "")
        if ref:
            tests = gen_tests(name, args, ref)
            if tests:
                new_tests[name] = tests
            else:
                skipped.append(name)
        else:
            skipped.append(name)

    print(f"Generated: {len(new_tests)} | Skipped: {len(skipped)}")

    # Write
    with open("research/prompt_tests_auto.py", "w") as f:
        f.write('"""Auto-generated test cases.\n\n')
        f.write(f'{len(new_tests)} functions, {sum(len(v) for v in new_tests.values())} test cases.\n')
        f.write('"""\n\nTESTS_AUTO = {\n')
        for name, tests in sorted(new_tests.items()):
            f.write(f'    {name!r}: [\n')
            for t in tests:
                f.write(f'        {t!r},\n')
            f.write('    ],\n')
        f.write('}\n')

    total = len(EXISTING_TESTS) + len(new_tests)
    print(f"Total coverage: {total} functions (was {len(EXISTING_TESTS)}, +{len(new_tests)})")


if __name__ == "__main__":
    main()
