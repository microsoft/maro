# Black: 22.3.0
# isort: 5.10.1

import os
import sys
from typing import List


def show_hint(msg: str) -> None:
    n = 80
    print()
    print("#" * n)
    left_count = (n - 2 - len(msg)) // 2
    right_count = n - 2 - len(msg) - left_count
    print("#" + " " * left_count + msg + " " * right_count + "#")
    print("#" * n)


def get_all_py_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = [os.path.join(dp, f) for dp, _, filenames in os.walk(path) for f in filenames]
    else:
        files = [path]

    return [f for f in files if f.endswith(".py")]


if __name__ == "__main__":
    paths = sys.argv[1:]
    files = []
    for path in paths:
        files += get_all_py_files(path)

    for r in range(2):
        show_hint(f"Black, round {r + 1}/2")
        for path in paths:
            os.system(f"black {path} -l 120")

        show_hint(f"Add trailing comma, round {r + 1}/2")
        for f in files:
            cmd = f"add-trailing-comma {f}"
            os.system(cmd)

    show_hint("isort")
    for path in paths:
        isort_cmd = f'isort {path} --indent "    " -l 120 --trailing-comma --use-parentheses --multi-line 6'
        os.system(isort_cmd)
