#!/usr/bin/env python3
"""Patch Cython's Python 3.13 traceback shim in generated core.cpp."""

from __future__ import annotations

import sys
from pathlib import Path


TRACEBACK_SIGNATURE = (
    "static void __Pyx_AddTraceback(const char *funcname, int c_line,\n"
    "                               int py_line, const char *filename) {\n"
)

TRACEBACK_GUARD = """#if PY_VERSION_HEX >= 0x030D0000
    (void)funcname;
    (void)c_line;
    (void)py_line;
    (void)filename;
    return;
#endif
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: PatchCythonTraceback.py <generated-core.cpp>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    text = path.read_text()
    if TRACEBACK_GUARD in text:
        return 0
    if TRACEBACK_SIGNATURE not in text:
        print(f"cannot find Cython traceback helper in {path}", file=sys.stderr)
        return 1

    text = text.replace(TRACEBACK_SIGNATURE, TRACEBACK_SIGNATURE + TRACEBACK_GUARD)
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
