#!/usr/bin/env python3
"""Embed gpu/metal/kmeans.metal into a C++ header.

The Metal command line toolchain (xcrun metal) is part of full Xcode, which is
not installed on every development machine. Compiling the shader from an
embedded source string at run time needs only Metal.framework, so the build
works with the Command Line Tools alone. The .metal file stays the single
source of truth.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIMITER = "METAL_SRC"


def main() -> int:
    source = (HERE / "kmeans.metal").read_text()
    if DELIMITER in source:
        raise SystemExit(f"{DELIMITER} must not appear in kmeans.metal")
    header = (
        "#pragma once\n"
        "// Generated from gpu/metal/kmeans.metal by embed_source.py. Do not edit.\n"
        f'static const char* const kKmeansMetalSource = R"{DELIMITER}({source}){DELIMITER}";\n'
    )
    target = HERE / "kernels_source.h"
    target.write_text(header)
    print(f"wrote {target.relative_to(HERE.parent.parent)} ({len(source)} bytes of Metal source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
