"""
diagnose_nccl.py — Find which copy of libnccl/libcublas the loader actually uses.

WHEN TO RUN THIS
    After fix_torch_cuda.py reports "Every pin matches" and `import torch`
    still dies with `undefined symbol: ncclCommResume`.

    Matching pins plus a missing symbol is not a contradiction, it is a
    diagnosis: the version pip records is not the version the dynamic loader
    resolves. Something earlier on the search path exports an older ABI, so
    libtorch_cuda.so binds against a libnccl that predates the symbol it
    needs.

    Three things put a second copy on the path, and all three are plausible
    here:

      * A LEFTOVER CUDA-12 WHEEL. Installing docling, onnxruntime-gpu or
        paddlepaddle pulls nvidia-*-cu12 alongside the cu13 set torch wants.
        Uninstalling the package does not always remove them, so both remain
        and load order decides which wins.
      * A SYSTEM NCCL. libnccl2 from apt, or a full CUDA toolkit install,
        landing in /usr/lib/x86_64-linux-gnu ahead of site-packages.
      * LD_LIBRARY_PATH, which beats everything -- including the RPATH torch
        sets on its own libraries.

    The symbol test is done with ctypes rather than `nm`, so it works without
    binutils installed.

Usage:
    python tools/diagnose_nccl.py
    python tools/diagnose_nccl.py --symbol ncclCommResume --lib nccl
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import site
import subprocess
import sys
from importlib import metadata


def _search_roots() -> list[str]:
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    roots.append(os.path.join(sys.prefix, "lib"))
    return [r for r in dict.fromkeys(roots) if os.path.isdir(r)]


def find_libraries(stem: str) -> list[str]:
    """Every copy of lib<stem>.so* under site-packages and on the ld cache."""
    found: list[str] = []
    for root in _search_roots():
        found += glob.glob(os.path.join(root, "**", f"lib{stem}.so*"),
                           recursive=True)

    try:
        cache = subprocess.run(["ldconfig", "-p"], capture_output=True,
                               text=True, timeout=10).stdout
        for line in cache.splitlines():
            if f"lib{stem}.so" in line and "=>" in line:
                found.append(line.split("=>")[-1].strip())
    except Exception:
        pass

    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry:
            found += glob.glob(os.path.join(entry, f"lib{stem}.so*"))

    return sorted(dict.fromkeys(os.path.realpath(f) for f in found
                                if os.path.exists(f)))


def exports(path: str, symbol: str) -> bool | None:
    """True/False, or None when the library cannot be loaded standalone."""
    try:
        handle = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    except OSError:
        return None
    try:
        getattr(handle, symbol)
        return True
    except AttributeError:
        return False


def duplicate_cuda_wheels() -> dict[str, list[str]]:
    """
    nvidia-* distributions whose SAME library ships for two CUDA majors.

    The suffix must be parsed, not stripped loosely: newer wheels drop it
    entirely (`nvidia-cublas`) while older ones keep it (`nvidia-cublas-cu12`).
    A naive rsplit("-cu") turns every unsuffixed name into the family "nvidia"
    and reports the whole CUDA stack as one giant conflict, which is worse
    than reporting nothing -- it sends you uninstalling wheels torch needs.
    """
    families: dict[str, set[str]] = {}
    versions: dict[str, list[str]] = {}

    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if not name.startswith("nvidia-"):
            continue
        match = re.match(r"^(nvidia-.+?)-cu(\d+)$", name)
        base, major = (match.group(1), match.group(2)) if match else (name, "-")
        families.setdefault(base, set()).add(major)
        versions.setdefault(base, []).append(f"{name}=={dist.version}")

    return {base: sorted(versions[base])
            for base, majors in families.items() if len(majors) > 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="nccl",
                        help="library stem, e.g. nccl or cublasLt")
    parser.add_argument("--symbol", default="ncclCommResume",
                        help="the symbol named in the ImportError")
    args = parser.parse_args()

    print("diagnose_nccl: running", flush=True)
    print(f"python          : {sys.executable}")
    print(f"LD_LIBRARY_PATH : {os.environ.get('LD_LIBRARY_PATH') or '(unset)'}")
    if os.environ.get("LD_LIBRARY_PATH"):
        print("                  ^ this OVERRIDES torch's own RPATH. If a "
              "libnccl lives\n                    in any of those "
              "directories, it wins regardless of pip.")

    dupes = duplicate_cuda_wheels()
    print(f"\nnvidia-* wheels installed for more than one CUDA major version:")
    if dupes:
        for family, versions in sorted(dupes.items()):
            print(f"  {family}: {', '.join(sorted(versions))}   <-- CONFLICT")
        print("  A cu12 wheel beside a cu13 one is the usual cause. Remove the "
              "set torch does\n  NOT want; keep the one fix_torch_cuda.py "
              "listed.")
    else:
        print("  none -- pip has only one CUDA generation installed")

    print(f"\ncopies of lib{args.lib}.so* visible to this interpreter:")
    libraries = find_libraries(args.lib)
    if not libraries:
        print(f"  none found. torch may dlopen it from its own bundled path; "
              f"check\n  {os.path.join(sys.prefix, 'lib')} manually.")
    for path in libraries:
        has = exports(path, args.symbol)
        verdict = {True: f"exports {args.symbol}",
                   False: f"MISSING {args.symbol}  <-- would break torch",
                   None: "could not load standalone (may still be fine)"}[has]
        # "in the environment" means inside a site-packages tree, not merely
        # under sys.prefix -- on a system interpreter those are the same
        # directory and the label would be meaningless.
        in_env = any(path.startswith(os.path.realpath(r)) for r in _search_roots())
        print(f"  [{'pkg' if in_env else 'SYS'}] {path}")
        print(f"         {verdict}")

    torch_lib = os.path.join(sys.prefix, "lib",
                             f"python{sys.version_info.major}."
                             f"{sys.version_info.minor}", "site-packages",
                             "torch", "lib", "libtorch_cuda.so")
    if os.path.exists(torch_lib):
        print(f"\nwhat the loader resolves for libtorch_cuda.so:")
        try:
            out = subprocess.run(["ldd", torch_lib], capture_output=True,
                                 text=True, timeout=20).stdout
            for line in out.splitlines():
                if args.lib in line or "not found" in line:
                    print(f"  {line.strip()}")
        except Exception as exc:
            print(f"  ldd unavailable ({exc})")

    print("\nWhat to do:")
    print("  * a SYS copy missing the symbol, listed before the venv copy ->")
    print("      unset LD_LIBRARY_PATH   (or drop that directory from it)")
    print("  * a cu12/cu13 conflict above ->")
    print("      pip uninstall -y <the cu12 packages>, then re-run "
          "fix_torch_cuda.py")
    print("  * nothing suspicious ->")
    print("      the wheel itself may be truncated: "
          "pip install --force-reinstall --no-cache-dir \\")
    print("        nvidia-nccl-cu13==<version fix_torch_cuda.py reported>")


if __name__ == "__main__":
    main()