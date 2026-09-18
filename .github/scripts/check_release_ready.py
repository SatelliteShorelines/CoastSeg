#!/usr/bin/env python3
"""Pre-release readiness checks for the CoastSeg repository.

Run before tagging a release (CI does this automatically in the
"Pre-release: build & install matrix" workflow):

    python .github/scripts/check_release_ready.py --python-versions 3.10 3.11 3.12

The checks catch the mistakes that are invisible locally but break a release:

1. Machine-local dependencies. Development sometimes points a dependency at a
   local checkout, e.g.
       coastsat-package = {path = "E:/3_development/.../coastsat_package"}
   That path exists on exactly one machine and is recorded verbatim in
   pixi.lock, so every other user's `pixi install` fails.
2. Absolute paths recorded in pixi.lock (the downstream symptom of #1).
3. Direct-reference requirements in [project.dependencies]. PyPI rejects any
   distribution whose metadata contains a `pkg @ git+...` or `pkg @ file://`
   requirement, but only *after* the release tag has been pushed.
4. requires-python, the trove classifiers, and the CI matrix disagreeing about
   which Python versions are supported.
5. On a tag build: the tag not matching `project.version`.

Stdlib only (tomllib, so Python 3.11+) -- this must run before anything is
installed. Exits non-zero with a report of every failure, not just the first.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# `coastseg = {path = "."}` is the workspace referring to itself -- always fine.
SELF_REFERENCE = "coastseg"

# Keys in [tool.pixi.pypi-dependencies] that make an environment
# non-reproducible for anyone other than the person who wrote them.
NON_REPRODUCIBLE_KEYS = ("path", "git", "url")

# A pixi.lock `- pypi: <source>` entry is fine when it is a URL or a path
# relative to the workspace root. Anything rooted at a drive letter, a POSIX
# absolute path, or a home directory came from a local checkout.
ABSOLUTE_SOURCE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|~|file://)")


class Report:
    """Collects pass/fail lines so every check runs before we exit."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, message: str) -> None:
        print(f"  PASS  {message}")

    def fail(self, message: str, detail: str = "") -> None:
        print(f"  FAIL  {message}")
        if detail:
            for line in detail.splitlines():
                print(f"        {line}")
        self.failures.append(message)


def find_line(text: str, key: str) -> int:
    """Return the 1-based line number where a TOML key is assigned, or 0."""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", re.MULTILINE)
    match = pattern.search(text)
    return text.count("\n", 0, match.start()) + 1 if match else 0


def parse_requires_python(spec: str) -> set[str]:
    """Expand a `>=3.10,<3.13` style specifier into {"3.10", "3.11", "3.12"}.

    Only the major.minor bounds that CoastSeg actually uses are supported; an
    unrecognised specifier raises so the check fails loudly rather than
    silently comparing against an empty set.
    """
    lower = upper = None
    upper_inclusive = False

    for clause in (c.strip() for c in spec.split(",")):
        match = re.fullmatch(r"(>=|>|<=|<)\s*3\.(\d+)(?:\.\*)?", clause)
        if not match:
            raise ValueError(f"unsupported requires-python clause: {clause!r}")
        operator, minor = match.group(1), int(match.group(2))
        if operator == ">=":
            lower = minor
        elif operator == ">":
            lower = minor + 1
        elif operator == "<":
            upper = minor
        else:  # "<="
            upper = minor
            upper_inclusive = True

    if lower is None or upper is None:
        raise ValueError(f"requires-python must have both bounds, got {spec!r}")
    if upper_inclusive:
        # `<=3.11` excludes 3.11.1 and is almost never what was meant, but
        # expand it the way pip reads it so the report matches reality.
        upper += 1
    return {f"3.{minor}" for minor in range(lower, upper)}


def check_pixi_pypi_dependencies(config: dict, text: str, report: Report) -> None:
    deps = config.get("tool", {}).get("pixi", {}).get("pypi-dependencies", {})
    offenders = []

    for name, spec in deps.items():
        if name == SELF_REFERENCE or not isinstance(spec, dict):
            continue
        for key in NON_REPRODUCIBLE_KEYS:
            if key in spec:
                line = find_line(text, name)
                where = f"pyproject.toml:{line}" if line else "pyproject.toml"
                offenders.append(f"{where}  {name} = {{{key} = {spec[key]!r}}}")

    if offenders:
        report.fail(
            "[tool.pixi.pypi-dependencies] has no machine-local dependencies",
            "\n".join(offenders)
            + "\n\nRemove the local override and restore the published"
            "\ndependency, then re-lock:  git checkout pixi.lock",
        )
    else:
        report.ok("[tool.pixi.pypi-dependencies] has no machine-local dependencies")


def check_pixi_lock(lock_path: Path, report: Report) -> None:
    if not lock_path.exists():
        report.fail(f"{lock_path.name} exists")
        return

    offenders = []
    for number, line in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^\s*-\s*(?:pypi|conda):\s*(\S+)", line)
        if match and ABSOLUTE_SOURCE.match(match.group(1)):
            offenders.append(f"{lock_path.name}:{number}  {match.group(1)}")

    if offenders:
        shown = offenders[:5]
        extra = len(offenders) - len(shown)
        detail = "\n".join(shown)
        if extra:
            detail += f"\n... and {extra} more"
        report.fail(
            f"{lock_path.name} records no absolute paths",
            detail + "\n\nThese resolve only on the machine that wrote them.",
        )
    else:
        report.ok(f"{lock_path.name} records no absolute paths")


def check_direct_references(config: dict, report: Report) -> None:
    """PyPI rejects metadata containing PEP 508 direct references."""
    offenders = [
        dep for dep in config.get("project", {}).get("dependencies", []) if "@" in dep
    ]
    if offenders:
        report.fail(
            "[project.dependencies] has no direct-reference requirements",
            "\n".join(offenders) + "\n\nPyPI rejects uploads whose metadata contains these.",
        )
    else:
        report.ok("[project.dependencies] has no direct-reference requirements")


def check_python_versions(config: dict, expected: list[str], report: Report) -> None:
    project = config.get("project", {})
    spec = project.get("requires-python", "")

    try:
        supported = parse_requires_python(spec)
    except ValueError as error:
        report.fail(f"requires-python is parseable ({spec!r})", str(error))
        return

    classifiers = {
        match.group(1)
        for match in (
            re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", classifier)
            for classifier in project.get("classifiers", [])
        )
        if match
    }

    def describe(versions: set[str]) -> str:
        return ", ".join(sorted(versions, key=lambda v: int(v.split(".")[1]))) or "(none)"

    if supported == classifiers:
        report.ok(f"requires-python {spec!r} matches the classifiers ({describe(supported)})")
    else:
        report.fail(
            f"requires-python {spec!r} matches the classifiers",
            f"requires-python implies: {describe(supported)}\n"
            f"classifiers declare:    {describe(classifiers)}",
        )

    wanted = set(expected)
    if supported == wanted:
        report.ok(f"CI matrix covers every supported version ({describe(wanted)})")
    else:
        report.fail(
            "CI matrix covers every supported version",
            f"requires-python implies: {describe(supported)}\n"
            f"matrix tests:           {describe(wanted)}\n\n"
            "Update the matrix in .github/workflows/release_env_matrix.yml,\n"
            "or update requires-python.",
        )


def check_tag_matches_version(config: dict, tag: str, report: Report) -> None:
    version = config.get("project", {}).get("version", "")
    expected = tag[1:] if tag.startswith("v") else tag

    if version == expected:
        report.ok(f"tag {tag} matches project.version {version}")
    else:
        report.fail(
            f"tag {tag} matches project.version",
            f"tag implies version: {expected}\n"
            f"pyproject.toml says: {version}\n\n"
            "Bump version in pyproject.toml, or delete and re-push the tag.",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-versions",
        nargs="+",
        default=["3.10", "3.11", "3.12"],
        metavar="X.Y",
        help="Python versions the CI matrix tests; must match requires-python.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Release tag being built (e.g. v2.0.4). When set, it must match project.version.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the checkout containing this script).",
    )
    args = parser.parse_args()

    pyproject_path = args.repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"error: {pyproject_path} not found", file=sys.stderr)
        return 2

    text = pyproject_path.read_text(encoding="utf-8")
    config = tomllib.loads(text)

    print(f"Release readiness checks for {args.repo_root}\n")
    report = Report()

    check_pixi_pypi_dependencies(config, text, report)
    check_pixi_lock(args.repo_root / "pixi.lock", report)
    check_direct_references(config, report)
    check_python_versions(config, args.python_versions, report)
    if args.tag:
        check_tag_matches_version(config, args.tag, report)
    else:
        print("  SKIP  tag/version agreement (not a tag build)")

    if report.failures:
        print(f"\n{len(report.failures)} check(s) failed -- this tree is not ready to release.")
        return 1

    print("\nAll release readiness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
