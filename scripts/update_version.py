#!/usr/bin/env python3
"""
Version management script for hill-climber package.

Usage:
    python scripts/update_version.py 0.2.0
    python scripts/update_version.py --check
"""

import argparse
import re
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory."""

    return Path(__file__).parent.parent


def get_version_from_init():
    """Read version from __init__.py."""

    init_file = get_project_root() / "hill_climber" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r"__version__ = ['\"]([^'\"]+)['\"]", content)

    if match:
        return match.group(1)

    raise ValueError("Could not find __version__ in __init__.py")


def get_version_from_pyproject():
    """Read version from pyproject.toml."""

    pyproject_file = get_project_root() / "pyproject.toml"
    content = pyproject_file.read_text()
    match = re.search(r'version = "([^"]+)"', content)

    if match:
        return match.group(1)

    raise ValueError("Could not find version in pyproject.toml")


def get_version_from_citation():
    """Read version from CITATION.cff."""

    citation_file = get_project_root() / "CITATION.cff"
    content = citation_file.read_text()
    match = re.search(r'^version: (.+)', content, re.MULTILINE)

    if match:
        return match.group(1).strip()

    raise ValueError("Could not find version in CITATION.cff")


def update_version_in_init(new_version):
    """Update version in __init__.py."""

    init_file = get_project_root() / "hill_climber" / "__init__.py"
    content = init_file.read_text()
    updated = re.sub(
        r"__version__ = ['\"]([^'\"]+)['\"]",
        f"__version__ = '{new_version}'",
        content
    )
    init_file.write_text(updated)
    print(f"✓ Updated __init__.py to version {new_version}")


def update_version_in_pyproject(new_version):
    """Update version in pyproject.toml."""

    pyproject_file = get_project_root() / "pyproject.toml"
    content = pyproject_file.read_text()
    updated = re.sub(
        r'version = "([^"]+)"',
        f'version = "{new_version}"',
        content
    )
    pyproject_file.write_text(updated)
    print(f"✓ Updated pyproject.toml to version {new_version}")


def update_version_in_citation(new_version):
    """Update version in CITATION.cff (not cff-version)."""

    citation_file = get_project_root() / "CITATION.cff"
    content = citation_file.read_text()
    updated = re.sub(
        r'^version: .+',
        f'version: {new_version}',
        content,
        flags=re.MULTILINE
    )
    citation_file.write_text(updated)
    print(f"✓ Updated CITATION.cff to version {new_version}")


def check_versions():
    """Check if versions are consistent."""

    try:
        init_version = get_version_from_init()
        pyproject_version = get_version_from_pyproject()
        citation_version = get_version_from_citation()
        
        print(f"__init__.py version:    {init_version}")
        print(f"pyproject.toml version: {pyproject_version}")
        print(f"CITATION.cff version:   {citation_version}")
        
        if init_version == pyproject_version == citation_version:
            print("\n✓ Versions are consistent!")
            return True

        else:
            print("\n✗ Versions are inconsistent!")
            return False

    except Exception as e:
        print(f"\n✗ Error checking versions: {e}")
        return False


def validate_version(version):
    """Validate semantic version format."""

    pattern = r'^\d+\.\d+\.\d+$'

    if not re.match(pattern, version):
        raise ValueError(
            f"Invalid version format: {version}\n"
            "Version must follow semantic versioning: MAJOR.MINOR.PATCH (e.g., 0.1.0)"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Update version numbers in hill-climber package"
    )

    parser.add_argument(
        "version",
        nargs="?",
        help="New version number (e.g., 0.2.0)"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if versions are consistent"
    )
    
    args = parser.parse_args()
    
    if args.check:
        success = check_versions()
        sys.exit(0 if success else 1)
    
    if not args.version:
        parser.print_help()
        sys.exit(1)
    
    try:
        validate_version(args.version)
        
        print(f"\nUpdating to version {args.version}...")
        update_version_in_init(args.version)
        update_version_in_pyproject(args.version)
        update_version_in_citation(args.version)
        
        print("\n✓ Version update complete!")
        print(f"\nNext steps:")
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git commit -am 'Bump version to {args.version}'")
        print(f"  3. Push: git push origin dev")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
