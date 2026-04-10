"""Bump version in pyproject.toml and insert a new version header in CHANGELOG.md."""

import argparse
import re
from datetime import date
from pathlib import Path

TOP_DIR = Path(__file__).resolve().parents[1]


def bump_pyproject(version: str) -> None:
    """
    Updates the version in pyproject.toml to the specified version string.

    Args:
        version (str): The new version string to set in pyproject.toml.
    """
    pyproj_file = TOP_DIR / "pyproject.toml"
    text = pyproj_file.read_text()
    pyproj_file.write_text(
        re.sub(r'^version = ".*"', f'version = "{version}"', text, count=1, flags=re.M)
    )


def bump_changelog(version: str, release_date: str) -> None:
    """
    Inserts a new version header in CHANGELOG.md for the specified version and release date.

    Args:
        version (str): The new version string to add to the changelog.
        release_date (str): The release date in YYYY-MM-DD format to insert in the changelog header.
    """
    changelog_file = TOP_DIR / "CHANGELOG.md"
    text = changelog_file.read_text()

    new_section = (
        "## [Unreleased]\n"
        "\n"
        "## Added\n"
        "\n"
        "## Changed\n"
        "\n"
        "## Fixed\n"
        "\n"
        "\n"
        f"## [{version}] - {release_date}"
    )

    text = text.replace("## [Unreleased]", new_section, 1)
    changelog_file.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a release.")
    parser.add_argument("version", help="Version string, e.g. 3.5.0")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Release date (default: today, YYYY-MM-DD)",
    )
    args = parser.parse_args()

    bump_pyproject(args.version)
    bump_changelog(args.version, args.date)

    print(f"Bumped to v{args.version} ({args.date})")


if __name__ == "__main__":
    main()
