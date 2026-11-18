"""Generate the flat requirements file consumed by ValueArena's worker build."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def render_requirements() -> str:
    with (ROOT / "pyproject.toml").open("rb") as manifest:
        project = tomllib.load(manifest)["project"]
    lines = [
        "# Generated from pyproject.toml by python scripts/generate_requirements.py.",
        "# Contains the runtime dependencies and the local extra for hosted workers.",
        *project["dependencies"],
        *project["optional-dependencies"]["local"],
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (ROOT / "requirements.txt").write_text(render_requirements(), encoding="utf-8", newline="\n")
