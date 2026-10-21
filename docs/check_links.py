"""Validate repository-relative links, GitHub anchors, and image descriptions.

Run from the repository root: python docs/check_links.py
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    """Ignore fenced code so literal examples are not interpreted as links."""
    return re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)


def anchors(text):
    found, counts = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", prose(text), re.M):
        heading = re.sub(r"[`*_]", "", heading.lower())
        heading = re.sub(r"[^\w\s-]", "", heading)
        key = re.sub(r"\s", "-", heading.strip())
        count = counts.get(key, 0)
        counts[key] = count + 1
        found.add(key + ("-" + str(count) if count else ""))
    found.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    return found


def main():
    paths = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    paths += list((ROOT / "examples").glob("*.md"))
    paths += list((ROOT / "legacy").glob("*.md"))
    paths += list((ROOT / "src").rglob("*.md"))
    errors = []
    for path in sorted(paths):
        text = prose(path.read_text(encoding="utf-8"))
        for image, label, href in re.findall(r"(!?)\[([^\]]*)\]\(([^\s)]+)\)", text):
            parts = urlsplit(href)
            if image and not label.strip():
                errors.append(f"{path.relative_to(ROOT)}: image needs descriptive alt text")
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing file {href}")
            elif parts.fragment and target.suffix == ".md":
                if unquote(parts.fragment) not in anchors(target.read_text(encoding="utf-8")):
                    errors.append(f"{path.relative_to(ROOT)}: missing anchor {href}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(paths)} Markdown files: links, anchors and image alt text passed")


if __name__ == "__main__":
    main()
