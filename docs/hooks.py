"""Render the README as Home and preserve GitHub-relative documentation links."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from markdown.extensions.toc import slugify

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def on_page_read_source(page, config):
    if page.file.src_uri == "index.md":
        return (ROOT / "README.md").read_text(encoding="utf-8")
    return None


def on_page_markdown(markdown, page, config, files):
    source = ROOT / "README.md" if page.file.src_uri == "index.md" else DOCS / page.file.src_uri
    destination = DOCS / page.file.src_uri

    def rewrite(match):
        label, href = match.groups()
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        target = (source.parent / parts.path).resolve()
        suffix = "?" + parts.query if parts.query else ""
        suffix += "#" + parts.fragment if parts.fragment else ""
        if target == DOCS or target == DOCS / "README.md":
            target = DOCS / "index.md"
        if target.is_relative_to(DOCS):
            import os

            path = os.path.relpath(target, destination.parent)
        elif target.is_relative_to(ROOT):
            kind = "tree" if target.is_dir() else "blob"
            path = config.repo_url.rstrip("/") + "/" + kind + "/main/"
            path += target.relative_to(ROOT).as_posix()
        else:
            return match.group(0)
        return "[" + label + "](" + path + suffix + ")"

    # Do not rewrite strings or Markdown examples inside fenced code blocks.
    lines, fence, math = [], False, False
    for line in markdown.splitlines():
        if re.match(r"^```math\s*$", line) and not fence:
            math = True
            lines.append("$$")
        elif line.startswith("```"):
            if math:
                math = False
                lines.append("$$")
            else:
                fence = not fence
                lines.append(line)
        elif fence or math:
            lines.append(line)
        else:
            heading = re.match(r"^#{1,6} (.+)$", line)
            if heading:
                clean = re.sub(r"[`*_]", "", heading[1].lower())
                github = re.sub(r"[^\w\s-]", "", clean)
                github = re.sub(r"\s", "-", github.strip())
                if github != slugify(clean, "-"):
                    lines.append('<a id="' + github + '"></a>')
            lines.append(re.sub(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^\s)]+)\)", rewrite, line))
    return "\n".join(lines)
