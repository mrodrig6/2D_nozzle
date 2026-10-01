"""The documentation's maths has to survive GitHub's Markdown, not just LaTeX.

Every equation in ``docs/theory.md`` was valid LaTeX *and* accepted by KaTeX in
strict mode while still displaying wrongly, so syntax checking is not the test
that matters.  Two failure modes bit:

* ``$$ ... $$`` blocks containing ``\\\\`` row separators -- matrices,
  ``aligned``, ``cases``.  Markdown processes the block's content before the
  maths renderer sees it, and ``\\\\`` is a Markdown escape, so the rows
  collapse.  A ```` ```math ```` fence is taken verbatim, which is why GitHub
  documents it.
* ``\\boldsymbol{\\mathcal{F}}``.  KaTeX accepts it, finds no bold calligraphic
  glyph, and silently renders it unbold -- indistinguishable from an ordinary
  calligraphic letter, so a reader sees the wrong notation rather than an error.

Neither is detectable by rendering one equation in isolation, which is what makes
them worth a test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = sorted(
    [Path("README.md"), *Path("docs").glob("*.md"), Path("examples/README.md")]
)
ROOT = Path(__file__).resolve().parents[1]


def spans(path: Path):
    """Yield ``(line, is_display, source)`` for every maths span in a document."""
    lines = (ROOT / path).read_text().split("\n")
    in_code = in_math = False
    start, buf = 0, []
    for i, line in enumerate(lines, 1):
        if re.match(r"\s*```math\s*$", line) and not in_code and not in_math:
            in_math, start, buf = True, i, []
            continue
        if re.match(r"\s*```", line):
            if in_math:
                in_math = False
                yield start, True, "\n".join(buf)
            else:
                in_code = not in_code
            continue
        if in_math:
            buf.append(line)
            continue
        if in_code:
            continue
        stripped = re.sub(r"`[^`]*`", lambda m: "\0" * len(m.group(0)), line)
        for m in re.finditer(r"\$([^$]+)\$", stripped):
            yield i, False, m.group(1)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p))
def test_display_maths_uses_a_fence_not_dollar_dollar(path):
    r"""``$$`` blocks let Markdown eat ``\\``; a ```math fence does not."""
    lines = (ROOT / path).read_text().split("\n")
    in_code = False
    bad = []
    for i, line in enumerate(lines, 1):
        if re.match(r"\s*```", line):
            in_code = not in_code
            continue
        if not in_code and line.strip() == "$$":
            bad.append(i)
    assert not bad, f"{path} uses $$ display maths at lines {bad}; use a ```math fence"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p))
def test_no_bold_calligraphic(path):
    """KaTeX renders ``\\boldsymbol{\\mathcal{X}}`` unbold, with no error."""
    for line, _, src in spans(path):
        assert "\\boldsymbol{\\mathcal" not in src, (
            f"{path}:{line} uses \\boldsymbol{{\\mathcal{{...}}}}, which KaTeX "
            "renders unbold -- write it with \\mathbf instead"
        )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p))
def test_inline_maths_is_well_formed(path):
    r"""GitHub's inline maths needs tight delimiters, one line, and no ``\\``."""
    for line, display, src in spans(path):
        if display:
            continue
        assert src == src.strip(), f"{path}:{line} inline maths padded: ${src}$"
        assert "\n" not in src, f"{path}:{line} inline maths spans lines"
        assert "\\\\" not in src, f"{path}:{line} inline maths contains a line break"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p))
def test_every_link_and_anchor_resolves(path):
    """A dead cross-reference is the other thing readers notice immediately."""
    text = (ROOT / path).read_text()
    anchors = set()
    for line in text.split("\n"):
        m = re.match(r"^#+\s+(.*)$", line)
        if m:
            slug = re.sub(r"[`*_]", "", m.group(1).lower())
            slug = re.sub(r"[^\w\s-]", "", slug)
            anchors.add(re.sub(r"\s+", "-", slug.strip()))
    dead = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in anchors]
    assert not dead, f"{path} links to missing anchors {dead}"

    missing = [
        target
        for target in re.findall(r"\]\((?!#|https?:|mailto:)([^)#]+)", text)
        if not (ROOT / path).parent.joinpath(target).exists()
    ]
    assert not missing, f"{path} links to missing files {missing}"
