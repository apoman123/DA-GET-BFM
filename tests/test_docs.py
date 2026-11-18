"""Check that the documentation's relative links and anchors resolve."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "model" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")


def github_slug(heading: str) -> str:
    """Return GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"`|\*\*|\*", "", heading.strip()).lower()
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = "".join(character for character in text if character.isalnum() or character in " -_")
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    """Return the anchors of a Markdown file's headings (outside code blocks), with GitHub's duplicate suffixes."""
    seen: dict[str, int] = {}
    result = set()
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if fenced or not match:
            continue
        slug = github_slug(match.group(2))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def links(path: Path) -> list[str]:
    """Return the link targets of a Markdown file, outside code blocks and inline code."""
    targets = []
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        targets += LINK.findall(line)
    return targets


class DocumentationLinkTest(unittest.TestCase):
    """Every relative link points to an existing file, and every anchor to an existing heading."""

    def test_relative_links_and_anchors_resolve(self) -> None:
        """Broken links are listed with the document they are in."""
        broken = []
        for document in DOCUMENTS:
            for target in links(document):
                if re.match(r"^[a-z]+:", target):
                    continue  # external URL
                file_part, _, anchor = target.partition("#")
                destination = (document.parent / file_part).resolve() if file_part else document
                if not destination.exists():
                    broken.append(f"{document.relative_to(ROOT)}: {target} (missing file)")
                    continue
                if anchor and destination.suffix == ".md" and anchor not in anchors(destination):
                    broken.append(f"{document.relative_to(ROOT)}: {target} (missing anchor)")
        self.assertEqual(broken, [])

    def test_slugs_follow_github(self) -> None:
        """The slug function reproduces GitHub's anchors for headings used here."""
        self.assertEqual(github_slug("Why training uses no estimated torque"), "why-training-uses-no-estimated-torque")
        self.assertEqual(github_slug("Deployment: `DaGetBfmPolicy`"), "deployment-dagetbfmpolicy")
        self.assertEqual(github_slug("R-inv. Inverse response"), "r-inv-inverse-response")
        self.assertEqual(github_slug("Training (online off-policy FB-CPR)"), "training-online-off-policy-fb-cpr")


if __name__ == "__main__":
    unittest.main()
