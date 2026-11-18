"""Build the submission-ready DOCX report and PPTX slide deck.

Run from the repository root::

    python -m pip install -e ".[docs]"
    python scripts/build_documents.py

Outputs:

* ``report/report.docx``
* ``slides/seam_carving_slides.pptx``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptxColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PInches
from pptx.util import Pt as PPt

REPORT_MD = ROOT / "report" / "report.md"
REPORT_DOCX = ROOT / "report" / "report.docx"
REPORT_DIR = REPORT_MD.parent
FIGURES = REPORT_DIR / "figures"
ASSETS = ROOT / "assets"
SLIDES = ROOT / "slides"
SLIDES_PPTX = SLIDES / "seam_carving_slides.pptx"

DARK = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0xE0, 0x7A, 0x1F)
GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF5, 0xF7, 0xFA)

PPT_DARK = PptxColor(0x1F, 0x3A, 0x5F)
PPT_ACCENT = PptxColor(0xE0, 0x7A, 0x1F)
PPT_GRAY = PptxColor(0x55, 0x55, 0x55)
PPT_LIGHT = PptxColor(0xF5, 0xF7, 0xFA)
PPT_WHITE = PptxColor(0xFF, 0xFF, 0xFF)


# ---------------------------------------------------------------------------
# DOCX conversion
# ---------------------------------------------------------------------------

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def _add_runs(paragraph, text: str) -> None:
    """Add Markdown inline formatting to a python-docx paragraph."""
    for token in INLINE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        elif token.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if match:
                label, url = match.groups()
                label = label.replace("`", "")
                run = paragraph.add_run(label)
                if url.startswith("http"):
                    run.font.color.rgb = RGBColor(0x1F, 0x5F, 0xA8)
                    run.underline = True
                    extra = paragraph.add_run(f" ({url})")
                    extra.font.size = Pt(8.5)
                    extra.font.color.rgb = GRAY
            else:
                paragraph.add_run(token)
        else:
            paragraph.add_run(token)


def _add_code_block(doc: Document, code_lines: list[str]) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.2)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run("\n".join(code_lines))
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    properties = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F2F2F2")
    properties.append(shading)


def _add_table(doc: Document, table_lines: list[str]) -> None:
    rows: list[list[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Table Grid"

    for row_index, cells in enumerate(rows):
        for column_index, text in enumerate(cells):
            cell = table.cell(row_index, column_index)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            _add_runs(paragraph, text)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                if row_index == 0:
                    run.bold = True
    doc.add_paragraph()


def _add_image(doc: Document, alt: str, url: str, lines: list[str], index: int) -> int:
    path = (REPORT_DIR / url).resolve()
    if not path.exists():
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(f"[Missing image: {url}]")
        run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
        return index

    picture = doc.add_picture(str(path), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption = alt
    lookahead = index + 1
    while lookahead < len(lines) and not lines[lookahead].strip():
        lookahead += 1
    if lookahead < len(lines):
        candidate = lines[lookahead].strip()
        if candidate.startswith("*") and candidate.endswith("*") and "Figure" in candidate:
            caption = candidate.strip("*")
            index = lookahead

    caption_paragraph = doc.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    return index


def build_docx() -> None:
    markdown = REPORT_MD.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12

    for level in range(1, 4):
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Calibri Light"
        style.font.color.rgb = DARK

    footer = section.footer.paragraphs[0]
    footer.text = "Assignment 2 - Image Compression by Seam Carving"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = GRAY

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            _add_code_block(doc, code_lines)
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            _add_table(doc, table_lines)
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                doc.add_heading(text, level=0)
            else:
                doc.add_heading(text, level=min(level, 3))
            index += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                alt, url = match.groups()
                index = _add_image(doc, alt, url, lines, index)
            index += 1
            continue

        if stripped.startswith(("- ", "* ")):
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(("- ", "* ")):
                items.append(lines[index].strip()[2:])
                index += 1
            for item in items:
                if item.startswith("[ ] "):
                    item = "\u2610 " + item[4:]
                paragraph = doc.add_paragraph(style="List Bullet")
                _add_runs(paragraph, item)
            continue

        if re.match(r"^\d+\.\s", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s", "", lines[index].strip()))
                index += 1
            for item in items:
                paragraph = doc.add_paragraph(style="List Number")
                _add_runs(paragraph, item)
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].strip())
                index += 1
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.3)
            run = paragraph.add_run(" ".join(quote_lines))
            run.italic = True
            run.font.color.rgb = GRAY
            continue

        if stripped == "---":
            index += 1
            continue

        paragraph_lines: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith(("#", "|", "```", "![", ">", "- ", "* "))
                or re.match(r"^\d+\.\s", candidate)
                or candidate == "---"
            ):
                break
            paragraph_lines.append(candidate)
            index += 1

        if paragraph_lines:
            paragraph = doc.add_paragraph()
            _add_runs(paragraph, " ".join(paragraph_lines))
        else:
            index += 1

    doc.save(REPORT_DOCX)
    print(f"wrote {REPORT_DOCX.relative_to(ROOT)} ({REPORT_DOCX.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# PPTX builder
# ---------------------------------------------------------------------------


def _pt_title(slide, text: str) -> None:
    box = slide.shapes.add_textbox(PInches(0.6), PInches(0.32), PInches(12.1), PInches(0.8))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.name = "Calibri"
    paragraph.font.size = PPt(30)
    paragraph.font.bold = True
    paragraph.font.color.rgb = PPT_DARK

    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        PInches(0.6),
        PInches(1.16),
        PInches(12.1),
        PInches(0.05),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = PPT_ACCENT
    line.line.fill.background()


def _pt_footer(slide, number: int) -> None:
    label = slide.shapes.add_textbox(PInches(0.6), PInches(7.12), PInches(8), PInches(0.25))
    paragraph = label.text_frame.paragraphs[0]
    paragraph.text = "Assignment 2 - Image Compression by Seam Carving"
    paragraph.font.size = PPt(10)
    paragraph.font.color.rgb = PPT_GRAY

    page = slide.shapes.add_textbox(PInches(12.25), PInches(7.12), PInches(0.5), PInches(0.25))
    paragraph = page.text_frame.paragraphs[0]
    paragraph.text = str(number)
    paragraph.alignment = PP_ALIGN.RIGHT
    paragraph.font.size = PPt(10)
    paragraph.font.color.rgb = PPT_GRAY


def _pt_bullets(slide, items: list[str], left=0.7, top=1.5, width=6.0, height=5.3, size=19):
    box = slide.shapes.add_textbox(
        PInches(left), PInches(top), PInches(width), PInches(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    for item_index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if item_index == 0 else frame.add_paragraph()
        paragraph.text = "\u2022 " + item
        paragraph.font.name = "Calibri"
        paragraph.font.size = PPt(size)
        paragraph.font.color.rgb = PPT_DARK
        paragraph.space_after = PPt(8)


def _pt_code(slide, code: str, left=6.9, top=1.5, width=5.9, height=2.4) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        PInches(left),
        PInches(top),
        PInches(width),
        PInches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = PPT_LIGHT
    shape.line.color.rgb = PPT_ACCENT
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = PInches(0.2)
    frame.margin_right = PInches(0.2)
    frame.margin_top = PInches(0.15)
    paragraph = frame.paragraphs[0]
    paragraph.text = code
    paragraph.font.name = "Consolas"
    paragraph.font.size = PPt(13)
    paragraph.font.color.rgb = PPT_DARK


def _pt_image(slide, path: Path, left, top, width):
    if path.exists():
        slide.shapes.add_picture(str(path), PInches(left), PInches(top), width=PInches(width))


def _pt_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def _pt_table(slide, rows: list[list[str]], left, top, width, height, size=13):
    table = slide.shapes.add_table(
        len(rows), len(rows[0]), PInches(left), PInches(top), PInches(width), PInches(height)
    ).table
    widths = [width / len(rows[0])] * len(rows[0])
    for column_index, column_width in enumerate(widths):
        table.columns[column_index].width = PInches(column_width)
    for row_index, row in enumerate(rows):
        for column_index, text in enumerate(row):
            cell = table.cell(row_index, column_index)
            cell.text = text
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = PPt(size)
                    run.font.name = "Calibri"
                    if row_index == 0:
                        run.font.bold = True
                        run.font.color.rgb = PPT_WHITE
            cell.fill.solid()
            if row_index == 0:
                cell.fill.fore_color.rgb = PPT_DARK
            else:
                cell.fill.fore_color.rgb = PPT_LIGHT if row_index % 2 else PPT_WHITE


def build_pptx() -> None:
    presentation = Presentation()
    presentation.slide_width = PInches(13.333)
    presentation.slide_height = PInches(7.5)
    blank = presentation.slide_layouts[6]

    # Slide 1: title.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Image Compression by Seam Carving")
    box = slide.shapes.add_textbox(PInches(0.7), PInches(1.55), PInches(6.2), PInches(1.2))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = "Dynamic programming  |  Greedy  |  Graph cut"
    paragraph.font.size = PPt(22)
    paragraph.font.color.rgb = PPT_ACCENT
    paragraph.font.bold = True
    paragraph = frame.add_paragraph()
    paragraph.text = "Content-aware resizing that keeps the important content."
    paragraph.font.size = PPt(17)
    paragraph.font.color.rgb = PPT_GRAY

    box = slide.shapes.add_textbox(PInches(0.7), PInches(3.0), PInches(6.2), PInches(1.7))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = "Team T23"
    paragraph.font.size = PPt(20)
    paragraph.font.color.rgb = PPT_DARK
    paragraph.font.bold = True
    paragraph = frame.add_paragraph()
    paragraph.text = (
        "Ong Jay Tong  |  Liew Wen Shan  |  Tristan Tham Rui Hong  |  "
        "Ching Wei An  |  Mohammad Shahrin Idzuan Bin Iswandi  |  "
        "Amirul Syafiq Bin Mohamed Faizul"
    )
    paragraph.font.size = PPt(10.5)
    paragraph.font.color.rgb = PPT_DARK
    paragraph = frame.add_paragraph()
    paragraph.text = "52 tests  |  3 algorithms  |  1 interactive app"
    paragraph.font.size = PPt(13)
    paragraph.font.color.rgb = PPT_GRAY

    _pt_image(slide, ASSETS / "scene.png", 7.1, 1.6, 5.6)
    _pt_footer(slide, 1)
    _pt_notes(
        slide,
        "Introduce the team and the project. We implement seam finding three ways and build a "
        "CLI, a GUI, a test suite, and report figures.",
    )

    # Slide 2: why seam carving.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Why seam carving?")
    _pt_bullets(
        slide,
        [
            "Cropping is fast but removes content.",
            "Scaling keeps content but distorts object shapes.",
            "Seam carving removes low-energy connected paths.",
            "Edges and objects stay where they are.",
        ],
        top=1.35,
        width=12.0,
        height=1.4,
        size=18,
    )
    _pt_image(slide, FIGURES / "comparison_large.png", 0.6, 2.95, 12.1)
    _pt_footer(slide, 2)
    _pt_notes(
        slide,
        "Point at the comparison: the crop loses the lighthouse, the scaled image is squashed, "
        "the seam-carved image keeps the lighthouse and the horizon.",
    )

    # Slide 3: energy map.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Energy map: where are the edges?")
    _pt_bullets(
        slide,
        [
            "Convert to grayscale: 0.299 R + 0.587 G + 0.114 B.",
            "Apply the Sobel kernels Gx and Gy.",
            "Energy = sqrt(Gx^2 + Gy^2).",
            "High energy = edge; low energy = flat region.",
            "Cost: O(H*W) time and space.",
        ],
        width=5.9,
    )
    _pt_image(slide, FIGURES / "energy.png", 6.9, 1.5, 5.8)
    _pt_footer(slide, 3)
    _pt_notes(
        slide,
        "The lighthouse and the horizon are bright in the energy map. The sky and the sea are "
        "dark, so the seam finder removes them first.",
    )

    # Slide 4: DP.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Dynamic programming")
    _pt_bullets(
        slide,
        [
            "A vertical seam: one pixel per row, adjacent rows differ by at most one column.",
            "total[i,j] = cheapest seam ending at (i,j).",
            "Base case: total[0,j] = e[0,j].",
            "The best seam ends at the minimum of the last row.",
            "Parent pointers rebuild the seam from bottom to top.",
            "Cost: O(H*W) per seam.",
        ],
        width=6.2,
        size=17,
    )
    _pt_code(
        slide,
        "total[i,j] = e[i,j] + min(\n"
        "    total[i-1,j-1],\n"
        "    total[i-1,j],\n"
        "    total[i-1,j+1]\n"
        ")\n\n"
        "parent[i] = argmin(...) - 1\n"
        "# -1 left, 0 up, +1 right",
        top=1.5,
        height=3.4,
    )
    _pt_footer(slide, 4)
    _pt_notes(
        slide,
        "Explain optimal substructure and overlapping subproblems. The parent pointer stores "
        "which of the three predecessors was chosen.",
    )

    # Slide 5: greedy.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Greedy: fast, but not optimal")
    _pt_bullets(
        slide,
        [
            "Start at the cheapest pixel in the top row.",
            "Each row: choose the cheapest of j-1, j, j+1.",
            "No lookahead: a cheap pixel now can force an expensive pixel later.",
            "Measured on 320x240, 80 seams:",
            "DP 0.680 s, energy 392,678.5.",
            "Greedy 0.217 s, energy 469,627.8.",
        ],
        width=6.2,
        size=17,
    )
    _pt_code(
        slide,
        "[[  0,  1, 100],\n"
        " [100, 50,   0],\n"
        " [100,100,   0]]\n\n"
        "Greedy: 0 + 50 + 0 = 50\n"
        "DP:     1 +  0 + 0 = 1",
        top=1.5,
        height=2.8,
    )
    _pt_footer(slide, 5)
    _pt_notes(
        slide,
        "The 3x3 counterexample proves greedy is not optimal. The experiment shows the same "
        "thing on the real scene: greedy is faster but removes more energy.",
    )

    # Slide 6: graph cut.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Graph cut: seam as a minimum s-t cut")
    _pt_bullets(
        slide,
        [
            "One node per pixel, plus a source and a sink.",
            "Right edge (i,j)->(i,j+1) has capacity e[i,j].",
            "Infinite backward and diagonal edges force a valid seam.",
            "Minimum cut = optimal seam; max-flow min-cut theorem.",
            "Dinic: BFS levels + DFS blocking flow.",
            "Seam: rightmost source-side pixel in each row.",
            "Last-column sink capacity is e[i,W-1], not infinity.",
        ],
        width=6.4,
        size=15,
    )
    _pt_image(slide, FIGURES / "comparison_methods_small.png", 6.9, 1.5, 5.8)
    _pt_footer(slide, 6)
    _pt_notes(
        slide,
        "Explain why the infinite edges are needed and why the last-column capacity is finite. "
        "DP and graph cut have the same energy, but graph cut is much slower.",
    )

    # Slide 7: demo.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Interactive demo and step-by-step seams")
    _pt_bullets(
        slide,
        [
            "GUI: open, save, reset, method selector, width and height.",
            "Insert one vertical or horizontal seam.",
            "Show the first removed seam.",
            "CLI: --width, --height, --method, --seams-dir, --gif.",
            "Step images and animated GIF for the bonus.",
        ],
        width=6.2,
        size=17,
    )
    _pt_code(
        slide,
        "python -m seam_carving.cli \\\n"
        "  assets/scene.png out/demo.png \\\n"
        "  --width 240 --seams-dir out/steps \\\n"
        "  --gif out/carving.gif",
        top=4.4,
        width=6.0,
        height=1.5,
    )
    _pt_image(slide, FIGURES / "seams_first_6.png", 6.9, 1.5, 5.8)
    _pt_footer(slide, 7)
    _pt_notes(
        slide,
        "Run the live demo. Show the lighthouse staying in place, then reset and compare with "
        "greedy. Show the first six seams and the animated GIF.",
    )

    # Slide 8: horizontal and insertion.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Horizontal seams and insertion")
    _pt_bullets(
        slide,
        [
            "Horizontal seam: one row per column.",
            "Transpose the image, run the vertical pipeline, transpose back.",
            "Insertion: find a low-energy seam and duplicate its pixels.",
            "The GUI inserts one seam at a time.",
            "Duplication is simple; averaging is smoother.",
        ],
        width=6.2,
        size=17,
    )
    _pt_image(slide, FIGURES / "horizontal.png", 6.4, 1.5, 6.3)
    _pt_footer(slide, 8)
    _pt_notes(
        slide,
        "Transposition reuses every vertical algorithm. Insertion is implemented in the "
        "library and exposed one seam at a time in the GUI.",
    )

    # Slide 9: complexity.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Complexity and analysis")
    _pt_table(
        slide,
        [
            ["Method", "Per seam", "Space", "Optimal?"],
            ["DP", "O(H*W)", "O(H*W)", "Yes"],
            ["Greedy", "O(H*W)", "O(H)", "No"],
            ["Graph cut", "max-flow", "O(H*W)", "Yes"],
        ],
        left=0.7,
        top=1.5,
        width=6.4,
        height=2.5,
        size=14,
    )
    _pt_bullets(
        slide,
        [
            "Total for k seams: O(k*H*W) because the energy map is recomputed.",
            "Full DP table is not required: two rows + parent pointers, or Hirschberg.",
            "Outer seam loop is sequential.",
            "Energy map and per-row columns can be parallelised.",
            "Python's GIL limits pure-Python threads; NumPy vectorisation is the practical win.",
        ],
        left=7.3,
        top=1.5,
        width=5.4,
        size=16,
    )
    _pt_footer(slide, 9)
    _pt_notes(
        slide,
        "Emphasise that DP and greedy have the same Big-O but different quality. Graph cut is "
        "optimal but its max-flow is much more expensive.",
    )

    # Slide 10: testing and bonus.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Testing, CI, and bonus features")
    _pt_bullets(
        slide,
        [
            "52 tests: brute force, DP vs graph cut, edge cases, CLI, GUI.",
            "Graph-cut total energy equals DP total energy.",
            "CI runs tests and a real CLI resize on Python 3.10, 3.11, 3.12.",
            "Bonus: step-by-step visualisation (GIF and per-seam images).",
            "Bonus: interactive Tkinter GUI.",
            "Bonus: CI/CD workflow with uploaded artifacts.",
        ],
        width=12.0,
        size=19,
    )
    _pt_footer(slide, 10)
    _pt_notes(
        slide,
        "Mention the brute-force check and the graph-cut equality. The CI workflow proves the "
        "project runs from a clean checkout.",
    )

    # Slide 11: conclusion.
    slide = presentation.slides.add_slide(blank)
    _pt_title(slide, "Conclusion")
    _pt_bullets(
        slide,
        [
            "DP is the practical choice: optimal for the energy and fast.",
            "Graph cut proves the same optimum but is much slower.",
            "Greedy is a useful fast baseline, but it is not optimal.",
            "The project includes a CLI, a GUI, tests, figures, and CI.",
        ],
        width=8.0,
        size=22,
    )
    _pt_image(slide, FIGURES / "comparison_large.png", 1.6, 4.35, 10.1)
    _pt_footer(slide, 11)
    _pt_notes(
        slide,
        "Close with the main comparison. Thank the audience and invite questions about the "
        "recurrence, the graph construction, the complexity, or the code.",
    )

    presentation.save(SLIDES_PPTX)
    print(f"wrote {SLIDES_PPTX.relative_to(ROOT)} ({SLIDES_PPTX.stat().st_size} bytes)")


def main() -> None:
    if not REPORT_MD.exists():
        raise SystemExit(f"missing {REPORT_MD}")
    build_docx()
    build_pptx()


if __name__ == "__main__":
    sys.exit(main())
