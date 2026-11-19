#!/usr/bin/env python3
"""Build CeramScan_Sistema_2026.docx directly from project_description.md and benchmark_results.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = REPO_ROOT / "grant_case" / "project_description.md"
BENCHMARK_PATH = REPO_ROOT / "reports" / "benchmark_results.json"
OUT_DOCX_PATH = REPO_ROOT / "grant_case" / "CeramScan_Sistema_2026.docx"

INK_PRIMARY = RGBColor(0x0F, 0x17, 0x2A)
INK_SECONDARY = RGBColor(0x33, 0x41, 0x55)
ACCENT_COBALT = RGBColor(0x1E, 0x3A, 0x8A)

IMAGE_RE = re.compile(r"^!\[([^\]]+)\]\(([^)]+)\)$")
INLINE_TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+(.+)$")

CLASS_LABELS_RU: dict[str, str] = {
    "pass": "pass (годное изделие)",
    "crack": "crack (трещина, цек)",
    "chip": "chip (скол кромки)",
    "glaze_bare": "glaze_bare (плешина)",
    "shape_warp": "shape_warp (перекос)",
    "glaze_drip": "glaze_drip (натек)",
    "pinhole_speck": "pinhole_speck (накол, мушка)",
}

CLASS_NOTES_RU: dict[str, str] = {
    "pass": "Утиль 900 °C и глазурь 1350 °C с кобальтом",
    "crack": "Включая трещины поверх бликов и росписи",
    "chip": "Детекция в зоне ободка m_rim",
    "glaze_bare": "Непокрытые участки глазури на поле",
    "shape_warp": "Выход контура за допуск > 1.5%",
    "glaze_drip": "Локальные наплывы с доработкой REWORK",
    "pinhole_speck": "Отделение точечных дефектов от кобальта",
}


def set_run_font(
    run: Any,
    *,
    size_pt: float = 12.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = INK_PRIMARY,
    font_name: str = "Times New Roman",
) -> None:
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    r_pr = run._r.get_or_add_rPr()
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        r_fonts.set(qn(attr), font_name)


def apply_paragraph_spacing(
    paragraph: Any,
    *,
    space_before_pt: float = 0.0,
    space_after_pt: float = 4.0,
    first_line_cm: float = 0.0,
    align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT,
    keep_with_next: bool = False,
) -> None:
    pf = paragraph.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(space_before_pt)
    pf.space_after = Pt(space_after_pt)
    pf.first_line_indent = Cm(first_line_cm) if first_line_cm > 0 else Cm(0)
    pf.keep_with_next = keep_with_next
    paragraph.alignment = align


def add_inline_formatted_runs(
    paragraph: Any,
    text: str,
    *,
    size_pt: float = 12.0,
    color: RGBColor = INK_PRIMARY,
    base_bold: bool = False,
) -> None:
    pos = 0
    for match in INLINE_TOKEN_RE.finditer(text):
        if match.start() > pos:
            r_plain = paragraph.add_run(text[pos : match.start()])
            set_run_font(r_plain, size_pt=size_pt, bold=base_bold, color=color)
        token = match.group(0)
        if token.startswith("**") and token.endswith("**"):
            r_bold = paragraph.add_run(token[2:-2])
            set_run_font(r_bold, size_pt=size_pt, bold=True, color=color)
        elif token.startswith("`") and token.endswith("`"):
            r_code = paragraph.add_run(token[1:-1])
            set_run_font(r_code, size_pt=size_pt, bold=base_bold, color=color)
        pos = match.end()
    if pos < len(text):
        r_tail = paragraph.add_run(text[pos:])
        set_run_font(r_tail, size_pt=size_pt, bold=base_bold, color=color)


def add_page_number_field(paragraph: Any) -> None:
    run_begin = paragraph.add_run()
    set_run_font(run_begin, size_pt=11.0, color=INK_SECONDARY)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run_begin._r.append(fld_begin)

    run_instr = paragraph.add_run()
    set_run_font(run_instr, size_pt=11.0, color=INK_SECONDARY)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    run_instr._r.append(instr)

    run_sep = paragraph.add_run()
    set_run_font(run_sep, size_pt=11.0, color=INK_SECONDARY)
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run_sep._r.append(fld_sep)

    run_val = paragraph.add_run("2")
    set_run_font(run_val, size_pt=11.0, color=INK_SECONDARY)

    run_end = paragraph.add_run()
    set_run_font(run_end, size_pt=11.0, color=INK_SECONDARY)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run_end._r.append(fld_end)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = INK_PRIMARY
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(4)

    r_pr = normal._element.get_or_add_rPr()
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        r_fonts.set(qn(attr), "Times New Roman")

    footer = section.footer
    footer_para = footer.paragraphs[0]
    apply_paragraph_spacing(
        footer_para,
        space_before_pt=0.0,
        space_after_pt=0.0,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_page_number_field(footer_para)


def set_cell_borders(cell: Any, color_hex: str = "CBD5E1", sz: str = "4") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'  <w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'  <w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f"</w:tcBorders>"
    )
    tc_pr.append(borders)


def set_cell_borderless_toc(cell: Any) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:bottom w:val="dotted" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'  <w:right w:val="none"/>'
        f"</w:tcBorders>"
    )
    tc_pr.append(borders)


def set_cell_margins(cell: Any, top: int = 60, bottom: int = 60, left: int = 90, right: int = 90) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f"</w:tcMar>"
    )
    tc_pr.append(mar)


def set_cell_shading(cell: Any, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_row_cant_split(row: Any, is_header: bool = False) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    if is_header:
        tr_pr.append(OxmlElement("w:tblHeader"))


def pick_overlay(
    overlay_files: list[str],
    rel_target: str,
) -> Path:
    candidate = (REPO_ROOT / "grant_case" / rel_target).resolve()
    if candidate.is_file():
        return candidate

    stem = Path(rel_target).stem.upper()
    tokens = [tok for tok in stem.split("-") if tok not in {"SYN", "REAL", "01", "02", "03", "04"}]
    for entry in overlay_files:
        p = REPO_ROOT / entry
        if p.is_file() and all(tok in p.stem.upper() for tok in tokens):
            return p
    for tok in tokens:
        for entry in overlay_files:
            p = REPO_ROOT / entry
            if p.is_file() and tok in p.stem.upper():
                return p
    for entry in overlay_files:
        p = REPO_ROOT / entry
        if p.is_file():
            return p
    raise FileNotFoundError(f"Overlay PNG not found for {rel_target}")


def load_benchmark_data() -> dict[str, Any]:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def build_dynamic_table3_rows(bench: dict[str, Any]) -> list[list[str]]:
    per_class = bench.get("per_class", {})
    ordered_classes = [
        "pass",
        "crack",
        "chip",
        "glaze_bare",
        "shape_warp",
        "glaze_drip",
        "pinhole_speck",
    ]
    rows: list[list[str]] = [
        ["Класс / контур проверки", "Число образцов", "Precision", "Recall", "F1", "Примечание по этапу и порогам"]
    ]
    for cls_key in ordered_classes:
        if cls_key not in per_class:
            continue
        metrics = per_class[cls_key]
        rows.append(
            [
                CLASS_LABELS_RU.get(cls_key, cls_key),
                str(int(metrics["support"])),
                f"{float(metrics['precision']):.4f}",
                f"{float(metrics['recall']):.4f}",
                f"{float(metrics['f1']):.4f}",
                CLASS_NOTES_RU.get(cls_key, "Проверено на стенде"),
            ]
        )

    sample_count = int(bench["sample_count"])
    rows.append(
        [
            f"Итого ОТК (sample_count = {sample_count})",
            str(sample_count),
            f"{float(bench['precision']):.4f}",
            f"{float(bench['recall']):.4f}",
            f"{float(bench['f1']):.4f}",
            (
                f"Точность вердиктов ГОСТ: {float(bench['verdict_accuracy']):.4f}; "
                f"среднее время {float(bench['avg_latency_ms']):.2f} мс/кадр, "
                f"p95 = {float(bench['p95_latency_ms']):.2f} мс/кадр"
            ),
        ]
    )

    auth = bench.get("auth_benchmark", {})
    pair_count = int(auth.get("pair_count", 0))
    rows.append(
        [
            f"Контур 289-ФЗ (pair_count = {pair_count})",
            f"{pair_count} пары",
            "-",
            "-",
            "-",
            (
                f"Точность сверки: {float(auth.get('auth_accuracy', 0.0)):.4f}; "
                f"среднее сходство оригинала {float(auth.get('genuine_mean_match', 0.0)):.2f}%, "
                f"подделки {float(auth.get('fake_mean_match', 0.0)):.2f}%"
            ),
        ]
    )
    return rows


def render_title_page(doc: Document, lines: list[str]) -> None:
    clean_lines = [ln.strip() for ln in lines if ln.strip()]
    header_lines: list[str] = []
    title_line = ""
    meta_lines: list[str] = []
    footer_line = ""

    for ln in clean_lines:
        if ln.startswith("**") and ln.endswith("**") and not title_line:
            title_line = ln[2:-2].strip()
        elif not title_line:
            header_lines.append(ln)
        elif ln.startswith("г. Гатчина,"):
            footer_line = ln
        else:
            meta_lines.append(ln)

    for idx, h_ln in enumerate(header_lines):
        p = doc.add_paragraph()
        apply_paragraph_spacing(
            p,
            space_before_pt=0.0,
            space_after_pt=4.0 if idx < len(header_lines) - 1 else 48.0,
            align=WD_ALIGN_PARAGRAPH.CENTER,
        )
        r = p.add_run(h_ln)
        set_run_font(r, size_pt=12.0, bold=(idx == 0), color=INK_PRIMARY)

    p_title = doc.add_paragraph()
    apply_paragraph_spacing(
        p_title,
        space_before_pt=36.0,
        space_after_pt=48.0,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    r_title = p_title.add_run(title_line)
    set_run_font(r_title, size_pt=16.0, bold=True, color=ACCENT_COBALT)

    for idx, m_ln in enumerate(meta_lines):
        p_meta = doc.add_paragraph()
        p_meta.paragraph_format.left_indent = Cm(5.0)
        apply_paragraph_spacing(
            p_meta,
            space_before_pt=0.0,
            space_after_pt=4.0 if idx < len(meta_lines) - 1 else 72.0,
            align=WD_ALIGN_PARAGRAPH.LEFT,
        )
        add_inline_formatted_runs(p_meta, m_ln, size_pt=12.0, color=INK_PRIMARY)

    if footer_line:
        p_foot = doc.add_paragraph()
        apply_paragraph_spacing(
            p_foot,
            space_before_pt=96.0,
            space_after_pt=0.0,
            align=WD_ALIGN_PARAGRAPH.CENTER,
        )
        r_foot = p_foot.add_run(footer_line)
        set_run_font(r_foot, size_pt=12.0, bold=True, color=INK_PRIMARY)

    doc.add_page_break()


def render_toc_table(doc: Document, rows: list[list[str]]) -> None:
    data_rows = rows[1:]
    table = doc.add_table(rows=len(data_rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_widths = (Cm(15.0), Cm(2.0))

    for r_idx, row_vals in enumerate(data_rows):
        row = table.rows[r_idx]
        set_row_cant_split(row, is_header=False)
        for c_idx, val in enumerate(row_vals[:2]):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_borderless_toc(cell)
            set_cell_margins(cell, top=35, bottom=35, left=40, right=40)
            p = cell.paragraphs[0]
            is_chapter = val.startswith("Глава ") or val.startswith("1.")
            align = WD_ALIGN_PARAGRAPH.RIGHT if c_idx == 1 else WD_ALIGN_PARAGRAPH.LEFT
            apply_paragraph_spacing(p, space_before_pt=0.0, space_after_pt=0.0, align=align)
            if c_idx == 0 and not val.startswith("1.") and not val.startswith("Глава "):
                p.paragraph_format.left_indent = Cm(0.8)
            elif c_idx == 0 and val.startswith("Глава "):
                p.paragraph_format.left_indent = Cm(0.4)
            r = p.add_run(val)
            set_run_font(r, size_pt=12.0, bold=(is_chapter and c_idx == 0), color=INK_PRIMARY)


def compute_col_widths(num_cols: int, rows: list[list[str]]) -> list[Cm]:
    total_cm = 17.0
    if num_cols == 5 and "Система / аналог" in rows[0][0]:
        return [Cm(3.0), Cm(2.4), Cm(3.7), Cm(6.3), Cm(1.6)]
    if num_cols == 5 and "Дефект по ГОСТ" in rows[0][0]:
        return [Cm(3.2), Cm(4.2), Cm(2.4), Cm(3.6), Cm(3.6)]
    if num_cols == 6:
        return [Cm(3.7), Cm(1.8), Cm(1.6), Cm(1.5), Cm(1.5), Cm(6.9)]
    if num_cols == 5 and "Параметр" in rows[0][0]:
        return [Cm(4.5), Cm(2.6), Cm(2.6), Cm(2.7), Cm(4.6)]
    width_each = total_cm / float(num_cols)
    return [Cm(width_each) for _ in range(num_cols)]


def render_data_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    num_cols = len(rows[0])
    col_widths = compute_col_widths(num_cols, rows)
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    for r_idx, row_vals in enumerate(rows):
        row = table.rows[r_idx]
        is_header = r_idx == 0
        set_row_cant_split(row, is_header=is_header)
        is_highlight = not is_header and (
            row_vals[0].startswith("Итого")
            or row_vals[0].startswith("А. ")
            or row_vals[0].startswith("Валовый")
            or row_vals[0].startswith("Чистый эффект")
            or row_vals[0].startswith("CeramScan")
        )
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_borders(cell)
            set_cell_margins(cell, top=55, bottom=55, left=75, right=75)
            if is_header:
                set_cell_shading(cell, "EFF6FF")
            elif is_highlight:
                set_cell_shading(cell, "F8FAFC")

            p = cell.paragraphs[0]
            apply_paragraph_spacing(
                p,
                space_before_pt=0.0,
                space_after_pt=0.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
            )
            add_inline_formatted_runs(
                p,
                val,
                size_pt=10.0 if num_cols >= 5 else 11.0,
                color=ACCENT_COBALT if is_header else INK_PRIMARY,
                base_bold=(is_header or is_highlight),
            )

    spacer = doc.add_paragraph()
    apply_paragraph_spacing(spacer, space_before_pt=0.0, space_after_pt=4.0)


def parse_markdown_table(lines: list[str], start_idx: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    idx = start_idx
    while idx < len(lines):
        ln = lines[idx].strip()
        if not ln.startswith("|"):
            break
        parts = [cell.strip() for cell in ln.strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in parts):
            rows.append(parts)
        idx += 1
    return rows, idx


def build_docx() -> Path:
    md_text = MD_PATH.read_text(encoding="utf-8")
    bench = load_benchmark_data()
    overlay_files = [str(p) for p in bench.get("overlay_files", [])]

    doc = Document()
    configure_document(doc)

    lines = md_text.splitlines()
    idx = 0
    in_title_section = False
    in_toc_section = False
    title_lines: list[str] = []
    in_code_block = False
    code_lines: list[str] = []

    while idx < len(lines):
        raw_line = lines[idx]
        stripped = raw_line.strip()

        if idx == 0 and stripped.startswith("# "):
            idx += 1
            continue

        if stripped.startswith("```"):
            if in_code_block:
                p_code = doc.add_paragraph()
                p_code.paragraph_format.left_indent = Cm(0.6)
                apply_paragraph_spacing(p_code, space_before_pt=2.0, space_after_pt=6.0)
                r_code = p_code.add_run("\n".join(code_lines))
                set_run_font(r_code, size_pt=10.0, color=INK_SECONDARY, font_name="Times New Roman")
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            idx += 1
            continue

        if in_code_block:
            code_lines.append(raw_line)
            idx += 1
            continue

        if stripped == "## 1.1 Титульный лист":
            in_title_section = True
            in_toc_section = False
            idx += 1
            continue

        if stripped.startswith("## "):
            if in_title_section:
                render_title_page(doc, title_lines)
                in_title_section = False
            heading_text = stripped[3:].strip()
            in_toc_section = heading_text == "1.2 Содержание"
            p_h2 = doc.add_paragraph()
            apply_paragraph_spacing(
                p_h2,
                space_before_pt=10.0 if not in_toc_section else 0.0,
                space_after_pt=6.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                keep_with_next=True,
            )
            r_h2 = p_h2.add_run(heading_text)
            set_run_font(r_h2, size_pt=14.0, bold=True, color=ACCENT_COBALT)
            idx += 1
            continue

        if in_title_section:
            title_lines.append(raw_line)
            idx += 1
            continue

        if stripped.startswith("### "):
            p_h3 = doc.add_paragraph()
            apply_paragraph_spacing(
                p_h3,
                space_before_pt=10.0,
                space_after_pt=5.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                keep_with_next=True,
            )
            r_h3 = p_h3.add_run(stripped[4:].strip())
            set_run_font(r_h3, size_pt=13.0, bold=True, color=INK_PRIMARY)
            idx += 1
            continue

        if stripped.startswith("#### "):
            p_h4 = doc.add_paragraph()
            apply_paragraph_spacing(
                p_h4,
                space_before_pt=8.0,
                space_after_pt=4.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                keep_with_next=True,
            )
            r_h4 = p_h4.add_run(stripped[5:].strip())
            set_run_font(r_h4, size_pt=12.0, bold=True, color=INK_PRIMARY)
            idx += 1
            continue

        if stripped.startswith("|"):
            rows, next_idx = parse_markdown_table(lines, idx)
            if in_toc_section:
                render_toc_table(doc, rows)
                in_toc_section = False
            elif rows and "Класс / контур проверки" in rows[0][0]:
                render_data_table(doc, build_dynamic_table3_rows(bench))
            else:
                render_data_table(doc, rows)
            idx = next_idx
            continue

        img_match = IMAGE_RE.match(stripped)
        if img_match:
            caption_text = img_match.group(1).strip()
            rel_img_path = img_match.group(2).strip()
            resolved_img = pick_overlay(overlay_files, rel_img_path)

            p_img = doc.add_paragraph()
            apply_paragraph_spacing(
                p_img,
                space_before_pt=6.0,
                space_after_pt=2.0,
                align=WD_ALIGN_PARAGRAPH.CENTER,
                keep_with_next=True,
            )
            p_img.add_run().add_picture(str(resolved_img), width=Cm(15.5))

            p_cap = doc.add_paragraph()
            apply_paragraph_spacing(
                p_cap,
                space_before_pt=0.0,
                space_after_pt=8.0,
                align=WD_ALIGN_PARAGRAPH.CENTER,
            )
            r_cap = p_cap.add_run(caption_text)
            set_run_font(r_cap, size_pt=11.0, italic=True, color=INK_SECONDARY)
            idx += 1
            continue

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("Таблица "):
            p_tbl_cap = doc.add_paragraph()
            apply_paragraph_spacing(
                p_tbl_cap,
                space_before_pt=6.0,
                space_after_pt=3.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                keep_with_next=True,
            )
            r_tbl_cap = p_tbl_cap.add_run(stripped)
            set_run_font(r_tbl_cap, size_pt=12.0, bold=True, color=INK_PRIMARY)
            idx += 1
            continue

        num_match = NUMBERED_ITEM_RE.match(stripped)
        if num_match:
            p_num = doc.add_paragraph()
            p_num.paragraph_format.left_indent = Cm(0.6)
            apply_paragraph_spacing(
                p_num,
                space_before_pt=0.0,
                space_after_pt=3.0,
                align=WD_ALIGN_PARAGRAPH.JUSTIFY,
            )
            add_inline_formatted_runs(p_num, stripped, size_pt=12.0, color=INK_PRIMARY)
            idx += 1
            continue

        if stripped.startswith("[") and "]" in stripped[:6]:
            p_src = doc.add_paragraph()
            p_src.paragraph_format.left_indent = Cm(0.5)
            p_src.paragraph_format.first_line_indent = Cm(-0.5)
            apply_paragraph_spacing(
                p_src,
                space_before_pt=0.0,
                space_after_pt=3.0,
                align=WD_ALIGN_PARAGRAPH.LEFT,
            )
            add_inline_formatted_runs(p_src, stripped, size_pt=11.5, color=INK_PRIMARY)
            idx += 1
            continue

        p_body = doc.add_paragraph()
        apply_paragraph_spacing(
            p_body,
            space_before_pt=0.0,
            space_after_pt=4.0,
            first_line_cm=1.0,
            align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        )
        add_inline_formatted_runs(p_body, stripped, size_pt=12.0, color=INK_PRIMARY)
        idx += 1

    OUT_DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX_PATH))
    return OUT_DOCX_PATH


if __name__ == "__main__":
    output = build_docx()
    print(f"Generated: {output}")
