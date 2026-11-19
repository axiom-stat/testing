from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSON_PATH = REPO_ROOT / "reports" / "benchmark_results.json"
OVERLAYS_DIR = REPO_ROOT / "reports" / "overlays"
OUTPUT_PPTX_PATH = REPO_ROOT / "grant_case" / "CeramScan_Sistema_2026.pptx"

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
LEFT_M = 0.65
CONTENT_W = 12.033
OVERLAY_ASPECT = 1284.0 / 684.0

# Arial + DejaVu Sans Mono render Cyrillic and ASCII math operators (<, >, +)
# cleanly in both LibreOffice Impress PDF export and Windows/macOS PowerPoint.
FONT_SANS = "Arial"
FONT_MONO = "DejaVu Sans Mono"

C_BG = RGBColor(0xF8, 0xFA, 0xFC)
C_SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
C_MUTED = RGBColor(0xF1, 0xF5, 0xF9)
C_INK = RGBColor(0x0F, 0x17, 0x2A)
C_INK_SEC = RGBColor(0x47, 0x55, 0x69)
C_COBALT = RGBColor(0x1D, 0x4E, 0xD8)
C_COBALT_DARK = RGBColor(0x1E, 0x3A, 0x8A)
C_COBALT_WASH = RGBColor(0xEF, 0xF6, 0xFF)
C_COBALT_LIGHT = RGBColor(0xBF, 0xDB, 0xFE)
C_HAIRLINE = RGBColor(0xCB, 0xD5, 0xE1)
C_PASS = RGBColor(0x05, 0x96, 0x69)
C_PASS_BG = RGBColor(0xEC, 0xFD, 0xF5)
C_PASS_INK = RGBColor(0x06, 0x5F, 0x46)
C_REWORK = RGBColor(0xB4, 0x53, 0x09)
C_REJECT = RGBColor(0xDC, 0x26, 0x26)
C_REJECT_BG = RGBColor(0xFE, 0xF2, 0xF2)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CLASS_ORDER = (
    "pass",
    "crack",
    "chip",
    "glaze_bare",
    "shape_warp",
    "glaze_drip",
    "pinhole_speck",
)

CLASS_LABELS_RU = {
    "pass": "Годные",
    "crack": "Трещина",
    "chip": "Скол",
    "glaze_bare": "Плешина",
    "shape_warp": "Перекос",
    "glaze_drip": "Натек",
    "pinhole_speck": "Мушка",
}


def load_benchmark(json_path: Path = BENCHMARK_JSON_PATH) -> dict[str, Any]:
    if not json_path.is_file():
        raise FileNotFoundError(f"Benchmark JSON not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def collect_overlay_paths(bench: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for raw in bench.get("overlay_files", []):
        candidate = REPO_ROOT / str(raw)
        if candidate.is_file() and candidate.suffix.lower() == ".png":
            paths.append(candidate)
    if not paths and OVERLAYS_DIR.is_dir():
        paths = sorted(p for p in OVERLAYS_DIR.glob("*.png") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No overlay PNG files found in {OVERLAYS_DIR}")
    return paths


def pick_overlay(
    candidates: list[Path],
    priority_groups: list[list[str]],
    fallback_idx: int,
    avoid: set[Path] | None = None,
) -> Path:
    blocked = avoid or set()
    for tokens in priority_groups:
        for path in candidates:
            if path in blocked:
                continue
            stem = path.stem.upper()
            if all(tok in stem for tok in tokens):
                return path
    for tokens in priority_groups:
        for path in candidates:
            stem = path.stem.upper()
            if all(tok in stem for tok in tokens):
                return path
    non_blocked = [p for p in candidates if p not in blocked]
    pool = non_blocked if non_blocked else candidates
    return pool[fallback_idx % len(pool)]


def set_slide_bg(slide: Any, color: RGBColor = C_BG) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _strip_shape_shadow(shape: Any) -> None:
    shape.shadow.inherit = False
    style_el = shape._element.find(qn("p:style"))
    if style_el is not None:
        effect_ref = style_el.find(qn("a:effectRef"))
        if effect_ref is not None:
            effect_ref.set("idx", "0")
    sp_pr = shape._element.find(qn("p:spPr"))
    if sp_pr is not None:
        existing_eff = sp_pr.find(qn("a:effectLst"))
        if existing_eff is not None:
            sp_pr.remove(existing_eff)
        sp_pr.append(OxmlElement("a:effectLst"))


def _apply_run_font(run: Any, font_name: str, font_size: float, color: RGBColor, bold: bool) -> None:
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    r_pr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = r_pr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            r_pr.append(el)
        el.set("typeface", font_name)


def add_rect(
    slide: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: RGBColor | None = None,
    border: RGBColor | None = None,
    border_pt: float = 1.0,
) -> Any:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    if fill is not None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if border is not None:
        shape.line.color.rgb = border
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    _strip_shape_shadow(shape)
    return shape


def add_textbox(
    slide: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    font_name: str = FONT_SANS,
    font_size: float = 12.0,
    color: RGBColor = C_INK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin_lr: float = 0.04,
    margin_tb: float = 0.02,
    line_spacing: float = 1.15,
) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_lr)
    tf.margin_right = Inches(margin_lr)
    tf.margin_top = Inches(margin_tb)
    tf.margin_bottom = Inches(margin_tb)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    run = p.add_run()
    run.text = text
    _apply_run_font(run, font_name, font_size, color, bold)
    return box


def add_paragraphs(
    slide: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    items: list[tuple[str, str, float, RGBColor, bool]],
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    space_after_pt: float = 4.0,
    margin_lr: float = 0.04,
    margin_tb: float = 0.02,
) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_lr)
    tf.margin_right = Inches(margin_lr)
    tf.margin_top = Inches(margin_tb)
    tf.margin_bottom = Inches(margin_tb)
    for idx, (text, font_name, font_size, color, bold) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.15
        p.space_after = Pt(space_after_pt if idx < len(items) - 1 else 0.0)
        run = p.add_run()
        run.text = text
        _apply_run_font(run, font_name, font_size, color, bold)
    return box


def add_chrome(
    slide: Any,
    slide_num: int,
    eyebrow: str,
    title: str,
    footer_left: str,
) -> None:
    add_textbox(
        slide,
        LEFT_M,
        0.26,
        9.4,
        0.30,
        eyebrow,
        font_name=FONT_MONO,
        font_size=9.5,
        color=C_INK_SEC,
        bold=True,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_textbox(
        slide,
        11.1,
        0.26,
        1.583,
        0.30,
        f"{slide_num:02d}/10",
        font_name=FONT_MONO,
        font_size=9.5,
        color=C_COBALT_DARK,
        bold=True,
        align=PP_ALIGN.RIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_rect(slide, LEFT_M, 0.58, CONTENT_W, 0.015, fill=C_HAIRLINE)
    add_textbox(
        slide,
        LEFT_M,
        0.68,
        CONTENT_W,
        0.74,
        title,
        font_name=FONT_SANS,
        font_size=20.0,
        color=C_INK,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
        margin_tb=0.0,
        line_spacing=1.12,
    )
    add_rect(slide, LEFT_M, 6.96, CONTENT_W, 0.015, fill=C_HAIRLINE)
    add_textbox(
        slide,
        LEFT_M,
        7.04,
        9.5,
        0.28,
        footer_left,
        font_name=FONT_SANS,
        font_size=9.0,
        color=C_INK_SEC,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_textbox(
        slide,
        10.4,
        7.04,
        2.283,
        0.28,
        "CeramScan",
        font_name=FONT_MONO,
        font_size=9.0,
        color=C_INK_SEC,
        bold=True,
        align=PP_ALIGN.RIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )


def style_table_cell(
    cell: Any,
    text: str,
    font_name: str = FONT_SANS,
    font_size: float = 11.0,
    color: RGBColor = C_INK,
    bold: bool = False,
    fill_color: RGBColor | None = None,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    border_bottom_hex: str = "CBD5E1",
) -> None:
    if fill_color is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill_color
    else:
        cell.fill.background()
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.12)
    cell.margin_right = Inches(0.12)
    cell.margin_top = Inches(0.06)
    cell.margin_bottom = Inches(0.06)

    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _apply_run_font(run, font_name, font_size, color, bold)

    tc_pr = cell._tc.get_or_add_tcPr()
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        existing = tc_pr.find(qn(f"a:{edge}"))
        if existing is not None:
            tc_pr.remove(existing)
    for edge in ("lnL", "lnR", "lnT"):
        ln = OxmlElement(f"a:{edge}")
        ln.append(OxmlElement("a:noFill"))
        tc_pr.append(ln)
    ln_b = OxmlElement("a:lnB")
    ln_b.set("w", "12700")
    solid = OxmlElement("a:solidFill")
    srgb = OxmlElement("a:srgbClr")
    srgb.set("val", border_bottom_hex)
    solid.append(srgb)
    ln_b.append(solid)
    tc_pr.append(ln_b)


def build_slide_1(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)

    add_rect(slide, LEFT_M, 0.65, 7.80, 0.52, fill=C_COBALT_WASH)
    add_rect(slide, LEFT_M, 0.65, 0.06, 0.52, fill=C_COBALT)
    add_textbox(
        slide,
        LEFT_M + 0.14,
        0.67,
        7.55,
        0.48,
        "Номинация «Технологии в креативных индустриях и народных промыслах»",
        font_name=FONT_SANS,
        font_size=10.5,
        color=C_COBALT_DARK,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )

    add_textbox(
        slide,
        LEFT_M,
        1.42,
        7.80,
        1.75,
        "CeramScan («КерамикСкан»): контроль качества и паспорт фарфора",
        font_name=FONT_SANS,
        font_size=26.5,
        color=C_INK,
        bold=True,
        line_spacing=1.12,
        margin_lr=0.0,
    )
    add_textbox(
        slide,
        LEFT_M,
        3.25,
        7.80,
        0.65,
        "ОТК по ГОСТ 28390-89 и защита ручной росписи под 289-ФЗ",
        font_name=FONT_SANS,
        font_size=13.5,
        color=C_INK_SEC,
        margin_lr=0.0,
    )

    add_rect(slide, LEFT_M, 4.25, 7.80, 2.50, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, LEFT_M, 4.25, 7.80, 0.05, fill=C_COBALT)
    add_paragraphs(
        slide,
        LEFT_M + 0.20,
        4.45,
        7.40,
        2.15,
        [
            ("Павловец Александр Иванович, 1 курс, г. Гатчина", FONT_SANS, 13.5, C_INK, True),
            ("Гатчинский Государственный университет, экономический факультет", FONT_SANS, 12.0, C_INK, False),
            ("Направление «экономика предприятий и организаций»", FONT_SANS, 12.0, C_INK_SEC, False),
        ],
        valign=MSO_ANCHOR.MIDDLE,
        space_after_pt=8.0,
    )

    sx = 8.85
    sw = 3.83
    add_rect(slide, sx, 0.65, sw, 6.10, fill=C_COBALT_DARK)
    add_textbox(
        slide,
        sx + 0.28,
        0.95,
        sw - 0.56,
        0.35,
        "БФ «СИСТЕМА» 2026",
        font_name=FONT_MONO,
        font_size=11.0,
        color=C_COBALT_LIGHT,
        bold=True,
    )
    add_rect(slide, sx + 0.28, 1.42, sw - 0.56, 0.015, fill=C_COBALT)
    add_paragraphs(
        slide,
        sx + 0.28,
        1.85,
        sw - 0.56,
        1.65,
        [
            ("900°C -> 1350°C", FONT_MONO, 19.0, C_WHITE, True),
            ("отсев до обжига", FONT_SANS, 12.5, C_COBALT_LIGHT, False),
        ],
        space_after_pt=6.0,
    )
    add_rect(slide, sx + 0.28, 3.70, sw - 0.56, 0.015, fill=C_COBALT)
    add_paragraphs(
        slide,
        sx + 0.28,
        4.15,
        sw - 0.56,
        1.65,
        [
            ("< 300 мс", FONT_MONO, 21.0, C_WHITE, True),
            ("на CPU без GPU", FONT_SANS, 12.5, C_COBALT_LIGHT, False),
        ],
        space_after_pt=6.0,
    )


def build_slide_2(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        2,
        "1.2 АКТУАЛЬНОСТЬ",
        "Ручной ОТК на тираже 130-140 тыс. упускает брак до обжига 1350°C",
        "Объединение «Гжель», 289-ФЗ",
    )

    add_rect(slide, LEFT_M, 1.65, 6.65, 5.05, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
    rows = [
        ("Потери печи", "дефект утиля 900°C сжигает роспись и глазурь", C_REJECT),
        ("Человеческий фактор", "синий орнамент маскирует цек глазури", C_REWORK),
        ("Маркетплейсы Ozon/WB", "штамповка нарушает нормы 289-ФЗ", C_COBALT),
    ]
    for idx, (head, desc, accent) in enumerate(rows):
        ry = 1.85 + idx * 1.60
        add_rect(slide, LEFT_M + 0.24, ry + 0.10, 0.06, 1.15, fill=accent)
        add_paragraphs(
            slide,
            LEFT_M + 0.44,
            ry + 0.05,
            5.95,
            1.25,
            [
                (head, FONT_SANS, 14.5, C_INK, True),
                (desc, FONT_SANS, 12.5, C_INK_SEC, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=4.0,
        )
        if idx < 2:
            add_rect(slide, LEFT_M + 0.24, ry + 1.48, 6.17, 0.015, fill=C_HAIRLINE)

    kx = 7.60
    kw = 5.083
    kpis = [
        ("130-140 тыс. шт", "изделий/год", C_COBALT_DARK, C_COBALT_WASH),
        ("900°C -> 1350°C", "порог брака", C_REJECT, C_REJECT_BG),
        ("01.10.2026", "289-ФЗ", C_INK, C_MUTED),
    ]
    for idx, (val, lbl, val_col, bg_col) in enumerate(kpis):
        ky = 1.65 + idx * 1.75
        add_rect(slide, kx, ky, kw, 1.55, fill=bg_col, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, kx, ky, 0.08, 1.55, fill=val_col)
        add_paragraphs(
            slide,
            kx + 0.28,
            ky + 0.12,
            kw - 0.45,
            1.30,
            [
                (val, FONT_MONO, 22.0, val_col, True),
                (lbl, FONT_SANS, 13.0, C_INK_SEC, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=3.0,
        )


def build_slide_3(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        3,
        "1.2 ЦЕЛЬ",
        "Цель: ОТК со смартфона до 300 мс без GPU и замены ручной росписи",
        "ПП РФ № 1374, 289-ФЗ",
    )

    add_rect(slide, LEFT_M, 1.65, CONTENT_W, 0.92, fill=C_COBALT_WASH, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, LEFT_M, 1.65, 0.08, 0.92, fill=C_COBALT)
    add_textbox(
        slide,
        LEFT_M + 0.24,
        1.70,
        CONTENT_W - 0.40,
        0.82,
        "Отечественная альтернатива Sacmi и задел под «Честный ЗНАК»",
        font_name=FONT_SANS,
        font_size=14.5,
        color=C_COBALT_DARK,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )

    steps = [
        ("01", "Утиль 900°C", "Скол, трещина, форма до обжига", C_REWORK),
        ("02", "Глазурь 1350°C", "Цек, плешина, допуски ГОСТ\n28390-89", C_COBALT),
        ("03", "Паспорт 289-ФЗ", "Отпечаток мазка и QR-метка", C_PASS),
    ]
    step_w = 3.74
    gap = 0.406
    for idx, (num, head, desc, accent) in enumerate(steps):
        sx = LEFT_M + idx * (step_w + gap)
        add_rect(slide, sx, 3.05, step_w, 3.15, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, sx, 3.05, step_w, 0.08, fill=accent)
        add_textbox(
            slide,
            sx + 0.24,
            3.35,
            1.2,
            0.42,
            num,
            font_name=FONT_MONO,
            font_size=20.0,
            color=accent,
            bold=True,
        )
        add_paragraphs(
            slide,
            sx + 0.24,
            3.95,
            step_w - 0.48,
            1.95,
            [
                (head, FONT_SANS, 16.0, C_INK, True),
                (desc, FONT_SANS, 13.0, C_INK_SEC, False),
            ],
            space_after_pt=10.0,
        )
        if idx < 2:
            add_textbox(
                slide,
                sx + step_w + 0.02,
                4.35,
                gap - 0.04,
                0.50,
                "->",
                font_name=FONT_MONO,
                font_size=16.0,
                color=C_COBALT,
                bold=True,
                align=PP_ALIGN.CENTER,
                valign=MSO_ANCHOR.MIDDLE,
                margin_lr=0.0,
            )


def build_slide_4(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        4,
        "1.3 АРХИТЕКТУРА",
        "Маска бликов LAB и Black-Hat фильтры отделяют брак на CPU",
        "ГОСТ 28390-89, < 300 мс CPU",
    )

    blocks = [
        ("Вырез тарелки 640x640", C_SURFACE, C_COBALT_DARK),
        ("Сброс бликов L > 235", C_SURFACE, C_COBALT),
        ("Black-Hat, Sato, эллипс", C_SURFACE, C_REWORK),
        ("Хук --weights ONNX", C_COBALT_WASH, C_COBALT_DARK),
    ]
    for idx, (label, bg_col, bar_col) in enumerate(blocks):
        by = 1.65 + idx * 1.32
        add_rect(slide, LEFT_M, by, 4.35, 1.05, fill=bg_col, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, LEFT_M, by, 0.08, 1.05, fill=bar_col)
        add_textbox(
            slide,
            LEFT_M + 0.24,
            by + 0.10,
            3.95,
            0.85,
            label,
            font_name=FONT_SANS,
            font_size=13.0,
            color=C_INK,
            bold=True,
            valign=MSO_ANCHOR.MIDDLE,
        )
        if idx < 3:
            add_rect(slide, LEFT_M + 2.15, by + 1.05, 0.03, 0.27, fill=C_COBALT)

    tx = 5.35
    tw = 7.333
    table_shape = slide.shapes.add_table(5, 3, Inches(tx), Inches(1.65), Inches(tw), Inches(5.05))
    table = table_shape.table
    table.columns[0].width = Inches(2.55)
    table.columns[1].width = Inches(2.75)
    table.columns[2].width = Inches(2.033)

    headers = ["Дефект", "Норма ГОСТ", "Вердикт"]
    for c_idx, h_txt in enumerate(headers):
        style_table_cell(
            table.cell(0, c_idx),
            h_txt,
            font_name=FONT_MONO,
            font_size=10.5,
            color=C_INK_SEC,
            bold=True,
            fill_color=C_MUTED,
            align=PP_ALIGN.RIGHT if c_idx == 2 else PP_ALIGN.LEFT,
            border_bottom_hex="94A3B8",
        )

    rows = [
        ("Трещина, цек", "Запрещены", "REJECT", C_REJECT),
        ("Скол, перекос", "Выход кромки", "REJECT", C_REJECT),
        ("Плешина, натек", "Сбой слоя", "REWORK", C_REWORK),
        ("Накол, мушка", "До 2 точек", "PASS", C_PASS),
    ]
    for r_idx, (d_name, g_rule, verdict, v_col) in enumerate(rows, start=1):
        row_bg = C_SURFACE if r_idx % 2 == 1 else C_BG
        style_table_cell(table.cell(r_idx, 0), d_name, font_size=12.0, bold=True, fill_color=row_bg)
        style_table_cell(table.cell(r_idx, 1), g_rule, font_size=12.0, color=C_INK_SEC, fill_color=row_bg)
        style_table_cell(
            table.cell(r_idx, 2),
            verdict,
            font_name=FONT_MONO,
            font_size=11.0,
            color=v_col,
            bold=True,
            fill_color=row_bg,
            align=PP_ALIGN.RIGHT,
        )


def build_slide_5(prs: Presentation, bench: dict[str, Any]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)

    n_samples = int(bench["sample_count"])
    avg_ms = float(bench["avg_latency_ms"])
    p95_ms = float(bench["p95_latency_ms"])
    f1_val = float(bench["f1"])
    verdict_acc = float(bench["verdict_accuracy"]) * 100.0
    dataset_label = str(bench.get("dataset_label", "синтетический стенд"))

    add_chrome(
        slide,
        5,
        "1.3 БЕНЧМАРК",
        f"Стенд {n_samples} образцов: {avg_ms:.2f} мс на кадр, F1 {f1_val:.2f}",
        f"Стенд: {dataset_label}",
    )

    kpis = [
        (f"{avg_ms:.2f} мс", "среднее", C_COBALT_DARK),
        (f"{p95_ms:.2f} мс", "p95", C_COBALT),
        (f"{f1_val:.2f}", "F1", C_PASS),
        (f"{verdict_acc:.0f}%", "ГОСТ", C_INK),
    ]
    card_w = 2.83
    gap = 0.237
    for idx, (val, lbl, col) in enumerate(kpis):
        kx = LEFT_M + idx * (card_w + gap)
        add_rect(slide, kx, 1.62, card_w, 1.15, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, kx, 1.62, card_w, 0.05, fill=col)
        add_paragraphs(
            slide,
            kx + 0.16,
            1.74,
            card_w - 0.32,
            0.95,
            [
                (val, FONT_MONO, 20.0, col, True),
                (lbl, FONT_SANS, 11.5, C_INK_SEC, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=2.0,
        )

    per_class = bench.get("per_class", {})
    ordered_keys = [k for k in CLASS_ORDER if k in per_class]
    extra_keys = [k for k in per_class if k not in ordered_keys]
    all_keys = ordered_keys + extra_keys

    table_shape = slide.shapes.add_table(
        len(all_keys) + 1,
        3,
        Inches(LEFT_M),
        Inches(2.98),
        Inches(CONTENT_W),
        Inches(3.75),
    )
    table = table_shape.table
    table.columns[0].width = Inches(4.80)
    table.columns[1].width = Inches(2.20)
    table.columns[2].width = Inches(5.033)

    for c_idx, h_txt in enumerate(["Класс", "Шт", "P/R/F1"]):
        style_table_cell(
            table.cell(0, c_idx),
            h_txt,
            font_name=FONT_MONO,
            font_size=10.5,
            color=C_INK_SEC,
            bold=True,
            fill_color=C_MUTED,
            align=PP_ALIGN.RIGHT if c_idx > 0 else PP_ALIGN.LEFT,
            border_bottom_hex="94A3B8",
        )

    for r_idx, cls_key in enumerate(all_keys, start=1):
        stats = per_class[cls_key]
        row_bg = C_SURFACE if r_idx % 2 == 1 else C_BG
        cls_name = CLASS_LABELS_RU.get(cls_key, cls_key)
        support = int(stats["support"])
        p_val = float(stats["precision"])
        r_val = float(stats["recall"])
        f_val = float(stats["f1"])
        style_table_cell(table.cell(r_idx, 0), cls_name, font_size=11.5, bold=True, fill_color=row_bg)
        style_table_cell(
            table.cell(r_idx, 1),
            str(support),
            font_name=FONT_MONO,
            font_size=11.0,
            fill_color=row_bg,
            align=PP_ALIGN.RIGHT,
        )
        style_table_cell(
            table.cell(r_idx, 2),
            f"{p_val:.2f}/{r_val:.2f}/{f_val:.2f}",
            font_name=FONT_MONO,
            font_size=11.0,
            color=C_PASS_INK,
            bold=True,
            fill_color=row_bg,
            align=PP_ALIGN.RIGHT,
        )


def build_slide_6(prs: Presentation, overlays: list[Path]) -> set[Path]:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        6,
        "1.3 РАЗМЕТКА",
        "Детектор выделяет трещины и сколы поверх бликов глазури и кобальта",
        "Оверлеи: reports/overlays/ (1284x684)",
    )

    left_img = pick_overlay(overlays, [["GLAZED", "CRACK"], ["CRACK"]], fallback_idx=0)
    right_img = pick_overlay(
        overlays,
        [["BISQUE", "CHIP"], ["CHIP"], ["GLAZE"]],
        fallback_idx=1,
        avoid={left_img},
    )

    panel_w = 5.82
    img_h = panel_w / OVERLAY_ASPECT
    panels = [
        (
            LEFT_M,
            left_img,
            "Глазурь 1350°C: трещина поверх росписи (REJECT)",
            "Зеркальный блик и кобальт отсечены без ложных тревог, дефект локализован рамкой",
        ),
        (
            LEFT_M + panel_w + 0.393,
            right_img,
            "Утиль 900°C: скол кромки до обжига (REJECT)",
            "Выбой кромки от эллипса пойман до росписи и печи 1350°C",
        ),
    ]
    for px, img_path, head, desc in panels:
        add_rect(slide, px - 0.02, 1.63, panel_w + 0.04, img_h + 0.04, fill=C_SURFACE, border=C_HAIRLINE)
        slide.shapes.add_picture(str(img_path), Inches(px), Inches(1.65), width=Inches(panel_w))
        cap_y = 1.65 + img_h + 0.16
        cap_h = 6.75 - cap_y
        add_rect(slide, px, cap_y, panel_w, cap_h, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, px, cap_y, 0.07, cap_h, fill=C_REJECT)
        add_paragraphs(
            slide,
            px + 0.20,
            cap_y + 0.08,
            panel_w - 0.34,
            cap_h - 0.16,
            [
                (head, FONT_SANS, 12.5, C_REJECT, True),
                (desc, FONT_SANS, 11.5, C_INK, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=4.0,
        )
    return {left_img, right_img}


def build_slide_7(prs: Presentation, bench: dict[str, Any], overlays: list[Path], used: set[Path]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)

    auth = bench.get("auth_benchmark", {})
    pair_count = int(auth.get("pair_count", 0))
    auth_acc = float(auth.get("auth_accuracy", 0.0)) * 100.0
    genuine_match = float(auth.get("genuine_mean_match", 0.0))
    fake_match = float(auth.get("fake_mean_match", 0.0))

    add_chrome(
        slide,
        7,
        "1.3 ПАСПОРТ 289-ФЗ",
        f"Отпечаток мазка отличает ручную роспись от штамповки ({genuine_match:.2f}% против {fake_match:.2f}%)",
        f"Замер: auth_benchmark ({pair_count} пары)",
    )

    lw = 5.05
    add_rect(slide, LEFT_M, 1.65, lw, 1.75, fill=C_COBALT_WASH, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, LEFT_M, 1.65, 0.08, 1.75, fill=C_COBALT)
    add_paragraphs(
        slide,
        LEFT_M + 0.22,
        1.75,
        lw - 0.36,
        1.55,
        [
            ("0.50 ORB + 0.25 pHash + 0.25 Кобальт", FONT_MONO, 11.5, C_COBALT_DARK, True),
            ("Порог подлинника: >= 62% и >= 12 KP", FONT_SANS, 12.0, C_INK, False),
        ],
        valign=MSO_ANCHOR.MIDDLE,
        space_after_pt=8.0,
    )

    metrics = [
        (f"{pair_count}", "пары", C_INK),
        (f"{auth_acc:.0f}%", "точность", C_COBALT_DARK),
        (f"{genuine_match:.2f}%", "подлинник", C_PASS),
        (f"{fake_match:.2f}%", "подделка", C_REJECT),
    ]
    cw = 2.425
    ch = 1.48
    for idx, (val, lbl, col) in enumerate(metrics):
        col_i = idx % 2
        row_i = idx // 2
        mx = LEFT_M + col_i * (cw + 0.20)
        my = 3.60 + row_i * (ch + 0.17)
        add_rect(slide, mx, my, cw, ch, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
        add_paragraphs(
            slide,
            mx + 0.16,
            my + 0.12,
            cw - 0.32,
            ch - 0.24,
            [
                (val, FONT_MONO, 18.0, col, True),
                (lbl, FONT_SANS, 12.0, C_INK_SEC, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=2.0,
        )

    rx = 6.00
    rw = 6.683
    img_h = rw / OVERLAY_ASPECT
    auth_img = pick_overlay(overlays, [["FAKE"], ["PASS"]], fallback_idx=2, avoid=used)
    add_rect(slide, rx - 0.02, 1.63, rw + 0.04, img_h + 0.04, fill=C_SURFACE, border=C_HAIRLINE)
    slide.shapes.add_picture(str(auth_img), Inches(rx), Inches(1.65), width=Inches(rw))

    note_y = 1.65 + img_h + 0.15
    note_h = 6.75 - note_y
    add_rect(slide, rx, note_y, rw, note_h, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, rx, note_y, 0.08, note_h, fill=C_PASS)
    add_paragraphs(
        slide,
        rx + 0.22,
        note_y + 0.08,
        rw - 0.38,
        note_h - 0.16,
        [
            ("QR-код CERAMSCAN для Ozon/WB", FONT_SANS, 12.5, C_INK, True),
            ("Веб-интерфейс: Пост ОТК, Подлинность, Дашборд на FastAPI", FONT_SANS, 11.5, C_INK_SEC, False),
        ],
        valign=MSO_ANCHOR.MIDDLE,
        space_after_pt=3.0,
    )


def build_slide_8(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        8,
        "1.3 ЭКОНОМИКА",
        "Расчетная модель до пилота: отсев 1.11% брака дает окупаемость 2.6 мес",
        "Расчетная модель до пилота",
    )

    table_shape = slide.shapes.add_table(3, 2, Inches(LEFT_M), Inches(1.75), Inches(CONTENT_W), Inches(3.00))
    table = table_shape.table
    table.columns[0].width = Inches(3.50)
    table.columns[1].width = Inches(8.533)

    lpr_rows = [
        ("БФ «Система»", "Открытый код GitHub под пилот", C_COBALT_WASH),
        ("Объединение «Гжель»", "Отсев 900°C (130-140 тыс.), защита 289-ФЗ", C_SURFACE),
        ("МТС / АФК «Система»", "CPU < 300 мс, pytest, ONNX", C_COBALT_WASH),
    ]
    for r_idx, (lpr, val, bg_col) in enumerate(lpr_rows):
        style_table_cell(table.cell(r_idx, 0), lpr, font_size=13.5, color=C_COBALT_DARK, bold=True, fill_color=bg_col)
        style_table_cell(table.cell(r_idx, 1), val, font_size=13.5, color=C_INK, fill_color=bg_col)

    add_rect(slide, LEFT_M, 5.10, CONTENT_W, 1.62, fill=C_COBALT_WASH, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, LEFT_M, 5.10, CONTENT_W, 0.06, fill=C_COBALT)
    econ = [
        ("TCO 300 тыс. руб", "за год"),
        ("2.22 руб/изд", "удельно"),
        ("2.6 мес", "окупаемость"),
        ("1.11%", "порог"),
    ]
    ew = 2.85
    egap = 0.14
    for idx, (val, lbl) in enumerate(econ):
        ex = LEFT_M + 0.18 + idx * (ew + egap)
        add_paragraphs(
            slide,
            ex,
            5.25,
            ew,
            1.32,
            [
                (val, FONT_MONO, 15.5, C_COBALT_DARK, True),
                (lbl, FONT_SANS, 12.0, C_INK_SEC, False),
            ],
            valign=MSO_ANCHOR.MIDDLE,
            space_after_pt=4.0,
        )


def build_slide_9(prs: Presentation, bench: dict[str, Any]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    avg_ms = float(bench["avg_latency_ms"])
    add_chrome(
        slide,
        9,
        "1.4 ВЫВОДЫ",
        "Выводы: прототип подтвержден на CPU и готов к фотопотоку завода",
        "Итог: стенд v0.1",
    )

    add_rect(slide, LEFT_M, 1.68, CONTENT_W, 1.15, fill=C_PASS_BG, border=C_HAIRLINE, border_pt=1.0)
    add_rect(slide, LEFT_M, 1.68, 0.08, 1.15, fill=C_PASS)
    add_textbox(
        slide,
        LEFT_M + 0.26,
        1.75,
        CONTENT_W - 0.44,
        1.00,
        f"Ядро на CPU ({avg_ms:.2f} мс), тесты pytest, ГОСТ 28390-89 и паспорт 289-ФЗ работают без GPU",
        font_name=FONT_SANS,
        font_size=14.0,
        color=C_PASS_INK,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )

    add_rect(slide, LEFT_M, 3.45, CONTENT_W, 0.03, fill=C_HAIRLINE)
    milestones = [
        ("Ноябрь 2026", "20-30 фото цеха в ingest_real_photos.py", C_COBALT),
        ("Q1 2027", "Пилот утиля 900°C на Гжели", C_COBALT_DARK),
        ("Q2 2027", "Калибровка весов\nONNX --weights", C_PASS),
        ("2027", "QR-паспорта партий для Ozon/WB", C_INK),
    ]
    mw = 2.82
    mgap = 0.251
    for idx, (date_txt, task_txt, col) in enumerate(milestones):
        mx = LEFT_M + idx * (mw + mgap)
        add_rect(slide, mx, 3.38, 0.22, 0.16, fill=col)
        add_rect(slide, mx, 3.74, mw, 2.55, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
        add_rect(slide, mx, 3.74, mw, 0.06, fill=col)
        add_paragraphs(
            slide,
            mx + 0.18,
            3.96,
            mw - 0.36,
            2.15,
            [
                (date_txt, FONT_MONO, 15.0, col, True),
                (task_txt, FONT_SANS, 12.5, C_INK, False),
            ],
            space_after_pt=10.0,
        )


def build_slide_10(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_BG)
    add_chrome(
        slide,
        10,
        "1.5 ИСТОЧНИКИ",
        "Нормативная база, стек прототипа и контакты автора",
        "Стипендия БФ «Система»",
    )

    add_rect(slide, LEFT_M, 1.65, 7.25, 5.05, fill=C_SURFACE, border=C_HAIRLINE, border_pt=1.0)
    sources = [
        "1. ГОСТ 28390-89: допуски дефектов фарфора",
        "2. 289-ФЗ: платформенная экономика с 01.10.2026",
        "3. ПП РФ № 1374: маркировка «Честный ЗНАК»",
        "4. Положение БФ «Система» 2026",
        "5. OpenCV, scikit-image, FastAPI, ONNX Runtime",
    ]
    for idx, src_txt in enumerate(sources):
        sy = 1.82 + idx * 0.96
        add_textbox(
            slide,
            LEFT_M + 0.24,
            sy,
            6.77,
            0.78,
            src_txt,
            font_name=FONT_SANS,
            font_size=13.0,
            color=C_INK,
            valign=MSO_ANCHOR.MIDDLE,
        )
        if idx < 4:
            add_rect(slide, LEFT_M + 0.24, sy + 0.86, 6.77, 0.015, fill=C_HAIRLINE)

    cx = 8.22
    cw = 4.463
    add_rect(slide, cx, 1.65, cw, 5.05, fill=C_COBALT_DARK)
    add_paragraphs(
        slide,
        cx + 0.28,
        1.95,
        cw - 0.56,
        4.45,
        [
            ("Павловец Александр Иванович", FONT_SANS, 16.0, C_WHITE, True),
            ("Гатчинский Государственный университет", FONT_SANS, 12.0, C_COBALT_LIGHT, False),
            ("1 курс, г. Гатчина", FONT_SANS, 12.0, C_COBALT_LIGHT, False),
            ("+79052010295", FONT_MONO, 13.5, C_WHITE, True),
            ("pavlovec000@gmail.com", FONT_MONO, 12.5, C_COBALT_LIGHT, False),
            ("github.com/axiom-stat/testing", FONT_MONO, 11.5, C_WHITE, True),
        ],
        valign=MSO_ANCHOR.MIDDLE,
        space_after_pt=12.0,
    )


def build_presentation(output_path: Path = OUTPUT_PPTX_PATH) -> Path:
    bench = load_benchmark()
    overlays = collect_overlay_paths(bench)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)

    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs, bench)
    used = build_slide_6(prs, overlays)
    build_slide_7(prs, bench, overlays, used)
    build_slide_8(prs)
    build_slide_9(prs, bench)
    build_slide_10(prs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    return output_path


def main() -> int:
    out = build_presentation()
    print(f"Generated {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
