from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

EM_DASH = "\u2014"
RU_WORD_RE = re.compile(r"[а-яёА-ЯЁ]{2,}")
NOMINAL_RE = re.compile(
    r"\b[а-яё]{4,}(?:ание|ение|ания|ения|анию|ению|анием|ением|ании|ении)\b",
    re.IGNORECASE,
)
YAV_RE = re.compile(r"\bявля[ею]тся\b", re.IGNORECASE)
BOLD_COLON_BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*[^*]{2,50}:\*\*\s+", re.MULTILINE)
BANNED_RU_AI_RE = re.compile(
    r"(в современном мире|в условиях быстро меняющейся|на сегодняшний день|"
    r"стоит отметить,\s*что|важно (?:понимать|помнить|отметить),\s*что|"
    r"можно с уверенностью сказать|подводя итог|раскрыть потенциал|"
    r"вывести на новый уровень|открывает новые горизонты|играет (?:важную|ключевую) роль)",
    re.IGNORECASE,
)
TEAM_VOICE_RE = re.compile(r"\b(мы с командой|наша команда|нашей команды|нашей командой)\b", re.IGNORECASE)
IGNORED_WALK_PARTS = frozenset({".venv", "venv", ".agents", ".git", "__pycache__", ".pytest_cache"})


def _convert_to_pdf(src_path: Path, out_dir: Path) -> Path:
    proc = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", str(src_path), "--outdir", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"soffice failed: {proc.stdout}\n{proc.stderr}"
    pdf_path = out_dir / f"{src_path.stem}.pdf"
    assert pdf_path.is_file(), f"Expected PDF not generated: {pdf_path}"
    return pdf_path


def _render_pdf_to_pngs(pdf_path: Path, out_dir: Path, prefix: str) -> list[Path]:
    out_prefix = out_dir / prefix
    proc = subprocess.run(
        ["pdftoppm", "-png", "-r", "150", str(pdf_path), str(out_prefix)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"pdftoppm failed: {proc.stdout}\n{proc.stderr}"
    pngs = sorted(out_dir.glob(f"{prefix}-*.png"))
    assert len(pngs) >= 1, f"No PNG pages rendered from {pdf_path}"
    return pngs


def _extract_docx_text(docx_path: Path) -> str:
    import docx

    document = docx.Document(str(docx_path))
    chunks: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    chunks.append(cell.text)
    return "\n".join(chunks)


def _extract_pptx_slide_texts(pptx_path: Path) -> list[str]:
    from pptx import Presentation

    prs = Presentation(str(pptx_path))
    slide_texts: list[str] = []
    for slide in prs.slides:
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = shape.text_frame.text.strip()
                if txt:
                    parts.append(txt)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        cell_txt = cell.text.strip()
                        if cell_txt:
                            parts.append(cell_txt)
        slide_texts.append("\n".join(parts))
    return slide_texts


def _format_variants(value: float) -> tuple[str, ...]:
    candidates = {
        f"{value:.1f}",
        f"{value:.2f}",
        f"{value:.1f}".replace(".", ","),
        f"{value:.2f}".replace(".", ","),
        str(int(round(value))),
    }
    return tuple(candidates)


def _iter_repo_text_files(repo_root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_WALK_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return files


def _check_ru_stylometry(text: str, label: str) -> None:
    assert EM_DASH not in text, f"Found U+2014 in {label}"
    assert not BANNED_RU_AI_RE.search(text), f"Found banned RU AI phrase in {label}"
    ru_words = RU_WORD_RE.findall(text)
    assert len(ru_words) > 0, f"Expected Russian words in {label}"
    nom_count = len(NOMINAL_RE.findall(text))
    assert (nom_count / len(ru_words)) < 0.05, (
        f"Nominalization ratio too high in {label}: {nom_count}/{len(ru_words)}"
    )
    yav_count = len(YAV_RE.findall(text))
    allowed_yav = max(1, len(ru_words) // 500)
    assert yav_count <= allowed_yav, (
        f"'является/являются' used {yav_count}x in {len(ru_words)} words in {label} (max {allowed_yav})"
    )


# =============================================================================
# Tier 1: Feature Coverage (24 test cases: 6 F12 + 6 F13 + 6 F14 + 6 F15)
# =============================================================================


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_project_description_author_identity(repo_root) -> None:
    text = (repo_root / "grant_case" / "project_description.md").read_text(encoding="utf-8")
    for token in (
        "Павловец Александр Иванович",
        "Гатчинск",
        "1 курс",
        "+79052010295",
        "pavlovec000@gmail.com",
        "экономика предприятий и организаций",
    ):
        assert token in text, f"Missing author identity token {token!r}"
    assert "Лифт в будущее" in text or "Система" in text


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_lpr_three_groups_coverage(repo_root) -> None:
    text = (repo_root / "grant_case" / "project_description.md").read_text(encoding="utf-8")
    for token in (
        "Пастухова",
        "Ксенофонтова",
        "Гжель",
        "130",
        "140",
        "900",
        "1350",
        "28390-89",
        "289-ФЗ",
        "Ozon",
        "WB",
        "300",
        "pytest",
    ):
        assert token in text, f"Missing LPR token {token!r}"
    assert "ONNX" in text or "--weights" in text
    assert "синтетический" in text.lower() or "до пилота" in text.lower()


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_docx_exists_and_appendix2_sections(repo_root) -> None:
    docx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.docx"
    assert docx_path.is_file() and docx_path.stat().st_size > 10_000

    text = _extract_docx_text(docx_path)
    text_lower = text.lower()
    for required in (
        "технологии в креативных индустриях и народных промыслах",
        "содержание",
        "введение",
        "заключение",
        "список использованных источников",
        "импортозамещен",
        "аналог",
        "масштабирован",
        "апроб",
    ):
        assert required in text_lower, f"Missing Appendix 2 section/topic: {required!r}"


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_docx_typography_tables_figures_and_pages(repo_root, tmp_path) -> None:
    import docx
    from pypdf import PdfReader

    docx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.docx"
    document = docx.Document(str(docx_path))

    assert len(document.tables) >= 1
    assert len(document.inline_shapes) >= 1

    with zipfile.ZipFile(docx_path, "r") as zf:
        styles_xml = zf.read("word/styles.xml").decode("utf-8", errors="replace")
        doc_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")

    normal_font = document.styles["Normal"].font
    has_tnr = normal_font.name == "Times New Roman" or "Times New Roman" in styles_xml
    has_12pt = (normal_font.size is not None and round(normal_font.size.pt) == 12) or ('w:val="24"' in styles_xml) or ('w:val="24"' in doc_xml)
    has_single = (document.styles["Normal"].paragraph_format.line_spacing == 1.0) or ('w:line="240"' in styles_xml) or ('w:line="240"' in doc_xml)
    assert has_tnr, "Times New Roman font not detected in .docx"
    assert has_12pt, "12 pt font size not detected in .docx"
    assert has_single, "1.0 line spacing not detected in .docx"

    pdf_path = _convert_to_pdf(docx_path, tmp_path)
    reader = PdfReader(str(pdf_path))
    assert 1 <= len(reader.pages) <= 15, f"DOCX PDF page count out of range: {len(reader.pages)}"


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_build_docx_script_executes(repo_root, venv_python) -> None:
    script_path = repo_root / "grant_case" / "build_docx.py"
    proc = subprocess.run(
        [str(venv_python), str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"build_docx.py failed: {proc.stdout}\n{proc.stderr}"
    out_docx = repo_root / "grant_case" / "CeramScan_Sistema_2026.docx"
    assert out_docx.is_file() and out_docx.stat().st_size > 10_000


@pytest.mark.tier1
@pytest.mark.f12
def test_t1_f12_docx_md_stylometry_clean(repo_root, slop_scanner_cmd) -> None:
    md_path = repo_root / "grant_case" / "project_description.md"
    proc = subprocess.run(
        [*slop_scanner_cmd, str(md_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, f"slop_scan.py failed:\n{proc.stdout}\n{proc.stderr}"
    _check_ru_stylometry(md_path.read_text(encoding="utf-8"), "project_description.md")


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_build_presentation_executes(repo_root, venv_python) -> None:
    script_path = repo_root / "grant_case" / "build_presentation.py"
    proc = subprocess.run(
        [str(venv_python), str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"build_presentation.py failed: {proc.stdout}\n{proc.stderr}"
    out_pptx = repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"
    assert out_pptx.is_file() and out_pptx.stat().st_size > 20_000


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_pptx_exactly_10_slides_16_9(repo_root, tmp_path) -> None:
    from pptx import Presentation
    from pypdf import PdfReader

    pptx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"
    prs = Presentation(str(pptx_path))
    assert len(prs.slides) == 10
    ratio = prs.slide_width / prs.slide_height
    assert abs(ratio - (16.0 / 9.0)) < 0.02

    pdf_path = _convert_to_pdf(pptx_path, tmp_path)
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) == 10


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_pptx_appendix3_structure_and_slide1(repo_root) -> None:
    slide_texts = _extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx")
    assert len(slide_texts) == 10

    s1 = slide_texts[0]
    assert "CeramScan" in s1 or "КерамикСкан" in s1
    for token in ("Павловец", "Гатчинск", "экономика", "1 курс"):
        assert token in s1, f"Missing Slide 1 metadata: {token!r}"

    s9 = slide_texts[8].lower()
    assert "выводы" in s9 or "итоги" in s9

    s10 = slide_texts[9]
    for token in ("28390-89", "289-ФЗ", "pavlovec000@gmail.com"):
        assert token in s10, f"Missing Slide 10 token: {token!r}"


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_pptx_embeds_real_png_overlays(repo_root) -> None:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    pptx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"
    prs = Presentation(str(pptx_path))
    picture_count = sum(
        1
        for slide in prs.slides
        for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    )
    assert picture_count >= 2

    with zipfile.ZipFile(pptx_path, "r") as zf:
        media_pngs = [name for name in zf.namelist() if name.startswith("ppt/media/") and name.lower().endswith(".png")]
        assert len(media_pngs) >= 2
        for media_name in media_pngs:
            payload = zf.read(media_name)
            assert len(payload) > 5_000
            assert payload[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_pptx_benchmark_numeric_parity(repo_root) -> None:
    bench = json.loads((repo_root / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    all_text = "\n".join(_extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))

    assert str(bench["sample_count"]) in all_text
    assert any(v in all_text for v in _format_variants(float(bench["avg_latency_ms"])))
    acc_or_f1 = _format_variants(float(bench["f1"]) * 100.0) + _format_variants(float(bench["f1"])) + _format_variants(float(bench["verdict_accuracy"]) * 100.0) + _format_variants(float(bench["verdict_accuracy"]))
    assert any(v in all_text for v in acc_or_f1)


@pytest.mark.tier1
@pytest.mark.f13
def test_t1_f13_pptx_word_budget_and_no_em_dash(repo_root) -> None:
    from pptx import Presentation

    prs = Presentation(str(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    slide_texts = _extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx")
    word_counts: list[int] = []
    for idx, (slide, txt) in enumerate(zip(prs.slides, slide_texts), start=1):
        assert EM_DASH not in txt, f"U+2014 found on slide {idx}"
        if slide.has_notes_slide:
            assert EM_DASH not in slide.notes_slide.notes_text_frame.text
        wc = len(re.findall(r"\S+", txt))
        assert wc <= 85, f"Slide {idx} exceeds 85 words: {wc}"
        word_counts.append(wc)

    body_mean = sum(word_counts[1:9]) / 8.0
    assert 25.0 <= body_mean <= 70.0, f"Slides 2..9 mean word count {body_mean:.1f} outside 25..70"


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_pitch_script_90s_and_7min_sections(repo_root) -> None:
    text = (repo_root / "grant_case" / "pitch_and_video_script.md").read_text(encoding="utf-8")
    assert "90" in text
    assert "7" in text
    for lpr_token in ("БФ «Система»", "Гжель", "МТС"):
        assert lpr_token in text, f"Missing LPR reference {lpr_token!r} in pitch_and_video_script.md"


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_pitch_12_jury_qas(repo_root) -> None:
    text = (repo_root / "grant_case" / "pitch_and_video_script.md").read_text(encoding="utf-8")
    qa_blocks = re.split(r"(?m)^(?:#{2,4}\s*|\*\*)?(?:Вопрос|Q)\s*\d+[\.:)]", text)
    answers = [blk.strip() for blk in qa_blocks[1:] if len(blk.split()) >= 15]
    assert len(answers) >= 12, f"Expected >= 12 Q&As with >= 15 words, found {len(answers)}"


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_pilot_letter_template_complete(repo_root) -> None:
    text = (repo_root / "grant_case" / "pilot_letter_template.md").read_text(encoding="utf-8")
    for token in ("Гжель", "900", "1350", "28390-89", "289-ФЗ", "Павловец", "+79052010295", "pavlovec000@gmail.com"):
        assert token in text, f"Missing token {token!r} in pilot_letter_template.md"


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_refinements_roadmap_nov30_cli(repo_root) -> None:
    text = (repo_root / "grant_case" / "REFINEMENTS_ROADMAP.md").read_text(encoding="utf-8")
    for token in (
        "30 ноября",
        "20",
        "30",
        "ingest_real_photos.py",
        "--input data/real_photos",
        "--rebuild-all",
        "pass",
        "chip",
        "crack",
        "glaze",
        "fake",
    ):
        assert token in text, f"Missing roadmap token {token!r}"


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_first_person_author_and_synthetic_qualifiers(repo_root) -> None:
    for name in ("pitch_and_video_script.md", "pilot_letter_template.md", "REFINEMENTS_ROADMAP.md"):
        text = (repo_root / "grant_case" / name).read_text(encoding="utf-8")
        assert not TEAM_VOICE_RE.search(text), f"Team voice detected in {name}"
    pitch = (repo_root / "grant_case" / "pitch_and_video_script.md").read_text(encoding="utf-8").lower()
    roadmap = (repo_root / "grant_case" / "REFINEMENTS_ROADMAP.md").read_text(encoding="utf-8").lower()
    assert "синтетическ" in pitch or "до пилота" in pitch
    assert "синтетическ" in roadmap or "до пилота" in roadmap


@pytest.mark.tier1
@pytest.mark.f14
def test_t1_f14_slop_scan_clean(repo_root, slop_scanner_cmd) -> None:
    targets = [
        str(repo_root / "grant_case" / "pitch_and_video_script.md"),
        str(repo_root / "grant_case" / "pilot_letter_template.md"),
        str(repo_root / "grant_case" / "REFINEMENTS_ROADMAP.md"),
    ]
    proc = subprocess.run(
        [*slop_scanner_cmd, "--ignore-inline-code", *targets],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, f"slop_scan.py failed on F14 docs:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_readme_author_synthetic_and_github_remote(repo_root) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    for token in (
        "CeramScan",
        "Павловец Александр Иванович",
        "git@github.com:axiom-stat/testing.git",
        "git push -u origin main",
    ):
        assert token in text, f"Missing README token {token!r}"
    assert "синтетический" in text.lower() or "до пилота" in text.lower()


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_readme_cli_commands_and_paths_exist(repo_root) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    expected_paths = (
        "scripts/generate_synthetic_ceramic.py",
        "scripts/run_benchmark.py",
        "scripts/ingest_real_photos.py",
        "src/api.py",
        "grant_case/build_docx.py",
        "grant_case/build_presentation.py",
    )
    for rel in expected_paths:
        assert rel in text, f"README missing path reference: {rel}"
        assert (repo_root / rel).is_file(), f"Referenced file missing on disk: {rel}"


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_gitignore_required_exclusions(repo_root) -> None:
    text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    required_patterns = (
        ".venv/",
        "__pycache__/",
        "*.db",
        "*.db-wal",
        "*.db-shm",
        ".pytest_cache/",
        ".agents/",
        "tmp_*.png",
        "reports/overlays/runtime/",
        "reports/passports/",
        "reports/qr/",
        "data/real_photos/**",
    )
    for pattern in required_patterns:
        assert pattern in text, f"Missing .gitignore rule: {pattern}"


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_gitignore_verified_in_isolated_git_repo(repo_root, tmp_path) -> None:
    ws = tmp_path / "git_ws"
    ws.mkdir()
    shutil.copy2(repo_root / ".gitignore", ws / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)

    ignored_samples = (
        ".venv/bin/python",
        "__pycache__/mod.pyc",
        "data/ceramscan.db",
        "tmp_preview.png",
        "reports/qr/GZHEL-1.png",
    )
    for rel in ignored_samples:
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        chk = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ws, check=False)
        assert chk.returncode == 0, f"Path should be ignored by .gitignore: {rel}"

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ws,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == "?? .gitignore"


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_slop_scan_full_deliverables(repo_root, slop_scanner_cmd) -> None:
    targets = [
        repo_root / "src" / "web",
        repo_root / "grant_case",
        repo_root / "README.md",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests" / "e2e",
        repo_root / "TEST_INFRA.md",
    ]
    for target in targets:
        assert target.exists(), f"Missing deliverable target for slop_scan: {target}"

    proc = subprocess.run(
        [*slop_scanner_cmd, *(str(p) for p in targets)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert proc.returncode == 0, f"slop_scan.py found issues:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.tier1
@pytest.mark.f15
def test_t1_f15_no_forbidden_external_project_refs(repo_root) -> None:
    forbidden = (
        "avito" + "_bot",
        "ducknoir" + "-tg",
        "facevs" + "-landing",
        "calccore" + "-platform",
    )
    files = _iter_repo_text_files(repo_root, (".py", ".md", ".html", ".css", ".js"))
    assert len(files) >= 1
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace").lower()
        for banned in forbidden:
            assert banned not in content, f"Forbidden reference {banned!r} in {path}"


# =============================================================================
# Tier 2: Boundary & Corner Cases (24 test cases: 6 F12 + 6 F13 + 6 F14 + 6 F15)
# =============================================================================


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_rebuild_idempotent_under_15_pages(repo_root, venv_python, tmp_path) -> None:
    from pypdf import PdfReader

    grant_dir = repo_root / "grant_case"
    before_names = {p.name for p in grant_dir.iterdir()}
    proc = subprocess.run(
        [str(venv_python), str(grant_dir / "build_docx.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"build_docx.py failed: {proc.stdout}\n{proc.stderr}"
    after_names = {p.name for p in grant_dir.iterdir()}
    stray = (after_names - before_names) - {"CeramScan_Sistema_2026.docx"}
    assert stray == set(), f"Stray files created in grant_case/: {stray}"

    pdf_path = _convert_to_pdf(grant_dir / "CeramScan_Sistema_2026.docx", tmp_path)
    reader = PdfReader(str(pdf_path))
    assert 1 <= len(reader.pages) <= 15


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_extracted_text_zero_slop(repo_root) -> None:
    text = _extract_docx_text(repo_root / "grant_case" / "CeramScan_Sistema_2026.docx")
    _check_ru_stylometry(text, "CeramScan_Sistema_2026.docx")


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_inline_figures_and_numbered_tables(repo_root) -> None:
    import io
    from PIL import Image

    docx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.docx"
    text = _extract_docx_text(docx_path)
    assert re.search(r"Таблица\s+\d+", text)
    assert re.search(r"Рисунок\s+\d+", text)

    with zipfile.ZipFile(docx_path, "r") as zf:
        media_imgs = [n for n in zf.namelist() if n.startswith("word/media/")]
        assert len(media_imgs) >= 1
        for name in media_imgs:
            raw = zf.read(name)
            assert len(raw) > 2_000
            img = Image.open(io.BytesIO(raw))
            img.verify()


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_page_numbering_xml_present(repo_root) -> None:
    docx_path = repo_root / "grant_case" / "CeramScan_Sistema_2026.docx"
    with zipfile.ZipFile(docx_path, "r") as zf:
        hf_files = [
            n
            for n in zf.namelist()
            if n.startswith("word/footer") or n.startswith("word/header")
        ]
        assert len(hf_files) >= 1, "No header/footer XML parts found in .docx"
        combined_xml = "\n".join(zf.read(n).decode("utf-8", errors="replace") for n in hf_files)
    assert "PAGE" in combined_xml


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_no_team_voice_and_no_bold_colon_bullets(repo_root) -> None:
    md_text = (repo_root / "grant_case" / "project_description.md").read_text(encoding="utf-8")
    docx_text = _extract_docx_text(repo_root / "grant_case" / "CeramScan_Sistema_2026.docx")
    assert not TEAM_VOICE_RE.search(md_text)
    assert not TEAM_VOICE_RE.search(docx_text)
    assert not BOLD_COLON_BULLET_RE.search(md_text)


@pytest.mark.tier2
@pytest.mark.f12
def test_t2_f12_docx_libreoffice_pdftoppm_nonblank_pages(repo_root, tmp_path) -> None:
    import numpy as np
    from PIL import Image

    pdf_path = _convert_to_pdf(repo_root / "grant_case" / "CeramScan_Sistema_2026.docx", tmp_path)
    pngs = _render_pdf_to_pngs(pdf_path, tmp_path, "docx_page")
    assert 1 <= len(pngs) <= 15
    for png in pngs:
        assert png.stat().st_size > 5_000
        arr = np.asarray(Image.open(png).convert("L"))
        assert float(arr.std()) > 5.0, f"Blank DOCX page rendered: {png}"


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_slide_count_exact_10(repo_root) -> None:
    from pptx import Presentation

    prs = Presentation(str(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    assert len(prs.slides) == 10


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_slide_word_bounds_and_nonempty(repo_root) -> None:
    slide_texts = _extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx")
    assert len(slide_texts) == 10
    for idx, txt in enumerate(slide_texts, start=1):
        wc = len(re.findall(r"\S+", txt))
        assert 15 <= wc <= 85, f"Slide {idx} word count {wc} outside 15..85"
        assert "TBD" not in txt and "Lorem" not in txt


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_text_zero_slop_and_no_heading_periods(repo_root) -> None:
    from pptx import Presentation

    prs = Presentation(str(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    for idx, slide in enumerate(prs.slides, start=1):
        if slide.shapes.title is not None:
            title_txt = slide.shapes.title.text.strip()
            assert not title_txt.endswith("."), f"Slide {idx} title ends with period: {title_txt!r}"
        slide_txt = "\n".join(
            s.text_frame.text.strip() for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()
        )
        assert EM_DASH not in slide_txt, f"U+2014 on slide {idx}"
        assert not BANNED_RU_AI_RE.search(slide_txt), f"Banned RU AI phrase on slide {idx}"


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_shapes_within_bounds_and_no_severe_overlap(repo_root) -> None:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(str(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    sw = prs.slide_width
    sh = prs.slide_height

    for idx, slide in enumerate(prs.slides, start=1):
        text_boxes: list[tuple[int, int, int, int]] = []
        for shape in slide.shapes:
            is_text = shape.has_text_frame and bool(shape.text_frame.text.strip())
            is_pic = shape.shape_type == MSO_SHAPE_TYPE.PICTURE
            if not (is_text or is_pic):
                continue
            left = int(shape.left)
            top = int(shape.top)
            right = left + int(shape.width)
            bottom = top + int(shape.height)
            assert left >= -10_000, f"Slide {idx} shape clipped left: {left}"
            assert top >= -10_000, f"Slide {idx} shape clipped top: {top}"
            assert right <= sw + 50_000, f"Slide {idx} shape exceeds width: {right} > {sw}"
            assert bottom <= sh + 50_000, f"Slide {idx} shape exceeds height: {bottom} > {sh}"
            if is_text:
                text_boxes.append((left, top, right, bottom))

        for pos, box_a in enumerate(text_boxes):
            area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            for box_b in text_boxes[pos + 1 :]:
                area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
                ix = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
                iy = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
                overlap_ratio = (ix * iy) / min(area_a, area_b)
                assert overlap_ratio <= 0.15, f"Slide {idx} text box overlap {overlap_ratio:.2f} > 0.15"


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_no_duplicate_layout_claims_or_broken_charts(repo_root) -> None:
    from pptx import Presentation

    prs = Presentation(str(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    for idx, slide in enumerate(prs.slides, start=1):
        assert len(slide.shapes) >= 2, f"Slide {idx} has fewer than 2 shapes"
        for shape in slide.shapes:
            assert shape.width > 0 and shape.height > 0, f"Zero-size shape on slide {idx}"


@pytest.mark.tier2
@pytest.mark.f13
def test_t2_f13_pptx_libreoffice_pdftoppm_10_nonblank_pngs(repo_root, tmp_path) -> None:
    import numpy as np
    from PIL import Image

    pdf_path = _convert_to_pdf(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx", tmp_path)
    pngs = _render_pdf_to_pngs(pdf_path, tmp_path, "slide")
    assert len(pngs) == 10
    for png in pngs:
        assert png.stat().st_size > 5_000
        arr = np.asarray(Image.open(png).convert("L"))
        assert float(arr.std()) > 5.0, f"Blank PPTX slide rendered: {png}"


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_pitch_exact_12_plus_numbered_qas(repo_root) -> None:
    text = (repo_root / "grant_case" / "pitch_and_video_script.md").read_text(encoding="utf-8")
    matches = re.findall(r"(?m)^(?:#{2,4}\s*|\*\*)?(?:Вопрос|Q)\s*(\d+)[\.:)]", text)
    assert len(set(matches)) >= 12


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_pitch_defense_under_10_min_limit(repo_root) -> None:
    text = (repo_root / "grant_case" / "pitch_and_video_script.md").read_text(encoding="utf-8")
    assert re.search(r"7[\s-]*мин", text, re.IGNORECASE)
    assert re.search(r"90[\s-]*сек", text, re.IGNORECASE)


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_pilot_letter_no_unfilled_english_slop(repo_root) -> None:
    text = (repo_root / "grant_case" / "pilot_letter_template.md").read_text(encoding="utf-8")
    for placeholder in ("John Doe", "Jane Doe", "Acme", "TBD", "Lorem"):
        assert placeholder not in text
    assert len(RU_WORD_RE.findall(text)) >= 50


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_roadmap_matches_real_photos_disk_dirs(repo_root) -> None:
    text = (repo_root / "grant_case" / "REFINEMENTS_ROADMAP.md").read_text(encoding="utf-8")
    real_dir = repo_root / "data" / "real_photos"
    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        assert cat in text
        assert (real_dir / cat / ".gitkeep").is_file(), f"Missing .gitkeep in {cat}"


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_f14_files_zero_em_dash_and_nominalization_lt_05(repo_root) -> None:
    for name in ("pitch_and_video_script.md", "pilot_letter_template.md", "REFINEMENTS_ROADMAP.md"):
        content = (repo_root / "grant_case" / name).read_text(encoding="utf-8")
        _check_ru_stylometry(content, name)


@pytest.mark.tier2
@pytest.mark.f14
def test_t2_f14_no_bold_colon_lists_in_f14_docs(repo_root) -> None:
    for name in ("pitch_and_video_script.md", "pilot_letter_template.md", "REFINEMENTS_ROADMAP.md"):
        content = (repo_root / "grant_case" / name).read_text(encoding="utf-8")
        assert not BOLD_COLON_BULLET_RE.search(content), f"Bold-colon list found in {name}"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_slop_scan_each_directory_independently(repo_root, slop_scanner_cmd) -> None:
    for rel in ("src/web", "grant_case", "README.md", "src", "scripts", "tests/e2e"):
        target = repo_root / rel
        assert target.exists(), f"Target missing: {target}"
        proc = subprocess.run(
            [*slop_scanner_cmd, str(target)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert proc.returncode == 0, f"slop_scan.py failed on {rel}:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_repo_wide_zero_em_dash(repo_root) -> None:
    files = _iter_repo_text_files(repo_root, (".py", ".md", ".html", ".css", ".js"))
    assert len(files) >= 1
    offenders = [str(p.relative_to(repo_root)) for p in files if EM_DASH in p.read_text(encoding="utf-8", errors="replace")]
    assert offenders == [], f"Files containing U+2014: {offenders}"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_no_secrets_or_private_keys_in_repo(repo_root) -> None:
    secret_patterns = (
        re.compile(r"BEGIN [A-Z0-9 ]*PRIVATE KEY"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b"),
        re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    )
    files = _iter_repo_text_files(repo_root, (".py", ".md", ".html", ".css", ".js", ".json", ".txt"))
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        for pat in secret_patterns:
            assert not pat.search(content), f"Potential secret pattern {pat.pattern!r} in {path}"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_gitignore_preserves_gitkeep_and_ignores_wal_runtime(repo_root, tmp_path) -> None:
    ws = tmp_path / "git_boundary"
    ws.mkdir()
    shutil.copy2(repo_root / ".gitignore", ws / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)

    keep_path = ws / "data" / "real_photos" / "pass" / ".gitkeep"
    keep_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path.write_text("", encoding="utf-8")
    keep_chk = subprocess.run(["git", "check-ignore", "-q", "data/real_photos/pass/.gitkeep"], cwd=ws, check=False)
    assert keep_chk.returncode == 1, ".gitkeep must NOT be ignored"

    must_ignore = (
        "data/real_photos/pass/img.jpg",
        "data/ceramscan.db-wal",
        "data/ceramscan.db-shm",
        "reports/overlays/runtime/a.png",
        "reports/passports/a.png",
        "reports/qr/a.png",
        "tmp_foo.png",
    )
    for rel in must_ignore:
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        chk = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ws, check=False)
        assert chk.returncode == 0, f"Artifact should be ignored: {rel}"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_git_repo_cleanliness_safe_pre_or_post_commit(repo_root, tmp_path) -> None:
    ws = tmp_path / "git_selfcheck"
    ws.mkdir()
    shutil.copy2(repo_root / ".gitignore", ws / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    assert subprocess.run(["git", "check-ignore", "-q", ".venv/x"], cwd=ws, check=False).returncode == 0
    assert subprocess.run(["git", "check-ignore", "-q", "data/x.db"], cwd=ws, check=False).returncode == 0

    if (repo_root / ".git").is_dir():
        ls_proc = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked = ls_proc.stdout.splitlines()
        banned_prefixes = (".venv/", "__pycache__/", ".agents/", "reports/qr/", "reports/passports/", "reports/overlays/runtime/")
        for item in tracked:
            assert not item.startswith(banned_prefixes), f"Forbidden tracked artifact: {item}"
            assert not item.endswith(".db"), f"Tracked SQLite DB: {item}"

        remote_proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if remote_proc.returncode == 0:
            assert remote_proc.stdout.strip() == "git@github.com:axiom-stat/testing.git"


@pytest.mark.tier2
@pytest.mark.f15
def test_t2_f15_readme_relative_paths_resolve(repo_root) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    raw_candidates = re.findall(
        r"`((?:src|scripts|grant_case|reports|data)/[A-Za-z0-9_./*<>-]+|requirements\.txt|PROJECT\.md|ARCHITECTURE\.md|DESIGN_SPEC\.md)`",
        text,
    )
    ignored_prefixes = (
        "data/real_photos/",
        "reports/overlays/runtime/",
        "reports/passports/",
        "reports/qr/",
    )
    committed_prefixes = (
        "src/",
        "scripts/",
        "grant_case/",
        "data/reference_materials/",
    )
    committed_exact = {
        "reports/benchmark_results.json",
        "reports/benchmark_summary.md",
        "requirements.txt",
        "PROJECT.md",
        "ARCHITECTURE.md",
        "DESIGN_SPEC.md",
    }
    candidates: list[str] = []
    for rel in raw_candidates:
        if any(ch in rel for ch in ("*", "<", ">", "{", "}")):
            continue
        if rel == "data/ceramscan.db" or rel.endswith((".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")):
            continue
        if rel.startswith(ignored_prefixes):
            continue
        if rel in committed_exact or rel.startswith(committed_prefixes):
            candidates.append(rel)

    assert len(candidates) >= 3
    for rel in candidates:
        assert (repo_root / rel.rstrip("/")).exists(), f"Path referenced in README.md does not exist: {rel}"


# =============================================================================
# Tier 3: Cross-Feature Pairwise Combinations (4 test cases)
# =============================================================================


@pytest.mark.tier3
@pytest.mark.f13
@pytest.mark.f8
def test_t3_f13_f08_pptx_metrics_match_benchmark_json(repo_root) -> None:
    bench = json.loads((repo_root / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    all_text = "\n".join(_extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))

    assert str(bench["sample_count"]) in all_text
    assert any(v in all_text for v in _format_variants(float(bench["avg_latency_ms"])))
    acc_candidates = (
        _format_variants(float(bench["f1"]) * 100.0)
        + _format_variants(float(bench["f1"]))
        + _format_variants(float(bench["verdict_accuracy"]) * 100.0)
        + _format_variants(float(bench["verdict_accuracy"]))
    )
    assert any(v in all_text for v in acc_candidates)
    auth_val = float(bench["auth_benchmark"]["auth_accuracy"])
    auth_candidates = _format_variants(auth_val * 100.0) + _format_variants(auth_val)
    assert any(v in all_text for v in auth_candidates)


@pytest.mark.tier3
@pytest.mark.f13
@pytest.mark.f8
def test_t3_f13_f08_pptx_overlays_match_reports_overlays(repo_root) -> None:
    import hashlib
    import io
    import numpy as np
    from PIL import Image

    overlay_paths = [
        p
        for p in (repo_root / "reports" / "overlays").glob("*.png")
        if p.is_file()
    ]
    assert len(overlay_paths) >= 1

    overlay_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in overlay_paths}
    overlay_arrays = [np.asarray(Image.open(p).convert("RGB")) for p in overlay_paths]

    matched = False
    with zipfile.ZipFile(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx", "r") as zf:
        media_pngs = [n for n in zf.namelist() if n.startswith("ppt/media/") and n.lower().endswith(".png")]
        assert len(media_pngs) >= 1
        for media_name in media_pngs:
            payload = zf.read(media_name)
            if hashlib.sha256(payload).hexdigest() in overlay_hashes:
                matched = True
                break
            media_arr = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            if any(media_arr.shape == ov.shape and np.array_equal(media_arr, ov) for ov in overlay_arrays):
                matched = True
                break
    assert matched, "No embedded PPTX PNG matches reports/overlays/*.png"


@pytest.mark.tier3
@pytest.mark.f12
@pytest.mark.f8
def test_t3_f12_f08_docx_metrics_match_benchmark_json(repo_root) -> None:
    bench = json.loads((repo_root / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    md_text = (repo_root / "grant_case" / "project_description.md").read_text(encoding="utf-8")
    docx_text = _extract_docx_text(repo_root / "grant_case" / "CeramScan_Sistema_2026.docx")

    for doc_text in (md_text, docx_text):
        assert str(bench["sample_count"]) in doc_text
        assert any(v in doc_text for v in _format_variants(float(bench["avg_latency_ms"])))


@pytest.mark.tier3
@pytest.mark.f14
@pytest.mark.f9
@pytest.mark.f15
def test_t3_f14_f09_roadmap_and_readme_cli_flags_match_scripts_help(repo_root, venv_python) -> None:
    roadmap = (repo_root / "grant_case" / "REFINEMENTS_ROADMAP.md").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    combined_docs = roadmap + "\n" + readme

    script_flags = {
        "scripts/generate_synthetic_ceramic.py": ("--output-dir", "--count"),
        "scripts/run_benchmark.py": ("--dataset",),
        "scripts/ingest_real_photos.py": ("--input", "--rebuild-all", "--export-yolo"),
    }
    for rel_script, flags in script_flags.items():
        proc = subprocess.run(
            [str(venv_python), str(repo_root / rel_script), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, f"{rel_script} --help failed: {proc.stderr}"
        for flag in flags:
            assert flag in proc.stdout, f"Flag {flag} missing from {rel_script} --help"
            assert flag in combined_docs, f"Flag {flag} missing from roadmap/README"


# =============================================================================
# Tier 4: Real-World User Workflows (4 test cases)
# =============================================================================


@pytest.mark.tier4
@pytest.mark.f12
@pytest.mark.f8
def test_t4_f12_docx_full_build_and_pdf_render_pipeline(
    repo_root,
    venv_python,
    slop_scanner_cmd,
    tmp_path,
) -> None:
    import numpy as np
    from PIL import Image
    from pypdf import PdfReader

    grant_dir = repo_root / "grant_case"
    build_proc = subprocess.run(
        [str(venv_python), str(grant_dir / "build_docx.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert build_proc.returncode == 0, f"build_docx.py failed: {build_proc.stderr}"

    docx_path = grant_dir / "CeramScan_Sistema_2026.docx"
    pdf_path = _convert_to_pdf(docx_path, tmp_path)
    reader = PdfReader(str(pdf_path))
    assert 1 <= len(reader.pages) <= 15

    pngs = _render_pdf_to_pngs(pdf_path, tmp_path, "t4_docx")
    assert len(pngs) == len(reader.pages)
    for png in pngs:
        assert float(np.asarray(Image.open(png).convert("L")).std()) > 5.0

    slop_proc = subprocess.run(
        [*slop_scanner_cmd, str(grant_dir / "project_description.md"), str(grant_dir / "build_docx.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert slop_proc.returncode == 0, f"slop_scan.py failed:\n{slop_proc.stdout}\n{slop_proc.stderr}"


@pytest.mark.tier4
@pytest.mark.f13
@pytest.mark.f8
def test_t4_f13_pptx_full_build_and_pdf_render_pipeline(
    repo_root,
    venv_python,
    tmp_path,
) -> None:
    import numpy as np
    from PIL import Image
    from pptx import Presentation
    from pypdf import PdfReader

    grant_dir = repo_root / "grant_case"
    build_proc = subprocess.run(
        [str(venv_python), str(grant_dir / "build_presentation.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert build_proc.returncode == 0, f"build_presentation.py failed: {build_proc.stderr}"

    pptx_path = grant_dir / "CeramScan_Sistema_2026.pptx"
    prs = Presentation(str(pptx_path))
    assert len(prs.slides) == 10

    pdf_path = _convert_to_pdf(pptx_path, tmp_path)
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) == 10

    pngs = _render_pdf_to_pngs(pdf_path, tmp_path, "t4_slide")
    assert len(pngs) == 10
    for png in pngs:
        assert png.stat().st_size > 5_000
        assert float(np.asarray(Image.open(png).convert("L")).std()) > 5.0

    bench = json.loads((repo_root / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    all_text = "\n".join(_extract_pptx_slide_texts(pptx_path))
    assert str(bench["sample_count"]) in all_text
    assert any(v in all_text for v in _format_variants(float(bench["avg_latency_ms"])))


@pytest.mark.tier4
@pytest.mark.f15
@pytest.mark.f11
@pytest.mark.f12
@pytest.mark.f13
@pytest.mark.f14
def test_t4_f15_full_antislop_and_hygiene_audit(repo_root, slop_scanner_cmd, tmp_path) -> None:
    targets = [
        repo_root / "src" / "web",
        repo_root / "grant_case",
        repo_root / "README.md",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests" / "e2e",
        repo_root / "TEST_INFRA.md",
    ]
    proc = subprocess.run(
        [*slop_scanner_cmd, *(str(p) for p in targets)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert proc.returncode == 0, f"Full slop_scan failed:\n{proc.stdout}\n{proc.stderr}"

    docx_text = _extract_docx_text(repo_root / "grant_case" / "CeramScan_Sistema_2026.docx")
    pptx_text = "\n".join(_extract_pptx_slide_texts(repo_root / "grant_case" / "CeramScan_Sistema_2026.pptx"))
    assert EM_DASH not in docx_text
    assert EM_DASH not in pptx_text

    ws = tmp_path / "hygiene_git"
    ws.mkdir()
    shutil.copy2(repo_root / ".gitignore", ws / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    for ignored in (".venv/x", "__pycache__/x.pyc", "data/test.db", "reports/qr/x.png", "tmp_x.png"):
        p = ws / ignored
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"1")
        assert subprocess.run(["git", "check-ignore", "-q", ignored], cwd=ws, check=False).returncode == 0


@pytest.mark.tier4
@pytest.mark.f14
@pytest.mark.f12
@pytest.mark.f13
@pytest.mark.f15
def test_t4_f14_complete_grant_submission_package_consistency(repo_root) -> None:
    grant_files = (
        "project_description.md",
        "build_docx.py",
        "build_presentation.py",
        "CeramScan_Sistema_2026.docx",
        "CeramScan_Sistema_2026.pptx",
        "pitch_and_video_script.md",
        "pilot_letter_template.md",
        "REFINEMENTS_ROADMAP.md",
    )
    grant_dir = repo_root / "grant_case"
    for name in grant_files:
        assert (grant_dir / name).is_file(), f"Missing grant deliverable: {name}"

    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    desc_text = (grant_dir / "project_description.md").read_text(encoding="utf-8")
    pitch_text = (grant_dir / "pitch_and_video_script.md").read_text(encoding="utf-8")
    pilot_text = (grant_dir / "pilot_letter_template.md").read_text(encoding="utf-8")
    docx_text = _extract_docx_text(grant_dir / "CeramScan_Sistema_2026.docx")
    pptx_text = "\n".join(_extract_pptx_slide_texts(grant_dir / "CeramScan_Sistema_2026.pptx"))
    bench = json.loads((repo_root / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))

    for doc_name, txt in (
        ("README.md", readme_text),
        ("project_description.md", desc_text),
        ("pitch_and_video_script.md", pitch_text),
        ("pilot_letter_template.md", pilot_text),
        ("CeramScan_Sistema_2026.docx", docx_text),
        ("CeramScan_Sistema_2026.pptx", pptx_text),
    ):
        assert "Павловец" in txt, f"Missing author surname in {doc_name}"
        assert "pavlovec000@gmail.com" in txt, f"Missing author email in {doc_name}"
        assert "28390-89" in txt, f"Missing GOST 28390-89 in {doc_name}"
        assert "289-ФЗ" in txt, f"Missing 289-FZ in {doc_name}"

    assert "Гатчинск" in desc_text and "Гатчинск" in pptx_text
    assert "+79052010295" in desc_text and "+79052010295" in pilot_text
    assert str(bench["sample_count"]) in pptx_text and str(bench["sample_count"]) in desc_text
