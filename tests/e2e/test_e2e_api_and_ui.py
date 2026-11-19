from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pytest

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
FONT_URL_RE = re.compile(r"url\(\s*['\"]?([^)'\"\s]+\.woff2)['\"]?\s*\)")


def _assert_png_bytes(payload: bytes) -> None:
    assert len(payload) > 100
    assert payload[:8] == PNG_MAGIC


def _assert_clean_telemetry(telemetry: dict[str, list[object]]) -> None:
    assert telemetry["console_errors"] == [], f"Console errors: {telemetry['console_errors']}"
    assert telemetry["page_errors"] == [], f"Page errors: {telemetry['page_errors']}"
    assert telemetry["failed_requests"] == [], f"Failed requests: {telemetry['failed_requests']}"
    assert telemetry["http_errors"] == [], f"HTTP >= 400 responses: {telemetry['http_errors']}"


def _trigger_demo_inspection(page: object) -> None:
    sample_buttons = page.locator("#demo-sample-list button")
    assert sample_buttons.count() >= 1
    sample_buttons.first.click()
    page.locator("#btn-run-inspect").click()
    page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=15000)


# =============================================================================
# Tier 1: Feature Coverage (14 test cases: 7 F10 + 7 F11)
# =============================================================================


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_health_endpoint(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["model_mode"] in ("deterministic", "onnx")
    assert body["db_ok"] is True


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_static_and_asset_mounts(api_client) -> None:
    root_resp = api_client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "")
    assert "#app-header" in root_resp.text or 'id="app-header"' in root_resp.text

    css_resp = api_client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "--accent-primary" in css_resp.text

    js_resp = api_client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert len(js_resp.text.strip()) > 100

    samples_resp = api_client.get("/api/demo-samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()["samples"]
    assert len(samples) >= 1
    asset_resp = api_client.get(samples[0]["image_url"])
    assert asset_resp.status_code == 200
    assert asset_resp.content[:8] == PNG_MAGIC or asset_resp.content[:3] == JPEG_MAGIC


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_demo_samples_schema(api_client) -> None:
    resp = api_client.get("/api/demo-samples")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["samples"], list) and len(data["samples"]) >= 1
    for sample in data["samples"]:
        assert {"id", "label", "stage", "image_url"}.issubset(sample.keys())
        assert sample["stage"] in ("bisque", "glazed")
        assert sample["image_url"].startswith("/")

    assert isinstance(data["passports"], list) and len(data["passports"]) >= 1
    for passport in data["passports"]:
        assert {
            "item_id",
            "label",
            "sku",
            "master_name",
            "probe_image_url",
            "is_counterfeit",
        }.issubset(passport.keys())
        assert isinstance(passport["is_counterfeit"], bool)


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_inspect_bisque_and_glazed(api_client, ceramic_image_factory) -> None:
    bisque_png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)
    glazed_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    expected_keys = {
        "inspection_id",
        "item_id",
        "stage",
        "verdict",
        "gost_grade",
        "model_mode",
        "contour_deviation_pct",
        "defect_count",
        "defects",
        "gost_rules_triggered",
        "metrics",
        "overlay_url",
        "overlay_base64",
    }

    for stage_name, payload in (("bisque", bisque_png), ("glazed", glazed_png)):
        resp = api_client.post(
            "/api/inspect",
            files={"file": (f"{stage_name}.png", payload, "image/png")},
            data={"stage": stage_name},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert expected_keys.issubset(body.keys())
        assert body["stage"] == stage_name
        assert body["verdict"] in ("PASS", "REWORK", "REJECT")
        assert isinstance(body["defects"], list)
        assert isinstance(body["gost_rules_triggered"], list)

        metrics = body["metrics"]
        assert {"latency_ms", "glare_ratio", "cobalt_ratio", "scale_calibrated"}.issubset(metrics.keys())
        assert metrics["latency_ms"] < 300.0

        decoded_overlay = base64.b64decode(body["overlay_base64"])
        _assert_png_bytes(decoded_overlay)

        overlay_resp = api_client.get(body["overlay_url"])
        assert overlay_resp.status_code == 200
        _assert_png_bytes(overlay_resp.content)


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_passport_register_auto_and_custom_id(api_client, ceramic_image_factory) -> None:
    cobalt_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    auto_resp = api_client.post(
        "/api/passport/register",
        files={"file": ("plate_auto.png", cobalt_png, "image/png")},
        data={"stage": "glazed"},
    )
    assert auto_resp.status_code == 200
    auto_body = auto_resp.json()
    assert re.match(r"^GZHEL-\d{8}-[0-9a-f]{6}$", auto_body["item_id"])
    assert auto_body["qr_payload"] == f"CERAMSCAN:{auto_body['item_id']}"
    assert auto_body["orb_count"] >= 0
    assert 0.0 <= auto_body["cobalt_ratio"] <= 1.0

    custom_resp = api_client.post(
        "/api/passport/register",
        files={"file": ("plate_custom.png", cobalt_png, "image/png")},
        data={
            "item_id": "GZHEL-E2E-001",
            "sku": "SKU-77",
            "master_name": "Иванов И.",
            "batch_note": "Печь-2",
            "stage": "glazed",
        },
    )
    assert custom_resp.status_code == 200
    custom_body = custom_resp.json()
    assert custom_body["item_id"] == "GZHEL-E2E-001"
    assert custom_body["sku"] == "SKU-77"
    assert custom_body["master_name"] == "Иванов И."
    assert custom_body["batch_note"] == "Печь-2"
    assert custom_body["qr_payload"] == "CERAMSCAN:GZHEL-E2E-001"

    qr_resp = api_client.get(custom_body["qr_image_url"])
    assert qr_resp.status_code == 200
    _assert_png_bytes(qr_resp.content)

    ref_resp = api_client.get(custom_body["reference_image_url"])
    assert ref_resp.status_code == 200
    _assert_png_bytes(ref_resp.content)


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_passport_verify_genuine_match(api_client, ceramic_image_factory) -> None:
    cobalt_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    reg_resp = api_client.post(
        "/api/passport/register",
        files={"file": ("ref.png", cobalt_png, "image/png")},
        data={
            "item_id": "GZHEL-E2E-VERIFY",
            "sku": "GZHEL-SKU-01",
            "master_name": "Мастер Гжель",
            "batch_note": "Обжиг-4",
            "stage": "glazed",
        },
    )
    assert reg_resp.status_code == 200

    ver_resp = api_client.post(
        "/api/passport/verify/GZHEL-E2E-VERIFY",
        files={"file": ("probe.png", cobalt_png, "image/png")},
    )
    assert ver_resp.status_code == 200
    body = ver_resp.json()
    expected_keys = {
        "item_id",
        "verdict",
        "match_percent",
        "orb_score",
        "phash_score",
        "color_score",
        "matched_kp",
        "latency_ms",
        "registered_sku",
        "registered_master",
        "registered_batch_note",
        "reference_image_url",
    }
    assert expected_keys.issubset(body.keys())
    assert body["item_id"] == "GZHEL-E2E-VERIFY"
    assert body["verdict"] == "Подлинник"
    assert body["match_percent"] >= 62.0
    assert body["matched_kp"] >= 12
    assert body["registered_sku"] == "GZHEL-SKU-01"

    ref_resp = api_client.get(body["reference_image_url"])
    assert ref_resp.status_code == 200
    _assert_png_bytes(ref_resp.content)


@pytest.mark.tier1
@pytest.mark.f10
def test_t1_f10_analytics_schema_and_filters(api_client, ceramic_image_factory) -> None:
    bisque_png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)
    glazed_clean_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    assert api_client.post(
        "/api/inspect",
        files={"file": ("b.png", bisque_png, "image/png")},
        data={"stage": "bisque"},
    ).status_code == 200
    assert api_client.post(
        "/api/inspect",
        files={"file": ("g.png", glazed_clean_png, "image/png")},
        data={"stage": "glazed"},
    ).status_code == 200

    resp = api_client.get("/api/analytics", params={"limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    summary = body["summary"]
    expected_summary_keys = {
        "total_inspections",
        "pass_count",
        "rework_count",
        "reject_count",
        "pass_rate_pct",
        "avg_latency_ms",
        "total_passports",
        "total_verifications",
        "counterfeit_flags",
    }
    assert expected_summary_keys.issubset(summary.keys())
    assert isinstance(body["by_defect_class"], list)
    assert isinstance(body["by_stage"], list)
    assert isinstance(body["by_shift"], list)
    assert isinstance(body["recent_inspections"], list)
    assert len(body["recent_inspections"]) <= 10

    bisque_resp = api_client.get("/api/analytics", params={"limit": 10, "stage": "bisque"})
    assert bisque_resp.status_code == 200
    for row in bisque_resp.json()["recent_inspections"]:
        assert row["stage"] == "bisque"

    pass_resp = api_client.get("/api/analytics", params={"limit": 10, "verdict": "PASS"})
    assert pass_resp.status_code == 200
    for row in pass_resp.json()["recent_inspections"]:
        assert row["verdict"] == "PASS"


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_dom_freeze_header_and_health(live_api_url, ui_page) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    required_ids = (
        "#app-header",
        "#brand-logo-svg",
        '#mode-tabs[role="tablist"]',
        '#mode-indicator[aria-hidden="true"]',
        "#tab-inspect",
        "#tab-passport",
        "#tab-dashboard",
        "#api-health-pill",
        '#status-announcer[aria-live="polite"]',
    )
    for selector in required_ids:
        assert page.locator(selector).count() == 1, f"Missing DOM element: {selector}"

    page.locator('#api-health-pill[data-status="ok"]').wait_for(state="visible", timeout=10000)
    assert page.locator("#tab-inspect").get_attribute("aria-selected") == "true"
    assert page.locator("#panel-passport").is_hidden()
    assert page.locator("#panel-dashboard").is_hidden()
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_self_hosted_fonts_and_tokens(repo_root, live_api_url, ui_page) -> None:
    html_text = (repo_root / "src" / "web" / "index.html").read_text(encoding="utf-8")
    css_text = (repo_root / "src" / "web" / "styles.css").read_text(encoding="utf-8")

    assert "fonts.googleapis.com" not in html_text
    assert "fonts.gstatic.com" not in html_text
    assert not re.search(r"<link[^>]+href=[\"']https?://", html_text, re.IGNORECASE)

    css_lower = css_text.lower()
    assert "--bg-canvas: #f8fafc" in css_lower
    assert "--accent-primary: #1d4ed8" in css_lower
    assert "--accent-anchor: #1e3a8a" in css_lower

    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    font_info = page.evaluate(
        """async () => {
            await document.fonts.ready;
            const families = [...document.fonts].map((f) => ({family: f.family, status: f.status}));
            const tabularSelectors = ['#inspect-latency-ms', '#passport-score-value', '#kpi-total'];
            const tabular = tabularSelectors.map((sel) => {
                const el = document.querySelector(sel);
                return el ? getComputedStyle(el).fontVariantNumeric : '';
            });
            return {families, tabular};
        }"""
    )
    family_names = {entry["family"].replace('"', "").replace("'", "") for entry in font_info["families"]}
    assert "IBM Plex Sans" in family_names
    assert "JetBrains Mono" in family_names
    for value in font_info["tabular"]:
        assert "tabular-nums" in value
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_mode1_inspect_controls_and_slider(live_api_url, ui_page) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    page.locator("#stage-glazed").click()
    assert page.locator("#stage-glazed").get_attribute("aria-checked") == "true"

    _trigger_demo_inspection(page)
    verdict_badge = page.locator("#inspect-verdict-badge")
    assert verdict_badge.inner_text().strip() in ("PASS", "REWORK", "REJECT")
    assert re.search(r"\d", page.locator("#inspect-latency-ms").inner_text())

    img_dims = page.evaluate(
        """() => {
            const orig = document.querySelector('#img-original');
            const over = document.querySelector('#img-overlay');
            return {
                origReady: Boolean(orig && orig.complete && orig.naturalWidth > 0),
                overReady: Boolean(over && over.complete && over.naturalWidth > 0),
            };
        }"""
    )
    assert img_dims["origReady"] is True
    assert img_dims["overReady"] is True

    for mode in ("original", "overlay", "split"):
        page.locator(f'#compare-mode-group [data-compare-mode="{mode}"]').click()
        assert page.locator("#compare-viewport").get_attribute("data-view") == mode
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_mode1_file_and_camera_inputs(
    live_api_url,
    ui_page,
    ceramic_image_factory,
    tmp_path,
) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    cam_input = page.locator("#inspect-camera-input")
    assert cam_input.get_attribute("accept") == "image/*"
    assert cam_input.get_attribute("capture") == "environment"

    upload_path = tmp_path / "bisque_upload.png"
    upload_path.write_bytes(ceramic_image_factory.make_bisque_plate_png(with_crack=True))
    page.locator("#inspect-file-input").set_input_files(str(upload_path))
    assert page.locator("#inspect-dropzone").get_attribute("data-state") == "ready"

    page.locator("#btn-run-inspect").click()
    page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=15000)
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_mode2_passport_dom_and_flows(
    isolated_api_server,
    ui_page,
    ceramic_image_factory,
    tmp_path,
) -> None:
    base_url, _paths = isolated_api_server
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(base_url, wait_until="networkidle")

    page.locator("#tab-passport").click()
    assert page.locator("#form-passport-verify").is_visible()
    assert page.locator("#form-passport-register").is_hidden()

    page.locator('#passport-action-tabs [data-action="register"]').click()
    assert page.locator("#form-passport-register").is_visible()
    assert page.locator("#form-passport-verify").is_hidden()

    ref_path = tmp_path / "cobalt_ref.png"
    ref_path.write_bytes(ceramic_image_factory.make_cobalt_plate_png(variant=0))

    page.locator("#register-item-id").fill("GZHEL-DOM-005")
    page.locator("#register-sku").fill("GZHEL-SKU-01")
    page.locator("#register-master-name").fill("Мастер Гжель")
    page.locator("#register-batch-note").fill("Тест-E2E")
    page.locator("#register-file-input").set_input_files(str(ref_path))
    page.locator("#btn-register-passport").click()

    qr_img = page.locator("#register-qr-img")
    qr_img.wait_for(state="visible", timeout=15000)
    assert qr_img.evaluate("el => el.complete && el.naturalWidth > 0") is True

    page.locator('#passport-action-tabs [data-action="verify"]').click()
    page.locator("#passport-id-input").fill("GZHEL-DOM-005")
    page.locator("#passport-file-input").set_input_files(str(ref_path))
    page.locator("#btn-verify-passport").click()

    banner = page.locator('#passport-verdict-banner[data-authentic="true"]')
    banner.wait_for(state="visible", timeout=15000)
    assert "Подлинник" in banner.inner_text()
    assert page.locator("#passport-score-value").inner_text().strip() != ""
    assert page.locator("#passport-meter-fill").count() == 1
    assert page.locator("#metric-orb-val").inner_text().strip() != ""
    assert page.locator("#metric-phash-val").inner_text().strip() != ""
    assert page.locator("#metric-cobalt-val").inner_text().strip() != ""
    assert page.locator("#passport-ref-img").evaluate("el => el.complete && el.naturalWidth > 0") is True
    assert page.locator("#passport-probe-img").evaluate("el => el.complete && el.naturalWidth > 0") is True
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_mode3_dashboard_dom_and_table(live_api_url, ui_page) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    _trigger_demo_inspection(page)
    page.locator("#tab-dashboard").click()
    page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)

    for selector in (
        "#dashboard-stand-badge",
        "#btn-refresh-analytics",
        "#kpi-total",
        "#kpi-pass-rate",
        "#kpi-defect-rate",
        "#kpi-avg-ms",
        "#dashboard-breakdown",
        "#filter-stage",
        "#filter-verdict",
        "#ledger-table",
        "#ledger-tbody",
        'label[for="filter-stage"]',
        'label[for="filter-verdict"]',
    ):
        assert page.locator(selector).count() == 1, f"Missing Mode 3 selector: {selector}"

    assert int(re.sub(r"\D", "", page.locator("#kpi-total").inner_text()) or "0") >= 1
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier1
@pytest.mark.f11
def test_t1_f11_inline_svg_icons_and_copy_no_emojis(repo_root, live_api_url, ui_page) -> None:
    web_dir = repo_root / "src" / "web"
    for filename in ("index.html", "styles.css", "app.js"):
        content = (web_dir / filename).read_text(encoding="utf-8")
        assert not EMOJI_RE.search(content), f"Emoji character found in {filename}"

    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    svg_stats = page.evaluate(
        """() => {
            const svgs = [...document.querySelectorAll('svg')];
            const valid = svgs.filter((el) =>
                el.getAttribute('viewBox') === '0 0 24 24' &&
                el.getAttribute('stroke') === 'currentColor'
            );
            const ariaHiddenCount = valid.filter((el) => el.getAttribute('aria-hidden') === 'true').length;
            return {total: svgs.length, valid: valid.length, ariaHiddenCount};
        }"""
    )
    assert svg_stats["valid"] >= 6
    assert svg_stats["ariaHiddenCount"] >= 6
    _assert_clean_telemetry(telemetry)


# =============================================================================
# Tier 2: Boundary & Corner Cases (16 test cases: 9 F10 + 7 F11)
# =============================================================================


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_inspect_empty_file_400(api_client, invalid_image_payloads) -> None:
    resp = api_client.post(
        "/api/inspect",
        files={"file": ("empty.png", invalid_image_payloads["empty"], "image/png")},
        data={"stage": "bisque"},
    )
    assert resp.status_code == 400
    assert resp.json().get("detail")


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_inspect_corrupt_bytes_400(api_client, invalid_image_payloads) -> None:
    resp = api_client.post(
        "/api/inspect",
        files={"file": ("bad.jpg", invalid_image_payloads["corrupt"], "image/jpeg")},
        data={"stage": "glazed"},
    )
    assert resp.status_code == 400


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_inspect_tiny_image_400(api_client, invalid_image_payloads) -> None:
    resp = api_client.post(
        "/api/inspect",
        files={"file": ("tiny.png", invalid_image_payloads["too_small_png"], "image/png")},
        data={"stage": "glazed"},
    )
    assert resp.status_code == 400


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_upload_exceeds_10mb_413(api_client, invalid_image_payloads) -> None:
    oversized = invalid_image_payloads["oversized_over_10mb"]

    inspect_resp = api_client.post(
        "/api/inspect",
        files={"file": ("huge.png", oversized, "image/png")},
        data={"stage": "glazed"},
    )
    assert inspect_resp.status_code == 413

    reg_resp = api_client.post(
        "/api/passport/register",
        files={"file": ("huge.png", oversized, "image/png")},
        data={"item_id": "GZHEL-HUGE-01", "stage": "glazed"},
    )
    assert reg_resp.status_code == 413


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_inspect_invalid_enum_422(api_client, ceramic_image_factory) -> None:
    png_bytes = ceramic_image_factory.make_bisque_plate_png(with_crack=False)
    resp = api_client.post(
        "/api/inspect",
        files={"file": ("plate.png", png_bytes, "image/png")},
        data={"stage": "kiln"},
    )
    assert resp.status_code == 422


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_item_id_traversal_and_sqli_rejected(api_client, ceramic_image_factory) -> None:
    png_bytes = ceramic_image_factory.make_cobalt_plate_png(variant=0)
    bad_ids = ("../../etc/passwd", "x'; DROP TABLE passports;--", "ab")

    for bad_id in bad_ids:
        r_insp = api_client.post(
            "/api/inspect",
            files={"file": ("p.png", png_bytes, "image/png")},
            data={"stage": "glazed", "item_id": bad_id},
        )
        assert r_insp.status_code in (400, 422), f"Unexpected status on inspect item_id={bad_id!r}"

        r_reg = api_client.post(
            "/api/passport/register",
            files={"file": ("p.png", png_bytes, "image/png")},
            data={"item_id": bad_id, "stage": "glazed"},
        )
        assert r_reg.status_code in (400, 422), f"Unexpected status on register item_id={bad_id!r}"

        r_ver = api_client.post(
            f"/api/passport/verify/{quote(bad_id, safe='')}",
            files={"file": ("p.png", png_bytes, "image/png")},
        )
        assert r_ver.status_code in (400, 404, 422), f"Unexpected status on verify item_id={bad_id!r}"

    health_resp = api_client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["db_ok"] is True
    assert api_client.get("/api/analytics").status_code == 200


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_passport_register_duplicate_409(api_client, ceramic_image_factory) -> None:
    cobalt_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    first = api_client.post(
        "/api/passport/register",
        files={"file": ("p1.png", cobalt_png, "image/png")},
        data={"item_id": "GZHEL-DUP-001", "stage": "glazed"},
    )
    assert first.status_code == 200

    second = api_client.post(
        "/api/passport/register",
        files={"file": ("p2.png", cobalt_png, "image/png")},
        data={"item_id": "GZHEL-DUP-001", "stage": "glazed"},
    )
    assert second.status_code == 409


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_passport_verify_unknown_404(api_client, ceramic_image_factory) -> None:
    cobalt_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)
    resp = api_client.post(
        "/api/passport/verify/GZHEL-NONEXISTENT-999",
        files={"file": ("probe.png", cobalt_png, "image/png")},
    )
    assert resp.status_code == 404


@pytest.mark.tier2
@pytest.mark.f10
def test_t2_f10_analytics_invalid_queries_422(api_client) -> None:
    bad_queries = (
        {"limit": 0},
        {"limit": 201},
        {"stage": "kiln"},
        {"verdict": "UNKNOWN"},
    )
    for params in bad_queries:
        resp = api_client.get("/api/analytics", params=params)
        assert resp.status_code == 422, f"Expected 422 for {params}, got {resp.status_code}"


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_css_no_banned_patterns(repo_root) -> None:
    web_dir = repo_root / "src" / "web"
    combined = "\n".join(
        (web_dir / name).read_text(encoding="utf-8")
        for name in ("index.html", "styles.css", "app.js")
    )
    banned_patterns = (
        r"backdrop-filter",
        r"-webkit-backdrop-filter",
        r"\b(Inter|InterVariable|Space Grotesk|Geist|Sora|Syne)\b",
        r"#7c3aed",
        r"#d97757",
        r"#f4f1ea",
        r"\bh-screen\b",
        r"\b(Jane Doe|John Doe|Acme Corp)\b",
    )
    for pat in banned_patterns:
        assert not re.search(pat, combined, re.IGNORECASE if pat.startswith("#") else 0), (
            f"Banned UI pattern detected: {pat}"
        )


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_font_face_files_exist_and_serve_200(repo_root, api_client) -> None:
    css_text = (repo_root / "src" / "web" / "styles.css").read_text(encoding="utf-8")
    font_urls = FONT_URL_RE.findall(css_text)
    assert len(font_urls) >= 4

    for rel_url in font_urls:
        clean_rel = rel_url.lstrip("./")
        disk_path = repo_root / "src" / "web" / clean_rel
        assert disk_path.is_file(), f"Missing font file: {disk_path}"
        assert disk_path.stat().st_size > 1000
        http_resp = api_client.get(f"/static/{clean_rel}")
        assert http_resp.status_code == 200
        assert len(http_resp.content) > 1000


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_mobile_320_and_375_no_overflow_and_tap_targets(live_api_url, ui_page) -> None:
    page, telemetry = ui_page

    for width, height in ((320, 640), (375, 812)):
        page.set_viewport_size({"width": width, "height": height})
        page.goto(live_api_url, wait_until="networkidle")

        _trigger_demo_inspection(page)
        inspect_check = page.evaluate(
            """() => {
                const overflow = document.documentElement.scrollWidth <= window.innerWidth;
                const selectors = [
                    '#mode-tabs button',
                    '#stage-toggle button',
                    '#btn-run-inspect',
                    '#compare-handle',
                ];
                const small = [];
                for (const sel of selectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height < 43.5) {
                            small.push({sel, height: rect.height});
                        }
                    }
                }
                return {overflow, small, scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth};
            }"""
        )
        assert inspect_check["overflow"] is True, f"Overflow at {width}x{height} inspect: {inspect_check}"
        assert inspect_check["small"] == [], f"Small tap targets at {width}x{height} inspect: {inspect_check['small']}"

        page.locator("#tab-passport").click()
        passport_check = page.evaluate(
            """() => {
                const overflow = document.documentElement.scrollWidth <= window.innerWidth;
                const btn = document.querySelector('#btn-verify-passport');
                const btnHeight = btn ? btn.getBoundingClientRect().height : 0;
                return {overflow, btnHeight};
            }"""
        )
        assert passport_check["overflow"] is True, f"Overflow at {width}x{height} passport"
        assert passport_check["btnHeight"] >= 43.5

        with page.expect_response(
            lambda r: "/api/analytics" in r.url and r.status == 200,
            timeout=10000,
        ):
            page.locator("#tab-dashboard").click()
        page.locator("#filter-stage").wait_for(state="visible", timeout=5000)
        dash_check = page.evaluate(
            """() => {
                const overflow = document.documentElement.scrollWidth <= window.innerWidth;
                const s1 = document.querySelector('#filter-stage').getBoundingClientRect().height;
                const s2 = document.querySelector('#filter-verdict').getBoundingClientRect().height;
                return {overflow, s1, s2};
            }"""
        )
        assert dash_check["overflow"] is True, f"Overflow at {width}x{height} dashboard"
        assert dash_check["s1"] >= 43.5 and dash_check["s2"] >= 43.5

    _assert_clean_telemetry(telemetry)


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_motion_and_reduced_motion_rules(repo_root) -> None:
    css_text = (repo_root / "src" / "web" / "styles.css").read_text(encoding="utf-8")
    js_text = (repo_root / "src" / "web" / "app.js").read_text(encoding="utf-8")

    assert "@media (prefers-reduced-motion: reduce)" in css_text
    assert "@media (hover: hover) and (pointer: fine)" in css_text
    assert not re.search(r"transition\s*:\s*all\b", css_text)
    assert "overflow-x: clip" in css_text
    assert "100dvh" in css_text
    assert not re.search(r"window\.addEventListener\(\s*['\"]scroll['\"]", js_text)


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_mode1_corrupt_upload_error_alert(
    live_api_url,
    ui_page,
    invalid_image_payloads,
    tmp_path,
) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(invalid_image_payloads["corrupt"])
    page.locator("#inspect-file-input").set_input_files(str(bad_path))
    page.locator("#btn-run-inspect").click()

    err_box = page.locator("#inspect-error")
    err_box.wait_for(state="visible", timeout=10000)
    assert err_box.get_attribute("role") == "alert"
    assert len(err_box.inner_text().strip()) > 0
    assert telemetry["page_errors"] == []


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_mode2_missing_passport_error_alert(
    live_api_url,
    ui_page,
    ceramic_image_factory,
    tmp_path,
) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    probe_path = tmp_path / "probe.png"
    probe_path.write_bytes(ceramic_image_factory.make_cobalt_plate_png(variant=0))

    page.locator("#tab-passport").click()
    page.locator("#passport-id-input").fill("GZHEL-NOPE-404")
    page.locator("#passport-file-input").set_input_files(str(probe_path))
    page.locator("#btn-verify-passport").click()

    err_box = page.locator("#passport-error")
    err_box.wait_for(state="visible", timeout=10000)
    assert err_box.get_attribute("role") == "alert"
    assert len(err_box.inner_text().strip()) > 0
    assert telemetry["page_errors"] == []


@pytest.mark.tier2
@pytest.mark.f11
def test_t2_f11_compare_range_extremes(live_api_url, ui_page) -> None:
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_api_url, wait_until="networkidle")

    _trigger_demo_inspection(page)
    for wipe_val in ("0", "50", "100"):
        state = page.evaluate(
            """(val) => {
                const rng = document.querySelector('#compare-range');
                rng.value = val;
                rng.dispatchEvent(new Event('input', {bubbles: true}));
                const vp = document.querySelector('#compare-viewport');
                const img = document.querySelector('#img-original');
                return {
                    wipeVar: vp ? vp.style.getPropertyValue('--wipe') : '',
                    clipPath: img ? getComputedStyle(img).clipPath : '',
                    overflow: document.documentElement.scrollWidth > window.innerWidth,
                };
            }""",
            wipe_val,
        )
        assert state["overflow"] is False
        assert f"{wipe_val}%" in state["wipeVar"] or state["clipPath"] != ""
    _assert_clean_telemetry(telemetry)


# =============================================================================
# Tier 3: Cross-Feature Pairwise Combinations (6 test cases)
# =============================================================================


@pytest.mark.tier3
@pytest.mark.f10
@pytest.mark.f6
def test_t3_f10_f06_inspect_persists_to_sqlite_analytics(
    isolated_api_server,
    ceramic_image_factory,
) -> None:
    import httpx

    base_url, _paths = isolated_api_server
    bisque_png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)
    glazed_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        r1 = client.post(
            "/api/inspect",
            files={"file": ("b.png", bisque_png, "image/png")},
            data={"stage": "bisque", "shift": "S2", "save_to_db": "true"},
        )
        r2 = client.post(
            "/api/inspect",
            files={"file": ("g.png", glazed_png, "image/png")},
            data={"stage": "glazed", "shift": "S2", "save_to_db": "true"},
        )
        assert r1.status_code == 200 and r2.status_code == 200
        id1 = r1.json()["inspection_id"]
        id2 = r2.json()["inspection_id"]

        analytics = client.get("/api/analytics").json()
        summary = analytics["summary"]
        assert summary["total_inspections"] == 2
        assert summary["pass_count"] + summary["rework_count"] + summary["reject_count"] == 2
        assert any(entry.get("shift") == "S2" for entry in analytics["by_shift"])
        recent_ids = {row["id"] for row in analytics["recent_inspections"]}
        assert {id1, id2}.issubset(recent_ids)

        bisque_only = client.get("/api/analytics", params={"stage": "bisque"}).json()
        assert len(bisque_only["recent_inspections"]) == 1


@pytest.mark.tier3
@pytest.mark.f10
@pytest.mark.f5
@pytest.mark.f6
def test_t3_f10_f05_passport_register_verify_genuine_fake_analytics(
    isolated_api_server,
    ceramic_image_factory,
) -> None:
    import httpx

    base_url, _paths = isolated_api_server
    genuine_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)
    fake_png = ceramic_image_factory.make_cobalt_plate_png(variant=1)

    item_id = "GZHEL-PAIR-01"
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        reg = client.post(
            "/api/passport/register",
            files={"file": ("ref.png", genuine_png, "image/png")},
            data={"item_id": item_id, "stage": "glazed"},
        )
        assert reg.status_code == 200
        reg_data = reg.json()

        qr_resp = client.get(reg_data["qr_image_url"])
        assert qr_resp.status_code == 200
        _assert_png_bytes(qr_resp.content)
        qr_arr = np.frombuffer(qr_resp.content, dtype=np.uint8)
        qr_img = cv2.imdecode(qr_arr, cv2.IMREAD_COLOR)
        assert qr_img is not None
        decoded_text, _pts, _straight = cv2.QRCodeDetector().detectAndDecode(qr_img)
        assert decoded_text == f"CERAMSCAN:{item_id}"

        ref_resp = client.get(reg_data["reference_image_url"])
        assert ref_resp.status_code == 200
        _assert_png_bytes(ref_resp.content)

        v_genuine = client.post(
            "/api/passport/verify/GZHEL-PAIR-01",
            files={"file": ("gen.png", genuine_png, "image/png")},
        ).json()
        v_fake = client.post(
            "/api/passport/verify/GZHEL-PAIR-01",
            files={"file": ("fake.png", fake_png, "image/png")},
        ).json()

        assert v_genuine["verdict"] == "Подлинник"
        assert v_genuine["match_percent"] >= 62.0
        assert v_fake["verdict"] == "Подозрение на контрафакт или штамповку"
        assert v_fake["match_percent"] < v_genuine["match_percent"]

        summary = client.get("/api/analytics").json()["summary"]
        assert summary["total_passports"] == 1
        assert summary["total_verifications"] == 2
        assert summary["counterfeit_flags"] == 1


@pytest.mark.tier3
@pytest.mark.f10
@pytest.mark.f4
def test_t3_f10_f04_inspect_save_to_db_false_does_not_persist(
    isolated_api_server,
    ceramic_image_factory,
) -> None:
    import httpx

    base_url, _paths = isolated_api_server
    plate_png = ceramic_image_factory.make_bisque_plate_png(with_crack=False)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        resp = client.post(
            "/api/inspect",
            files={"file": ("plate.png", plate_png, "image/png")},
            data={"stage": "bisque", "save_to_db": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["inspection_id"] is None
        overlay_resp = client.get(body["overlay_url"])
        assert overlay_resp.status_code == 200
        _assert_png_bytes(overlay_resp.content)

        analytics = client.get("/api/analytics").json()
        assert analytics["summary"]["total_inspections"] == 0


@pytest.mark.tier3
@pytest.mark.f11
@pytest.mark.f10
def test_t3_f11_f10_ui_inspect_updates_dashboard_table(isolated_api_server, ui_page) -> None:
    base_url, _paths = isolated_api_server
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(base_url, wait_until="networkidle")

    _trigger_demo_inspection(page)
    verdict = page.locator("#inspect-verdict-badge").inner_text().strip()

    page.locator("#tab-dashboard").click()
    page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)
    assert int(re.sub(r"\D", "", page.locator("#kpi-total").inner_text()) or "0") >= 1
    assert verdict in page.locator("#ledger-tbody tr").first.inner_text()
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier3
@pytest.mark.f11
@pytest.mark.f10
def test_t3_f11_f10_ui_register_qr_then_verify_dual_preview(
    isolated_api_server,
    ui_page,
    ceramic_image_factory,
    tmp_path,
) -> None:
    base_url, _paths = isolated_api_server
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(base_url, wait_until="networkidle")

    cobalt_path = tmp_path / "ui_777.png"
    cobalt_path.write_bytes(ceramic_image_factory.make_cobalt_plate_png(variant=0))

    page.locator("#tab-passport").click()
    page.locator('#passport-action-tabs [data-action="register"]').click()
    page.locator("#register-item-id").fill("GZHEL-UI-777")
    page.locator("#register-file-input").set_input_files(str(cobalt_path))
    page.locator("#btn-register-passport").click()

    qr_img = page.locator("#register-qr-img")
    qr_img.wait_for(state="visible", timeout=15000)
    assert qr_img.evaluate("el => el.complete && el.naturalWidth > 0") is True

    page.locator('#passport-action-tabs [data-action="verify"]').click()
    page.locator("#passport-id-input").fill("GZHEL-UI-777")
    page.locator("#passport-file-input").set_input_files(str(cobalt_path))
    page.locator("#btn-verify-passport").click()

    page.locator('#passport-verdict-banner[data-authentic="true"]').wait_for(state="visible", timeout=15000)
    assert page.locator("#passport-ref-img").evaluate("el => el.complete && el.naturalWidth > 0") is True
    assert page.locator("#passport-probe-img").evaluate("el => el.complete && el.naturalWidth > 0") is True
    _assert_clean_telemetry(telemetry)


@pytest.mark.tier3
@pytest.mark.f11
@pytest.mark.f10
def test_t3_f11_f10_ui_dashboard_filters_match_api(
    isolated_api_server,
    ui_page,
    ceramic_image_factory,
) -> None:
    import httpx

    base_url, _paths = isolated_api_server
    bisque_png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)
    glazed_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        assert client.post(
            "/api/inspect",
            files={"file": ("b.png", bisque_png, "image/png")},
            data={"stage": "bisque"},
        ).status_code == 200
        assert client.post(
            "/api/inspect",
            files={"file": ("g.png", glazed_png, "image/png")},
            data={"stage": "glazed"},
        ).status_code == 200

    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(base_url, wait_until="networkidle")
    page.locator("#tab-dashboard").click()
    page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)

    with page.expect_response(lambda r: "/api/analytics" in r.url and "stage=bisque" in r.url):
        page.locator("#filter-stage").select_option("bisque")

    page.wait_for_timeout(200)
    row_texts = page.locator("#ledger-tbody tr").all_inner_texts()
    assert len(row_texts) == 1
    assert "Утиль" in row_texts[0] or "bisque" in row_texts[0].lower()
    _assert_clean_telemetry(telemetry)


# =============================================================================
# Tier 4: Real-World User Workflows (2 test cases)
# =============================================================================


@pytest.mark.tier4
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
def test_t4_f11_mobile_375x812_full_operator_journey(isolated_api_server, ui_page) -> None:
    import httpx

    base_url, _paths = isolated_api_server
    demo_payload = httpx.get(f"{base_url}/api/demo-samples", timeout=15.0).json()
    genuine_demo = next(p for p in demo_payload["passports"] if not p["is_counterfeit"])
    fake_demo = next(p for p in demo_payload["passports"] if p["is_counterfeit"])

    page, telemetry = ui_page
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(base_url, wait_until="networkidle")

    page.locator('#api-health-pill[data-status="ok"]').wait_for(state="visible", timeout=10000)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    # Mode 1: Bisque stage inspection + compare slider wipe
    page.locator("#stage-bisque").click()
    _trigger_demo_inspection(page)
    page.evaluate(
        """() => {
            const rng = document.querySelector('#compare-range');
            rng.value = '30';
            rng.dispatchEvent(new Event('input', {bubbles: true}));
        }"""
    )
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    # Mode 2: Genuine & counterfeit demo passport verification
    page.locator("#tab-passport").click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    page.locator(f'#demo-passport-list button[data-passport-id="{genuine_demo["item_id"]}"]').click()
    page.wait_for_timeout(200)
    if page.locator("#passport-verdict-banner").is_hidden() and page.locator("#btn-verify-passport").is_enabled():
        page.locator("#btn-verify-passport").click()
    page.locator('#passport-verdict-banner[data-authentic="true"]').wait_for(state="visible", timeout=15000)

    page.locator(f'#demo-passport-list button[data-passport-id="{fake_demo["item_id"]}"]').click()
    page.wait_for_timeout(200)
    if (
        page.locator("#passport-verdict-banner").get_attribute("data-authentic") == "true"
        and page.locator("#btn-verify-passport").is_enabled()
    ):
        page.locator("#btn-verify-passport").click()
    page.locator('#passport-verdict-banner[data-authentic="false"]').wait_for(state="visible", timeout=15000)

    # Mode 3: Mobile stacked ledger & filter
    page.locator("#tab-dashboard").click()
    page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)
    page.locator("#filter-stage").select_option("bisque")
    page.wait_for_timeout(200)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    _assert_clean_telemetry(telemetry)


@pytest.mark.tier4
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
def test_t4_f11_desktop_1440x900_full_lab_journey(
    isolated_api_server,
    ui_page,
    ceramic_image_factory,
    tmp_path,
) -> None:
    base_url, _paths = isolated_api_server
    page, telemetry = ui_page
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(base_url, wait_until="networkidle")

    page.locator('#api-health-pill[data-status="ok"]').wait_for(state="visible", timeout=10000)
    fonts_loaded = page.evaluate(
        """async () => {
            await document.fonts.ready;
            const names = new Set([...document.fonts].map((f) => f.family.replace(/['"]/g, '')));
            return names.has('IBM Plex Sans') && names.has('JetBrains Mono');
        }"""
    )
    assert fonts_loaded is True
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    # Mode 1: Upload plate PNG and cycle view modes
    plate_path = tmp_path / "lab_plate.png"
    plate_path.write_bytes(ceramic_image_factory.make_cobalt_plate_png(variant=0))
    page.locator("#stage-glazed").click()
    page.locator("#inspect-file-input").set_input_files(str(plate_path))
    page.locator("#btn-run-inspect").click()
    page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=15000)
    for view_mode in ("split", "original", "overlay"):
        page.locator(f'#compare-mode-group [data-compare-mode="{view_mode}"]').click()
        assert page.locator("#compare-viewport").get_attribute("data-view") == view_mode

    # Mode 2: Register new passport then verify probe
    page.locator("#tab-passport").click()
    page.locator('#passport-action-tabs [data-action="register"]').click()
    page.locator("#register-item-id").fill("GZHEL-LAB-900")
    page.locator("#register-file-input").set_input_files(str(plate_path))
    page.locator("#btn-register-passport").click()
    page.locator("#register-qr-img").wait_for(state="visible", timeout=15000)

    page.locator('#passport-action-tabs [data-action="verify"]').click()
    page.locator("#passport-id-input").fill("GZHEL-LAB-900")
    page.locator("#passport-file-input").set_input_files(str(plate_path))
    page.locator("#btn-verify-passport").click()
    page.locator('#passport-verdict-banner[data-authentic="true"]').wait_for(state="visible", timeout=15000)

    # Mode 3: Refresh analytics & verify table columns
    page.locator("#tab-dashboard").click()
    page.locator("#btn-refresh-analytics").click()
    page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)
    assert page.locator("#ledger-table thead th").count() >= 5
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth") is True

    _assert_clean_telemetry(telemetry)
