# CeramScan Design System & UI Specification

Two-Layer normative design and frontend specification for CeramScan («КерамикСкан»), targeting a Zero-Build HTML5, CSS3, and Vanilla JS workspace (`src/web/index.html`, `src/web/styles.css`, `src/web/app.js`, `src/web/fonts/*.woff2`) served by FastAPI (`src/api.py`).

## Layer 1 - Normative Tokens, Contrast Math & Motion Rules

### 1.1 Design Brief & Baseline Dials

- Product surface - Three-mode ceramic quality control and anti-counterfeit web workspace served by `src/api.py` for factory smartphone posts, marketplace verification, and shift analytics
- Visual direction - Cool-neutral porcelain laboratory ground paired with a single Gzhel cobalt accent, crisp `1px` hairline divisions, and monospace telemetry
- Baseline dials - `DESIGN_VARIANCE: 8` (asymmetric `5:7` and `4:8` splits on desktop, single column below `768px`), `MOTION_INTENSITY: 6` (GPU-only tactile physics and `clip-path` split comparison), `VISUAL_DENSITY: 7` (borderless hairline tables and compact numeric strips in Mode 3)
- Anti-goals - Zero violet/indigo-cyan gradients, zero pure `#000000`, zero terracotta/cream palettes, zero blur filters, zero generic default sans-serif stacks, zero viewport height utilities that break mobile Safari, zero fake device chrome, zero emojis as UI icons, zero unverified metrics

### 1.2 Color Palette & Verified WCAG 2.2 AA Contrast Math

All foreground and background pairs use verified sRGB relative luminance (`L = 0.2126 R + 0.7152 G + 0.0722 B`). Because `#64748B` measures `4.34:1` on `#F1F5F9` (below `4.5:1` for normal text), `#64748B` is restricted strictly to non-text `--border-control` (`>= 3.0:1`), while all secondary text on `#F8FAFC`, `#FFFFFF`, and `#F1F5F9` uses `--ink-secondary` (`#475569`, `>= 6.92:1`).

| Token | Hex | Role | Contrast Pair | Measured Ratio | WCAG 2.2 AA Target |
|---|---|---|---|---|---|
| `--bg-canvas` | `#F8FAFC` | Page ground | Ground | - | - |
| `--bg-surface` | `#FFFFFF` | Elevated work sheet / table | Ground | - | - |
| `--bg-surface-muted` | `#F1F5F9` | Recessed dropzone / table head | Ground | - | - |
| `--bg-surface-subtle` | `#E2E8F0` | Skeleton track / meter track | Track | - | - |
| `--ink-primary` | `#0F172A` | Primary headings & data | on `#F8FAFC` / `#FFFFFF` / `#F1F5F9` | `17.06:1` / `17.85:1` / `16.30:1` | `PASS` (`>= 4.5:1`) |
| `--ink-secondary` | `#475569` | Captions, labels, table heads | on `#F8FAFC` / `#FFFFFF` / `#F1F5F9` | `7.24:1` / `7.58:1` / `6.92:1` | `PASS` (`>= 4.5:1`) |
| `--accent-primary` | `#1D4ED8` | Primary CTA, meter fill, links | `#FFFFFF` on `#1D4ED8` / `#1D4ED8` on `#FFFFFF` | `6.70:1` / `6.70:1` | `PASS` (`>= 4.5:1`) |
| `--accent-hover` | `#1E40AF` | Primary CTA hover fill | `#FFFFFF` on `#1E40AF` | `8.72:1` | `PASS` (`>= 4.5:1`) |
| `--accent-anchor` | `#1E3A8A` | Brand anchor, active tab text | `#FFFFFF` on `#1E3A8A` / on `#EFF6FF` | `10.36:1` / `9.52:1` | `PASS` (`>= 4.5:1`) |
| `--accent-tint` | `#EFF6FF` | Active mode pill / sample wash | `#1D4ED8` on `#EFF6FF` | `6.16:1` | `PASS` (`>= 4.5:1`) |
| `--border-hairline` | `#CBD5E1` | Structural `1px` dividers | Decorative hairline on `#FFFFFF` | `1.48:1` | Hairline divider |
| `--border-control` | `#64748B` | Input, tab, button outlines | on `#FFFFFF` / `#F1F5F9` | `4.76:1` / `4.34:1` | `PASS` (`>= 3.0:1` UI) |
| `--status-pass-bg` | `#ECFDF5` | `PASS` / `Подлинник` surface | - | - | - |
| `--status-pass-border` | `#059669` | `PASS` border / indicator | on `#ECFDF5` / `#FFFFFF` | `3.58:1` / `3.77:1` | `PASS` (`>= 3.0:1` UI) |
| `--status-pass-ink` | `#065F46` | `PASS` badge & verdict text | on `#ECFDF5` | `7.29:1` | `PASS` (`>= 4.5:1`) |
| `--status-rework-bg` | `#FFFBEB` | `REWORK` surface | - | - | - |
| `--status-rework-border` | `#B45309` | `REWORK` border / indicator | on `#FFFBEB` | `4.84:1` | `PASS` (`>= 3.0:1` UI) |
| `--status-rework-ink` | `#92400E` | `REWORK` badge & verdict text | on `#FFFBEB` / `#FEF3C7` | `6.84:1` / `6.37:1` | `PASS` (`>= 4.5:1`) |
| `--status-reject-bg` | `#FEF2F2` | `REJECT` / counterfeit surface | - | - | - |
| `--status-reject-border` | `#DC2626` | `REJECT` border / indicator | on `#FEF2F2` | `4.41:1` | `PASS` (`>= 3.0:1` UI) |
| `--status-reject-ink` | `#991B1B` | `REJECT` badge & verdict text | on `#FEF2F2` | `7.60:1` | `PASS` (`>= 4.5:1`) |

### 1.3 Typography & Self-Hosted Zero-404 Font Strategy

Neither `'IBM Plex Sans'` nor `'JetBrains Mono'` ships in default macOS system font directories. To guarantee `0` console errors and `0` HTTP `404` or failed network requests even offline or inside headless Chrome sandboxes, `src/web/` avoids runtime `<link>` calls to external font CDNs and ships self-hosted SIL OFL 1.1 `.woff2` subsets inside `src/web/fonts/`:

1. Store static `.woff2` subsets (`cyrillic`, `cyrillic-ext`, `latin`, `latin-ext`) inside `src/web/fonts/` during M3 (`ibm-plex-sans-400-cyrillic.woff2`, `ibm-plex-sans-400-latin.woff2`, `ibm-plex-sans-500-cyrillic.woff2`, `ibm-plex-sans-500-latin.woff2`, `ibm-plex-sans-600-cyrillic.woff2`, `ibm-plex-sans-600-latin.woff2`, `ibm-plex-sans-700-cyrillic.woff2`, `ibm-plex-sans-700-latin.woff2`, `jetbrains-mono-400-cyrillic.woff2`, `jetbrains-mono-400-latin.woff2`, `jetbrains-mono-600-cyrillic.woff2`, `jetbrains-mono-600-latin.woff2`).
2. Declare `@font-face` blocks at the top of `src/web/styles.css` pointing via relative paths `./fonts/*.woff2` with `font-display: swap` and exact `unicode-range` descriptors. Every `@font-face` `url(...)` must match an existing file on disk in `src/web/fonts/`.
3. Fallback stacks and numeric rules:
   - UI stack: `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
   - Telemetry / monospace stack: `'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace`
   - Numeric discipline: enforce `font-variant-numeric: tabular-nums` on every latency timer (`ms`), match percentage (`%`), defect counter, GOST tolerance, timestamp, and table column.
4. Modular scale (`1.25` ratio, `clamp()` fluid bounds) and `45ch - 65ch` prose measure:
   - `--text-xs`: `0.75rem` (`12px`), `line-height: 1.35`, `letter-spacing: 0.02em`
   - `--text-sm`: `0.875rem` (`14px`), `line-height: 1.45`
   - `--text-base`: `1rem` (`16px`), `line-height: 1.5`
   - `--text-lg`: `1.25rem` (`20px`), `line-height: 1.3`
   - `--text-xl`: `clamp(1.375rem, 1.1rem + 1.1vw, 1.75rem)`, `line-height: 1.2`, `letter-spacing: -0.02em`
   - `--text-2xl`: `clamp(1.625rem, 1.25rem + 1.6vw, 2.25rem)`, `line-height: 1.12`, `letter-spacing: -0.025em`

### 1.4 Complete Normative `:root` CSS Token Block

```css
:root {
  /* Palette: Cool Lab Neutrals & Gzhel Cobalt */
  --bg-canvas: #F8FAFC;
  --bg-surface: #FFFFFF;
  --bg-surface-muted: #F1F5F9;
  --bg-surface-subtle: #E2E8F0;

  --ink-primary: #0F172A;
  --ink-secondary: #475569;
  --ink-on-accent: #FFFFFF;

  --accent-primary: #1D4ED8;
  --accent-hover: #1E40AF;
  --accent-anchor: #1E3A8A;
  --accent-tint: #EFF6FF;

  --border-hairline: #CBD5E1;
  --border-subtle: #E2E8F0;
  --border-control: #64748B;
  --border-active: #1D4ED8;

  /* Semantic Verdicts (GOST 28390-89 & 289-FZ) */
  --status-pass-bg: #ECFDF5;
  --status-pass-border: #059669;
  --status-pass-ink: #065F46;

  --status-rework-bg: #FFFBEB;
  --status-rework-border: #B45309;
  --status-rework-ink: #92400E;

  --status-reject-bg: #FEF2F2;
  --status-reject-border: #DC2626;
  --status-reject-ink: #991B1B;

  /* Typography */
  --font-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;

  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.25rem;
  --text-xl: clamp(1.375rem, 1.1rem + 1.1vw, 1.75rem);
  --text-2xl: clamp(1.625rem, 1.25rem + 1.6vw, 2.25rem);

  /* 4px / 8px Spacing Scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;

  /* Radii & Subtle Tinted Elevation */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-pill: 9999px;
  --shadow-sheet: 0 1px 2px rgba(15, 23, 42, 0.05), 0 8px 24px -12px rgba(15, 23, 42, 0.08);
  --shadow-focus: 0 0 0 3px rgba(29, 78, 216, 0.24);

  /* GPU-Only Motion System */
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-spring: cubic-bezier(0.22, 1, 0.36, 1);
  --dur-instant: 0ms;
  --dur-fast: 140ms;
  --dur-med: 260ms;
  --dur-spring: 360ms;

  /* Layout Constraints */
  --container-max: 1360px;
  --header-height: 60px;
}

@supports (transition-timing-function: linear(0, 1)) {
  :root {
    --ease-spring: linear(
      0, 0.009, 0.035 2.1%, 0.141, 0.281 6.7%, 0.723 12.9%,
      0.938 16.7%, 1.017, 1.077, 1.121, 1.149 24.3%,
      1.159, 1.163, 1.161, 1.154 29.9%, 1.129 32.8%,
      1.051 39.6%, 1.017 43.1%, 0.991, 0.977 51%,
      0.974 53.8%, 0.975 57.1%, 0.997 69.8%, 1.003 76.9%, 1
    );
  }
}
```

### 1.5 GPU-Only Motion Rules & Mobile Safety Guards

1. Permitted animated properties: strictly `transform`, `opacity`, and `clip-path` (plus `background-color` and `border-color` on state change). Never animate `width`, `height`, `top`, `left`, `margin`, `padding`, or write `transition: all`.
2. Timing budget:
   - `0ms` (`--dur-instant`): live wipe slider pointer drag (`clip-path: inset(...)` updated directly in `pointermove` via `requestAnimationFrame`).
   - `140ms` (`--dur-fast`, `--ease-out-expo`): button press (`scale(0.98)`), segmented mode indicator slide (`transform: translateX(...)`), stage switch (`Утиль 900°C` / `Глазурь 1350°C`).
   - `260ms - 360ms` (`--dur-med` / `--dur-spring`): mode panel entrance (`opacity: 0 -> 1`, `translateY(8px -> 0)`), authenticity meter (`transform: scaleX(var(--score, 0))` with `transform-origin: left center`), `requestAnimationFrame` `tabular-nums` count-up (`360ms`), staggered `.reveal` items capped at `--stagger: 0..3`.
3. Hover guard: wrap all hover lifts (`transform: translateY(-2px)`) inside `@media (hover: hover) and (pointer: fine)` so touch taps at `375x812` never leave stuck hover frames.
4. Scroll reveals: use one-shot `IntersectionObserver` (`threshold: 0.14`, `rootMargin: '0px 0px -40px 0px'`, `io.unobserve(entry.target)` on reveal). Never bind `window.addEventListener('scroll')`.
5. Mobile CSS floor:
   - `html, body { min-height: 100dvh; overflow-x: clip; }` (no `overflow-x: hidden` clipping sticky children, no fixed `100vh` jump).
   - Desktop multi-column grids use `minmax(0, 1fr)` (e.g., `grid-template-columns: minmax(0, 5fr) minmax(0, 7fr)`) and collapse to `minmax(0, 1fr)` below `768px`.
   - Headings, filenames, and passport IDs set `overflow-wrap: anywhere; min-width: 0`.
   - All interactive controls (`button`, `[role="tab"]`, `input`, `select`, slider thumb, upload labels) enforce `min-height: 44px` on `< 768px`, `white-space: nowrap` on nav/tab/button labels, and `min-width: 44px` for square controls.
6. Mandatory Reduced Motion block (placed at the very end of `src/web/styles.css`):

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
  .reveal {
    opacity: 1;
    transform: none;
  }
}
```

---

## Layer 2 - Wireframes, DOM Contracts, SVG Primitives & Copy Rules

### 2.1 Macrostructure & Header Shell (`1440x900` & `375x812`)

Single sticky utility header (`height: 60px` desktop, `background: var(--bg-surface)`, `border-bottom: 1px solid var(--border-hairline)`, solid surface without blur filters).

```text
Desktop 1440x900 Header (60px):
+----------------------------------------------------------------------------------+
| [SVG] CeramScan [Стенд v0.1]  |  [ Пост ОТК | Подлинность | Дашборд ]  | ● CPU   |
+----------------------------------------------------------------------------------+

Mobile 375x812 Header (2-row stack, zero overflow):
+-------------------------------------------+
| [SVG] CeramScan [Стенд v0.1]      ● CPU   |
| [  Пост ОТК  | Подлинность |  Дашборд  ]  |
+-------------------------------------------+
```

- Desktop `1440x900`: left brand block (`#brand-logo-svg` + `CeramScan` + `Стенд v0.1` pill), center segmented mode switch (`#mode-tabs` with sliding `#mode-indicator` behind `#tab-inspect`, `#tab-passport`, `#tab-dashboard`), right `#api-health-pill` showing `/health` connectivity and CV mode (`OpenCV CPU` / `ONNX`).
- Mobile `375x812`: Row 1 aligns brand mark and `#api-health-pill` (`justify-content: space-between`); Row 2 renders a full-width 3-column grid (`grid-template-columns: repeat(3, minmax(0, 1fr))`) with labels `Пост ОТК`, `Подлинность`, `Дашборд` (`min-height: 44px`, `font-size: 13px`, `white-space: nowrap`) keeping `scrollWidth <= 375`.

### 2.2 Mode 1: `Пост ОТК (Смартфон)` (`#panel-inspect`)

Desktop `1440x900` (`5:7` asymmetric split, `gap: 24px`):

```text
+-------------------------------+  +---------------------------------------------+
| [Этап обжига]                 |  | [REJECT] Брак | ГОСТ 28390-89   18.4 мс     |
| [Утиль 900°C | Глазурь 1350°C]|  |---------------------------------------------|
|                               |  | [Сравнение]  [Исходник]  [Оверлей]          |
| [ Dropzone / Выбрать фото ]   |  | +-----------------------------------------+ |
| [ Снять с камеры (cam)    ]   |  | |   Исходник      |  Оверлей (4:3)       | |
|                               |  | |                 <|>                     | |
| [Быстрый образец: 4-5 чипов]  |  | +-----------------------------------------+ |
|                               |  | Дефект        | Зона   | Размер / ГОСТ       |
| [   Проверить изделие (CTA) ] |  | Трещина / цек | Поле   | 42 мм (не допуск.)  |
+-------------------------------+  +---------------------------------------------+
```

- Left `5fr` control sheet (`--bg-surface`, `1px solid --border-hairline`):
  1. Stage selector (`#stage-toggle`): `Утиль 900°C` (`#stage-bisque`, `data-stage="bisque"`, default) and `Глазурь 1350°C` (`#stage-glazed`, `data-stage="glazed"`).
  2. Upload & smartphone camera zone (`#inspect-dropzone`): dashed `1px` border (`--border-control`), file input (`#inspect-file-input`, `accept="image/*"`), camera trigger (`#btn-camera-capture` invoking `<input id="inspect-camera-input" type="file" accept="image/*" capture="environment">`), and thumbnail preview.
  3. Quick demo sample strip (`#demo-sample-list`): `4-5` compact buttons loaded from `GET /api/demo-samples` (`samples`), `min-height: 44px` touch targets, `white-space: nowrap`.
  4. Primary CTA (`#btn-run-inspect`, full width, `--accent-primary`, `Проверить изделие`) plus short note (`Оценка по ГОСТ 28390-89`).
- Right `7fr` visual proof stage:
  1. Verdict & telemetry header (`border-bottom: 1px solid var(--border-subtle)`): `#inspect-verdict-badge` (`PASS` / `REWORK` / `REJECT`), GOST summary (`#inspect-gost-summary`), measured CPU latency (`#inspect-latency-ms`, `tabular-nums`).
  2. View mode bar (`#compare-mode-group`): `Сравнение`, `Исходник`, `Оверлей`.
  3. Interactive `"Исходник vs Оверлей"` stage (`#compare-viewport`, `aspect-ratio: 4 / 3`, `--bg-surface-muted`):
     - Base image: `<img id="img-overlay">` (from `overlay_url`).
     - Clipped image: `<img id="img-original">` with `clip-path: inset(0 calc(100% - var(--wipe, 50%)) 0 0)`.
     - Tactile handle (`#compare-handle`, `44x44px` hit box) synced with `<input id="compare-range" type="range" min="0" max="100" value="50">`.
     - Explicit `#inspect-empty`, `#inspect-skeleton`, and `#inspect-error` (`role="alert"`) containers.
  4. Defect table (`#inspect-defect-list`, `divide-y` list): class (`Трещина` / `Скол кромки` / `Плешина`), zone (`Кромка` / `Поле`), size/count vs GOST limit.

Mobile `375x812`: Single `1fr` stack in exact reading order: stage switch -> camera & file buttons -> 2-column demo chips -> primary CTA -> verdict strip -> `4:3` compare viewport (`touch-action: pan-y` on viewport, `touch-action: none` on `#compare-handle`) -> defect list.

### 2.3 Mode 2: `Проверка подлинности (Ozon/WB)` (`#panel-passport`)

Desktop `1440x900` (`5:7` asymmetric split, `gap: 24px`):
- Left `5fr` control sheet:
  1. Sub-mode switch (`#passport-action-tabs`): `Сверить узор` (`data-action="verify"`, default) vs `Выдать паспорт` (`data-action="register"`).
  2. Verify form (`#form-passport-verify`):
     - Label above field (`gap: 8px`): `Код паспорта` (`#passport-id-input`, placeholder `GZHEL-2026-001`) + quick demo passport buttons (`#demo-passport-list`, populated from `GET /api/demo-samples` `passports`).
     - Photo upload (`#passport-file-input`, `accept="image/*"`).
     - Primary CTA (`#btn-verify-passport`, `Сверить узор`).
  3. Register form (`#form-passport-register`, hidden in verify sub-mode):
     - Optional custom ID (`#register-item-id`, placeholder `GZHEL-2026-002`, auto-generated if empty).
     - SKU input (`#register-sku`, default `GZHEL-SKU-01`).
     - Artisan name input (`#register-master-name`, default `Мастер Гжель`).
     - Batch/kiln note (`#register-batch-note`).
     - Photo upload (`#register-file-input`, `accept="image/*"`).
     - Primary CTA (`#btn-register-passport`, `Создать паспорт`).
     - Result card displaying generated QR PNG (`#register-qr-img` from `qr_image_url`), `qr_payload`, and registered `item_id`.
- Right `7fr` 289-FZ verification stage:
  1. Verdict banner (`#passport-verdict-banner`): `Подлинник` (`--status-pass-*`) or `Подозрение на контрафакт или штамповку` (`--status-reject-*`).
  2. Cobalt Match Meter (`#passport-meter-wrap`): large `tabular-nums` percentage (`#passport-score-value`, `requestAnimationFrame` count-up) + horizontal track (`--bg-surface-subtle`, height `12px`, `border-radius: 9999px`) with `#passport-meter-fill` animated solely via `transform: scaleX(var(--score, 0))`.
  3. Sub-metric breakdown (`divide-y` hairline rows): `Признаки ORB` (`#metric-orb-val`), `Структура pHash` (`#metric-phash-val`), `Палитра кобальта` (`#metric-cobalt-val`), each with a `scaleX` bar and `%` readout.
  4. Dual preview strip (`grid-template-columns: repeat(2, minmax(0, 1fr))` on desktop & mobile, `gap: 12px`): `Эталон паспорта` (`#passport-ref-img`, wired directly to `reference_image_url` returned by `POST /api/passport/verify/{item_id}`) vs `Проверяемый кадр` (`#passport-probe-img`), plus explicit `#passport-empty`, `#passport-skeleton`, and `#passport-error` (`role="alert"`).

Mobile `375x812`: Single `1fr` stack preserving verify/register controls, verdict banner, match meter, 3-row metric breakdown, and side-by-side dual preview strip.

### 2.4 Mode 3: `Дашборд ОТК` (`#panel-dashboard`)

High-density lab cockpit (`VISUAL_DENSITY: 7`), structured with `border-t` and `divide-y` hairlines rather than nested card containers:

- Top disclaimer & refresh bar: `#dashboard-stand-badge` (`Синтетический стенд v0.1 / Журнал ОТК`) + `#btn-refresh-analytics` (`Обновить`).
- KPI strip (`#dashboard-kpi-strip`): `4` cells (`repeat(4, minmax(0, 1fr))` desktop, `repeat(2, minmax(0, 1fr))` mobile `< 768px`), separated by `1px` `--border-hairline` borders, `tabular-nums`:
  1. `Всего проверок` (`#kpi-total`)
  2. `Годно (PASS)` (`#kpi-pass-rate`)
  3. `Доработка / Брак` (`#kpi-defect-rate`)
  4. `Средний цикл, мс` (`#kpi-avg-ms`)
- Lower `4:8` split (`minmax(0, 4fr) minmax(0, 8fr)` desktop, `1fr` mobile):
  - Left `4fr` breakdown (`#dashboard-breakdown`): horizontal `scaleX` bars for `По этапам` (`Утиль 900°C` / `Глазурь 1350°C`), `Частые дефекты`, and `По сменам`.
  - Right `8fr` inspection ledger (`#dashboard-ledger-wrap`):
    - Filter controls wired to `GET /api/analytics?limit=25&stage=...&verdict=...`: `#filter-stage` (`""`, `"bisque"`, `"glazed"`) and `#filter-verdict` (`""`, `"PASS"`, `"REWORK"`, `"REJECT"`), `min-height: 44px`.
    - Desktop `>= 768px`: semantic `<table>` (`#ledger-table`, `border-collapse: collapse`, `divide-y` rows) with columns `Время`, `ID кадра`, `Этап`, `Дефекты`, `Цикл, мс`, `Вердикт`.
    - Mobile `< 768px`: CSS converts `tbody tr` into stacked `divide-y` rows (`display: block`) with `td::before { content: attr(data-label); }` so `scrollWidth <= 375`.

### 2.5 Authoritative DOM ID & `data-*` Freeze Map

M3 frontend code and E2E Playwright tests lock onto these exact selectors:

| Area | Selector / ID | Element & Contract | Required `data-*` / ARIA Attributes |
|---|---|---|---|
| Header | `#app-header` | `<header>` shell | - |
| Header | `#mode-tabs` | Segmented tablist container | `role="tablist"` |
| Header | `#mode-indicator` | Sliding cobalt pill | `aria-hidden="true"` |
| Header | `#tab-inspect` | Mode 1 tab button | `role="tab"`, `data-mode="inspect"`, `aria-selected`, `aria-controls="panel-inspect"` |
| Header | `#tab-passport` | Mode 2 tab button | `role="tab"`, `data-mode="passport"`, `aria-selected`, `aria-controls="panel-passport"` |
| Header | `#tab-dashboard` | Mode 3 tab button | `role="tab"`, `data-mode="dashboard"`, `aria-selected`, `aria-controls="panel-dashboard"` |
| Header | `#api-health-pill` | Live `/health` status pill | `data-status="ok\|offline\|loading"` |
| Mode 1 | `#panel-inspect` | Mode 1 `<section>` | `role="tabpanel"`, `aria-labelledby="tab-inspect"` |
| Mode 1 | `#stage-toggle` | Stage radio group | `role="radiogroup"` |
| Mode 1 | `#stage-bisque` | `Утиль 900°C` button | `data-stage="bisque"`, `aria-checked="true"` |
| Mode 1 | `#stage-glazed` | `Глазурь 1350°C` button | `data-stage="glazed"`, `aria-checked="false"` |
| Mode 1 | `#inspect-dropzone` | Dropzone container | `data-state="idle\|dragover\|ready"` |
| Mode 1 | `#inspect-file-input` | File `<input type="file">` | `accept="image/*"` |
| Mode 1 | `#inspect-camera-input` | Camera `<input type="file">` | `accept="image/*"`, `capture="environment"` |
| Mode 1 | `#btn-camera-capture` | Camera trigger button | `type="button"` |
| Mode 1 | `#demo-sample-list` | Demo preset chip container | Buttons use `data-sample-id="..."`, `data-stage="..."` |
| Mode 1 | `#btn-run-inspect` | Submit inspect CTA | `type="button"`, `disabled` while loading |
| Mode 1 | `#compare-mode-group` | View switch bar | `data-compare-mode="split\|original\|overlay"` buttons |
| Mode 1 | `#compare-viewport` | Split comparison box | `data-view="split\|original\|overlay"` |
| Mode 1 | `#img-original` | Source image `<img>` | `alt="Исходный кадр керамики"` |
| Mode 1 | `#img-overlay` | Defect overlay `<img>` | `alt="Разметка дефектов CeramScan"` |
| Mode 1 | `#compare-handle` | Draggable divider knob | `aria-hidden="true"` |
| Mode 1 | `#compare-range` | Accessible wipe `<input type="range">` | `min="0"`, `max="100"`, `value="50"` |
| Mode 1 | `#inspect-verdict-badge` | Verdict badge | `data-verdict="PASS\|REWORK\|REJECT"` |
| Mode 1 | `#inspect-gost-summary` | GOST grade summary text | - |
| Mode 1 | `#inspect-latency-ms` | CPU latency readout | `tabular-nums` |
| Mode 1 | `#inspect-defect-list` | Defect breakdown list | `<ul>` with `divide-y` |
| Mode 1 | `#inspect-empty` / `#inspect-skeleton` / `#inspect-error` | State containers | `hidden` toggled, `#inspect-error` has `role="alert"` |
| Mode 2 | `#panel-passport` | Mode 2 `<section>` | `role="tabpanel"`, `aria-labelledby="tab-passport"`, `hidden` when inactive |
| Mode 2 | `#passport-action-tabs` | Verify vs Register switch | `data-action="verify\|register"` buttons |
| Mode 2 | `#form-passport-verify` | Verify form container | `hidden` when in register sub-mode |
| Mode 2 | `#passport-id-input` | Passport ID text `<input>` | `type="text"`, `autocomplete="off"` |
| Mode 2 | `#demo-passport-list` | Demo pair buttons | `data-passport-id="..."`, `data-probe-url="..."` |
| Mode 2 | `#passport-file-input` | Verification photo `<input>` | `accept="image/*"` |
| Mode 2 | `#btn-verify-passport` | Submit verify CTA | `type="button"` |
| Mode 2 | `#form-passport-register` | Register form container | `hidden` when in verify sub-mode |
| Mode 2 | `#register-item-id` | Optional custom item ID `<input>` | `type="text"`, `autocomplete="off"` |
| Mode 2 | `#register-sku` | SKU text `<input>` | `type="text"`, default `"GZHEL-SKU-01"` |
| Mode 2 | `#register-master-name` | Artisan name `<input>` | `type="text"`, default `"Мастер Гжель"` |
| Mode 2 | `#register-batch-note` | Batch/kiln note `<input>` | `type="text"` |
| Mode 2 | `#register-file-input` | Reference photo `<input>` | `accept="image/*"` |
| Mode 2 | `#btn-register-passport` | Submit register CTA | `type="button"` |
| Mode 2 | `#register-qr-img` | Generated QR PNG `<img>` | `alt="QR-код паспорта изделия"` |
| Mode 2 | `#passport-verdict-banner` | 289-FZ verdict banner | `data-authentic="true\|false"` |
| Mode 2 | `#passport-score-value` | Animated `%` readout | `tabular-nums` |
| Mode 2 | `#passport-meter-fill` | Cobalt `scaleX` bar fill | `style="--score: 0.0..1.0"` |
| Mode 2 | `#metric-orb-val` / `#metric-phash-val` / `#metric-cobalt-val` | Sub-score `%` readouts | `tabular-nums` |
| Mode 2 | `#passport-ref-img` / `#passport-probe-img` | Dual preview `<img>` elements | `alt="Эталон паспорта"` / `alt="Проверяемый кадр"` |
| Mode 2 | `#passport-empty` / `#passport-skeleton` / `#passport-error` | State containers | `#passport-error` has `role="alert"` |
| Mode 3 | `#panel-dashboard` | Mode 3 `<section>` | `role="tabpanel"`, `aria-labelledby="tab-dashboard"`, `hidden` when inactive |
| Mode 3 | `#dashboard-stand-badge` | Synthetic stand badge | - |
| Mode 3 | `#btn-refresh-analytics` | Manual refresh button | `type="button"` |
| Mode 3 | `#kpi-total` / `#kpi-pass-rate` / `#kpi-defect-rate` / `#kpi-avg-ms` | KPI values | `tabular-nums` |
| Mode 3 | `#dashboard-breakdown` | Stage/defect/shift bars | Horizontal `scaleX` bars |
| Mode 3 | `#filter-stage` / `#filter-verdict` | Ledger `<select>` filters | Paired `<label>` elements |
| Mode 3 | `#ledger-table` / `#ledger-tbody` | Inspection ledger | `td[data-label]` for mobile stack |
| Mode 3 | `#dashboard-empty` / `#dashboard-skeleton` / `#dashboard-error` | State containers | `#dashboard-error` has `role="alert"` |
| Global | `#status-announcer` | Live region for SR updates | `aria-live="polite"`, `class="sr-only"` |

### 2.6 Interactive Component State Matrix

Every control implements all 5 interactive states strictly with semantic tokens:

| State | Buttons (`.btn-primary` / `.btn-secondary`) | Mode Tabs (`.mode-tab`) & Stage Switches | Inputs, Selects & Dropzone |
|---|---|---|---|
| `default` | `--accent-primary` (`#1D4ED8`) or `--bg-surface` with `--border-control` | `--ink-secondary` text; active uses `--ink-primary` + `--accent-anchor` | `--bg-surface` or `--bg-surface-muted`, `1px solid --border-control` |
| `hover` (`@media (hover: hover) and (pointer: fine)`) | `--accent-hover` (`#1E40AF`) + `transform: translateY(-2px)` | `--ink-primary` text, subtle `--bg-surface-muted` tint | `--border-active` (`#1D4ED8`) |
| `active` | `transform: scale(0.98)` | `transform: scale(0.98)` | Dropzone `dragover`: `--accent-tint` bg + `--accent-primary` dashed border |
| `focus-visible` | `outline: 2px solid var(--accent-primary); outline-offset: 2px; box-shadow: var(--shadow-focus);` | Same crisp 2px cobalt focus ring | Same crisp 2px cobalt focus ring |
| `disabled` | `opacity: 0.52; cursor: not-allowed; transform: none;` | `opacity: 0.52; pointer-events: none;` | `background: var(--bg-surface-muted); opacity: 0.65;` |

### 2.7 Custom Inline SVG Icon Spec (Zero Emojis)

All UI icons use crisp inline `<svg>` geometry with uniform parameters: `width="18" height="18"` (`24x24` for brand mark), `viewBox="0 0 24 24"`, `fill="none"`, `stroke="currentColor"`, `stroke-width="1.75"`, `stroke-linecap="round"`, `stroke-linejoin="round"`, `aria-hidden="true"`.

| Icon ID | Usage | Geometry Description |
|---|---|---|
| `icon-brand-ceramscan` | Header brand mark (`24x24`) | Ceramic vessel contour arc (`#1E3A8A`) + cobalt scan crosshair reticle (`#1D4ED8`) |
| `icon-camera` | Smartphone camera button | Camera body rect + top prism + center lens circle |
| `icon-upload` | File upload dropzone | Upward tray + vertical stem + arrow chevron |
| `icon-split` | `"Исходник vs Оверлей"` view switch | Square frame bisected vertically with opposing left/right arrows |
| `icon-shield` | 289-FZ authenticity tab & badge | Clean shield path + inner check path |
| `icon-qr` | Passport registration / QR view | Four corner finder squares + center data dot |
| `icon-chart` | OTK Dashboard tab | Three vertical bar primitives on baseline |
| `icon-check` / `icon-alert` / `icon-x` | `PASS` / `REWORK` / `REJECT` badges | Circle, triangle, and cross primitives matching status ink |
| `icon-refresh` | Dashboard reload button | Circular arc with directional arrowhead |

### 2.8 Anti-AI-Slop & Russian Copy Guardrails

Every UI label stays within `2-4` words on a single line (`white-space: nowrap` on tabs and buttons), omits trailing periods on headings, labels, and CTAs, avoids U+2014 connectors, avoids passive copula verbs, and uses concrete workshop and CV nouns (`снимок`, `кадр`, `проверка`, `сверка`, `мазок`, `узор`, `брак`, `допуск`, `сходство`, `замер`).

Approved Russian UI strings:
- Header & Tabs: `CeramScan`, `Стенд v0.1`, `Пост ОТК`, `Подлинность`, `Дашборд ОТК`
- Mode 1: `Этап обжига`, `Утиль 900°C`, `Глазурь 1350°C`, `Снять с камеры`, `Выбрать фото`, `Быстрый образец`, `Проверить изделие`, `Сравнение`, `Исходник`, `Оверлей`, `Журнал дефектов`
- Mode 2: `Сверить узор`, `Выдать паспорт`, `Код паспорта`, `Артикул (SKU)`, `Мастер / участок`, `Партия / заметка`, `Загрузить снимок`, `Создать паспорт`, `Подлинник`, `Подозрение на контрафакт или штамповку`, `Индекс сходства`, `Признаки ORB`, `Структура pHash`, `Палитра кобальта`, `Эталон паспорта`, `Проверяемый кадр`
- Mode 3: `Синтетический стенд v0.1 / Журнал ОТК`, `Всего проверок`, `Годно (PASS)`, `Доработка / Брак`, `Средний цикл, мс`, `По этапам`, `Частые дефекты`, `По сменам`, `Обновить`

---

## 5-Dimensional Self-Critique (`design-system-spec-architect` Rubric)

| Axis | Score | Concrete Rationale |
|---|---|---|
| 1. Philosophy | `5/5` | Cohesive cool-neutral porcelain laboratory palette with restrained Gzhel cobalt `#1D4ED8` / `#1E3A8A`, free of generic templates, terracotta/cream palettes, or blur filters |
| 2. Hierarchy | `5/5` | Desktop `5:7` asymmetric split puts controls left and visual proof right; mobile `375x812` orders stage -> camera/sample -> primary CTA -> verdict -> `4:3` interactive slider top-to-bottom within two screens |
| 3. Detail | `5/5` | Every foreground/background contrast pair passes WCAG 2.2 AA (`>= 4.5:1` text, `>= 3.0:1` UI borders; `#64748B` excluded from text on `#F1F5F9`), self-hosted `.woff2` files prevent offline 404s, and `tabular-nums` + `1.75` SVG strokes stay uniform |
| 4. Function | `5/5` | DOM IDs and `data-*` contracts reconcile Mode 2 register/verify (`reference_image_url`) and Mode 3 filters (`?stage=&verdict=`) directly with `ARCHITECTURE.md`; all 5 states, empty/loading/error views, `>= 44px` touch targets, and keyboard `<input type="range">` parity are locked |
| 5. Innovation | `5/5` | Signature `"Исходник vs Оверлей"` zero-lag `clip-path` wipe slider proves CV defect detection in 3 seconds, paired with GPU `scaleX` 289-FZ authenticity meter and responsive `divide-y` cockpit ledger |
