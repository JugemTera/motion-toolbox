"""Multi-format evaluation report generator.

Generates evaluation reports in HTML (interactive dashboard), LaTeX, and JSON.

The HTML report matches the design of ``dashboard/index.html``:
  - Dark theme with the same CSS variables
  - Chart.js radar chart for physical metrics
  - Finger-level category breakdown (per-joint data → index/middle/ring/pinky/thumb)
  - Optional action-level categories (caller-supplied dict)

Usage:
    >>> gen = EvaluationReportGenerator(method_name="GRAB baseline")
    >>> gen.generate_all(
    ...     output_dir="./reports",
    ...     physical_report=phys_report,
    ...     categories={"Grasping": grab_report, "Passing": pass_report},
    ...     title="Hand Physical Metrics",
    ... )
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Finger joint ranges within a single 32-joint bimanual hand (0-15 left, 16-31 right)
_FINGER_SLICES: dict[str, list[int]] = {
    "L Wrist": [0],
    "L Index": [1, 2, 3],
    "L Middle": [4, 5, 6],
    "L Ring": [7, 8, 9],
    "L Pinky": [10, 11, 12],
    "L Thumb": [13, 14, 15],
    "R Wrist": [16],
    "R Index": [17, 18, 19],
    "R Middle": [20, 21, 22],
    "R Ring": [23, 24, 25],
    "R Pinky": [26, 27, 28],
    "R Thumb": [29, 30, 31],
}

# Which direction is "better" for each metric key
_METRIC_DIRECTION: dict[str, str] = {
    "hand_smoothness_jerk": "lower",
    "inter_hand_penetration": "lower",
    "finger_flexion": "higher",
    "joint_limit_violations": "lower",
    # semantic
    "FID": "lower",
    "matching_score": "lower",
    "R_precision_top1": "higher",
    "R_precision_top2": "higher",
    "R_precision_top3": "higher",
    "diversity": "higher",
    "multimodality": "higher",
}

_METRIC_LABELS: dict[str, str] = {
    "hand_smoothness_jerk": "Smoothness",
    "inter_hand_penetration": "Penetration",
    "finger_flexion": "Flexion",
    "joint_limit_violations": "Joint Limits",
    "FID": "FID",
    "matching_score": "Match Score",
    "R_precision_top1": "R-Prec @1",
    "R_precision_top2": "R-Prec @2",
    "R_precision_top3": "R-Prec @3",
    "diversity": "Diversity",
    "multimodality": "MultiModal",
}

_METRIC_ICONS: dict[str, str] = {
    "hand_smoothness_jerk": "activity",
    "inter_hand_penetration": "layers",
    "finger_flexion": "hand",
    "joint_limit_violations": "alert-triangle",
}


def _label(key: str) -> str:
    return _METRIC_LABELS.get(key, key.replace("_", " ").title())


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


class EvaluationReportGenerator:
    """Generate HTML / LaTeX / JSON evaluation reports.

    Args:
        method_name: Name of the primary method being evaluated.
        comparison_results: Optional dict of ``method_name -> physical_report``
            for side-by-side comparison in the radar and table.
    """

    def __init__(
        self,
        method_name: str = "Ours",
        comparison_results: dict[str, Any] | None = None,
    ) -> None:
        self.method_name = method_name
        self.comparison_results = comparison_results or {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        output_dir: str | Path,
        semantic_report: Any | None = None,
        physical_report: Any | None = None,
        categories: dict[str, Any] | None = None,
        title: str = "Motion Evaluation Report",
        basename: str = "evaluation_report",
    ) -> dict[str, Path]:
        """Generate HTML, LaTeX, and JSON reports.

        Args:
            output_dir: Directory to save outputs.
            semantic_report: Optional semantic evaluation results.
            physical_report: Physical evaluation results (has .physical_metrics dict).
            categories: Optional ``{category_name: physical_report}`` dict for the
                category breakdown panel (mirrors the full-body taxonomy categories).
            title: Report title shown in the header.
            basename: Base filename (no extension).

        Returns:
            Dict of ``{format: Path}``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        html_path = output_dir / f"{basename}.html"
        self._generate_html(html_path, semantic_report, physical_report, categories, title)
        outputs["html"] = html_path

        latex_path = output_dir / f"{basename}.tex"
        self._generate_latex(latex_path, semantic_report, physical_report, title)
        outputs["latex"] = latex_path

        json_path = output_dir / f"{basename}.json"
        self._generate_json(json_path, semantic_report, physical_report)
        outputs["json"] = json_path

        try:
            from weasyprint import HTML as WP

            pdf_path = output_dir / f"{basename}.pdf"
            WP(filename=str(html_path)).write_pdf(str(pdf_path))
            outputs["pdf"] = pdf_path
            print(f"PDF report saved to: {pdf_path}")
        except ImportError:
            print("PDF generation skipped: weasyprint not installed. Install with: pip install weasyprint")
        except Exception as e:
            print(f"PDF generation failed: {e}")

        print(f"HTML report saved to: {html_path}")
        print(f"LaTeX table saved to: {latex_path}")
        print(f"JSON data saved to: {json_path}")
        return outputs

    # ──────────────────────────────────────────────────────────────────────────
    # HTML generation
    # ──────────────────────────────────────────────────────────────────────────

    def _build_semantic_section(self, semantic_report: Any) -> str:
        """Build an HTML card for semantic metrics (FID, R-Precision, etc.)."""
        metrics = getattr(semantic_report, "metrics", {})
        if not metrics:
            return ""

        dataset = getattr(semantic_report, "dataset_name", "")
        n_samples = getattr(semantic_report, "num_samples", "")
        n_reps = getattr(semantic_report, "num_replications", "")

        cards_html = ""
        for name, r in metrics.items():
            if name == "diversity_gt":
                continue
            direction = getattr(r, "direction", "lower")
            arrow = "↓ lower" if direction == "lower" else "↑ higher"
            mean = getattr(r, "mean", None)
            ci = getattr(r, "conf_interval_95", None)
            label = _label(name)
            val_str = _fmt(mean)
            ci_str = f" ± {ci:.4f}" if ci is not None else ""
            cards_html += f"""
            <div class="metric-card">
              <div class="metric-card-info">
                <div class="metric-card-name">{label}</div>
                <div>
                  <span class="metric-card-val">{val_str}</span>
                  <span class="metric-card-dir" style="font-size:10px">{ci_str}</span>
                  <span class="metric-card-dir">({arrow})</span>
                </div>
              </div>
            </div>"""

        meta = f"{dataset} · {n_samples} samples · {n_reps} replications" if dataset else ""
        return f"""
    <div class="card">
      <div class="card-header">
        <i data-lucide="bar-chart-2" style="width:13px;height:13px"></i>
        Semantic Metrics
        <span style="margin-left:auto;font-size:10px;font-weight:400">{meta}</span>
      </div>
      <div class="card-body">
        <div class="metric-cards" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px">
{cards_html}
        </div>
      </div>
    </div>"""

    def _generate_html(
        self,
        path: Path,
        semantic_report: Any | None,
        physical_report: Any | None,
        categories: dict[str, Any] | None,
        title: str,
    ) -> None:
        phys = getattr(physical_report, "physical_metrics", {}) if physical_report else {}
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        # All methods for comparison (primary + comparison_results)
        all_methods: dict[str, dict] = {self.method_name: phys}
        for name, rpt in self.comparison_results.items():
            all_methods[name] = getattr(rpt, "physical_metrics", {})

        metric_keys = list(phys.keys())
        semantic_section_html = self._build_semantic_section(semantic_report) if semantic_report is not None else ""

        # Build radar data (normalize each metric to [0,1], higher = better)
        radar_data = self._build_radar_data(all_methods, metric_keys)

        # Build finger breakdown from primary report's per_joint data
        finger_data = self._build_finger_data(phys, metric_keys)

        # Build category data
        cat_data = self._build_category_data(categories, metric_keys) if categories else {}

        # Serialize to JSON for embedding
        def _to_json(obj: Any) -> str:
            return json.dumps(obj, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

        radar_js = _to_json(radar_data)
        finger_js = _to_json(finger_data)
        cat_js = _to_json(cat_data)
        methods_js = _to_json(list(all_methods.keys()))

        # Comparison table rows
        table_rows = self._build_table_rows(all_methods, metric_keys)

        # Metric summary cards (primary method only)
        summary_cards = self._build_summary_cards(phys, metric_keys)

        # Pre-compute conditional HTML strings (no backslashes inside f-string expressions)
        cat_tab_html = (
            '<div class="tab" onclick="switchTab(this,\'category\')">Category Breakdown</div>' if cat_data else ""
        )
        cat_panel_html = (
            '<div class="tab-panel" id="tab-category"><div class="cat-grid" id="catGrid"></div></div>'
            if cat_data
            else ""
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/lucide@0.344.0/dist/umd/lucide.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:         #0a0a0a;
      --surface:    #111111;
      --surface-2:  #1a1a1a;
      --surface-3:  #222222;
      --border:     #2a2a2a;
      --border-2:   #333333;
      --text:       #fafafa;
      --text-muted: #888888;
      --text-dim:   #555555;
      --accent:     #ffffff;
      --accent-low: rgba(255,255,255,0.06);
      --c0: #ffffff; --c1: #60a5fa; --c2: #34d399;
      --c3: #f87171; --c4: #fbbf24; --c5: #a78bfa;
      --radius: 6px; --radius-lg: 10px;
    }}
    html {{ background: var(--bg); color: var(--text);
           font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
           font-size: 14px; line-height: 1.5; }}
    body {{ min-height: 100vh; }}
    .app {{ display: flex; flex-direction: column; min-height: 100vh; }}

    /* Header */
    .header {{
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
      padding: 14px 24px; border-bottom: 1px solid var(--border);
      background: var(--surface); position: sticky; top: 0; z-index: 100;
    }}
    .header-title {{ font-size: 15px; font-weight: 700; letter-spacing: 0.04em; display: flex; align-items: center; gap: 8px; }}
    .header-meta {{ font-size: 12px; color: var(--text-muted); margin-left: auto; }}

    /* Main grid */
    .main {{ flex: 1; padding: 24px; display: flex; flex-direction: column; gap: 24px; max-width: 1400px; margin: 0 auto; width: 100%; }}

    /* Cards */
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; }}
    .card-header {{
      display: flex; align-items: center; gap: 8px; padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
      text-transform: uppercase; color: var(--text-muted);
    }}
    .card-body {{ padding: 20px; }}

    /* Top row: radar + summary */
    .top-row {{ display: grid; grid-template-columns: 1fr 300px; gap: 20px; }}
    @media (max-width: 860px) {{ .top-row {{ grid-template-columns: 1fr; }} }}
    .top-row > * {{ min-width: 0; }}

    /* Radar */
    .radar-wrap {{ position: relative; display: flex; align-items: center; justify-content: center; }}
    .radar-wrap canvas {{ max-width: 420px; max-height: 420px; }}

    /* Metric summary cards */
    .metric-cards {{ display: flex; flex-direction: column; gap: 8px; }}
    .metric-card {{
      display: flex; align-items: center; gap: 10px;
      padding: 10px 14px; background: var(--surface-2);
      border: 1px solid var(--border); border-radius: var(--radius);
    }}
    .metric-card-icon {{ opacity: 0.45; flex-shrink: 0; }}
    .metric-card-info {{ flex: 1; min-width: 0; }}
    .metric-card-name {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }}
    .metric-card-val {{ font-size: 15px; font-weight: 700; }}
    .metric-card-dir {{ font-size: 10px; color: var(--text-dim); margin-left: 4px; }}
    .metric-card-ratio {{ font-size: 11px; color: var(--text-muted); margin-top: 1px; }}

    /* Tabs */
    .tabs {{ display: flex; gap: 4px; border-bottom: 1px solid var(--border); padding: 0 20px; }}
    .tab {{
      padding: 10px 14px; font-size: 12px; font-weight: 500; cursor: pointer;
      border-bottom: 2px solid transparent; color: var(--text-muted);
      transition: color 0.15s, border-color 0.15s;
    }}
    .tab.active {{ color: var(--text); border-bottom-color: var(--accent); }}
    .tab-panel {{ display: none; padding: 20px; }}
    .tab-panel.active {{ display: block; }}

    /* Finger grid */
    .finger-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
    .finger-card {{
      background: var(--surface-2); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 12px;
    }}
    .finger-name {{ font-size: 12px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }}
    .finger-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
    .finger-metrics {{ display: flex; flex-direction: column; gap: 5px; }}
    .finger-metric-row {{ display: flex; align-items: center; gap: 6px; }}
    .finger-metric-label {{ font-size: 10px; color: var(--text-dim); width: 70px; flex-shrink: 0; }}
    .bar-wrap {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 2px; transition: width 0.4s; }}
    .finger-metric-val {{ font-size: 10px; color: var(--text-muted); text-align: right; width: 42px; flex-shrink: 0; }}

    /* Category grid */
    .cat-controls {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }}
    .cat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
    .cat-card {{
      background: var(--surface-2); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 14px; cursor: pointer;
      transition: border-color 0.15s, background 0.15s;
    }}
    .cat-card:hover {{ border-color: #444; background: var(--surface-3); }}
    .cat-card.active {{ border-color: var(--accent); }}
    .cat-name {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
    .cat-count {{ font-size: 11px; color: var(--text-muted); margin-bottom: 8px; }}
    .cat-metrics {{ display: flex; flex-direction: column; gap: 4px; }}
    .cat-metric-row {{ display: flex; align-items: center; gap: 6px; font-size: 10px; }}
    .cat-metric-label {{ color: var(--text-dim); width: 64px; flex-shrink: 0; }}
    .cat-metric-val {{ color: var(--text-muted); margin-left: auto; }}

    /* Comparison table */
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    thead th {{
      padding: 8px 14px; text-align: right;
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--text-muted);
      border-bottom: 1px solid var(--border); white-space: nowrap;
    }}
    thead th:first-child {{ text-align: left; }}
    tbody td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: right; }}
    tbody td:first-child {{ text-align: left; font-weight: 500; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: var(--surface-2); }}
    .best {{ color: #34d399; font-weight: 700; }}
    .dir-badge {{
      display: inline-block; font-size: 9px; padding: 1px 5px;
      border-radius: 3px; background: var(--surface-3);
      color: var(--text-dim); margin-left: 4px; vertical-align: middle;
    }}

    /* Footer */
    .footer {{ text-align: center; color: var(--text-dim); font-size: 11px; padding: 20px; }}

    /* Select */
    .select-input {{
      background: var(--surface-2); border: 1px solid var(--border-2);
      color: var(--text); border-radius: var(--radius);
      padding: 6px 10px; font-size: 12px; cursor: pointer;
    }}
  </style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="header-title">
      <i data-lucide="bar-chart-2" style="width:16px;height:16px"></i>
      {title}
    </div>
    <div class="header-meta">{self.method_name} &nbsp;·&nbsp; {ts}</div>
  </header>

  <main class="main">

    <!-- ── Semantic metrics (only when semantic_report is provided) ── -->
{semantic_section_html}
    <!-- ── Top row: radar + summary cards ─────────────────────── -->
    <div class="top-row">
      <div class="card">
        <div class="card-header">
          <i data-lucide="radio" style="width:13px;height:13px"></i>
          Physical Metrics Radar
        </div>
        <div class="card-body radar-wrap">
          <canvas id="radarChart"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <i data-lucide="activity" style="width:13px;height:13px"></i>
          Summary
        </div>
        <div class="card-body">
          <div class="metric-cards">
{summary_cards}
          </div>
        </div>
      </div>
    </div>

    <!-- ── Breakdown tabs ──────────────────────────────────────── -->
    <div class="card">
      <div class="tabs">
        <div class="tab active" onclick="switchTab(this,'finger')">Finger Breakdown</div>
        {cat_tab_html}
        <div class="tab" onclick="switchTab(this,'table')">Comparison Table</div>
      </div>

      <!-- Finger tab -->
      <div class="tab-panel active" id="tab-finger">
        <div class="finger-grid" id="fingerGrid"></div>
      </div>

      <!-- Category tab (only if categories provided) -->
      {cat_panel_html}

      <!-- Table tab -->
      <div class="tab-panel" id="tab-table">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
{table_rows["header"]}
              </tr>
            </thead>
            <tbody>
{table_rows["body"]}
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>
  <div class="footer">Generated {ts} · motion_toolbox EvaluationReportGenerator</div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────
const RADAR_DATA   = {radar_js};
const FINGER_DATA  = {finger_js};
const CAT_DATA     = {cat_js};
const METHODS      = {methods_js};
const MODEL_COLORS = ['#ffffff','#60a5fa','#34d399','#f87171','#fbbf24','#a78bfa'];
const FINGER_COLORS = {{
  'L Wrist':'#94a3b8','L Index':'#60a5fa','L Middle':'#34d399',
  'L Ring':'#fbbf24','L Pinky':'#f87171','L Thumb':'#a78bfa',
  'R Wrist':'#64748b','R Index':'#3b82f6','R Middle':'#10b981',
  'R Ring':'#d97706','R Pinky':'#ef4444','R Thumb':'#8b5cf6',
}};

// ── Radar chart ───────────────────────────────────────────────
lucide.createIcons();

const radarCtx = document.getElementById('radarChart').getContext('2d');
const radarChart = new Chart(radarCtx, {{
  type: 'radar',
  data: {{
    labels: RADAR_DATA.labels,
    datasets: RADAR_DATA.datasets.map((d, i) => ({{
      label: d.label,
      data: d.data,
      borderColor: MODEL_COLORS[i % MODEL_COLORS.length],
      backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length] + '22',
      borderWidth: 2,
      pointBackgroundColor: MODEL_COLORS[i % MODEL_COLORS.length],
      pointRadius: 3,
    }})),
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{
        labels: {{ color: '#888', font: {{ size: 11 }} }},
      }},
      tooltip: {{
        callbacks: {{
          label: (ctx) => ` ${{ctx.dataset.label}}: ${{(ctx.raw * 100).toFixed(1)}}%`,
        }},
      }},
    }},
    scales: {{
      r: {{
        min: 0, max: 1,
        grid: {{ color: '#2a2a2a' }},
        angleLines: {{ color: '#2a2a2a' }},
        pointLabels: {{ color: '#888', font: {{ size: 11 }} }},
        ticks: {{ display: false }},
      }},
    }},
  }},
}});

// ── Finger breakdown ──────────────────────────────────────────
function renderFingerGrid() {{
  const grid = document.getElementById('fingerGrid');
  if (!grid || !FINGER_DATA.fingers) return;
  const metricKeys = FINGER_DATA.metric_keys || [];
  grid.innerHTML = FINGER_DATA.fingers.map(f => {{
    const color = FINGER_COLORS[f.name] || '#888';
    const metricsHtml = metricKeys.map(k => {{
      const val = f.metrics[k];
      if (val === null || val === undefined) return '';
      const pct = (val * 100).toFixed(0);
      return `<div class="finger-metric-row">
        <span class="finger-metric-label">${{FINGER_DATA.labels[k] || k}}</span>
        <div class="bar-wrap"><div class="bar" style="width:${{pct}}%;background:${{color}}88"></div></div>
        <span class="finger-metric-val">${{FINGER_DATA.raw[f.name]?.[k]?.toFixed(4) ?? '—'}}</span>
      </div>`;
    }}).join('');
    return `<div class="finger-card">
      <div class="finger-name">
        <span class="finger-dot" style="background:${{color}}"></span>${{f.name}}
      </div>
      <div class="finger-metrics">${{metricsHtml}}</div>
    </div>`;
  }}).join('');
}}
renderFingerGrid();

// ── Category breakdown ────────────────────────────────────────
function renderCatGrid() {{
  const grid = document.getElementById('catGrid');
  if (!grid || !Object.keys(CAT_DATA).length) return;
  grid.innerHTML = Object.entries(CAT_DATA).map(([catName, catInfo]) => {{
    const metricsHtml = Object.entries(catInfo.metrics).map(([k, v]) => `
      <div class="cat-metric-row">
        <span class="cat-metric-label">${{FINGER_DATA.labels?.[k] || k}}</span>
        <span class="cat-metric-val">${{v?.toFixed(4) ?? '—'}}</span>
      </div>`).join('');
    return `<div class="cat-card">
      <div class="cat-name">${{catName}}</div>
      <div class="cat-count">${{catInfo.count ?? ''}} sequences</div>
      <div class="cat-metrics">${{metricsHtml}}</div>
    </div>`;
  }}).join('');
}}
renderCatGrid();

// ── Tab switching ─────────────────────────────────────────────
function switchTab(el, id) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  const panel = document.getElementById('tab-' + id);
  if (panel) panel.classList.add('active');
}}
</script>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")

    # ──────────────────────────────────────────────────────────────────────────
    # Data builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_radar_data(self, all_methods: dict[str, dict], metric_keys: list[str]) -> dict:
        """Normalize metrics to [0,1] (1 = best) for radar chart."""
        if not metric_keys:
            return {"labels": [], "datasets": []}

        # Collect all raw values per metric across all methods
        raw: dict[str, list[float]] = {k: [] for k in metric_keys}
        for phys in all_methods.values():
            for k in metric_keys:
                r = phys.get(k)
                if r is not None:
                    v = getattr(r, "sequence_average", None)
                    if v is not None:
                        raw[k].append(float(v))

        # Compute per-metric min/max for normalization
        mn = {k: min(vs) if vs else 0.0 for k, vs in raw.items()}
        mx = {k: max(vs) if vs else 1.0 for k, vs in raw.items()}

        def _norm(k: str, v: float) -> float:
            lo, hi = mn[k], mx[k]
            if hi == lo:
                return 0.5
            norm = (v - lo) / (hi - lo)  # 0 = worst, 1 = best for "higher"
            return norm if _METRIC_DIRECTION.get(k, "lower") == "higher" else 1.0 - norm

        datasets = []
        for method_name, phys in all_methods.items():
            points = []
            for k in metric_keys:
                r = phys.get(k)
                v = getattr(r, "sequence_average", None) if r is not None else None
                points.append(_norm(k, float(v)) if v is not None else 0.0)
            datasets.append({"label": method_name, "data": points})

        return {
            "labels": [_label(k) for k in metric_keys],
            "datasets": datasets,
        }

    def _build_finger_data(self, phys: dict, metric_keys: list[str]) -> dict:
        """Build per-finger metric data from per_joint arrays."""
        fingers = []
        raw_vals: dict[str, dict[str, float]] = {}

        # Collect per-joint data for metrics that have it
        per_joint: dict[str, Any] = {}
        for k in metric_keys:
            r = phys.get(k)
            if r is not None and hasattr(r, "per_joint") and r.per_joint is not None:
                try:
                    arr = list(r.per_joint)
                    if len(arr) == 32:
                        per_joint[k] = arr
                except Exception:
                    pass

        if not per_joint:
            return {"fingers": [], "metric_keys": [], "labels": {}, "raw": {}}

        avail_keys = list(per_joint.keys())

        # Max value per metric for normalization (bar chart width)
        mx: dict[str, float] = {}
        for k, arr in per_joint.items():
            mx[k] = max(abs(v) for v in arr) or 1.0

        for finger_name, joint_idxs in _FINGER_SLICES.items():
            metrics_norm: dict[str, float | None] = {}
            raw_vals[finger_name] = {}
            for k in avail_keys:
                arr = per_joint[k]
                vals = [arr[i] for i in joint_idxs if i < len(arr)]
                avg = sum(vals) / len(vals) if vals else 0.0
                raw_vals[finger_name][k] = avg
                metrics_norm[k] = min(abs(avg) / mx[k], 1.0)
            fingers.append({"name": finger_name, "metrics": metrics_norm})

        return {
            "fingers": fingers,
            "metric_keys": avail_keys,
            "labels": {k: _label(k) for k in avail_keys},
            "raw": raw_vals,
        }

    def _build_category_data(self, categories: dict[str, Any], metric_keys: list[str]) -> dict:
        """Build per-category metric summaries from caller-supplied category reports."""
        result = {}
        for cat_name, cat_report in categories.items():
            phys = getattr(cat_report, "physical_metrics", cat_report if isinstance(cat_report, dict) else {})
            metrics = {}
            count = getattr(cat_report, "count", None)
            for k in metric_keys:
                r = phys.get(k)
                v = getattr(r, "sequence_average", r) if r is not None else None
                metrics[k] = float(v) if v is not None else None
            result[cat_name] = {"metrics": metrics, "count": count}
        return result

    def _build_summary_cards(self, phys: dict, metric_keys: list[str]) -> str:
        cards = []
        for k in metric_keys:
            r = phys.get(k)
            if r is None:
                continue
            val = getattr(r, "sequence_average", None)
            ratio = getattr(r, "ratio", None)
            direction = _METRIC_DIRECTION.get(k, "lower")
            icon = _METRIC_ICONS.get(k, "activity")
            ratio_html = f'<div class="metric-card-ratio">{ratio:.1%} frames</div>' if ratio is not None else ""
            cards.append(f"""            <div class="metric-card">
              <div class="metric-card-icon"><i data-lucide="{icon}" style="width:16px;height:16px"></i></div>
              <div class="metric-card-info">
                <div class="metric-card-name">{_label(k)}</div>
                <div>
                  <span class="metric-card-val">{_fmt(val)}</span>
                  <span class="metric-card-dir">({"↓ lower" if direction == "lower" else "↑ higher"})</span>
                </div>
                {ratio_html}
              </div>
            </div>""")
        return "\n".join(cards)

    def _build_table_rows(self, all_methods: dict[str, dict], metric_keys: list[str]) -> dict[str, str]:
        # Header
        header_cells = "".join(f"<th>{name}</th>" for name in all_methods.keys())

        # Find best value per metric
        best: dict[str, float | None] = {}
        for k in metric_keys:
            direction = _METRIC_DIRECTION.get(k, "lower")
            vals = []
            for phys in all_methods.values():
                r = phys.get(k)
                v = getattr(r, "sequence_average", None) if r is not None else None
                if v is not None:
                    vals.append(float(v))
            if vals:
                best[k] = min(vals) if direction == "lower" else max(vals)

        # Body rows
        body_rows = []
        for k in metric_keys:
            direction = _METRIC_DIRECTION.get(k, "lower")
            dir_badge = f'<span class="dir-badge">{"↓" if direction == "lower" else "↑"}</span>'
            cells = f"<td>{_label(k)}{dir_badge}</td>"
            for phys in all_methods.values():
                r = phys.get(k)
                v = getattr(r, "sequence_average", None) if r is not None else None
                fv = _fmt(float(v)) if v is not None else "—"
                is_best = v is not None and best.get(k) is not None and abs(float(v) - best[k]) < 1e-9
                cls = ' class="best"' if is_best and len(all_methods) > 1 else ""
                cells += f"<td{cls}>{fv}</td>"
            body_rows.append(f"<tr>{cells}</tr>")

        return {"header": header_cells, "body": "\n".join(body_rows)}

    # ──────────────────────────────────────────────────────────────────────────
    # LaTeX
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_latex(
        self,
        path: Path,
        semantic_report: Any | None,
        physical_report: Any | None,
        title: str,
    ) -> None:
        phys = getattr(physical_report, "physical_metrics", {}) if physical_report else {}
        metric_keys = list(phys.keys())

        all_methods: dict[str, dict] = {self.method_name: phys}
        for name, rpt in self.comparison_results.items():
            all_methods[name] = getattr(rpt, "physical_metrics", {})

        col_spec = "l" + "c" * len(all_methods)
        header = " & ".join(["Metric"] + [n.replace("_", r"\_") for n in all_methods]) + r" \\"

        # Best per metric
        best: dict[str, float | None] = {}
        for k in metric_keys:
            direction = _METRIC_DIRECTION.get(k, "lower")
            vals = []
            for phys_m in all_methods.values():
                r = phys_m.get(k)
                v = getattr(r, "sequence_average", None) if r is not None else None
                if v is not None:
                    vals.append(float(v))
            if vals:
                best[k] = min(vals) if direction == "lower" else max(vals)

        rows = []
        for k in metric_keys:
            direction = _METRIC_DIRECTION.get(k, "lower")
            arrow = r"$\downarrow$" if direction == "lower" else r"$\uparrow$"
            cells = [f"{_label(k)} {arrow}"]
            for phys_m in all_methods.values():
                r = phys_m.get(k)
                v = getattr(r, "sequence_average", None) if r is not None else None
                fv = _fmt(float(v)) if v is not None else "—"
                is_best = v is not None and best.get(k) is not None and abs(float(v) - best[k]) < 1e-9
                cells.append(rf"\textbf{{{fv}}}" if is_best and len(all_methods) > 1 else fv)
            rows.append(" & ".join(cells) + r" \\")

        rows_str = ("\n    ").join(rows)
        latex = rf"""% Auto-generated evaluation table — {title}
% Required packages: booktabs
\begin{{table}}[htbp]
  \centering
  \caption{{{title}}}
  \label{{tab:eval}}
  \begin{{tabular}}{{{col_spec}}}
    \toprule
    {header}
    \midrule
    {rows_str}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
        path.write_text(latex, encoding="utf-8")

    # ──────────────────────────────────────────────────────────────────────────
    # JSON
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_json(
        self,
        path: Path,
        semantic_report: Any | None,
        physical_report: Any | None,
    ) -> None:
        phys = getattr(physical_report, "physical_metrics", {}) if physical_report else {}
        data: dict[str, Any] = {
            "method_name": self.method_name,
            "generated_at": datetime.now().isoformat(),
            "physical": {
                k: {
                    "sequence_average": float(getattr(r, "sequence_average", 0)),
                    "ratio": float(r.ratio) if getattr(r, "ratio", None) is not None else None,
                }
                for k, r in phys.items()
            },
        }
        if semantic_report is not None:
            sem = getattr(semantic_report, "to_dict", lambda: {})()
            if sem:
                data["semantic"] = sem
        if self.comparison_results:
            data["comparisons"] = {}
            for name, rpt in self.comparison_results.items():
                cphys = getattr(rpt, "physical_metrics", {})
                data["comparisons"][name] = {
                    k: {
                        "sequence_average": float(getattr(r, "sequence_average", 0)),
                        "ratio": float(r.ratio) if getattr(r, "ratio", None) is not None else None,
                    }
                    for k, r in cphys.items()
                }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ──────────────────────────────────────────────────────────────────────────
    # Deprecated / kept for backwards-compat
    # ──────────────────────────────────────────────────────────────────────────

    def generate_html(self, output_path, semantic_report=None, physical_report=None, title="Report"):
        self._generate_html(Path(output_path), semantic_report, physical_report, None, title)

    def generate_latex(self, output_path, semantic_report=None, physical_report=None, title="Report"):
        self._generate_latex(Path(output_path), semantic_report, physical_report, title)

    def generate_json(self, output_path, semantic_report=None, physical_report=None):
        self._generate_json(Path(output_path), semantic_report, physical_report)

    def generate_pdf(self, html_path, pdf_path):
        from weasyprint import HTML as WP

        WP(filename=str(html_path)).write_pdf(str(pdf_path))
