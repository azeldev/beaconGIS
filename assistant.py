"""Dockable Assessment Reports panel: statistics, offline template-based
reports, CSV/PDF/HTML/TXT export. No external API calls."""
from qgis.PyQt.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                                  QTextEdit, QPushButton, QLabel,
                                  QFileDialog, QMessageBox, QGroupBox)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont, QTextCursor
from qgis.core import QgsProject, QgsVectorLayer
from datetime import datetime
import csv


# Severity tiers — heuristic mapping from damage statistics to an overall
# situation classification. Used by the report generator to pick the
# response-recommendation template. Tuned to be conservative (over-classify
# rather than under).
SEVERITY_TIERS = [
    # (label, max_destroyed_pct, max_damaged_pct, tier_index)
    ("Minimal",      1.0,   5.0, 0),
    ("Localized",    5.0,  15.0, 1),
    ("Moderate",    15.0,  30.0, 2),
    ("Severe",      30.0,  50.0, 3),
    ("Catastrophic", 100.0, 100.0, 4),
]


def classify_severity(stats):
    """Map a stats dict to (tier_label, tier_index)."""
    dpct = stats.get('destroyed_pct', 0.0)
    apct = 100.0 * stats.get('damaged', 0) / max(stats.get('total_buildings', 1), 1)
    for label, max_d, max_a, idx in SEVERITY_TIERS:
        if dpct <= max_d and apct <= max_a:
            return label, idx
    return SEVERITY_TIERS[-1][0], SEVERITY_TIERS[-1][2]


class AssessmentReportsDock(QDockWidget):
    """Dockable Assessment Reports panel.

    Displays live statistics from the latest detection run and produces
    offline template-driven SitRep-style reports in three styles
    (Pattern analysis / Full SitRep / Response priorities). All reports
    are generated locally from the stats dict — no network calls, no
    API keys, no telemetry."""

    def __init__(self, iface, parent=None):
        super().__init__("Assessment Reports", parent)
        self.iface = iface
        self.current_results = None
        self.current_features = None
        self.current_layer_name = None

        self.setup_ui()
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(360)

    def set_assessment_data(self, features, layer_name):
        """Receive feature dicts + layer name from the detection pass."""
        self.current_features = features
        self.current_layer_name = layer_name
        self.current_results = self._compute_stats_from_features(features)
        self._render_statistics()

    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        title = QLabel("Assessment Reports")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        stats_group = QGroupBox("📊 Current Assessment Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel(
            "No assessment loaded.\nRun Building Damage Assessment first.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            "background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        stats_layout.addWidget(self.stats_label)

        refresh_btn = QPushButton("🔄 Refresh Statistics")
        refresh_btn.clicked.connect(self.refresh_statistics)
        stats_layout.addWidget(refresh_btn)

        layout.addWidget(stats_group)

        report_group = QGroupBox("📝 Generated Reports")
        report_layout = QVBoxLayout(report_group)

        self.report_display = QTextEdit()
        self.report_display.setReadOnly(True)
        self.report_display.setMinimumHeight(220)
        self.report_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.report_display.setHtml("""
            <p style='color: #666;'>Run a detection first, then click a button below to generate a report.</p>
            <p style='color: #666;'>All reports are generated locally from your assessment data. No data leaves your machine.</p>
        """)
        report_layout.addWidget(self.report_display)

        actions = QVBoxLayout()

        self.pattern_btn = QPushButton("🔍 Damage Pattern Analysis")
        self.pattern_btn.clicked.connect(self.report_patterns)
        actions.addWidget(self.pattern_btn)

        self.sitrep_btn = QPushButton("📰 Full SitRep")
        self.sitrep_btn.clicked.connect(self.report_sitrep)
        actions.addWidget(self.sitrep_btn)

        self.priorities_btn = QPushButton("🚨 Response Priorities")
        self.priorities_btn.clicked.connect(self.report_priorities)
        actions.addWidget(self.priorities_btn)

        report_layout.addLayout(actions)
        layout.addWidget(report_group)

        # Narrative report exports (PDF / HTML / TXT).
        export_group = QGroupBox("📤 Export Report")
        export_layout = QHBoxLayout(export_group)

        self.export_pdf_btn = QPushButton("Save as PDF")
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        export_layout.addWidget(self.export_pdf_btn)

        self.export_html_btn = QPushButton("Save as HTML")
        self.export_html_btn.clicked.connect(self.export_html)
        export_layout.addWidget(self.export_html_btn)

        self.export_txt_btn = QPushButton("Save as TXT")
        self.export_txt_btn.clicked.connect(self.export_txt)
        export_layout.addWidget(self.export_txt_btn)

        layout.addWidget(export_group)

        # Raw per-polygon data export (CSV).
        data_export_group = QGroupBox("📊 Export Data")
        data_export_layout = QHBoxLayout(data_export_group)

        self.export_csv_btn = QPushButton("Save as CSV")
        self.export_csv_btn.setToolTip(
            "Export per-polygon damage data (id, class, area, confidence, "
            "centroid, WKT geometry) with a self-describing header section. "
            "Opens cleanly in Excel and Survey123/Kobo workflows."
        )
        self.export_csv_btn.clicked.connect(self.export_csv)
        data_export_layout.addWidget(self.export_csv_btn)

        layout.addWidget(data_export_group)

        layout.addStretch()
        self.setWidget(main_widget)

    def refresh_statistics(self):
        """Recompute stats — prefer in-memory features, fall back to scanning
        QgsProject for a 'Building Damage Assessment' vector layer."""
        if self.current_features is not None:
            self.current_results = self._compute_stats_from_features(
                self.current_features)
        else:
            self.current_results = self.get_assessment_data()
        self._render_statistics()

    def _compute_stats_from_features(self, features):
        """Build the stats dict from raw feature dicts."""
        stats = {
            'total_buildings': 0,
            'no_damage': 0,
            'minor_damage': 0,
            'major_damage': 0,
            'destroyed': 0,
            'total_area': 0.0,
            'destroyed_area': 0.0,
            'major_area': 0.0,
            'no_damage_pct': 0.0,
            'minor_damage_pct': 0.0,
            'major_damage_pct': 0.0,
            'destroyed_pct': 0.0,
            'mean_conf': 0.0,
            'damaged': 0,
        }
        if not features:
            return stats

        confs = []
        for feat in features:
            c = int(feat.get('damage_class', 0))
            area = float(feat.get('area', 0.0))
            stats['total_buildings'] += 1
            stats['total_area'] += area
            confs.append(float(feat.get('confidence', 0.0)))
            if c == 1:
                stats['no_damage'] += 1
            elif c == 2:
                stats['minor_damage'] += 1
            elif c == 3:
                stats['major_damage'] += 1
                stats['major_area'] += area
            elif c == 4:
                stats['destroyed'] += 1
                stats['destroyed_area'] += area

        total = max(stats['total_buildings'], 1)
        stats['no_damage_pct'] = 100 * stats['no_damage'] / total
        stats['minor_damage_pct'] = 100 * stats['minor_damage'] / total
        stats['major_damage_pct'] = 100 * stats['major_damage'] / total
        stats['destroyed_pct'] = 100 * stats['destroyed'] / total
        stats['mean_conf'] = (sum(confs) / len(confs)) if confs else 0.0
        stats['damaged'] = (stats['minor_damage'] + stats['major_damage']
                            + stats['destroyed'])
        return stats

    def _render_statistics(self):
        """Repaint stats_label from self.current_results."""
        if not self.current_results:
            self.stats_label.setText(
                "No assessment loaded.\n\n"
                "Run Building Damage Assessment first."
            )
            return

        stats = self.current_results
        total = stats['total_buildings']
        damaged_pct = (100 * stats.get('damaged', 0) / max(total, 1))
        severity, _ = classify_severity(stats)

        layer_line = (f"<b>📍 {self.current_layer_name}</b><br>"
                      f"━━━━━━━━━━━━━━━━━━━━━<br>"
                      if self.current_layer_name else
                      "<b>Assessment Summary</b><br>"
                      "━━━━━━━━━━━━━━━━━━━━━<br>")

        stats_text = f"""
{layer_line}
📊 <b>Total Buildings:</b> {total}<br>
📐 <b>Total Area:</b> {stats['total_area']:,.0f} m²<br>
⚠️ <b>Damaged (any level):</b> {stats.get('damaged', 0)} ({damaged_pct:.1f}%)<br>
🎯 <b>Mean Confidence:</b> {stats.get('mean_conf', 0.0):.2f}<br>
🚦 <b>Severity tier:</b> {severity}<br><br>

<b>Damage Breakdown:</b><br>
🟢 No Damage: {stats['no_damage']} ({stats['no_damage_pct']:.1f}%)<br>
🟡 Minor Damage: {stats['minor_damage']} ({stats['minor_damage_pct']:.1f}%)<br>
🟠 Major Damage: {stats['major_damage']} ({stats['major_damage_pct']:.1f}%)<br>
🔴 Destroyed: {stats['destroyed']} ({stats['destroyed_pct']:.1f}%)<br><br>

<b>Affected Area:</b><br>
🔴 Destroyed Area: {stats['destroyed_area']:,.0f} m²<br>
🟠 Major Damage Area: {stats['major_area']:,.0f} m²
"""
        self.stats_label.setText("")
        self.stats_label.setTextFormat(Qt.RichText)
        self.stats_label.setText(stats_text)

    def get_assessment_data(self):
        """Legacy: scan QgsProject for a named damage layer and tally stats."""
        layers = QgsProject.instance().mapLayersByName("Building Damage Assessment")
        if not layers:
            return None
        layer = layers[0]
        if not isinstance(layer, QgsVectorLayer):
            return None

        stats = {
            'total_buildings': 0,
            'no_damage': 0,
            'minor_damage': 0,
            'major_damage': 0,
            'destroyed': 0,
            'total_area': 0,
            'destroyed_area': 0,
            'major_area': 0,
            'no_damage_pct': 0,
            'minor_damage_pct': 0,
            'major_damage_pct': 0,
            'destroyed_pct': 0,
            'mean_conf': 0.0,
            'damaged': 0,
        }
        field_names = [f.name() for f in layer.fields()]
        for feature in layer.getFeatures():
            stats['total_buildings'] += 1
            damage_class = feature['damage_class']
            area = feature['area_m2'] if 'area_m2' in field_names else 0
            stats['total_area'] += area
            if damage_class == 1:
                stats['no_damage'] += 1
            elif damage_class == 2:
                stats['minor_damage'] += 1
            elif damage_class == 3:
                stats['major_damage'] += 1
                stats['major_area'] += area
            elif damage_class == 4:
                stats['destroyed'] += 1
                stats['destroyed_area'] += area

        if stats['total_buildings'] > 0:
            for key in ('no_damage', 'minor_damage', 'major_damage', 'destroyed'):
                stats[f'{key}_pct'] = 100 * stats[key] / stats['total_buildings']
            stats['damaged'] = (stats['minor_damage'] + stats['major_damage']
                                + stats['destroyed'])
        return stats

    # ------------------------------------------------------------------
    # Offline report generators (template-driven, no LLM)
    # ------------------------------------------------------------------
    def report_patterns(self):
        self.refresh_statistics()
        if not self.current_results:
            QMessageBox.warning(self, "No Data",
                                "Run Building Damage Assessment first.")
            return
        report = self._build_pattern_report(self.current_results)
        self._render_report(report, "Damage Pattern Analysis")

    def report_sitrep(self):
        self.refresh_statistics()
        if not self.current_results:
            QMessageBox.warning(self, "No Data",
                                "Run Building Damage Assessment first.")
            return
        report = self._build_sitrep(self.current_results)
        self._render_report(report, "Full SitRep")

    def report_priorities(self):
        self.refresh_statistics()
        if not self.current_results:
            QMessageBox.warning(self, "No Data",
                                "Run Building Damage Assessment first.")
            return
        report = self._build_priorities(self.current_results)
        self._render_report(report, "Response Priorities")

    def _render_report(self, markdown_text, title):
        """Display a generated report in the panel, replacing previous content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = self._markdown_to_html(markdown_text)
        wrapper = f"""
<div style='background-color: #f0f7f4; padding: 12px; border-radius: 5px; margin: 4px 0;'>
  <p style='color: #555; font-size: 10px; margin: 0;'>📄 {title} — generated {timestamp}</p>
  <hr style='border: none; border-top: 1px solid #ccc; margin: 6px 0;'>
  {html}
</div>
"""
        self.report_display.setHtml(wrapper)
        cursor = self.report_display.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.report_display.setTextCursor(cursor)

    def _markdown_to_html(self, text):
        """Minimal Markdown -> HTML for the in-panel display. Handles headings,
        bullet lists, bold, italics, line breaks."""
        lines = text.split('\n')
        out = []
        in_list = False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith('### '):
                if in_list:
                    out.append('</ul>')
                    in_list = False
                out.append(f"<h4>{stripped[4:]}</h4>")
            elif stripped.startswith('## '):
                if in_list:
                    out.append('</ul>')
                    in_list = False
                out.append(f"<h3>{stripped[3:]}</h3>")
            elif stripped.startswith('# '):
                if in_list:
                    out.append('</ul>')
                    in_list = False
                out.append(f"<h2>{stripped[2:]}</h2>")
            elif stripped.startswith('- '):
                if not in_list:
                    out.append('<ul>')
                    in_list = True
                out.append(f"<li>{stripped[2:]}</li>")
            elif stripped == '':
                if in_list:
                    out.append('</ul>')
                    in_list = False
                out.append('<br>')
            else:
                if in_list:
                    out.append('</ul>')
                    in_list = False
                out.append(f"<p>{stripped}</p>")
        if in_list:
            out.append('</ul>')
        html = '\n'.join(out)
        # bold + italics (simple, non-nested)
        import re
        html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
        html = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', html)
        return html

    def _build_pattern_report(self, stats):
        """Damage-pattern-focused report. Emphasizes the shape of the
        damage distribution rather than recommendations."""
        severity, _ = classify_severity(stats)
        total = stats['total_buildings']
        damaged = stats['damaged']
        dpct = (100 * damaged / max(total, 1))

        # Dominant class
        class_counts = [
            ('No Damage', stats['no_damage'], stats['no_damage_pct']),
            ('Minor', stats['minor_damage'], stats['minor_damage_pct']),
            ('Major', stats['major_damage'], stats['major_damage_pct']),
            ('Destroyed', stats['destroyed'], stats['destroyed_pct']),
        ]
        dominant = max(class_counts, key=lambda c: c[1])

        # Damage skew: how much of the damage is concentrated in destroyed?
        if damaged > 0:
            skew_pct = 100 * stats['destroyed'] / damaged
            if skew_pct > 60:
                skew_desc = ("Damage is **heavily skewed to total destruction** "
                             "— most affected buildings are unrecoverable.")
            elif skew_pct > 30:
                skew_desc = ("A substantial fraction of damaged buildings are "
                             "destroyed; recovery operations should expect "
                             "significant rebuild rather than repair.")
            else:
                skew_desc = ("Most damaged buildings show partial damage rather "
                             "than total destruction; repair and structural "
                             "stabilization should dominate the response.")
        else:
            skew_desc = "No damaged buildings detected in this assessment."

        # Confidence note
        conf = stats.get('mean_conf', 0.0)
        if conf > 0.75:
            conf_note = f"Mean model confidence is high ({conf:.2f})."
        elif conf > 0.5:
            conf_note = (f"Mean model confidence is moderate ({conf:.2f}). "
                         f"Recommend cross-checking against high-confidence "
                         f"reference imagery before publishing.")
        else:
            conf_note = (f"Mean model confidence is low ({conf:.2f}). "
                         f"Treat this output as preliminary screening only.")

        return f"""# Damage Pattern Analysis

**Layer**: {self.current_layer_name or 'current assessment'}
**Severity tier**: {severity}
**Total buildings**: {total} ({dpct:.1f}% damaged at any level)
**Total footprint**: {stats['total_area']:,.0f} m²

## Distribution

The dominant class is **{dominant[0]}** with {dominant[1]} buildings ({dominant[2]:.1f}% of the scene). The full breakdown:

- No Damage: {stats['no_damage']} ({stats['no_damage_pct']:.1f}%)
- Minor Damage: {stats['minor_damage']} ({stats['minor_damage_pct']:.1f}%)
- Major Damage: {stats['major_damage']} ({stats['major_damage_pct']:.1f}%)
- Destroyed: {stats['destroyed']} ({stats['destroyed_pct']:.1f}%)

## Interpretation

{skew_desc}

Total damaged footprint (Major + Destroyed): {stats['destroyed_area'] + stats['major_area']:,.0f} m².

## Model reliability

{conf_note}

## Caveats

This is an AI-generated draft assessment. Outputs must be reviewed by a qualified analyst before being used to drive response, allocation, or policy decisions. Off-nadir or cross-sensor imagery can systematically inflate the Destroyed class via rooftop parallax — verify against the source imagery before publishing.
"""

    def _build_sitrep(self, stats):
        """Full SitRep-style situation report with executive summary,
        findings, and impact assessment."""
        severity, sev_idx = classify_severity(stats)
        total = stats['total_buildings']
        damaged = stats['damaged']
        dpct = (100 * damaged / max(total, 1))

        if sev_idx == 0:
            exec_summary = (
                f"Damage to the surveyed area is **minimal**. Of {total} "
                f"buildings detected, {damaged} ({dpct:.1f}%) show any "
                f"damage; {stats['destroyed']} ({stats['destroyed_pct']:.1f}%) "
                f"are destroyed. Routine assessment and document-only response "
                f"are appropriate.")
        elif sev_idx == 1:
            exec_summary = (
                f"Damage is **localized**. Of {total} buildings, {damaged} "
                f"({dpct:.1f}%) are damaged with {stats['destroyed']} "
                f"({stats['destroyed_pct']:.1f}%) destroyed. Single-team "
                f"neighborhood-scale response, 1–3 day operation.")
        elif sev_idx == 2:
            exec_summary = (
                f"**Moderate-scale damage** detected. {damaged} of {total} "
                f"buildings ({dpct:.1f}%) damaged; {stats['destroyed']} "
                f"destroyed ({stats['destroyed_pct']:.1f}%). Multi-team "
                f"response over 1–2 weeks; mutual-aid coordination warranted.")
        elif sev_idx == 3:
            exec_summary = (
                f"**Severe damage** across the surveyed area. {damaged} of "
                f"{total} buildings ({dpct:.1f}%) damaged with {stats['destroyed']} "
                f"destroyed ({stats['destroyed_pct']:.1f}%). Full activation, "
                f"mutual aid request, and structured displacement support "
                f"required.")
        else:
            exec_summary = (
                f"**Catastrophic damage** to the surveyed area. {damaged} of "
                f"{total} buildings ({dpct:.1f}%) damaged with {stats['destroyed']} "
                f"({stats['destroyed_pct']:.1f}%) destroyed. State/federal "
                f"and possibly international response required; expect a "
                f"multi-month recovery timeline.")

        return f"""# Situation Report — Building Damage Assessment

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Source layer**: {self.current_layer_name or 'current assessment'}
**Method**: AI-assisted draft via the beaconGIS QGIS plugin 
**Severity tier**: {severity}

## 1. Executive summary

{exec_summary}

## 2. Methodology

The damage classification was produced by a two-network deep-learning pipeline:

- **Localization** identifies building footprints from the pre-disaster image.
- **Classification** compares pre- and post-disaster imagery and assigns one of four damage classes per building: No Damage, Minor, Major, Destroyed.
- An ensemble of independently-trained classification models is averaged for a more robust per-pixel softmax than any single model produces.

The model was trained on the xView2 / xBD dataset with Inria Aerial Image Labeling mid-training. Official xView2 scorer Combined F1 = 0.7312.

## 3. Findings

**Building inventory (this scene)**

- Total detected: {total}
- Total footprint area: {stats['total_area']:,.0f} m²

**Damage distribution**

- No Damage: {stats['no_damage']} ({stats['no_damage_pct']:.1f}%)
- Minor Damage: {stats['minor_damage']} ({stats['minor_damage_pct']:.1f}%)
- Major Damage: {stats['major_damage']} ({stats['major_damage_pct']:.1f}%)
- Destroyed: {stats['destroyed']} ({stats['destroyed_pct']:.1f}%)

**Damaged footprint area (Major + Destroyed)**: {stats['destroyed_area'] + stats['major_area']:,.0f} m²

## 4. Reliability indicators

- Mean per-building classification confidence: {stats.get('mean_conf', 0.0):.2f}

## 5. Impact assessment

{exec_summary}

## 6. Recommended actions

See the "Response Priorities" report for a tier-specific action breakdown.

## 7. Caveats and limitations

- **Draft assessment only.** AI outputs require analyst review before being used for response, allocation, or policy decisions.
- **Imagery alignment.** Off-nadir, oblique, or cross-sensor pre/post pairs can systematically inflate the Destroyed class via rooftop parallax. Verify against the source imagery before publishing.
- **Class accuracy is not uniform.** Minor-damage detection has the lowest historical F1 (~0.44–0.50); use manual filtering by class + confidence to triage uncertain buildings.
- **No human-confirmed ground truth.** This SitRep is generated from model predictions only.
"""

    def _build_priorities(self, stats):
        """Response-priority report — tier-specific action items."""
        severity, sev_idx = classify_severity(stats)
        damaged = stats['damaged']

        priorities_by_tier = {
            0: [
                "Document and archive imagery for after-action review.",
                "Update damage-tracking dashboards with current assessment.",
                "No immediate field deployment required.",
            ],
            1: [
                "Deploy a single damage-verification team to the affected neighborhood within 24 hours.",
                "Cross-check the lowest-confidence detected buildings against on-the-ground reports.",
                "Identify any destroyed buildings sheltering vulnerable populations as a priority.",
                "Pre-position one mutual-aid liaison in case the assessment expands.",
            ],
            2: [
                "Activate multi-team field deployment (3–5 teams) within 12 hours.",
                "Coordinate mutual-aid request through the regional emergency operations center.",
                "Set up displaced-persons reception at the nearest 2–3 unaffected community sites.",
                "Begin structural triage on all Major-class buildings (occupancy decisions within 48 hours).",
                "Survey utility infrastructure (power, water, gas) in the affected footprint.",
                "Expect a 1–2 week field operation; brief logistics accordingly.",
            ],
            3: [
                "Full operational activation: state/regional emergency declaration recommended.",
                "Mutual-aid request to neighboring jurisdictions and federal partners.",
                "Mass-care setup: shelter for an estimated population displaced by the destroyed/major-class buildings (use occupancy estimates from local records).",
                "Search-and-rescue prioritization on all Destroyed-class buildings within the first 72 hours.",
                "Standing damage-assessment cell with daily SitRep cadence for the first 7 days.",
                "Critical infrastructure restoration as parallel workstream (power, water, comms, hospitals, schools).",
                "Begin debris-removal planning by day 5; expect a 4–8 week field operation.",
            ],
            4: [
                "Catastrophe response: federal / national activation recommended.",
                "International humanitarian-aid liaison if cross-border assistance is anticipated.",
                "Surge medical capacity to the affected area within 48 hours (mass-casualty contingency).",
                "Mass-displacement support: shelter, food, water, sanitation, communications for an at-scale displaced population.",
                "Search-and-rescue at maximum deployment for the first 72–96 hours.",
                "Restoration of life-safety infrastructure (water, sanitation, comms, medical) as the top parallel priority.",
                "Public-information cell to manage media, official statements, and family-reunification information.",
                "Long-term recovery planning to begin in parallel during week 2; expect multi-month sustained operations.",
            ],
        }

        priorities = priorities_by_tier[sev_idx]
        timeline_label = {
            0: "Routine (no field deployment)",
            1: "1–3 days",
            2: "1–2 weeks",
            3: "4–8 weeks",
            4: "Months to years",
        }[sev_idx]

        bullets = '\n'.join(f"- {p}" for p in priorities)

        return f"""# Response Priorities

**Layer**: {self.current_layer_name or 'current assessment'}
**Severity tier**: {severity}
**Affected buildings**: {damaged} of {stats['total_buildings']} ({100 * damaged / max(stats['total_buildings'], 1):.1f}%)
**Estimated operational timeline**: {timeline_label}

## Recommended actions (in order of priority)

{bullets}

## Resource considerations

- **Footprint of damaged buildings (Major + Destroyed)**: {stats['destroyed_area'] + stats['major_area']:,.0f} m²
- **Destroyed-only footprint**: {stats['destroyed_area']:,.0f} m²
- These areas drive debris-removal, shelter, and rebuild-cost estimates. Pair with local occupancy and building-value records to derive personnel and budget requirements.

## Reliability gate

Mean model confidence: {stats.get('mean_conf', 0.0):.2f}.

In QGIS, sort the damage layer by the `confidence` attribute and manually review the lowest-confidence buildings before publishing this priority list externally.

## Caveats

These recommendations are template-derived from the severity-tier classification of the AI assessment. They are starting points for an experienced responder, not a substitute for situational judgment. Always reconcile against on-the-ground reports.
"""

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_csv(self):
        """Export per-polygon data as CSV with a self-describing header."""
        if self.current_features is None:
            if self.current_results is None:
                self.refresh_statistics()
            QMessageBox.warning(
                self, "No Data",
                "No assessment data available to export. Run Building Damage "
                "Assessment first, then try again."
            )
            return

        default_name = (self.current_layer_name.replace(':', '-')
                        if self.current_layer_name else
                        "damage_assessment") + ".csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Damage Data as CSV", default_name, "CSV (*.csv)"
        )
        if not path:
            return

        stats = (self.current_results or
                 self._compute_stats_from_features(self.current_features))

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([f"# Layer: {self.current_layer_name or 'unknown'}"])
                w.writerow([f"# Generated: {datetime.now().isoformat()}"])
                w.writerow([f"# Total buildings: {stats['total_buildings']}"])
                w.writerow([f"# Damaged (any level): {stats.get('damaged', 0)}"])
                w.writerow([f"# Destroyed: {stats['destroyed']}"])
                w.writerow([f"# Mean confidence: {stats.get('mean_conf', 0.0):.4f}"])
                w.writerow([f"# Total footprint (m^2): {stats['total_area']:.1f}"])
                for key, label in [('no_damage', 'No Damage'),
                                    ('minor_damage', 'Minor'),
                                    ('major_damage', 'Major'),
                                    ('destroyed', 'Destroyed')]:
                    w.writerow([f"# {label}: {stats[key]}"])
                w.writerow([])
                w.writerow(['id', 'damage_class', 'damage_name', 'area_m2',
                            'confidence', 'centroid_x', 'centroid_y', 'wkt'])
                for idx, feat in enumerate(self.current_features, start=1):
                    g = feat.get('geometry')
                    centroid = g.centroid().asPoint() if g is not None else None
                    cx = centroid.x() if centroid else ''
                    cy = centroid.y() if centroid else ''
                    wkt = g.asWkt() if g is not None else ''
                    w.writerow([
                        idx,
                        feat.get('damage_class', 0),
                        feat.get('damage_name', ''),
                        round(float(feat.get('area', 0.0)), 2),
                        round(float(feat.get('confidence', 0.0)), 4),
                        cx, cy, wkt,
                    ])
            QMessageBox.information(
                self, "Saved", f"CSV report written to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Could not save CSV report:\n\n{e}"
            )

    def export_pdf(self):
        try:
            from qgis.PyQt.QtPrintSupport import QPrinter
            from qgis.PyQt.QtGui import QTextDocument

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Report as PDF",
                f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf)")

            if file_path:
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)

                doc = QTextDocument()
                doc.setHtml(self.create_export_html())
                doc.print_(printer)

                QMessageBox.information(self, "Success",
                                        f"Report saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error",
                                f"Failed to export PDF: {str(e)}")

    def export_html(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report as HTML",
            f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html)")

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.create_export_html())
                QMessageBox.information(self, "Success",
                                        f"Report saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error",
                                    f"Failed to save: {str(e)}")

    def export_txt(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report as Text",
            f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)")

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.report_display.toPlainText())
                QMessageBox.information(self, "Success",
                                        f"Report saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error",
                                    f"Failed to save: {str(e)}")

    def create_export_html(self):
        """Standalone HTML for PDF/HTML export. Includes stats table + the
        currently-displayed report content."""
        stats_html = ""
        if self.current_results:
            stats = self.current_results
            severity, _ = classify_severity(stats)
            stats_html = f"""
<h2>Assessment Statistics</h2>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><td><b>Total Buildings</b></td><td>{stats['total_buildings']}</td></tr>
  <tr><td><b>Total Area</b></td><td>{stats['total_area']:.0f} m²</td></tr>
  <tr><td><b>Severity tier</b></td><td>{severity}</td></tr>
  <tr><td style="background-color: #90EE90;">No Damage</td><td>{stats['no_damage']} ({stats['no_damage_pct']:.1f}%)</td></tr>
  <tr><td style="background-color: #FFFF00;">Minor Damage</td><td>{stats['minor_damage']} ({stats['minor_damage_pct']:.1f}%)</td></tr>
  <tr><td style="background-color: #FFA500;">Major Damage</td><td>{stats['major_damage']} ({stats['major_damage_pct']:.1f}%)</td></tr>
  <tr><td style="background-color: #FF6B6B;">Destroyed</td><td>{stats['destroyed']} ({stats['destroyed_pct']:.1f}%)</td></tr>
</table>
"""

        return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Building Damage Assessment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ color: #333; border-bottom: 2px solid #333; }}
    h2 {{ color: #666; margin-top: 24px; }}
    table {{ border-collapse: collapse; margin: 20px 0; }}
    .report-content {{ margin-top: 30px; }}
  </style>
</head>
<body>
  <h1>🏗️ Building Damage Assessment Report</h1>
  <p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  {stats_html}
  <div class="report-content">
    <h2>Generated Report</h2>
    {self.report_display.toHtml()}
  </div>
</body>
</html>
"""
