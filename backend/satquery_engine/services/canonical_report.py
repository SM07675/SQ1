"""Nine report sections rendered with humanized, simple language from the canonical analysis result."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4


def write_manifest(output_dir, payload):
    path = output_dir / "analysis.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Friendly label mappings for human-readable reports
# ---------------------------------------------------------------------------

_FRIENDLY_STATUS = {
    "supported": "✅ Results look good",
    "supported_with_limitations": "✅ Results available (with some notes)",
    "low_confidence": "⚠️ Results have low confidence",
    "insufficient_evidence": "❌ Not enough evidence to answer",
    "invalid_input": "❌ Image could not be processed",
    "unsupported_task": "❌ This type of analysis is not supported",
    "disputed": "⚠️ Results are disputed",
    "model_unavailable": "⚠️ AI model was not available",
    "degraded_analysis": "⚠️ Partial analysis completed",
}

_FRIENDLY_METRIC_NAMES = {
    "count": "Total Count",
    "count_state": "Count Status",
    "area_m2": "Area (sq. metres)",
    "selected_pixels": "Detected Pixels",
    "region_count": "Number of Regions",
    "coverage_percent": "Coverage (%)",
    "before_count": "Count (Before)",
    "after_count": "Count (After)",
    "persistent_count": "Unchanged Count",
    "possible_new_count": "Possibly New",
    "possible_removed_count": "Possibly Removed",
    "net_count_change": "Net Change",
    "net_footprint_area_m2": "Net Area Change (sq. m)",
    "changed_pixels": "Changed Pixels",
    "changed_percent": "Changed Area (%)",
    "total_pixels": "Total Pixels",
    "gain_pixels": "Gained Pixels",
    "loss_pixels": "Lost Pixels",
    "net_percentage_points": "Net Change (%)",
    "resolution_m": "Resolution (metres/pixel)",
    "water_percent": "Water Coverage (%)",
    "land_percent": "Classified Land Coverage (%)",
    "unknown_percent": "Unclassified Coverage (%)",
    "possible_land_percent": "Possible Land Upper Bound (%)",
    "water_percent": "Estimated Water Coverage (%)",
    "vegetation_percent": "Green/Vegetation (%)",
    "built_up_percent": "Built-up Area (%)",
}

_FRIENDLY_CLASS_NAMES = {
    "water": "💧 Water Bodies",
    "vegetation": "🌿 Grass & Vegetation",
    "built_up": "🏗️ Buildings & Structures",
    "unknown": "🗺️ Unclassified Area",
    "woodland": "🌲 Forest & Woodland",
    "road": "🛣️ Roads",
    "bare_pervious": "🏜️ Bare Soil / Land",
    "agriculture": "🌾 Agricultural Fields",
    "snow": "❄️ Snow & Ice",
    "swimming_pool": "🏊 Swimming Pools",
}


def _friendly_metric(key):
    return _FRIENDLY_METRIC_NAMES.get(key, key.replace("_", " ").title())


def _friendly_class(name):
    return _FRIENDLY_CLASS_NAMES.get(name, name.replace("_", " ").title())


def _confidence_explanation(score, kind=None):
    """Convert a numeric confidence score to a human-friendly explanation."""
    if kind == "uncalibrated_evidence_strength":
        return f"Evidence strength {score:.2f} (uncalibrated); this is not an accuracy probability."
    if score >= 0.8:
        return f"{score:.1%} — High confidence. The analysis is well-supported by the image data."
    elif score >= 0.5:
        return f"{score:.1%} — Moderate confidence. Results are reasonable but should be verified."
    elif score >= 0.3:
        return f"{score:.1%} — Low confidence. Results are approximate and may have errors."
    else:
        return f"{score:.1%} — Very low confidence. Results should be treated as rough estimates only."


def _simplify_limitation(text):
    """Convert technical limitation text into simpler language."""
    replacements = {
        "RGB water proxy uses multi-cue optical evidence. True multispectral NIR/SWIR bands were absent.":
            "The image only has basic colors (RGB), so water detection used visual clues rather than specialized satellite sensors. Results may be less accurate.",
        "Vegetation and water classified using RGB optical proxy; true multispectral bands unavailable.":
            "Plant and water areas were identified using basic color analysis. Specialized sensor data was not available for more precise detection.",
        "BigEarthNet scene-level inference is unavailable":
            "The AI scene classification model was not available for this analysis.",
        "Land classes are estimates from available spatial evidence. Unknown pixels may be land or water and are excluded from classified land and water totals.":
            "Land types shown are estimates. Unclassified areas could be land or water and are excluded from the classified totals.",
    }
    for old, new in replacements.items():
        if old in text:
            return new
    return text


def write_pdf_report(output_dir: Path, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "GeoProof_Report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallText", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=5, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="FriendlyBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=10, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="Highlight", fontName="Helvetica-Bold", fontSize=10, leading=14, spaceAfter=8, textColor=colors.HexColor("#1a5276")))
    styles["BodyText"].leading = 14; styles["BodyText"].spaceAfter = 8
    styles["Title"].textColor = colors.HexColor("#123047")
    story = []

    def p(value, style="FriendlyBody"):
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style])

    def heading(text):
        story.append(p(text, "Heading2"))

    def page(number, title):
        if story:
            story.append(PageBreak())
        story.append(p(f"SATQUERY / {number:02d}", "SmallText"))
        story.append(p(title, "Title"))
        story.append(Spacer(1, 14))

    def table(rows):
        if not rows:
            return
        t = Table([[p(a, "SmallText"), p(b, "SmallText")] for a, b in rows], colWidths=[165, 330], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf4f8")),
            ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#cdd9e0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    def visual(artifact, height=540):
        relative = artifact["url"].split(f'/{payload["result_id"]}/', 1)[-1]
        local = (output_dir / relative).resolve()
        if not local.is_relative_to(output_dir.resolve()) or not local.is_file():
            story.append(p("The requested evidence image is unavailable."))
            return
        w, h = ImageReader(str(local)).getSize()
        scale = min(495 / w, height / h)
        story.append(Image(str(local), width=w * scale, height=h * scale))
        story.append(Spacer(1, 8))
        story.append(p(artifact["name"], "SmallText"))

    verdict = payload["verdict"]
    artifacts = payload.get("artifacts", [])
    overlays = [a for a in artifacts if a["mime_type"] == "image/png" and "overlay" in a["name"]]
    previews = [a for a in artifacts if a["mime_type"] == "image/png" and a["name"].startswith("preview")]

    # ── Page 1: Summary for the user ─────────────────────────────────
    page(1, "Your Analysis Results")
    story.append(p("What you asked:", "Highlight"))
    story.append(p(payload["query"]))
    story.append(Spacer(1, 8))
    story.append(p("What we found:", "Highlight"))
    story.append(p(verdict["answer"]))
    story.append(Spacer(1, 8))

    friendly_status = _FRIENDLY_STATUS.get(verdict["status"], verdict["status"].replace("_", " ").title())
    conf_text = _confidence_explanation(verdict["confidence"], verdict.get("confidence_kind"))

    table([
        ("Result Status", friendly_status),
        ("How confident are we?", conf_text),
        ("Analysis ID", payload["result_id"]),
        ("Date & Time", payload["generated_at"]),
    ])
    heading("Please Note")
    story.append(p(
        "This report shows what the satellite image analysis found. "
        "The colored areas on the map are the system's best estimates based on the available image data. "
        "Some areas may not be perfectly accurate, especially when using basic RGB images without specialized satellite sensors."
    ))

    # ── Page 2: Main annotated image ─────────────────────────────────
    page(2, "Annotated Satellite Image")
    story.append(p(
        "The image below shows the analysis results overlaid on your satellite image. "
        "Different colors represent different types of land, water, and structures."
    ))
    story.append(Spacer(1, 8))
    if overlays:
        visual(overlays[-1] if any("building_match" in a["url"] for a in overlays) else overlays[0], height=390)
    elif previews:
        visual(previews[0], height=390)
        story.append(p("Original image shown. No detection overlay was produced for this analysis."))
    else:
        story.append(p("No image artifact was available for this request."))

    # Color legend
    heading("Color Legend")
    legend_rows = [
        ("Deep Blue", "Water bodies (rivers, lakes, ponds)"),
        ("Bright Green", "Grass and vegetation"),
        ("Dark Green", "Forest and woodland"),
        ("Warm Red", "Buildings and structures"),
        ("Bright Yellow", "Roads"),
        ("Dark Yellow/Sandy", "Unclassified area"),
        ("Tan/Brown", "Bare soil"),
        ("Light Olive", "Agricultural fields"),
    ]
    table(legend_rows)

    # ── Page 3: Key measurements ─────────────────────────────────────
    page(3, "Key Measurements")
    story.append(p("Here are the important numbers from the analysis:"))
    story.append(Spacer(1, 6))

    keys = {"count", "count_state", "area_m2", "selected_pixels", "region_count", "coverage_percent",
            "valid_pixels", "land_pixels", "land_percent", "unknown_pixels", "unknown_percent", "possible_land_percent", "water_percent",
            "before_count", "after_count", "persistent_count",
            "possible_new_count", "possible_removed_count", "net_count_change",
            "net_footprint_area_m2", "changed_pixels", "changed_percent", "total_pixels",
            "gain_pixels", "loss_pixels", "net_percentage_points", "resolution_m"}

    for name, stats in payload.get("statistics", {}).items():
        heading(name.replace("_", " ").title())
        visible_keys = keys - {"selected_pixels", "coverage_percent", "area_m2"} if name == "land_cover" else keys
        table([
            (_friendly_metric(k), format(v, ".2f") if isinstance(v, float) else str(v))
            for k, v in stats.items() if k in visible_keys
        ])
        if stats.get("breakdown"):
            heading("Land Cover Breakdown")
            table([
                (_friendly_class(name),
                 f'{values["percent"]:.2f}% of valid pixels ({values["pixels"]:,} pixels)' +
                 (f' — about {values["area_m2"]:,.0f} square metres' if values.get("area_m2") is not None else ''))
                for name, values in stats["breakdown"].items() if values.get("pixels", 0) > 0
            ])
    if not payload.get("statistics"):
        story.append(p("No measurements were produced for this analysis."))

    # ── Page 4: Additional evidence images ───────────────────────────
    page(4, "Supporting Images")
    story.append(p("These additional images show different aspects of the analysis:"))
    story.append(Spacer(1, 6))
    evidence_images = previews + overlays[1:]
    for i, a in enumerate(evidence_images):
        if i and i % 2 == 0:
            story.append(PageBreak())
            heading("Supporting Images (continued)")
        visual(a, height=265)
    if not evidence_images:
        story.append(p("No additional supporting images were generated."))

    # ── Page 5: Location & map information ───────────────────────────
    page(5, "Location & Map Information")
    story.append(p("Details about the geographic location and coordinate system of the analyzed image:"))
    story.append(Spacer(1, 6))
    for asset in payload.get("assets", []):
        heading(asset["filename"])
        table([
            ("Coordinate System", asset.get("crs") or "Unknown — pixel coordinates only"),
            ("Image Boundaries", str(asset.get("bounds"))),
            ("Pixel Size", str(asset.get("resolution"))),
            ("Image ID", str(asset.get("asset_id", "Not specified"))),
        ])
    story.append(p(
        "Geographic coordinates allow us to measure real-world areas (in square metres). "
        "If the coordinate system is unknown, we can only report pixel-based measurements."
    ))

    # ── Page 6: How the analysis was done ────────────────────────────
    page(6, "How This Analysis Was Done")
    story.append(p(
        "Your satellite image was analyzed using a combination of AI models and scientific algorithms. "
        "Here's what each step found:"
    ))
    story.append(Spacer(1, 6))
    for item in payload.get("evidence", []):
        heading(str(item.get("producer", "Analysis Step")).replace("_", " ").title())
        story.append(p(item.get("summary", "")))
        method_name = item.get("metrics", {}).get("method", "Automated analysis")
        table([("Method Used", str(method_name))])

    heading("AI Models Used")
    models = payload.get("models_used", [])
    if models:
        story.append(p("The following AI models and algorithms were used in this analysis:"))
        for m in models:
            story.append(p(f"• {m}", "SmallText"))
    else:
        story.append(p("No AI models were needed — the analysis used deterministic algorithms only."))

    # ── Page 7: Processing steps ─────────────────────────────────────
    page(7, "Processing Steps")
    story.append(p("Here's the sequence of steps the system performed:"))
    story.append(Spacer(1, 6))
    for step in payload.get("trace", []):
        status_icon = "✅" if step["status"] == "ok" else "⏭️" if step["status"] == "skipped" else "❌"
        table([
            ("Step", step["component"].replace("_", " ").title()),
            ("What it did", step["action"].replace("_", " ")),
            ("Result", f'{status_icon} {step["status"].replace("_", " ").title()}'),
            ("Time taken", f'{step["duration_ms"]:,} ms' if step["duration_ms"] > 0 else "Instant"),
        ])

    # ── Page 8: Data quality & things to know ────────────────────────
    page(8, "Data Quality & Things to Know")
    story.append(p(
        "Every analysis has limitations. Here's what you should know about the accuracy and quality of these results:"
    ))
    story.append(Spacer(1, 8))

    heading("Image Quality")
    for asset in payload.get("assets", []):
        heading(asset["filename"])
        nodata_pct = asset.get("nodata_percent", 0)
        quality_note = "Excellent" if nodata_pct < 1 else "Good" if nodata_pct < 5 else "Fair" if nodata_pct < 15 else "Poor"
        table([
            ("Number of Bands", str(asset.get("bands", "Unknown"))),
            ("Missing Data", f'{nodata_pct:.1f}% — {quality_note}'),
            ("Sensor", str(asset.get("sensor") or "Not specified")),
            ("Date Captured", str(asset.get("acquisition_date") or "Not specified")),
        ])

    heading("Important Notes")
    limitations = verdict.get("limitations", []) + payload.get("quality", {}).get("blockers", [])
    for item in dict.fromkeys(limitations):
        simplified = _simplify_limitation(str(item))
        story.append(p(f"• {simplified}"))

    story.append(Spacer(1, 8))
    story.append(p(
        "The confidence score shown in this report is a measure of how strong the evidence is — "
        "it is NOT a probability of being correct. "
        "Real-world accuracy depends on image quality, sensor type, and local conditions."
    ))

    # ── Page 9: Downloadable files ───────────────────────────────────
    page(9, "Available Downloads")
    story.append(p("The following files were created during this analysis and are available for download:"))
    story.append(Spacer(1, 6))
    for a in artifacts:
        friendly_name = a["name"].replace("_", " ").replace(".png", "").replace(".tif", "").replace(".geojson", "").replace(".json", "").replace(".pdf", "").title()
        story.append(p(f"📁 {friendly_name} ({a['name']})", "SmallText"))
    story.append(Spacer(1, 12))
    story.append(p(
        "You can download these files from SATQUERY. "
        "The 'analysis.json' file contains all results in machine-readable format. "
        "GeoJSON files can be opened in any GIS software like QGIS or Google Earth."
    ))

    def footer(c, doc):
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#52677a"))
        c.drawString(50, 27, "SATQUERY | Satellite Image Analysis Report")
        c.drawRightString(A4[0] - 50, 27, f"Page {doc.page}")

    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=50, leftMargin=50,
        topMargin=45, bottomMargin=45,
        title="SATQUERY Analysis Report", author="SATQUERY"
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return path


