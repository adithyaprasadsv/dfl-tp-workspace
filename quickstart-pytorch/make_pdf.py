import os
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors

ROOT = Path("results")          # change if your root folder has a different name
OUTPUT_PDF = "all_figures.pdf"

def collect_pngs(root: Path):
    """Walk the tree and collect all .png files under any 'figures' folder."""
    entries = []
    for png in sorted(root.rglob("figures/*.png")):
        # Build label from parts *relative to root*, drop the extension
        rel = png.relative_to(root)
        label = "-".join(rel.with_suffix("").parts)   # e.g. tp1_results-vary_algo-figures-central_acc_algo
        entries.append((png, label))
    return entries

def build_pdf(entries, output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FigTitle",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=6,
    )

    page_w = landscape(A4)[0] - 3*cm   # usable width
    page_h = landscape(A4)[1] - 4*cm   # usable height (leave room for title)

    story = []
    for i, (png_path, label) in enumerate(entries):
        story.append(Paragraph(label, title_style))

        # Scale image to fit the page while preserving aspect ratio
        with Image.open(png_path) as im:
            iw, ih = im.size
        scale = min(page_w / iw, page_h / ih)
        story.append(RLImage(str(png_path), width=iw*scale, height=ih*scale))

        if i < len(entries) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"Done — {len(entries)} figures written to '{output_path}'")

if __name__ == "__main__":
    entries = collect_pngs(ROOT)
    if not entries:
        print("No .png files found under any 'figures/' folder. Check your ROOT path.")
    else:
        print(f"Found {len(entries)} images:")
        for _, label in entries:
            print(f"  {label}")
        build_pdf(entries, OUTPUT_PDF)