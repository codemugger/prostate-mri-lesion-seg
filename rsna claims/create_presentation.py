"""
Generate: CARPL Architecture & Deployment Deep-Dive Presentation
Style: matches Technical Meeting - MRI PSA - 3 Feb 2026.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

TEAL = RGBColor(0x00, 0x66, 0x66)
DARK_TEAL = RGBColor(0x00, 0x44, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
MEDIUM_GRAY = RGBColor(0x99, 0x99, 0x99)
ACCENT_ORANGE = RGBColor(0xE0, 0x7C, 0x3E)
ACCENT_BLUE = RGBColor(0x33, 0x99, 0xCC)
ACCENT_GREEN = RGBColor(0x2E, 0x8B, 0x57)
ACCENT_RED = RGBColor(0xCC, 0x33, 0x33)
ACCENT_PURPLE = RGBColor(0x66, 0x33, 0x99)
ACCENT_YELLOW = RGBColor(0xCC, 0xA3, 0x00)

SLIDE_W = Emu(12192000)
SLIDE_H = Emu(6858000)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H


def add_header_bar(slide, title_text, subtitle_text=None):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, Emu(1097280)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_TEAL
    bar.line.fill.background()

    tf = slide.shapes.add_textbox(Emu(457200), Emu(200000), Emu(11277295), Emu(700000)).text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = WHITE

    if subtitle_text:
        tf2 = slide.shapes.add_textbox(Emu(457200), Emu(750000), Emu(11277295), Emu(300000)).text_frame
        p2 = tf2.paragraphs[0]
        run2 = p2.add_run()
        run2.text = subtitle_text
        run2.font.size = Pt(14)
        run2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xDD)


def add_slide_number(slide, num):
    tf = slide.shapes.add_textbox(Emu(11200000), Emu(6500000), Emu(900000), Emu(300000)).text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = str(num)
    run.font.size = Pt(10)
    run.font.color.rgb = MEDIUM_GRAY


def add_body_text(slide, text, left=457200, top=1371600, width=11277295, height=4800000, font_size=16):
    tf = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height)).text_frame
    tf.word_wrap = True
    lines = text.strip().split('\n')
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 4

        if stripped.startswith('**') and stripped.endswith('**'):
            run = p.add_run()
            run.text = stripped[2:-2]
            run.font.bold = True
            run.font.size = Pt(font_size + 2)
            run.font.color.rgb = TEAL
        elif stripped.startswith('>>> '):
            run = p.add_run()
            run.text = stripped[4:]
            run.font.size = Pt(font_size - 1)
            run.font.italic = True
            run.font.color.rgb = MEDIUM_GRAY
        else:
            run = p.add_run()
            run.text = stripped
            run.font.size = Pt(font_size)
            run.font.color.rgb = BLACK

        if indent_level > 0:
            p.level = indent_level
    return tf


def add_box(slide, left, top, width, height, text, fill_color, text_color=WHITE, font_size=11, bold=False, align=PP_ALIGN.CENTER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(left), Emu(top), Emu(width), Emu(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False

    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = align
    tf.paragraphs[0].space_before = Pt(2)
    tf.paragraphs[0].space_after = Pt(2)

    for i, line in enumerate(text.split('\n')):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
            p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.color.rgb = text_color
        run.font.bold = bold
    return shape


def add_arrow_right(slide, left, top, width=200000, height=50000):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Emu(left), Emu(top), Emu(width), Emu(height))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = TEAL
    arrow.line.fill.background()
    return arrow


def add_arrow_down(slide, left, top, width=50000, height=200000):
    arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Emu(left), Emu(top), Emu(width), Emu(height))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = TEAL
    arrow.line.fill.background()
    return arrow


def add_section_title(slide, text, left=457200, top=1200000, font_size=20):
    tf = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(11277295), Emu(500000)).text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = TEAL
    return tf


slide_num = 0

# ============================================================
# SLIDE 1: TITLE SLIDE
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = DARK_TEAL
bar.line.fill.background()

accent_bar = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Emu(0), Emu(3200000), SLIDE_W, Emu(80000)
)
accent_bar.fill.solid()
accent_bar.fill.fore_color.rgb = ACCENT_ORANGE
accent_bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(1600000), Emu(10500000), Emu(1500000)).text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
run = p.add_run()
run.text = "CARPL Architecture & Deployment Deep-Dive"
run.font.size = Pt(36)
run.font.bold = True
run.font.color.rgb = WHITE

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(3500000), Emu(10500000), Emu(600000)).text_frame
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "MRI Prostate Segmentation Assistant (MRI PSA) Project"
run2.font.size = Pt(22)
run2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xDD)

tf3 = slide.shapes.add_textbox(Emu(800000), Emu(4500000), Emu(10500000), Emu(1200000)).text_frame
tf3.word_wrap = True
p3 = tf3.paragraphs[0]
run3 = p3.add_run()
run3.text = "SGH Department of Data Science & Diagnostic Radiology"
run3.font.size = Pt(16)
run3.font.color.rgb = RGBColor(0xAA, 0xBB, 0xBB)
p4 = tf3.add_paragraph()
run4 = p4.add_run()
run4.text = "February 2026"
run4.font.size = Pt(14)
run4.font.color.rgb = RGBColor(0xAA, 0xBB, 0xBB)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 2: AGENDA
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Agenda")

items = [
    ("1", "What is CARPL?", "Platform overview, capabilities, and why it matters"),
    ("2", "SGH Architecture & Data Flow", "RAPIER, CARPL, CHROMA, Biobot Mona Lisa, UroFusion - how it all connects"),
    ("3", "MRI PSA Pipeline Deep-Dive", "End-to-end data journey from PACS to biopsy planning"),
    ("4", "Deployment & MLOps Considerations", "Monitoring, retraining, versioning, and operational concerns"),
    ("5", "Questions for CARPL & Radiology Teams", "Critical unknowns for deployment readiness"),
]

y_start = 1400000
for i, (num, title, desc) in enumerate(items):
    y = y_start + i * 1000000

    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Emu(500000), Emu(y), Emu(450000), Emu(450000))
    circle.fill.solid()
    circle.fill.fore_color.rgb = TEAL
    circle.line.fill.background()
    ctf = circle.text_frame
    ctf.paragraphs[0].alignment = PP_ALIGN.CENTER
    ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = ctf.paragraphs[0].add_run()
    r.text = num
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = WHITE

    ttf = slide.shapes.add_textbox(Emu(1100000), Emu(y + 30000), Emu(10000000), Emu(250000)).text_frame
    tr = ttf.paragraphs[0].add_run()
    tr.text = title
    tr.font.size = Pt(20)
    tr.font.bold = True
    tr.font.color.rgb = DARK_TEAL

    dtf = slide.shapes.add_textbox(Emu(1100000), Emu(y + 280000), Emu(10000000), Emu(200000)).text_frame
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(13)
    dr.font.color.rgb = MEDIUM_GRAY

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 3: SECTION DIVIDER - PART 1
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = TEAL
bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(2200000), Emu(10500000), Emu(800000)).text_frame
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
run = p.add_run()
run.text = "Part 1"
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(0xAA, 0xDD, 0xDD)

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(2800000), Emu(10500000), Emu(1200000)).text_frame
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "What is CARPL?"
run2.font.size = Pt(40)
run2.font.bold = True
run2.font.color.rgb = WHITE

tf3 = slide.shapes.add_textbox(Emu(800000), Emu(3900000), Emu(10500000), Emu(600000)).text_frame
p3 = tf3.paragraphs[0]
run3 = p3.add_run()
run3.text = "Clinical AI Research & Production Lab"
run3.font.size = Pt(18)
run3.font.color.rgb = RGBColor(0xCC, 0xEE, 0xEE)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 4: CARPL OVERVIEW
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "CARPL: Clinical AI Research & Production Lab")

add_section_title(slide, "What is CARPL?", top=1200000)

add_body_text(slide, """CARPL is a cloud-based radiology AI platform that enables hospitals to:
    \u2022 Evaluate, deploy, and manage AI models in clinical radiology workflows
    \u2022 Integrate with hospital PACS/RIS systems for seamless data routing
    \u2022 Provide a standardised environment for AI model lifecycle management
    \u2022 Support DICOM-native input/output for clinical compatibility

**Key Capabilities:**
    \u2022 Model Deployment: Containerised AI model hosting (Docker/MAP containers)
    \u2022 Data Routing: Automated DICOM routing from PACS to AI models
    \u2022 Results Delivery: AI outputs pushed back to PACS/clinical viewers
    \u2022 Model Registry: Version-controlled model storage and deployment
    \u2022 Monitoring Dashboard: Track model performance and usage metrics
    \u2022 Multi-model Support: Run multiple AI models simultaneously""",
    top=1600000, font_size=15)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 5: CARPL - WHY IT MATTERS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Why CARPL Matters for SGH")

col_w = 3400000
gap = 300000
start_x = 457200
y_top = 1400000

titles = ["Without CARPL", "With CARPL", "For MRI PSA"]
colors = [ACCENT_RED, ACCENT_GREEN, ACCENT_BLUE]
contents = [
    "Manual model deployment\nNo standardised pipeline\nDifficult PACS integration\nNo monitoring/logging\nSlow iteration cycles\nCompliance challenges",
    "Automated deployment\nStandardised Docker/MAP\nNative DICOM routing\nBuilt-in monitoring\nRapid model updates\nAudit trail & governance",
    "Deploy NIH baseline model\nRoute SGH prostate MRIs\nReturn RTSTRUCT to PACS\nTrack model performance\nSwap to SGH-trained model\nScale across departments"
]

for i in range(3):
    x = start_x + i * (col_w + gap)

    header_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y_top), Emu(col_w), Emu(500000))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = colors[i]
    header_box.line.fill.background()
    htf = header_box.text_frame
    htf.paragraphs[0].alignment = PP_ALIGN.CENTER
    htf.vertical_anchor = MSO_ANCHOR.MIDDLE
    hr = htf.paragraphs[0].add_run()
    hr.text = titles[i]
    hr.font.size = Pt(16)
    hr.font.bold = True
    hr.font.color.rgb = WHITE

    body_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y_top + 520000), Emu(col_w), Emu(4200000))
    body_box.fill.solid()
    body_box.fill.fore_color.rgb = LIGHT_GRAY
    body_box.line.color.rgb = colors[i]
    body_box.line.width = Pt(2)

    btf = body_box.text_frame
    btf.word_wrap = True
    for j, line in enumerate(contents[i].split('\n')):
        if j == 0:
            p = btf.paragraphs[0]
        else:
            p = btf.add_paragraph()
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = "\u2022 " + line
        r.font.size = Pt(13)
        r.font.color.rgb = BLACK

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 6: CARPL ARCHITECTURE COMPONENTS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "CARPL Platform Architecture")

box_w = 2400000
box_h = 1100000
gap_x = 350000
gap_y = 250000
start_x = 700000
start_y = 1400000

components = [
    ("DICOM Router", "Routes studies from\nPACS to AI models", ACCENT_BLUE),
    ("Model Registry", "Stores versioned\nDocker/MAP containers", TEAL),
    ("Inference Engine", "Runs AI models on\nincoming DICOM data", ACCENT_GREEN),
    ("Results Manager", "Formats & returns\noutputs to PACS", ACCENT_ORANGE),
    ("Monitoring &\nAnalytics", "Tracks throughput,\nlatency, errors", ACCENT_PURPLE),
    ("Admin Console", "User management,\nmodel config, audit", DARK_TEAL),
    ("Data Store", "Temporary DICOM\nstorage & caching", ACCENT_BLUE),
    ("Integration APIs", "REST/DICOM interfaces\nfor external systems", ACCENT_RED),
]

for i, (title, desc, color) in enumerate(components):
    row = i // 4
    col = i % 4
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y + 800000)

    add_box(slide, x, y, box_w, Emu(450000).emu, title, color, font_size=13, bold=True)

    desc_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y + 470000), Emu(box_w), Emu(box_h - 470000))
    desc_box.fill.solid()
    desc_box.fill.fore_color.rgb = LIGHT_GRAY
    desc_box.line.color.rgb = color
    desc_box.line.width = Pt(1)
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(11)
    dr.font.color.rgb = BLACK

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 7: SECTION DIVIDER - PART 2
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = TEAL
bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(2200000), Emu(10500000), Emu(800000)).text_frame
p = tf.paragraphs[0]
run = p.add_run()
run.text = "Part 2"
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(0xAA, 0xDD, 0xDD)

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(2800000), Emu(10500000), Emu(1200000)).text_frame
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "SGH Architecture & Ecosystem"
run2.font.size = Pt(40)
run2.font.bold = True
run2.font.color.rgb = WHITE

tf3 = slide.shapes.add_textbox(Emu(800000), Emu(3900000), Emu(10500000), Emu(600000)).text_frame
p3 = tf3.paragraphs[0]
run3 = p3.add_run()
run3.text = "RAPIER, CARPL, CHROMA, Biobot, Mona Lisa, UroFusion"
run3.font.size = Pt(18)
run3.font.color.rgb = RGBColor(0xCC, 0xEE, 0xEE)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 8: KEY PLAYERS & SYSTEMS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Key Systems & Stakeholders in the MRI PSA Ecosystem")

systems = [
    ("RAPIER", "Radiology-Pathology Information\nExchange Resource\n(AISG-funded infrastructure project)", ACCENT_BLUE, 457200, 1400000),
    ("CARPL", "Clinical AI Research &\nProduction Lab\n(Radiology AI deployment platform)", TEAL, 4057200, 1400000),
    ("CHROMA", "SGH Supercomputer Cluster\n(Model training compute\nresource)", ACCENT_GREEN, 7657200, 1400000),
    ("Biobot", "Biobot Surgical Pte Ltd\n(Robotic biopsy system\nmanufacturer)", ACCENT_ORANGE, 457200, 3700000),
    ("Mona Lisa", "Biobot's MRI-guided\nTargeted Biopsy Robot\n(Surgical system)", ACCENT_RED, 4057200, 3700000),
    ("UroFusion", "Biobot's Software Suite\n(Contouring, planning,\nDICOM RTSTRUCT import)", ACCENT_PURPLE, 7657200, 3700000),
]

bw = 3200000
bh = 1800000

for name, desc, color, x, y in systems:
    add_box(slide, x, y, bw, 550000, name, color, font_size=16, bold=True)

    desc_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y + 570000), Emu(bw), Emu(bh - 570000))
    desc_shape.fill.solid()
    desc_shape.fill.fore_color.rgb = LIGHT_GRAY
    desc_shape.line.color.rgb = color
    desc_shape.line.width = Pt(1.5)
    dtf = desc_shape.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(12)
    dr.font.color.rgb = BLACK

note = slide.shapes.add_textbox(Emu(457200), Emu(5800000), Emu(11000000), Emu(400000)).text_frame
nr = note.paragraphs[0].add_run()
nr.text = "Synapxe: IT infrastructure provider; approved the CARPL platform deployment within SGH Diagnostic Radiology"
nr.font.size = Pt(11)
nr.font.italic = True
nr.font.color.rgb = MEDIUM_GRAY

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 9: HOW THE SYSTEMS CONNECT
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "How the Systems Connect: Logical Architecture")

y_row1 = 1500000
y_row2 = 2800000
y_row3 = 4200000
y_row4 = 5500000
bw = 2200000
bh = 700000
mid_x = 5000000

add_box(slide, 500000, y_row1, bw, bh, "Clinical PACS\n(SGH Radiology)", ACCENT_BLUE, font_size=12, bold=True)
add_arrow_right(slide, 2750000, y_row1 + 250000, 400000, 200000)
add_box(slide, 3200000, y_row1, 2400000, bh, "TTP\nDe-identification", MEDIUM_GRAY, font_size=12, bold=True)
add_arrow_right(slide, 5650000, y_row1 + 250000, 400000, 200000)
add_box(slide, 6100000, y_row1, 2400000, bh, "RAPIER / CARPL\n(AI Platform)", TEAL, font_size=12, bold=True)
add_arrow_right(slide, 8550000, y_row1 + 250000, 400000, 200000)
add_box(slide, 9000000, y_row1, 2600000, bh, "AI Model\n(Docker/MAP Container)", ACCENT_GREEN, font_size=12, bold=True)

add_arrow_down(slide, 7200000, y_row1 + bh, 200000, 350000)

add_box(slide, 6100000, y_row2, 2400000, bh, "Inference Results\n(RTSTRUCT / NIfTI)", ACCENT_ORANGE, font_size=12, bold=True)
add_arrow_right(slide, 3700000, y_row2 + 250000, 400000, 200000)
add_box(slide, 500000, y_row2, 3150000, bh, "Radiologist Review\n& Refinement (UroFusion)", ACCENT_PURPLE, font_size=12, bold=True)

add_arrow_down(slide, 2000000, y_row2 + bh, 200000, 350000)

add_box(slide, 500000, y_row3, 3150000, bh, "Curated Dataset\n(Refined Contours)", DARK_TEAL, font_size=12, bold=True)
add_arrow_right(slide, 3700000, y_row3 + 250000, 400000, 200000)
add_box(slide, 4150000, y_row3, 2400000, bh, "CHROMA\n(SGH Supercomputer)", ACCENT_GREEN, font_size=12, bold=True)
add_arrow_right(slide, 6600000, y_row3 + 250000, 400000, 200000)
add_box(slide, 7050000, y_row3, 2800000, bh, "SGH-Optimised Model\n(Retrained)", TEAL, font_size=12, bold=True)

add_box(slide, 500000, y_row4, 5500000, 600000, "Biobot Mona Lisa Robotic System  \u2190  Vetted RTSTRUCT from radiologist", ACCENT_RED, font_size=12, bold=True)

note_tf = slide.shapes.add_textbox(Emu(6200000), Emu(y_row4), Emu(5500000), Emu(600000)).text_frame
note_tf.word_wrap = True
nr = note_tf.paragraphs[0].add_run()
nr.text = "All imaging data remains within SGH campus (no third-party transfers). CARPL workstation sited in Dept of Diagnostic Radiology with restricted access."
nr.font.size = Pt(10)
nr.font.italic = True
nr.font.color.rgb = MEDIUM_GRAY

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 10: DATA FLOW - DETAILED
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "End-to-End Data Flow: From Patient Scan to Biopsy Planning")

steps = [
    ("1. Patient MRI Scan", "mpMRI acquired at SGH\n(T2W, ADC, DWI)", ACCENT_BLUE),
    ("2. PACS Storage", "DICOM study stored\nin clinical PACS", ACCENT_BLUE),
    ("3. TTP De-identification", "Trusted Third Party\nremoves patient identifiers", MEDIUM_GRAY),
    ("4. CARPL Ingestion", "De-identified DICOM\nrouted to CARPL platform", TEAL),
    ("5. AI Inference", "MRI PSA model processes\nstudy (organ + lesion seg)", ACCENT_GREEN),
    ("6. RTSTRUCT Export", "Segmentation masks\nconverted to DICOM RTSTRUCT", ACCENT_ORANGE),
    ("7. Radiologist Review", "Specialist reviews & refines\ncontours in UroFusion", ACCENT_PURPLE),
    ("8. Mona Lisa Import", "Vetted contours imported\ninto robotic biopsy system", ACCENT_RED),
]

cols = 4
rows = 2
bw = 2500000
bh = 1300000
gap_x = 300000
gap_y = 400000
start_x = 400000
start_y = 1400000

for i, (title, desc, color) in enumerate(steps):
    row = i // cols
    col = i % cols
    x = start_x + col * (bw + gap_x)
    y = start_y + row * (bh + gap_y)

    add_box(slide, x, y, bw, 400000, title, color, font_size=12, bold=True)

    desc_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y + 420000), Emu(bw), Emu(bh - 420000))
    desc_shape.fill.solid()
    desc_shape.fill.fore_color.rgb = LIGHT_GRAY
    desc_shape.line.color.rgb = color
    desc_shape.line.width = Pt(1)
    dtf = desc_shape.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(11)
    dr.font.color.rgb = BLACK

    if col < cols - 1:
        add_arrow_right(slide, x + bw + 20000, y + bh // 2 - 50000, gap_x - 50000, 150000)

if rows > 1:
    add_arrow_down(slide, start_x + (cols - 1) * (bw + gap_x) + bw + 50000,
                   start_y + bh + 20000, 150000, gap_y - 50000)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 11: CLOUD / ON-PREM INFRASTRUCTURE
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Infrastructure: Where Does Everything Live?")

zones = [
    ("SGH Clinical Network (On-Premises)", ACCENT_BLUE, 457200, 1400000, 5500000, 2000000,
     ["Clinical PACS (DICOM archive)", "Radiology Information System (RIS)", "Clinical Workstations", "Synapxe-managed network"]),
    ("SGH Research Zone (On-Premises / Restricted)", TEAL, 457200, 3600000, 5500000, 2000000,
     ["RAPIER / CARPL Platform (AI workstation)", "De-identified DICOM storage", "Model inference environment", "Sited in Dept of Diagnostic Radiology"]),
    ("SGH Compute (CHROMA Cluster)", ACCENT_GREEN, 6257200, 1400000, 5500000, 2000000,
     ["GPU-accelerated training nodes", "Model training & validation", "Experiment tracking", "Dataset version management"]),
    ("Biobot Systems (External/On-site)", ACCENT_ORANGE, 6257200, 3600000, 5500000, 2000000,
     ["UroFusion software (contouring)", "Mona Lisa robotic system", "RTSTRUCT import capability", "Surgical planning interface"]),
]

for title, color, x, y, w, h, items in zones:
    border = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    border.fill.solid()
    border.fill.fore_color.rgb = LIGHT_GRAY
    border.line.color.rgb = color
    border.line.width = Pt(2)

    title_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(400000))
    title_box.fill.solid()
    title_box.fill.fore_color.rgb = color
    title_box.line.fill.background()
    ttf = title_box.text_frame
    ttf.paragraphs[0].alignment = PP_ALIGN.CENTER
    ttf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tr = ttf.paragraphs[0].add_run()
    tr.text = title
    tr.font.size = Pt(13)
    tr.font.bold = True
    tr.font.color.rgb = WHITE

    itf = slide.shapes.add_textbox(Emu(x + 150000), Emu(y + 500000), Emu(w - 300000), Emu(h - 550000)).text_frame
    itf.word_wrap = True
    for j, item in enumerate(items):
        if j == 0:
            p = itf.paragraphs[0]
        else:
            p = itf.add_paragraph()
        p.space_before = Pt(4)
        r = p.add_run()
        r.text = "\u2022 " + item
        r.font.size = Pt(12)
        r.font.color.rgb = BLACK

key_note = slide.shapes.add_textbox(Emu(457200), Emu(5800000), Emu(11000000), Emu(500000)).text_frame
key_note.word_wrap = True
kr = key_note.paragraphs[0].add_run()
kr.text = "Data Sovereignty: All patient data (even de-identified) remains within SGH campus. No cloud transfers. CARPL is an on-premises deployment approved by Synapxe."
kr.font.size = Pt(12)
kr.font.bold = True
kr.font.color.rgb = ACCENT_RED

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 12: MRI PSA PIPELINE DEEP-DIVE
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "MRI PSA Pipeline: MONAI Deploy DAG Architecture")

pipeline_steps = [
    ("DICOM\nData Loader", "Load DICOM\nstudies", ACCENT_BLUE),
    ("Series\nSelector", "Regex-based\nT2/ADC/HIGHB", ACCENT_BLUE),
    ("Series to\nVolume", "Convert to 3D\nin-memory images", ACCENT_BLUE),
    ("Organ Seg\n(Prostate)", "RAS, 1.0mm\nSliding window", TEAL),
    ("Lesion Seg\n(5-Fold)", "0.5mm, ROI crop\nRRUNet3D", ACCENT_GREEN),
    ("PI-RADS\nClassifier", "3D ResNet\n64\u00b3 crops", ACCENT_ORANGE),
    ("RTSTRUCT\nExport", "NIfTI \u2192 DICOM\nRT Structure", ACCENT_RED),
]

bw = 1450000
bh = 1200000
start_x = 300000
y = 1800000
gap = 150000

for i, (title, desc, color) in enumerate(pipeline_steps):
    x = start_x + i * (bw + gap)
    add_box(slide, x, y, bw, 500000, title, color, font_size=11, bold=True)

    desc_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y + 520000), Emu(bw), Emu(bh - 520000))
    desc_shape.fill.solid()
    desc_shape.fill.fore_color.rgb = LIGHT_GRAY
    desc_shape.line.color.rgb = color
    desc_shape.line.width = Pt(1)
    dtf = desc_shape.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(10)
    dr.font.color.rgb = BLACK

    if i < len(pipeline_steps) - 1:
        add_arrow_right(slide, x + bw + 5000, y + bh // 2 - 50000, gap - 10000, 120000)

detail_text = """MONAI Deploy DAG: DICOMDataLoaderOperator \u2192 DICOMSeriesSelectorOperator \u2192 DICOMSeriesToVolumeOperator \u2192 ProstateSegOperator \u2192 ProstateLesionSegOperator \u2192 ProstateLesionClassifierOperator

Key Details:
\u2022 Organ Seg: T2-only input, RAS orientation, 1.0mm isotropic, sliding window ROI (128,128,16), 50% overlap, TorchScript model
\u2022 Lesion Seg: T2+ADC+HIGHB (3ch), 0.5mm isotropic, organ-masked ROI with 32-voxel margin, 5-fold RRUNet3D ensemble, threshold=0.6345
\u2022 Classifier: 3D ResNet on 64\u00b3 crops per lesion, 4-class \u2192 PI-RADS 2-5, rule: PI-RADS 4 + axis>40mm \u2192 PI-RADS 5
\u2022 RTSTRUCT: Hole-filled contours exported as DICOM RT Structure Set, verified in 3D Slicer"""

add_body_text(slide, detail_text, top=3300000, height=3200000, font_size=11)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 13: MODEL INVENTORY
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Model Inventory & Artifacts")

add_section_title(slide, "7 Models in the MRI PSA Pipeline", top=1200000)

models = [
    ("Organ Segmentation", "models/organ/model.ts", "TorchScript", "T2 (1ch)", "3-class (BG/TZ/PZ)", "RRUNet3D", "1.0mm isotropic"),
    ("Lesion Seg Fold 0", "models/fold0/model_best_fold0.pth.tar", "PyTorch", "T2+ADC+HIGHB (3ch)", "2-class (BG/Lesion)", "RRUNet3D", "0.5mm isotropic"),
    ("Lesion Seg Fold 1", "models/fold1/model_best_fold1.pth.tar", "PyTorch", "T2+ADC+HIGHB (3ch)", "2-class (BG/Lesion)", "RRUNet3D", "0.5mm isotropic"),
    ("Lesion Seg Fold 2", "models/fold2/model_best_fold2.pth.tar", "PyTorch", "T2+ADC+HIGHB (3ch)", "2-class (BG/Lesion)", "RRUNet3D", "0.5mm isotropic"),
    ("Lesion Seg Fold 3", "models/fold3/model_best_fold3.pth.tar", "PyTorch", "T2+ADC+HIGHB (3ch)", "2-class (BG/Lesion)", "RRUNet3D", "0.5mm isotropic"),
    ("Lesion Seg Fold 4", "models/fold4/model_best_fold4.pth.tar", "PyTorch", "T2+ADC+HIGHB (3ch)", "2-class (BG/Lesion)", "RRUNet3D", "0.5mm isotropic"),
    ("PI-RADS Classifier", "models/classifier/model_best.pth.tar", "PyTorch", "5ch (T2+ADC+HIGHB+organ+lesion)", "4-class (PI-RADS 2-5)", "3D ResNet", "0.5mm, 64\u00b3 crops"),
]

headers = ["Model", "Path", "Format", "Input", "Output", "Arch", "Resolution"]
col_widths = [1500000, 2300000, 900000, 1800000, 1600000, 1100000, 1200000]
start_x = 200000
header_y = 1700000
row_h = 370000

for j, (header, cw) in enumerate(zip(headers, col_widths)):
    x = start_x + sum(col_widths[:j])
    hbox = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(header_y), Emu(cw), Emu(row_h))
    hbox.fill.solid()
    hbox.fill.fore_color.rgb = DARK_TEAL
    hbox.line.color.rgb = WHITE
    hbox.line.width = Pt(0.5)
    htf = hbox.text_frame
    htf.word_wrap = True
    htf.vertical_anchor = MSO_ANCHOR.MIDDLE
    htf.paragraphs[0].alignment = PP_ALIGN.CENTER
    hr = htf.paragraphs[0].add_run()
    hr.text = header
    hr.font.size = Pt(9)
    hr.font.bold = True
    hr.font.color.rgb = WHITE

for i, row_data in enumerate(models):
    y = header_y + row_h + i * row_h
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (val, cw) in enumerate(zip(row_data, col_widths)):
        x = start_x + sum(col_widths[:j])
        cell = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(cw), Emu(row_h))
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        cell.line.color.rgb = MEDIUM_GRAY
        cell.line.width = Pt(0.3)
        ctf = cell.text_frame
        ctf.word_wrap = True
        ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        ctf.paragraphs[0].alignment = PP_ALIGN.CENTER
        cr = ctf.paragraphs[0].add_run()
        cr.text = val
        cr.font.size = Pt(7)
        cr.font.color.rgb = BLACK

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 14: DATA ENGINEERING DETAILS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Data Engineering: Series Selection & Preprocessing")

add_body_text(slide, """\u2022 DICOM Series Selection (Regex-based Rules):
    \u2022 T2-Weighted: Matched via SeriesDescription (e.g., "t2", "T2W", "T2_TSE")
    \u2022 ADC Map: Matched via ImageType containing "ADC" or SeriesDescription patterns
    \u2022 High b-value DWI: Matched via b-value tags or SeriesDescription ("HIGHB", "b1500", etc.)
    \u2022 Rules tuned for ProstateX naming; must be adapted for SGH naming conventions

\u2022 Geometry Standardisation:
    \u2022 T2 reoriented to RAS (Right-Anterior-Superior)
    \u2022 Organ seg: resample to 1.0mm isotropic
    \u2022 Lesion seg: ADC/HIGHB aligned to T2 geometry (SimpleITK), then all to 0.5mm isotropic
    \u2022 ROI computed from organ mask with 32-voxel margin

\u2022 Data Volume Estimates (SGH):
    \u2022 1 scan: ~9.55 MB (T2W + ADC + 4 DWIs), or ~6.56 MB (T2W + ADC + DWI high only)
    \u2022 5,000 scans: ~32-47 GB (uncompressed DICOM)
    \u2022 Entire study (incl. all series): ~3,000 images per study sent to CARPL

\u2022 Critical Consideration for SGH Deployment:
    \u2022 Series naming varies across scanners and protocols
    \u2022 Must validate regex rules against SGH DICOM metadata
    \u2022 Fallback/QA strategy needed for unmatched series""",
    top=1300000, font_size=13)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 15: SECTION DIVIDER - PART 3
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = TEAL
bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(2200000), Emu(10500000), Emu(800000)).text_frame
p = tf.paragraphs[0]
run = p.add_run()
run.text = "Part 3"
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(0xAA, 0xDD, 0xDD)

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(2800000), Emu(10500000), Emu(1200000)).text_frame
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "Deployment, MLOps &\nOperational Readiness"
run2.font.size = Pt(40)
run2.font.bold = True
run2.font.color.rgb = WHITE

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 16: DEPLOYMENT ARCHITECTURE
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Deployment Architecture: How We Run the Model")

add_body_text(slide, """\u2022 Containerised Deployment (Docker / MONAI Application Package):
    \u2022 Model packaged as MAP (MONAI Application Package) using Holoscan CLI
    \u2022 Contains: model weights, inference code, dependencies, DICOM I/O logic
    \u2022 Reproducible and portable across environments
    \u2022 GPU-accelerated inference (NVIDIA GPU with \u226512 GB VRAM recommended)

\u2022 CARPL Integration:
    \u2022 MAP deployed on CARPL infrastructure within SGH
    \u2022 DICOM studies routed from PACS to CARPL automatically
    \u2022 Inference results (RTSTRUCT) pushed back to PACS or viewer
    \u2022 We (DDS team) are the primary operators of the deployed model

\u2022 Build & Run Commands:
    \u2022 Local: ./scripts/test_local.sh -i <input> -o <output> -m <models>
    \u2022 MAP Build: ./scripts/test_MAP.sh -i <input> -o <output> -m <models> -b
    \u2022 MAP Run: ./scripts/test_MAP.sh -i <input> -o <output> -m <models>

\u2022 Resource Requirements:
    \u2022 GPU: NVIDIA with \u226512 GB VRAM (inference)
    \u2022 CPU: Multi-core for DICOM processing and SimpleITK resampling
    \u2022 Storage: Temporary DICOM + NIfTI outputs per study (~50-100 MB per case)
    \u2022 Network: DICOM routing between PACS and CARPL workstation""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 17: MLOPS CONSIDERATIONS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "MLOps & Monitoring Considerations")

col_w = 5300000
gap = 400000

left_tf = slide.shapes.add_textbox(Emu(457200), Emu(1300000), Emu(col_w), Emu(500000)).text_frame
lr = left_tf.paragraphs[0].add_run()
lr.text = "Model Lifecycle Management"
lr.font.size = Pt(18)
lr.font.bold = True
lr.font.color.rgb = TEAL

add_body_text(slide, """\u2022 Model Versioning:
    \u2022 Track model checkpoints with hashes
    \u2022 Pin deployed model version on CARPL
    \u2022 Maintain rollback capability
\u2022 Retraining Pipeline:
    \u2022 Curated SGH dataset \u2192 CHROMA training
    \u2022 5-fold cross-validation protocol
    \u2022 Validation on held-out test set
    \u2022 Performance benchmarking vs baseline
\u2022 Model Registry:
    \u2022 Store model artifacts + configs
    \u2022 Code commit hash per release
    \u2022 Dataset version linked to model""",
    left=457200, top=1800000, width=col_w, height=4000000, font_size=12)

right_tf = slide.shapes.add_textbox(Emu(457200 + col_w + gap), Emu(1300000), Emu(col_w), Emu(500000)).text_frame
rr = right_tf.paragraphs[0].add_run()
rr.text = "Monitoring & Operations"
rr.font.size = Pt(18)
rr.font.bold = True
rr.font.color.rgb = TEAL

add_body_text(slide, """\u2022 Performance Monitoring:
    \u2022 Inference latency per case
    \u2022 Success/failure rate
    \u2022 Series selection hit rate
    \u2022 GPU utilisation metrics
\u2022 Quality Monitoring:
    \u2022 Radiologist correction frequency
    \u2022 Dice score vs manual contours
    \u2022 Drift detection (input distribution)
    \u2022 Failure mode logging
\u2022 Alerting:
    \u2022 Pipeline failures / timeouts
    \u2022 Missing series detection
    \u2022 Output QA checks (empty masks, etc.)""",
    left=457200 + col_w + gap, top=1800000, width=col_w, height=4000000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 18: RETRAINING WORKFLOW
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Retraining Workflow: From Feedback to Improved Model")

steps = [
    ("1. Collect\nFeedback", "Radiologist corrections\nstored as ground truth", ACCENT_BLUE),
    ("2. Curate\nDataset", "QA annotations\nVersion dataset", TEAL),
    ("3. Train on\nCHROMA", "5-fold CV\nDiceCE + Novograd", ACCENT_GREEN),
    ("4. Validate &\nBenchmark", "Held-out test set\nvs. baseline model", ACCENT_ORANGE),
    ("5. Package\nMAP", "Containerise new\nmodel weights", ACCENT_PURPLE),
    ("6. Deploy to\nCARPL", "Swap model version\nMonitor performance", ACCENT_RED),
]

bw = 1700000
bh = 1300000
start_x = 350000
y = 1600000
gap_x = 200000

for i, (title, desc, color) in enumerate(steps):
    x = start_x + i * (bw + gap_x)
    add_box(slide, x, y, bw, 500000, title, color, font_size=11, bold=True)

    desc_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y + 520000), Emu(bw), Emu(bh - 520000))
    desc_shape.fill.solid()
    desc_shape.fill.fore_color.rgb = LIGHT_GRAY
    desc_shape.line.color.rgb = color
    desc_shape.line.width = Pt(1)
    dtf = desc_shape.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
    dr = dtf.paragraphs[0].add_run()
    dr.text = desc
    dr.font.size = Pt(10)
    dr.font.color.rgb = BLACK

    if i < len(steps) - 1:
        add_arrow_right(slide, x + bw + 5000, y + bh // 2 - 50000, gap_x - 10000, 120000)

feedback_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
    Emu(start_x), Emu(y + bh + 400000), SLIDE_W - Emu(start_x * 2), Emu(2500000))
feedback_box.fill.solid()
feedback_box.fill.fore_color.rgb = LIGHT_GRAY
feedback_box.line.color.rgb = TEAL
feedback_box.line.width = Pt(2)

ftf = feedback_box.text_frame
ftf.word_wrap = True
fr_title = ftf.paragraphs[0].add_run()
fr_title.text = "Training Configuration (Current / Baseline)"
fr_title.font.size = Pt(14)
fr_title.font.bold = True
fr_title.font.color.rgb = TEAL

config_items = [
    "Architecture: RRUNet3D (residual + recurrent + attention UNet)",
    "Loss: DiceCELoss | Optimizer: Novograd | Scheduler: OneCycleLR",
    "Training: batch_size=1, max_epochs=500, dropout=0.15, 5-fold CV",
    "Organ: in_ch=1 (T2), out_ch=3 (BG/TZ/PZ), 1.0mm",
    "Lesion: in_ch=3 (T2+ADC+HIGHB), out_ch=2 (BG/lesion), 0.5mm",
    "Compute: SGH CHROMA supercomputer cluster (GPU nodes)",
]

for item in config_items:
    p = ftf.add_paragraph()
    p.space_before = Pt(2)
    r = p.add_run()
    r.text = "\u2022 " + item
    r.font.size = Pt(10)
    r.font.color.rgb = BLACK

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 19: DATA GOVERNANCE & SECURITY
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Data Governance, Security & Compliance")

add_body_text(slide, """\u2022 De-identification Protocol:
    \u2022 Trusted Third Party (TTP) extracts DICOM and removes patient identifiers
    \u2022 Compliant with SingHealth policy
    \u2022 Researchers blinded to patient identifiers throughout

\u2022 Data Sovereignty:
    \u2022 All imaging data remains within SGH campus
    \u2022 No third-party cloud transfers
    \u2022 CARPL workstation physically sited in Dept of Diagnostic Radiology
    \u2022 Restricted access (approved by Synapxe)

\u2022 IRB & Regulatory:
    \u2022 IRB Reference: 2025-1468
    \u2022 IRB Exemption obtained (Oct 2025)
    \u2022 Research Collaboration Agreement between SGH and Biobot
    \u2022 SGH Innovation Grant 2025 (ends Sep 2026)

\u2022 IP & Collaboration:
    \u2022 Foreground IP jointly owned by SGH and Biobot (equal undivided share)
    \u2022 Background IP remains with originating party
    \u2022 DICOM-standard outputs ensure interoperability
    \u2022 Open-source NIH model used as baseline (publicly available)""",
    top=1300000, font_size=13)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 20: SECTION DIVIDER - PART 4
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = TEAL
bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(2200000), Emu(10500000), Emu(800000)).text_frame
p = tf.paragraphs[0]
run = p.add_run()
run.text = "Part 4"
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(0xAA, 0xDD, 0xDD)

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(2800000), Emu(10500000), Emu(1200000)).text_frame
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "Questions for CARPL &\nRadiology Teams"
run2.font.size = Pt(40)
run2.font.bold = True
run2.font.color.rgb = WHITE

tf3 = slide.shapes.add_textbox(Emu(800000), Emu(3900000), Emu(10500000), Emu(600000)).text_frame
p3 = tf3.paragraphs[0]
run3 = p3.add_run()
run3.text = "Critical unknowns we must clarify before deployment"
run3.font.size = Pt(18)
run3.font.color.rgb = RGBColor(0xCC, 0xEE, 0xEE)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 21: QUESTIONS - DATA FLOW & INTEGRATION
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Questions: Data Flow & CARPL Integration")

add_body_text(slide, """**Data Routing & Ingestion:**
\u2022 How are DICOM studies routed from PACS to CARPL? (Push/pull? Manual trigger or automated rule?)
\u2022 Is there a DICOM node/listener on CARPL that accepts incoming studies?
\u2022 Can we filter at the PACS level to only send prostate MRI studies, or does CARPL receive everything?
\u2022 What happens if the full study (~3000 images) is sent but we only need T2W, ADC, DWI high?
\u2022 Does CARPL handle series-level filtering, or must our model container do all filtering?

**CARPL Platform Specifics:**
\u2022 What container runtime does CARPL use? (Docker, Podman, containerd?)
\u2022 What GPU is available on the CARPL workstation? (Model/VRAM)
\u2022 Is there a model registry/versioning system built into CARPL?
\u2022 How do we deploy a new model version? (CLI, web UI, API?)
\u2022 What are the storage limits for temporary inference data?

**Results Delivery:**
\u2022 How are RTSTRUCT results returned to PACS/clinical viewers?
\u2022 Can CARPL push RTSTRUCT directly to UroFusion, or must it go via PACS?
\u2022 What DICOM conformance checks does CARPL perform on outputs?
\u2022 How do radiologists access the AI outputs? (Separate viewer, PACS overlay, UroFusion?)""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 22: QUESTIONS - OPERATIONS & MONITORING
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Questions: Operations, Monitoring & Support")

add_body_text(slide, """**Operational Responsibility (We are the main operators!):**
\u2022 Who has SSH/admin access to the CARPL workstation?
\u2022 What is the escalation path if the model container crashes or hangs?
\u2022 Is there 24/7 monitoring, or do we rely on manual checks?
\u2022 How do we handle model failures gracefully? (Fallback to manual workflow?)
\u2022 What SLA/uptime expectations exist for the AI model service?

**Monitoring & Logging:**
\u2022 Does CARPL provide built-in logging for inference requests and results?
\u2022 Can we access container logs (stdout/stderr) for debugging?
\u2022 Is there an alerting mechanism for failures or anomalies?
\u2022 Can we track inference latency, throughput, and GPU utilisation over time?
\u2022 How do we detect data drift or degraded model performance in production?

**Maintenance & Updates:**
\u2022 What is the process to update model weights without rebuilding the full container?
\u2022 Can we do hot-swaps of model versions, or is downtime required?
\u2022 Who manages OS/driver updates on the CARPL workstation?
\u2022 How are security patches applied to the inference environment?
\u2022 What backup/recovery procedures exist for the CARPL workstation?""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 23: QUESTIONS - DATA ENGINEERING & MLOPS
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Questions: Data Engineering & MLOps")

add_body_text(slide, """**Data Storage & Management:**
\u2022 Where are de-identified DICOM studies stored long-term on CARPL? (Local disk, NAS, object store?)
\u2022 What is the retention policy for inference inputs and outputs?
\u2022 How do we manage the growing dataset of radiologist-corrected contours for retraining?
\u2022 Is there a structured database linking studies to inference results and radiologist feedback?
\u2022 How do we export curated datasets from CARPL to CHROMA for training?

**Retraining & Model Lifecycle:**
\u2022 What is the expected retraining cadence? (Monthly, quarterly, on-demand?)
\u2022 How do we transfer training data securely between CARPL and CHROMA?
\u2022 Is there an automated pipeline for model validation before deployment?
\u2022 How do we run A/B testing between old and new model versions?
\u2022 What governance is needed to approve a model version for clinical use?

**Data Quality & Edge Cases:**
\u2022 How do we handle studies with missing series (e.g., no high b-value DWI)?
\u2022 What if the series selection regex fails on SGH-specific naming conventions?
\u2022 How do we handle multi-scanner/multi-protocol variability across SGH?
\u2022 What QA checks should run automatically before inference proceeds?
\u2022 How do we log and track failed/rejected studies for root-cause analysis?""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 24: QUESTIONS - CLINICAL WORKFLOW & RADIOLOGY
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Questions: Clinical Workflow & Radiology Integration")

add_body_text(slide, """**Radiologist Workflow:**
\u2022 At what point in the reporting workflow should AI contours appear?
\u2022 Should the AI run automatically on all prostate MRIs, or only on-demand?
\u2022 How long is acceptable latency from scan completion to AI contour availability?
\u2022 What is the expected review/refinement time per case with AI contours?
\u2022 How do we capture radiologist corrections in a structured, machine-readable format?

**Biobot / UroFusion Integration:**
\u2022 What DICOM RTSTRUCT fields does UroFusion require for import?
\u2022 Does UroFusion validate the RTSTRUCT geometry against the original DICOM series?
\u2022 Can UroFusion handle multi-class organ contours (TZ/PZ), or only whole-gland?
\u2022 What is the expected data format for lesion annotations in UroFusion?
\u2022 How does Mona Lisa ingest the final vetted contours from UroFusion?

**Clinical Validation:**
\u2022 What metrics will radiology use to evaluate AI contour quality? (Dice, HD95, visual?)
\u2022 How many cases are needed for initial clinical validation on SGH data?
\u2022 Is there a planned reader study or inter-rater variability assessment?
\u2022 What is the process to escalate AI failures to the development team?
\u2022 How do we ensure the AI does not introduce systematic bias in contours?""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 25: QUESTIONS - SCALABILITY & FUTURE
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Questions: Scalability & Future Considerations")

add_body_text(slide, """**Scaling the Pipeline:**
\u2022 What is the expected throughput? (Cases per day/hour)
\u2022 Can CARPL handle concurrent inference requests, or is it sequential?
\u2022 What happens during peak radiology hours? (Queuing, priority, throttling?)
\u2022 Is there capacity for batch processing (e.g., backlog of historical cases)?
\u2022 Can we scale to multiple CARPL workstations if demand grows?

**Multi-Model Deployment:**
\u2022 Can CARPL host multiple AI models simultaneously? (e.g., prostate + other organs)
\u2022 How are models isolated from each other? (Separate containers, shared GPU?)
\u2022 Is there a scheduling/orchestration layer for multi-model workflows?

**Future Roadmap:**
\u2022 Will CARPL support model marketplace / sharing between institutions?
\u2022 Is federated learning a consideration for multi-site model improvement?
\u2022 Can CARPL integrate with experiment tracking tools (MLflow, W&B)?
\u2022 What is the CARPL team's roadmap for new features and integrations?
\u2022 Is there a community/user group for CARPL best practices?

**Cost & Sustainability:**
\u2022 What are the ongoing operational costs for running CARPL?
\u2022 Who funds the GPU hardware refresh cycle?
\u2022 What is the licensing model for CARPL platform updates?
\u2022 How does this compare to commercial AI solutions (e.g., Lucida Medical at USD$50/case)?""",
    top=1300000, font_size=12)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 26: SUMMARY - PRIORITY QUESTIONS MATRIX
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Priority Questions Matrix: Who to Ask What")

categories = [
    ("CARPL Team", ACCENT_BLUE, [
        "DICOM routing configuration",
        "Container runtime & GPU specs",
        "Model deployment process",
        "Monitoring & logging capabilities",
        "Storage & retention policies",
    ]),
    ("Radiology Team", TEAL, [
        "Workflow integration point",
        "Acceptable latency / SLA",
        "Annotation capture format",
        "Clinical validation protocol",
        "Reader study design",
    ]),
    ("Biobot Team", ACCENT_ORANGE, [
        "RTSTRUCT import requirements",
        "UroFusion compatibility",
        "Mona Lisa data ingestion",
        "Multi-class contour support",
        "End-to-end testing plan",
    ]),
    ("DDS Team (Us)", ACCENT_GREEN, [
        "Series selection for SGH data",
        "Model retraining schedule",
        "Performance monitoring plan",
        "Failure handling procedures",
        "Dataset versioning strategy",
    ]),
]

col_w = 2700000
gap = 200000
start_x = 400000

for i, (team, color, questions) in enumerate(categories):
    x = start_x + i * (col_w + gap)

    add_box(slide, x, 1400000, col_w, 500000, team, color, font_size=14, bold=True)

    qbox = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(1920000), Emu(col_w), Emu(4200000))
    qbox.fill.solid()
    qbox.fill.fore_color.rgb = LIGHT_GRAY
    qbox.line.color.rgb = color
    qbox.line.width = Pt(1.5)

    qtf = qbox.text_frame
    qtf.word_wrap = True
    for j, q in enumerate(questions):
        if j == 0:
            p = qtf.paragraphs[0]
        else:
            p = qtf.add_paragraph()
        p.space_before = Pt(10)
        r = p.add_run()
        r.text = "\u2022 " + q
        r.font.size = Pt(11)
        r.font.color.rgb = BLACK

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 27: PROJECT TIMELINE (CONTEXT)
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_header_bar(slide, "Project Timeline (SGH Innovation Grant 2025)")

milestones = [
    ("Oct 2025", "IRB Exemption", "COMPLETED", ACCENT_GREEN),
    ("Dec 2025", "Establish Partnership Agreement (SGH-Biobot)", "COMPLETED", ACCENT_GREEN),
    ("Jan 2026", "SGH Data Extraction", "IN PROGRESS", ACCENT_ORANGE),
    ("Jan 2026", "Onboard NIH Model to CARPL", "IN PROGRESS", ACCENT_ORANGE),
    ("Mar 2026", "Biobot UroFusion Import Capability", "PLANNED", ACCENT_BLUE),
    ("Jun 2026", "Generate & Refine Contours (Curated Dataset)", "PLANNED", ACCENT_BLUE),
    ("Aug 2026", "Train SGH-Optimised Model on CHROMA", "PLANNED", ACCENT_BLUE),
    ("Aug 2026", "Onboard SGH Model to CARPL", "PLANNED", ACCENT_BLUE),
    ("Sep 2026", "Trial Deployment in SGH Radiology", "TARGET", ACCENT_RED),
]

start_y = 1400000
row_h = 520000

for i, (date, desc, status, color) in enumerate(milestones):
    y = start_y + i * row_h

    date_tf = slide.shapes.add_textbox(Emu(457200), Emu(y), Emu(1200000), Emu(row_h)).text_frame
    date_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dr = date_tf.paragraphs[0].add_run()
    dr.text = date
    dr.font.size = Pt(12)
    dr.font.bold = True
    dr.font.color.rgb = DARK_TEAL

    desc_tf = slide.shapes.add_textbox(Emu(1800000), Emu(y), Emu(6500000), Emu(row_h)).text_frame
    desc_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    descr = desc_tf.paragraphs[0].add_run()
    descr.text = desc
    descr.font.size = Pt(13)
    descr.font.color.rgb = BLACK

    status_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(8800000), Emu(y + 80000), Emu(2400000), Emu(row_h - 160000))
    status_box.fill.solid()
    status_box.fill.fore_color.rgb = color
    status_box.line.fill.background()
    stf = status_box.text_frame
    stf.paragraphs[0].alignment = PP_ALIGN.CENTER
    stf.vertical_anchor = MSO_ANCHOR.MIDDLE
    sr = stf.paragraphs[0].add_run()
    sr.text = status
    sr.font.size = Pt(11)
    sr.font.bold = True
    sr.font.color.rgb = WHITE

    if i < len(milestones) - 1:
        line_y = y + row_h
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(457200), Emu(line_y - 10000), Emu(11000000), Emu(20000))
        line.fill.solid()
        line.fill.fore_color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
        line.line.fill.background()

highlight = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
    Emu(350000), Emu(start_y + 2 * row_h - 30000), Emu(11200000), Emu(row_h * 2 + 60000))
highlight.fill.solid()
highlight.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xE0)
highlight.line.color.rgb = ACCENT_ORANGE
highlight.line.width = Pt(2)

z_order = slide.shapes._spTree
z_order.remove(highlight._element)
z_order.insert(2, highlight._element)

add_slide_number(slide, slide_num)

# ============================================================
# SLIDE 28: THANK YOU
# ============================================================
slide_num += 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
bar.fill.solid()
bar.fill.fore_color.rgb = DARK_TEAL
bar.line.fill.background()

accent_bar = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Emu(0), Emu(3200000), SLIDE_W, Emu(80000)
)
accent_bar.fill.solid()
accent_bar.fill.fore_color.rgb = ACCENT_ORANGE
accent_bar.line.fill.background()

tf = slide.shapes.add_textbox(Emu(800000), Emu(2000000), Emu(10500000), Emu(1000000)).text_frame
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
run = p.add_run()
run.text = "Thank You"
run.font.size = Pt(48)
run.font.bold = True
run.font.color.rgb = WHITE

tf2 = slide.shapes.add_textbox(Emu(800000), Emu(3600000), Emu(10500000), Emu(1500000)).text_frame
tf2.word_wrap = True
p2 = tf2.paragraphs[0]
p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = "Questions & Discussion"
r2.font.size = Pt(24)
r2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xDD)

p3 = tf2.add_paragraph()
p3.alignment = PP_ALIGN.CENTER
p3.space_before = Pt(20)
r3 = p3.add_run()
r3.text = "SGH Department of Data Science & Diagnostic Radiology"
r3.font.size = Pt(14)
r3.font.color.rgb = RGBColor(0xAA, 0xBB, 0xBB)

p4 = tf2.add_paragraph()
p4.alignment = PP_ALIGN.CENTER
r4 = p4.add_run()
r4.text = "MRI PSA Project | SGH Innovation Grant 2025"
r4.font.size = Pt(12)
r4.font.color.rgb = RGBColor(0xAA, 0xBB, 0xBB)

add_slide_number(slide, slide_num)

# ============================================================
# SAVE
# ============================================================
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "CARPL Architecture & Deployment Deep-Dive.pptx")
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {slide_num}")
