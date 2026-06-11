"""
make_slides.py — frame each rendered diagram PNG onto a 1920x1080 (16:9) slide
with a title bar, ready to drop into PowerPoint as a full-slide image.

Usage:  py -3 make_slides.py
Inputs: png/*.png  (tight diagrams rendered by mermaid-cli)
Output: slides/*-16x9.png  (1920x1080)
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
MARGIN = 60
TITLE_H = 96
BG = (255, 255, 255)
INK = (21, 34, 68)
SUB = (91, 102, 117)
RULE = (26, 86, 219)

FONTS = r"C:\Windows\Fonts"
def font(name, size):
    for cand in (name, "segoeui.ttf", "arial.ttf"):
        p = os.path.join(FONTS, cand)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

TITLE_FONT = font("segoeuib.ttf", 44)
SUB_FONT   = font("segoeui.ttf", 26)

# (source png, title, subtitle)
SLIDES = [
    ("png/01-drift-problem.png",
     "The Drift Problem",
     "A frozen decoder degrades as the signal distribution accumulates shift day-over-day (sequential)."),
    ("png/02-three-papers-to-N.png",
     "Three Influences, One Axis",
     "DietCorp · Do LMs Need Sleep? · ZenBrain  →  N = consolidation depth, at constant wake latency."),
    ("png/03-wake-sleep-loop.slide.png",
     "Wake / Sleep Loop",
     "Wake = one forward pass (latency independent of N).  Sleep = N AdamW steps on patch-embed only."),
    ("png/04-decision-gates.slide.png",
     "Decision Gates",
     "H_main: C2 (N>1) beats C1 (N=1) on later days with flat wake latency.  Memory: C3 < C2.  C4 = ceiling."),
]

def make(src, title, subtitle, out):
    slide = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(slide)
    # title
    d.text((MARGIN, 30), title, font=TITLE_FONT, fill=INK)
    d.text((MARGIN, 30 + 52), subtitle, font=SUB_FONT, fill=SUB)
    d.line([(MARGIN, TITLE_H + 36), (W - MARGIN, TITLE_H + 36)], fill=RULE, width=3)

    # content area
    top = TITLE_H + 60
    cw, ch = W - 2 * MARGIN, H - top - MARGIN
    img = Image.open(src).convert("RGBA")
    iw, ih = img.size
    scale = min(cw / iw, ch / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x = MARGIN + (cw - nw) // 2
    y = top + (ch - nh) // 2
    slide.paste(img, (x, y), img)
    slide.save(out, "PNG")
    return out, scale

os.makedirs("slides", exist_ok=True)
for src, title, subtitle in SLIDES:
    base = os.path.basename(src).replace(".slide.png", "").replace(".png", "")
    out, scale = make(src, title, subtitle, f"slides/{base}-16x9.png")
    print(f"{out:42s} (fit scale {scale:.3f})")
print("done -> slides/")
