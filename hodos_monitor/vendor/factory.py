#!/usr/bin/env python3
"""Tiny-brain crop factory: render individual glyphs and widgets with perfect labels.

Every function is seeded (numpy RandomState / random.Random passed in). The renderer
knows the truth, so every crop ships with a perfect label. Real captures are NEVER
used here -- they are validation-only (see spec section 4c).

Glyph normalization replicates the eye's `_square()` (text.py, read-only reference):
ink-bbox crop -> resize to 24x24. The reader model trains on exactly what the eye's
segmenter hands to the old gallery matcher, making it a true drop-in.
"""
import io
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

# --- charset: exactly the eye's _CHARS (text.py) --------------------------------
# Built by concatenation to avoid backslash-escape ambiguity; verified
# char-for-char against the eye's source at import time when IRIX_TEXT_SRC
# points at an eye text.py checkout (see _verify()).
CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "abcdefghijklmnopqrstuvwxyz"
         "0123456789"
         ".,:;!?@#%&()[]{}+-*/=<>_" + "'" + '"' + "|" + "\\" + "$")
N_CLASSES_GLYPH = len(CHARS)
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}

import os

_IRIX_TEXT_SRC = os.environ.get("IRIX_TEXT_SRC", "")


def _verify_charset():
    src = open(_IRIX_TEXT_SRC).read()
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("_CHARS = ("))
    stmt = "\n".join(lines[start:start + 4])
    eye = eval(stmt.split("=", 1)[1].strip())
    assert list(eye) == list(CHARS), (
        "charset drift vs eye: eye=%r mine=%r" % (eye, CHARS))
    return True


if _IRIX_TEXT_SRC and os.path.exists(_IRIX_TEXT_SRC):
    _verify_charset()
# else: dev-machine cross-check skipped (charset is inline above, intact). Set
# IRIX_TEXT_SRC=/path/to/irix/src/irix/text.py to enable the verification.

# --- affordance classes ---------------------------------------------------------
AFFORD_CLASSES = ["button", "link", "icon_button", "toggle", "tab",
                  "static_text", "decorative"]
N_CLASSES_AFFORD = len(AFFORD_CLASSES)
AFF2IDX = {c: i for i, c in enumerate(AFFORD_CLASSES)}

GLYPH_SIZE = 24   # matches eye N=24
WIDGET_SIZE = 48


def _curated_fonts():
    """Latin UI fonts present on this box. Render tools, not training data."""
    import glob
    keep = []
    for p in sorted(glob.glob('/usr/share/fonts/truetype/**/*.ttf', recursive=True)):
        n = p.lower()
        if 'emoji' in n or 'arabic' in n or 'lao' in n or 'kufi' in n:
            continue
        if ('dejavu' in n or 'liberation' in n or 'notosans-' in n
                or 'notoserif-' in n):
            try:
                ImageFont.truetype(p, 32)
                keep.append(p)
            except Exception:
                pass
    # cap Noto variants to keep the pool diverse but bounded
    noto = [p for p in keep if 'noto' in p.lower()]
    base = [p for p in keep if 'noto' not in p.lower()]
    random.Random(0).shuffle(noto)
    return base + noto[:16]


FONTS = _curated_fonts()

WORDS = ["Save", "Cancel", "Apply", "Delete", "Next", "Back", "Submit", "Reset",
         "Download", "Share", "Edit", "Close", "Open", "View", "Dismiss", "Send",
         "New", "Add", "List", "Draw", "More", "Scan", "Help", "Settings",
         "Messages", "Photos", "Profile", "Search", "Alerts", "Home"]


# --- degradation: the robustness curriculum ------------------------------------
def degrade(gray_img, rng, strong=False):
    """PIL grayscale Image -> degraded PIL grayscale Image. Seeded via rng."""
    im = gray_img
    # blur
    s = rng.uniform(0, 1.4 if strong else 0.9)
    if s > 0.15:
        im = im.filter(ImageFilter.GaussianBlur(s))
    # brightness / contrast
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.82, 1.18))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.78, 1.25))
    # downscale-upscale (resolution loss)
    if rng.random() < 0.45:
        f = rng.uniform(0.5, 0.92)
        w, h = im.size
        im = im.resize((max(4, int(w * f)), max(4, int(h * f))), Image.BILINEAR)
        im = im.resize((w, h), Image.BILINEAR)
    # jpeg artifacts
    if rng.random() < 0.4:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(rng.uniform(40, 92)))
        buf.seek(0)
        im = Image.open(buf).convert("L")
    # additive noise
    if rng.random() < 0.6:
        a = np.asarray(im).astype(np.float32)
        a += rng.normal(0, rng.uniform(2, 9), a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # slight rotation
    ang = rng.uniform(-2.0, 2.0)
    if abs(ang) > 0.4:
        im = im.rotate(ang, resample=Image.BILINEAR, expand=False,
                       fillcolor=int(np.asarray(im).mean()))
    return im


def _square_norm(arr):
    """Replicates eye _square(): ink-bbox crop -> 24x24. arr: 2D float in [0,1], ink=1."""
    ys, xs = np.where(arr > 0.5)
    if len(xs) == 0:
        return np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.float32)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    sq = im.resize((GLYPH_SIZE, GLYPH_SIZE), Image.BILINEAR)
    return (np.asarray(sq).astype(np.float32) / 255.0)


def render_glyph(char, rng):
    """Render one glyph -> (24x24 binary float32, class_idx). Seeded."""
    assert char in CHAR2IDX
    px = int(rng.uniform(22, 44))
    font = ImageFont.truetype(rng.choice(FONTS), px)
    # canvas with margin; random ink color on random bg (light/dark/low-contrast)
    theme = rng.random()
    if theme < 0.45:
        bg, fg = int(rng.uniform(235, 255)), int(rng.uniform(0, 40))
    elif theme < 0.8:
        bg, fg = int(rng.uniform(0, 30)), int(rng.uniform(220, 255))
    else:  # low contrast
        bg = int(rng.uniform(0, 255))
        fg = int(np.clip(bg + rng.choice([-1, 1]) * rng.uniform(28, 70), 0, 255))
    cw = max(8, int(font.getlength(char)) + px)
    im = Image.new("L", (cw + px, px * 2), bg)
    d = ImageDraw.Draw(im)
    d.text((px // 2, px // 2), char, fill=fg, font=font)
    im = degrade(im, rng)
    a = np.asarray(im).astype(np.float32)
    # binarize like the eye's pipeline (fixed threshold proxy)
    ink = (a < 127).astype(np.float32) if bg > 127 else (a > 127).astype(np.float32)
    # jitter the ink bbox slightly to simulate imperfect segmentation
    sq = _square_norm(ink)
    # pixel-flip noise post-binarization (segmentation speckle)
    if rng.random() < 0.5:
        flip = rng.random(sq.shape) < rng.uniform(0.0, 0.02)
        sq = np.where(flip, 1.0 - sq, sq)
    return sq.astype(np.float32), CHAR2IDX[char]


# --- widget rendering v2 -------------------------------------------------------
# v2 (2026-09-18): expanded per-class variants, unified neutral backgrounds (the
# v1 painters each drew their own bg tint -- a class-correlated signal the net
# could cheat on), UI-panel contexts, denser degradation, wider box jitter.
# The glyph pipeline above (render_glyph / degrade) is UNTOUCHED: bit-identical
# renders. render_widget()'s signature, AFFORD_CLASSES, and WIDGET_SIZE are
# unchanged so Kaggle packaging and the classifier arch keep working.
import math


def _rr(d, box, radius, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _text(d, xy, s, font, fill, bold=False):
    d.text(xy, s, fill=fill, font=font)
    if bold:
        d.text((xy[0] + 1, xy[1]), s, fill=fill, font=font)


def _panel_bg(d, S, rng):
    """Neutral UI backdrop. Identical distribution for every class, so the
    background carries no class signal -- the net must read the widget."""
    r = rng.random()
    if r < 0.40:
        base = int(rng.uniform(225, 250))     # light flat
    elif r < 0.75:
        base = int(rng.uniform(12, 34))       # dark flat
    else:
        base = int(rng.uniform(110, 145))     # mid gray flat
    d.rectangle([0, 0, S, S], fill=(base, base, base))
    kind = rng.random()
    if kind < 0.30:
        return  # flat
    if kind < 0.55:
        # card: centered rounded panel, tone shifted
        c = int(np.clip(base + rng.uniform(-28, 28), 0, 255))
        m = int(rng.uniform(6, 14))
        _rr(d, [m, m, S - m, S - m], 12, (c, c, c))
    elif kind < 0.75:
        # toolbar strip: horizontal band
        c = int(np.clip(base + rng.uniform(-24, 24), 0, 255))
        hb = int(rng.uniform(18, 30))
        yy = int(rng.uniform(0, S - hb))
        d.rectangle([0, yy, S, yy + hb], fill=(c, c, c))
    else:
        # list rows: faint separators
        c = int(np.clip(base + rng.uniform(-20, 20), 0, 255))
        step = int(rng.uniform(18, 28))
        for yy in range(int(rng.uniform(10, 20)), S, step):
            d.line([8, yy, S - 8, yy], fill=(c, c, c), width=2)


# --- icon glyphs (shared by icon_button + a few others) --------------------------
def _icon_glyph(d, cx, cy, r, g, gc):
    if g == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=gc, width=3)
    elif g == "plus":
        d.line([cx - r, cy, cx + r, cy], fill=gc, width=4)
        d.line([cx, cy - r, cx, cy + r], fill=gc, width=4)
    elif g == "bars":
        for yy in (-r * 0.6, 0, r * 0.6):
            d.line([cx - r, cy + yy, cx + r, cy + yy], fill=gc, width=3)
    elif g == "tri":
        d.polygon([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)],
                  outline=gc, width=3)
    elif g == "check":
        d.line([cx - r, cy, cx - r * 0.2, cy + r * 0.7], fill=gc, width=4)
        d.line([cx - r * 0.2, cy + r * 0.7, cx + r, cy - r * 0.6], fill=gc, width=4)
    elif g == "x":
        d.line([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7],
               fill=gc, width=4)
        d.line([cx - r * 0.7, cy + r * 0.7, cx + r * 0.7, cy - r * 0.7],
               fill=gc, width=4)
    elif g == "magnifier":
        d.ellipse([cx - r * 0.8, cy - r * 0.8, cx + r * 0.2, cy + r * 0.2],
                  outline=gc, width=3)
        d.line([cx + r * 0.15, cy + r * 0.15, cx + r, cy + r], fill=gc, width=3)
    elif g == "gear":
        d.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55],
                  outline=gc, width=3)
        for a in range(0, 360, 45):
            x1 = cx + math.cos(math.radians(a)) * r * 0.55
            y1 = cy + math.sin(math.radians(a)) * r * 0.55
            x2 = cx + math.cos(math.radians(a)) * r
            y2 = cy + math.sin(math.radians(a)) * r
            d.line([x1, y1, x2, y2], fill=gc, width=3)
    elif g == "bell":
        d.arc([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.5],
              start=200, end=340, fill=gc, width=3)
        d.line([cx - r * 0.7, cy + r * 0.1, cx + r * 0.7, cy + r * 0.1],
               fill=gc, width=3)
        d.ellipse([cx - 2, cy + r * 0.1, cx + 2, cy + r * 0.35], fill=gc)
    elif g == "heart":
        d.ellipse([cx - r * 0.7, cy - r * 0.6, cx, cy + r * 0.1], fill=gc)
        d.ellipse([cx, cy - r * 0.6, cx + r * 0.7, cy + r * 0.1], fill=gc)
        d.polygon([(cx - r * 0.62, cy), (cx + r * 0.62, cy), (cx, cy + r * 0.8)],
                  fill=gc)
    elif g == "arrow_l":
        d.line([cx + r * 0.7, cy, cx - r * 0.7, cy], fill=gc, width=4)
        d.line([cx - r * 0.7, cy, cx - r * 0.1, cy - r * 0.5], fill=gc, width=4)
        d.line([cx - r * 0.7, cy, cx - r * 0.1, cy + r * 0.5], fill=gc, width=4)
    elif g == "arrow_r":
        d.line([cx - r * 0.7, cy, cx + r * 0.7, cy], fill=gc, width=4)
        d.line([cx + r * 0.7, cy, cx + r * 0.1, cy - r * 0.5], fill=gc, width=4)
        d.line([cx + r * 0.7, cy, cx + r * 0.1, cy + r * 0.5], fill=gc, width=4)
    elif g == "star":
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
        d.polygon(pts, outline=gc, width=2)
    elif g == "dots":
        for xx in (-r * 0.7, 0, r * 0.7):
            d.ellipse([cx + xx - 2, cy - 2, cx + xx + 2, cy + 2], fill=gc)
    elif g == "share":
        p1 = (cx - r * 0.6, cy)
        p2 = (cx + r * 0.6, cy - r * 0.6)
        p3 = (cx + r * 0.6, cy + r * 0.6)
        for a, b in ((p1, p2), (p1, p3)):
            d.line([a[0], a[1], b[0], b[1]], fill=gc, width=3)
        for p in (p1, p2, p3):
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=gc)


_BUTTON_ACCENTS = [(66, 133, 244), (52, 168, 83), (234, 67, 53), (156, 39, 176),
                   (249, 115, 22), (0, 150, 136), (30, 30, 34)]


def _draw_button(d, box, rng, im):
    x, y, w, h = box
    style = rng.choice(["pill", "rect", "outline", "flat", "gradient", "pill"])
    state = rng.choice(["normal", "normal", "normal", "hover", "pressed",
                        "disabled"])
    accent = rng.choice(_BUTTON_ACCENTS)
    dark_accent = tuple(max(0, c - 30) for c in accent)
    if state == "disabled":
        fill, edge, tc = (205, 205, 212), None, (140, 140, 148)
    elif style == "outline":
        fill, edge, tc = None, accent, accent
    elif style == "flat":
        fill, edge, tc = None, None, accent
    elif style == "gradient":
        fill, edge, tc = accent, None, (255, 255, 255)
    else:
        fill = accent if state == "normal" else dark_accent
        edge, tc = None, (255, 255, 255)
    rad = h // 2 if style == "pill" else 10
    if style == "gradient" and state != "disabled":
        for i in range(h):
            t = i / max(1, h - 1)
            c = tuple(int(accent[j] * (1 - t) + dark_accent[j] * t)
                      for j in range(3))
            d.line([x, y + i, x + w, y + i], fill=c)
        _rr(d, [x, y, x + w, y + h], 8, None, outline=dark_accent, width=2)
    else:
        _rr(d, [x, y, x + w, y + h], rad, fill, outline=edge,
            width=3 if edge else 1)
    ty = y + 2 if state == "pressed" else y
    lab = rng.choice(WORDS)
    fs = max(10, int(h * 0.40))
    f = ImageFont.truetype(rng.choice(FONTS), fs)
    tw = d.textlength(lab, font=f)
    ix = 0
    if rng.random() < 0.25 and w > 64:
        r = int(h * 0.16)
        icx, icy = x + int(h * 0.30), ty + h / 2
        d.ellipse([icx - r, icy - r, icx + r, icy + r], fill=tc)
        ix = int(h * 0.42)
    tx = x + (w - tw) / 2 + ix / 2
    _text(d, (tx, ty + h * 0.24), lab, f, tc, bold=rng.random() < 0.3)


_LINK_WORDS = ["Learn more", "Privacy policy", "View details",
               "Read the full story", "Share this article", "Terms of service",
               "Forgot password?", "Create account", "See all results",
               "Documentation"]


def _draw_link(d, box, rng, im):
    x, y, w, h = box
    style = rng.choice(["underline", "underline", "plain", "visited", "arrow"])
    lc = (130, 70, 190) if style == "visited" else (40, 110, 220)
    lab = rng.choice(_LINK_WORDS)
    fs = max(9, int(min(h * 0.5, 26)))
    f = ImageFont.truetype(rng.choice(FONTS), fs)
    tw = d.textlength(lab, font=f)
    while tw > w and fs > 8:
        fs -= 1
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        tw = d.textlength(lab, font=f)
    d.text((x, y + h * 0.12), lab, fill=lc, font=f)
    if style in ("underline", "arrow"):
        d.line([x, y + h * 0.82, x + tw, y + h * 0.82], fill=lc, width=2)
    if style == "arrow":
        ax, ay, r = x + tw + 6, y + h * 0.45, h * 0.14
        d.line([ax, ay, ax + r * 2, ay], fill=lc, width=2)
        d.line([ax + r * 2, ay, ax + r * 1.2, ay - r * 0.8], fill=lc, width=2)
        d.line([ax + r * 2, ay, ax + r * 1.2, ay + r * 0.8], fill=lc, width=2)


def _draw_icon_button(d, box, rng, im):
    x, y, w, h = box
    style = rng.choice(["circle", "sq", "outline", "bare", "circle", "sq"])
    dark = rng.random() < 0.5
    fc = (40, 40, 48) if dark else (248, 248, 250)
    gc = (245, 245, 248) if dark else (45, 45, 52)
    if style == "circle":
        d.ellipse([x, y, x + w, y + h], fill=fc)
    elif style == "sq":
        _rr(d, [x, y, x + w, y + h], 10, fc)
    elif style == "outline":
        _rr(d, [x, y, x + w, y + h], 10, None, outline=(140, 140, 150), width=3)
    # bare: no chip, glyph floats on the panel
    cx, cy, r = x + w / 2, y + h / 2, min(w, h) * 0.24
    g = rng.choice(["circle", "plus", "bars", "tri", "check", "x", "magnifier",
                    "gear", "bell", "heart", "arrow_l", "arrow_r", "star",
                    "dots", "share"])
    _icon_glyph(d, cx, cy, r, g, gc)
    if rng.random() < 0.15:  # notification badge
        bx, by, br = x + w - 6, y + 6, 6
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=(230, 60, 60))


def _draw_toggle(d, box, rng, im):
    x, y, w, h = box
    variant = rng.choice(["switch", "switch", "switch", "checkbox", "radio"])
    on = rng.random() < 0.5
    disabled = rng.random() < 0.15
    if variant == "switch":
        fill = ((150, 150, 158) if disabled
                else ((60, 170, 90) if on else (178, 178, 186)))
        _rr(d, [x, y, x + w, y + h], h // 2, fill)
        kx = x + w - h * 0.32 if on else x + h * 0.32
        d.ellipse([kx - h * 0.28, y + h * 0.12, kx + h * 0.28, y + h * 0.88],
                  fill=(255, 255, 255))
    elif variant == "checkbox":
        s = min(w, h)
        cx0, cy0 = x + (w - s) / 2, y + (h - s) / 2
        bc = ((150, 150, 158) if disabled
              else ((60, 130, 200) if on else (120, 120, 130)))
        _rr(d, [cx0, cy0, cx0 + s, cy0 + s], 6, bc if on else None,
            outline=bc, width=3)
        if on:
            _icon_glyph(d, cx0 + s / 2, cy0 + s / 2, s * 0.28, "check",
                        (255, 255, 255))
    else:  # radio
        s = min(w, h)
        cx0, cy0 = x + (w - s) / 2, y + (h - s) / 2
        bc = ((150, 150, 158) if disabled
              else ((60, 130, 200) if on else (120, 120, 130)))
        d.ellipse([cx0, cy0, cx0 + s, cy0 + s], outline=bc, width=3)
        if on:
            d.ellipse([cx0 + s * 0.28, cy0 + s * 0.28,
                       cx0 + s * 0.72, cy0 + s * 0.72], fill=bc)


_TAB_WORDS = ["Home", "Search", "Alerts", "Profile", "Settings", "Messages",
              "Photos"]


def _draw_tab(d, box, rng, im):
    x, y, w, h = box
    style = rng.choice(["pill", "bar", "pill", "icon"])
    sel = rng.random() < 0.5
    dark = rng.random() < 0.5
    if rng.random() < 0.5:  # faint neighbor tabs at the edges
        nc = (70, 70, 78) if dark else (200, 200, 206)
        for off in (-1, 1):
            nx = x + off * (w + 8)
            _rr(d, [nx, y, nx + w, y + h], 8, None, outline=nc, width=2)
    tc = (245, 245, 248) if dark else (30, 30, 34)
    lab = rng.choice(_TAB_WORDS)
    if style == "bar":
        fs = max(9, int(h * 0.34))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        tw = d.textlength(lab, font=f)
        _text(d, (x + (w - tw) / 2, y + h * 0.14), lab, f, tc, bold=sel)
        if sel:
            bc = rng.choice([(66, 133, 244), (52, 168, 83), (234, 67, 53)])
            d.rectangle([x + w * 0.2, y + h - 6, x + w * 0.8, y + h - 2],
                        fill=bc)
    else:
        fill = ((52, 52, 62) if (dark and sel)
                else ((38, 38, 44) if dark
                      else ((255, 255, 255) if sel else (228, 228, 232))))
        _rr(d, [x, y, x + w, y + h], 8, fill)
        fs = max(9, int(h * 0.36))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        tw = d.textlength(lab, font=f)
        _text(d, (x + (w - tw) / 2, y + h * 0.24), lab, f, tc, bold=sel)
        if style == "icon":
            r = h * 0.10
            d.ellipse([x + w / 2 - r, y + h * 0.10 - r,
                       x + w / 2 + r, y + h * 0.10 + r], fill=tc)


def _draw_static_text(d, box, rng, im):
    x, y, w, h = box
    variant = rng.choice(["body", "body", "heading", "caption", "bullets",
                          "quote"])
    dark = rng.random() < 0.5
    tc = (225, 225, 230) if dark else (40, 40, 44)
    dim = (150, 150, 158) if dark else (120, 120, 128)
    if variant == "heading":
        fs = max(12, int(h * 0.4))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        _text(d, (x + 4, y + 4), rng.choice(
            ["Settings", "Profile", "Messages", "Inbox", "Photos",
             "Now playing"]), f, tc, bold=True)
    elif variant == "caption":
        fs = max(8, int(h * 0.2))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        _text(d, (x + 4, y + 4), rng.choice(
            ["Updated 2h ago", "3 new messages", "Version 4.2.1",
             "No results found"]), f, dim, bold=False)
    elif variant == "bullets":
        fs = max(9, int(h * 0.2))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        yy = y + 4
        for wd in [rng.choice(WORDS) for _ in range(3)]:
            if yy > y + h - 10:
                break
            d.ellipse([x + 6, yy + 4, x + 10, yy + 8], fill=dim)
            d.text((x + 14, yy), wd, fill=tc, font=f)
            yy += int(h * 0.3)
    elif variant == "quote":
        d.rectangle([x + 4, y + 4, x + 8, y + h - 4], fill=dim)
        fs = max(9, int(h * 0.2))
        f = ImageFont.truetype(rng.choice(FONTS), fs)
        d.text((x + 14, y + 6), rng.choice(
            ["Be yourself", "Stay curious", "Keep going"]), fill=tc, font=f)
    else:  # body paragraph
        f = ImageFont.truetype(rng.choice(FONTS), max(9, int(h * 0.22)))
        words = [rng.choice(WORDS) for _ in range(8)]
        yy, line = y + 4, ""
        for wd in words:
            t = (line + " " + wd).strip()
            if d.textlength(t, font=f) > w - 8 and line:
                d.text((x + 4, yy), line, fill=tc, font=f)
                yy += int(h * 0.26)
                line = wd
            else:
                line = t
            if yy > y + h - 12:
                break
        if line and yy <= y + h - 8:
            d.text((x + 4, yy), line, fill=tc, font=f)


def _draw_decorative(d, box, rng, im):
    x, y, w, h = box
    variant = rng.choice(["art", "art", "divider", "gradient", "dots", "blob",
                          "placeholder"])
    if variant == "art":
        yy, xx = np.mgrid[0:h, 0:w]
        r = (np.sin(xx / 23.0 + rng.uniform(0, 6)) * 60
             + np.cos(yy / 19.0) * 50 + 127).clip(0, 255)
        g = (np.sin((xx + yy) / 29.0 + rng.uniform(0, 6)) * 60 + 127).clip(0, 255)
        b = (np.cos(xx / 21.0 - yy / 27.0) * 60 + 127).clip(0, 255)
        art = np.clip(np.dstack([r, g, b])
                      + rng.randint(0, 12, size=(h, w, 1)), 0, 255).astype(np.uint8)
        art_im = Image.fromarray(art).filter(ImageFilter.GaussianBlur(2))
        im.paste(art_im, (x, y))
    elif variant == "divider":
        c = (170, 170, 178)
        d.line([x + 6, y + h / 2, x + w - 6, y + h / 2], fill=c, width=3)
    elif variant == "gradient":
        c0 = np.array(rng.choice([(66, 133, 244), (156, 39, 176),
                                  (249, 115, 22), (52, 168, 83)]),
                      dtype=np.float32)
        for i in range(h):
            t = i / max(1, h - 1)
            c = tuple(int(c0[j] * (1 - t) + (c0[j] * 0.4 + 100) * t)
                      for j in range(3))
            d.line([x, y + i, x + w, y + i], fill=c)
    elif variant == "dots":
        c = (160, 160, 168)
        for yy in range(y + 4, y + h, 10):
            for xx in range(x + 4, x + w, 10):
                d.ellipse([xx - 1, yy - 1, xx + 1, yy + 1], fill=c)
    elif variant == "blob":
        bx = x + w * rng.uniform(0.3, 0.7)
        by = y + h * rng.uniform(0.3, 0.7)
        br = min(w, h) * rng.uniform(0.3, 0.45)
        col = tuple(int(v) for v in rng.choice(
            [(120, 160, 230), (230, 160, 120), (160, 220, 170)]))
        tmp = Image.new("RGB", (w, h), (0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.ellipse([bx - x - br, by - y - br, bx - x + br, by - y + br],
                   fill=col)
        tmp = tmp.filter(ImageFilter.GaussianBlur(8))
        im.paste(tmp, (x, y))
    else:  # image placeholder
        _rr(d, [x, y, x + w, y + h], 8, (210, 210, 216),
            outline=(170, 170, 178), width=2)
        _icon_glyph(d, x + w / 2, y + h / 2 - 4, min(w, h) * 0.2, "tri",
                    (150, 150, 158))
        d.line([x + w / 2 - min(w, h) * 0.2, y + h / 2 + 8,
                x + w / 2 + min(w, h) * 0.2, y + h / 2 + 8],
               fill=(150, 150, 158), width=2)


_DRAW = {"button": _draw_button, "link": _draw_link,
         "icon_button": _draw_icon_button, "toggle": _draw_toggle,
         "tab": _draw_tab, "static_text": _draw_static_text,
         "decorative": _draw_decorative}


def degrade_widget(crop, rng):
    """Widget-only degradation: v1 degrade() plus extra ops.

    degrade() itself is UNCHANGED (the glyph path calls it directly), so this
    wrapper only ADDS ops after it. rng draws happen after degrade()'s, so
    glyph renders are unaffected.
    """
    im = degrade(crop, rng, strong=rng.random() < 0.35)
    # motion blur
    if rng.random() < 0.25:
        L = int(rng.uniform(3, 9))
        k = np.ones(L) / L
        a = np.asarray(im).astype(np.float32)
        if rng.random() < 0.5:
            a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"),
                                    1, a)
        else:
            a = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"),
                                    0, a)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # extra-crush JPEG pass
    if rng.random() < 0.15:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(rng.uniform(25, 45)))
        buf.seek(0)
        im = Image.open(buf).convert("L")
    # pixelation
    if rng.random() < 0.20:
        w, h = im.size
        f = rng.uniform(0.25, 0.5)
        im = im.resize((max(4, int(w * f)), max(4, int(h * f))),
                       Image.NEAREST).resize((w, h), Image.NEAREST)
    # slight shear
    if rng.random() < 0.20:
        sh = rng.uniform(-0.06, 0.06)
        w, h = im.size
        im = im.transform((w, h), Image.AFFINE, (1, sh, -sh * h / 2, 0, 1, 0),
                          resample=Image.BILINEAR)
    return im


def render_widget(cls_name, rng):
    """Render one widget -> (48x48 grayscale float32 in [0,1], class_idx).

    v2: unified neutral backgrounds (no class-correlated bg), UI-panel
    contexts, expanded per-class variants, denser degradation, wider box
    jitter (shift +-10%, scale 0.85-1.15) to simulate the eye's imperfect
    boxes. Draws at ~2x on a padded canvas, then crops to 48x48.
    """
    S = 96
    im = Image.new("RGB", (S, S), (128, 128, 128))
    d = ImageDraw.Draw(im)
    _panel_bg(d, S, rng)
    bw = int(rng.uniform(36, 84))
    bh = int(rng.uniform(20, 60))
    bx, by = (S - bw) // 2, (S - bh) // 2
    _DRAW[cls_name](d, (bx, by, bw, bh), rng, im)
    jx = int(rng.uniform(-0.1, 0.1) * bw)
    jy = int(rng.uniform(-0.1, 0.1) * bh)
    s = rng.uniform(0.85, 1.15)
    cx, cy = S // 2 + jx, S // 2 + jy
    cw, chh = int(48 * s), int(48 * s)
    crop = im.crop((cx - cw // 2, cy - chh // 2, cx + cw // 2, cy + chh // 2))
    crop = crop.resize((WIDGET_SIZE, WIDGET_SIZE), Image.BILINEAR).convert("L")
    crop = degrade_widget(crop, rng)
    a = np.asarray(crop).astype(np.float32) / 255.0
    return a, AFF2IDX[cls_name]
