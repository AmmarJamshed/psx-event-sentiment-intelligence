"""Render a clean LinkedIn-ready PNG with Pillow (crisp Windows fonts)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "dashboard" / "psx_impact_linkedin.png"

W, H = 1080, 1350

BG = (244, 241, 234)
CARD = (255, 253, 248)
INK = (28, 25, 23)
MUTED = (87, 83, 78)
AMBER = (181, 71, 8)
BLUE = (23, 92, 211)
TEAL = (14, 116, 144)
LINE = (231, 224, 212)
TRACK = (238, 231, 219)

FONT_REG = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REG
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


def round_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline=None, width: int = 2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def bar(draw, x, y, max_w, h, frac, color):
    round_rect(draw, (x, y, x + max_w, y + h), radius=h // 2, fill=TRACK)
    w = max(8, int(max_w * frac))
    round_rect(draw, (x, y, x + w, y + h), radius=h // 2, fill=color)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Soft accents (corners only)
    d.ellipse((-80, -80, 220, 220), fill=(255, 232, 204))
    d.ellipse((920, -60, 1180, 200), fill=(219, 234, 254))

    # Hero
    round_rect(d, (40, 40, 1040, 290), radius=28, fill=CARD, outline=LINE, width=2)
    d.text((70, 60), "PSX Event Impact", font=font(22, True), fill=MUTED)
    d.text((70, 100), "What shakes Pakistan's", font=font(44, True), fill=INK)
    d.text((70, 155), "stock market?", font=font(44, True), fill=INK)
    d.text((70, 225), "Answer: Money / economy news moves it most", font=font(24, True), fill=AMBER)

    # KPI chips
    kpis = [("118", "Events studied"), ("3.06%", "Avg week move"), ("~60%", "ML direction hit")]
    chip_w = 300
    gap = 20
    start_x = 40
    for i, (val, lab) in enumerate(kpis):
        x0 = start_x + i * (chip_w + gap)
        round_rect(d, (x0, 315, x0 + chip_w, 410), radius=20, fill=CARD, outline=LINE, width=2)
        tw = d.textlength(val, font=font(34, True))
        d.text((x0 + (chip_w - tw) / 2, 335), val, font=font(34, True), fill=INK)
        lw = d.textlength(lab, font=font(18))
        d.text((x0 + (chip_w - lw) / 2, 375), lab, font=font(18), fill=MUTED)

    # Category card
    round_rect(d, (40, 435, 1040, 820), radius=28, fill=CARD, outline=LINE, width=2)
    d.text((70, 460), "Which event type shakes stocks most?", font=font(24, True), fill=INK)
    d.text((70, 495), "Average |1-week KSE-100 move| after each event type", font=font(17), fill=MUTED)

    cats = [
        ("Money / economy", 3.06, True),
        ("War / security", 2.23, False),
        ("Disasters / society", 1.99, False),
        ("Politics / elections", 1.91, False),
        ("Rules / regulators", 1.74, False),
    ]
    max_v = max(c[1] for c in cats)
    y = 545
    for name, val, win in cats:
        d.text((70, y - 4), name, font=font(20, True), fill=INK)
        bar(d, 360, y, 480, 22, val / max_v, AMBER if win else BLUE)
        d.text((860, y - 4), f"{val:.2f}%", font=font(20, True), fill=INK)
        y += 48

    # Bottom two cards
    round_rect(d, (40, 845, 520, 1095), radius=24, fill=CARD, outline=LINE, width=2)
    d.text((65, 870), "How long shocks last", font=font(22, True), fill=INK)
    horizons = [("Day 0", 0.97), ("+1d", 1.66), ("+1w", 2.31), ("+1m", 4.62)]
    max_h = max(h[1] for h in horizons)
    y = 920
    for lab, val in horizons:
        d.text((65, y - 2), lab, font=font(17), fill=MUTED)
        bar(d, 150, y, 260, 18, val / max_h, TEAL)
        d.text((430, y - 2), f"{val:.1f}%", font=font(17, True), fill=INK)
        y += 40

    round_rect(d, (560, 845, 1040, 1095), radius=24, fill=CARD, outline=LINE, width=2)
    d.text((590, 870), "Computer recipe (ML)", font=font(22, True), fill=INK)
    d.text((590, 915), "Best robot: Random Forest", font=font(18), fill=MUTED)
    d.text((590, 955), "Watches mostly:", font=font(18), fill=MUTED)
    d.text((590, 995), "1) Recent trend   2) Yesterday", font=font(19, True), fill=INK)
    d.text((590, 1035), "3) Volatility   4) Oil / world mkts", font=font(19, True), fill=INK)

    # Takeaway
    round_rect(d, (40, 1120, 1040, 1310), radius=28, fill=CARD, outline=LINE, width=2)
    d.text((70, 1145), "One-line takeaway", font=font(18, True), fill=MUTED)
    d.text((70, 1185), "Economy news shakes PSX most on average;", font=font(23, True), fill=INK)
    d.text((70, 1225), "security shocks can still cause the worst single weeks.", font=font(23, True), fill=INK)
    d.text(
        (70, 1270),
        "Research summary  |  association != causation  |  not investment advice",
        font=font(15),
        fill=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, format="PNG", optimize=True)
    # also save a uniquely named copy so Windows Photos doesn't show a stale cache
    alt = OUT.with_name("psx_impact_linkedin_clean.png")
    img.save(alt, format="PNG", optimize=True)
    print(f"Wrote {OUT}")
    print(f"Wrote {alt}")


if __name__ == "__main__":
    main()
