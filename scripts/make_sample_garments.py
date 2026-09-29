"""Generate placeholder garment PNGs + catalog.json so the kiosk runs out of the box.

Replace these with real product photos (flat-lay or ghost-mannequin, transparent background)
before pointing the kiosk at a real try-on API - the models need realistic garment images.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "catalog"
W, H = 600, 700

# Front-facing tee. The wearer's LEFT side is on the image's RIGHT, like a camera sees a person.
TEE = [
    (240, 40), (270, 70), (300, 80), (330, 70), (360, 40),  # neckline
    (445, 68), (565, 225), (505, 285), (440, 225),  # wearer's left shoulder + sleeve
    (452, 680), (148, 680),  # hem
    (160, 225), (95, 285), (35, 225), (155, 68),  # wearer's right sleeve + shoulder
]

# Normalised garment points that line up with MediaPipe pose landmarks.
TEE_ANCHORS = {
    "left_shoulder": [0.71, 0.15],
    "right_shoulder": [0.29, 0.15],
    "mid_hip": [0.50, 0.92],
}

SAMPLES = [
    ("tee-navy", "Classic Tee - Navy", (32, 48, 92), None, "navy blue cotton crew-neck t-shirt"),
    ("tee-red", "Classic Tee - Crimson", (176, 30, 44), None, "crimson red cotton crew-neck t-shirt"),
    ("tee-sand", "Classic Tee - Sand", (214, 196, 160), None, "sand beige cotton crew-neck t-shirt"),
    ("tee-stripe", "Breton Stripe Tee", (245, 245, 240), (28, 40, 80), "white t-shirt with navy horizontal stripes"),
]


def make_tee(color, stripe=None) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(TEE, fill=255)

    fabric = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(fabric)
    if stripe:
        for y in range(90, H, 44):
            d.rectangle([0, y, W, y + 18], fill=stripe)

    # Soft vertical shading so it reads as cloth rather than a flat shape.
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for x in range(W):
        sd.line([(x, 0), (x, H)], fill=int(60 * (abs(x - W / 2) / (W / 2)) ** 2))
    fabric = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), fabric, shade)

    # Collar rib + hem line.
    d = ImageDraw.Draw(fabric)
    darker = tuple(max(0, int(c * 0.7)) for c in color)
    d.line(TEE[:5], fill=darker, width=10, joint="curve")
    d.line([(150, 668), (450, 668)], fill=darker, width=4)

    out = fabric.convert("RGBA")
    out.putalpha(mask.filter(ImageFilter.GaussianBlur(1.2)))
    return out


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    garments = []
    for gid, name, color, stripe, desc in SAMPLES:
        rel = f"images/{gid}.png"
        make_tee(color, stripe).save(OUT / rel)
        garments.append({
            "id": gid,
            "name": name,
            "category": "tops",
            "image": rel,
            "description": desc,
            "overlay": {"anchors": TEE_ANCHORS},
        })
    (OUT / "catalog.json").write_text(json.dumps({"garments": garments}, indent=2) + "\n")
    print(f"Wrote {len(garments)} garments to {OUT}")


if __name__ == "__main__":
    main()
