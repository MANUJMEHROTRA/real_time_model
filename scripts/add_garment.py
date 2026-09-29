"""Add a product photo to the catalog.

    python scripts/add_garment.py photo.jpg --id polo-blue --name "Colour-block Polo" \
        --category tops --description "light blue and navy colour-block polo shirt"

Works best with front-view garment photos on a plain white/light background. It:
  * copies the original to catalog/images/<id>.jpg   (sent to the try-on model)
  * cuts out the background -> catalog/images/<id>.png (used for the live overlay)
  * estimates overlay anchors from the garment silhouette (tweak them in catalog.json if the fit looks off)
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog"
BG_MARK = 128


def cut_out(img: Image.Image, threshold: int) -> Image.Image:
    """Alpha mask of the garment: flood-fill near-white pixels connected to the image border."""
    grey = img.convert("L").point(lambda v: 255 if v >= threshold else 0)
    w, h = grey.size
    for x in range(0, w, 8):
        for y in (0, h - 1):
            if grey.getpixel((x, y)) == 255:
                ImageDraw.floodfill(grey, (x, y), BG_MARK)
    for y in range(0, h, 8):
        for x in (0, w - 1):
            if grey.getpixel((x, y)) == 255:
                ImageDraw.floodfill(grey, (x, y), BG_MARK)
    mask = grey.point(lambda v: 0 if v == BG_MARK else 255)
    # Close pinholes, then soften the edge.
    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))


def row_extent(mask: Image.Image, y: int) -> tuple[int, int]:
    bbox = mask.crop((0, y, mask.width, y + 1)).getbbox()
    return (bbox[0], bbox[2]) if bbox else (0, mask.width)


def estimate_anchors(mask: Image.Image, category: str) -> dict:
    x0, y0, x1, y1 = mask.getbbox()
    H = y1 - y0
    W, Hi = mask.size
    norm = lambda x, y: [round(x / W, 3), round(y / Hi, 3)]  # noqa: E731

    if category == "bottoms":
        l, r = row_extent(mask, int(y0 + 0.04 * H))
        inset = 0.12 * (r - l)
        # Wearer's left is on the image's right.
        return {"left_hip": norm(r - inset, y0 + 0.08 * H),
                "right_hip": norm(l + inset, y0 + 0.08 * H),
                "mid_ankle": norm((l + r) / 2, y0 + 0.97 * H)}

    # Tops / one-pieces: the torso width below the sleeves locates the shoulder joints.
    tl, tr = row_extent(mask, int(y0 + (0.75 if category == "tops" else 0.4) * H))
    inset = 0.1 * (tr - tl)
    hip_y = y0 + (0.93 if category == "tops" else 0.42) * H
    return {"left_shoulder": norm(tr - inset, y0 + 0.13 * H),
            "right_shoulder": norm(tl + inset, y0 + 0.13 * H),
            "mid_hip": norm((tl + tr) / 2, hip_y)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo", type=Path)
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--category", choices=["tops", "bottoms", "one-pieces"], default="tops")
    p.add_argument("--description", help="short text description (used by some models)")
    p.add_argument("--threshold", type=int, default=235, help="background brightness cut-off (0-255)")
    args = p.parse_args()

    img = Image.open(args.photo).convert("RGB")
    mask = cut_out(img, args.threshold)
    (CATALOG / "images").mkdir(parents=True, exist_ok=True)
    img.save(CATALOG / "images" / f"{args.id}.jpg", quality=95)
    overlay = img.convert("RGBA")
    overlay.putalpha(mask)
    overlay.crop(mask.getbbox()).save(CATALOG / "images" / f"{args.id}.png")

    # Anchors are measured on the cropped overlay image.
    cropped_mask = mask.crop(mask.getbbox())
    entry = {
        "id": args.id,
        "name": args.name,
        "category": args.category,
        "image": f"images/{args.id}.png",
        "product_image": f"images/{args.id}.jpg",
        "description": args.description or args.name,
        "overlay": {"anchors": estimate_anchors(cropped_mask, args.category)},
    }

    catalog_path = CATALOG / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"garments": []}
    catalog["garments"] = [g for g in catalog["garments"] if g["id"] != args.id] + [entry]
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Added {args.id}: anchors {entry['overlay']['anchors']}")


if __name__ == "__main__":
    main()
