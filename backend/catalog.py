import base64
import json
import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import CATALOG_DIR

# Categories understood by every backend; each backend maps these to its own vocabulary.
CATEGORIES = {"tops", "bottoms", "one-pieces"}


@dataclass(frozen=True)
class Garment:
    id: str
    name: str
    category: str
    image: str  # path relative to catalog/, used for the live overlay (transparent PNG)
    product_image: str  # path relative to catalog/, sent to the try-on model
    description: str
    overlay: dict

    @property
    def product_path(self) -> Path:
        return CATALOG_DIR / self.product_image

    def product_data_uri(self) -> str:
        return file_to_data_uri(self.product_path)

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "image": f"/catalog/{self.image}",
            "overlay": self.overlay,
        }


def file_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Garment]:
    raw = json.loads((CATALOG_DIR / "catalog.json").read_text())
    garments = {}
    for item in raw["garments"]:
        if item["category"] not in CATEGORIES:
            raise ValueError(f"{item['id']}: category must be one of {sorted(CATEGORIES)}")
        garments[item["id"]] = Garment(
            id=item["id"],
            name=item["name"],
            category=item["category"],
            image=item["image"],
            product_image=item.get("product_image", item["image"]),
            description=item.get("description", item["name"]),
            overlay=item["overlay"],
        )
    return garments
