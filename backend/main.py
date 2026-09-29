import base64
import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .catalog import load_catalog
from .tryon import TryOnBackend, TryOnError, TryOnRequest, get_backend

log = logging.getLogger("tryon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Virtual Try-On Kiosk")

_backend: TryOnBackend | None = None
_backend_error: str | None = None
try:
    _backend = get_backend(config.TRYON_BACKEND)
    log.info("Try-on backend: %s", _backend.name)
except TryOnError as e:
    _backend_error = str(e)
    log.error("Try-on backend unavailable: %s", e)

MAX_IMAGE_CHARS = 12_000_000  # ~9 MB decoded; a 1080p JPEG is well under this


class TryOnBody(BaseModel):
    garment_id: str
    person_image: str = Field(max_length=MAX_IMAGE_CHARS)
    preview_image: str | None = Field(default=None, max_length=MAX_IMAGE_CHARS)


@app.get("/api/health")
def health():
    return {"backend": config.TRYON_BACKEND, "ready": _backend is not None, "error": _backend_error}


@app.get("/api/catalog")
def catalog():
    return [g.public() for g in load_catalog().values()]


@app.post("/api/tryon")
async def tryon(body: TryOnBody):
    if _backend is None:
        raise HTTPException(503, f"Try-on backend not configured: {_backend_error}")
    garment = load_catalog().get(body.garment_id)
    if garment is None:
        raise HTTPException(404, f"Unknown garment '{body.garment_id}'")
    if not body.person_image.startswith("data:image/"):
        raise HTTPException(400, "person_image must be an image data URI")

    t0 = time.perf_counter()
    try:
        result = await _backend.run(
            TryOnRequest(person_image=body.person_image, garment=garment, preview_image=body.preview_image)
        )
    except TryOnError as e:
        log.warning("Try-on failed for %s: %s", garment.id, e)
        raise HTTPException(502, str(e))
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info("Try-on %s via %s in %d ms", garment.id, result.backend, elapsed_ms)

    if config.SAVE_RESULTS:
        _save(garment.id, body.person_image, result.image)
    return {"image": result.image, "backend": result.backend, "elapsed_ms": elapsed_ms}


def _save(garment_id: str, person: str, result: str) -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    stem = config.OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{garment_id}"
    for suffix, uri in (("person", person), ("result", result)):
        if uri.startswith("data:"):
            ext = "png" if uri.startswith("data:image/png") else "jpg"
            stem.with_name(f"{stem.name}-{suffix}.{ext}").write_bytes(base64.b64decode(uri.split(",", 1)[1]))
        else:
            stem.with_name(f"{stem.name}-{suffix}.url.txt").write_text(uri)


app.mount("/catalog", StaticFiles(directory=config.CATALOG_DIR), name="catalog")
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
