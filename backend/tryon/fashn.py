import asyncio
import time

import httpx

from .. import config
from .base import TryOnBackend, TryOnError, TryOnRequest, TryOnResult

API = "https://api.fashn.ai/v1"
POLL_INTERVAL = 1.0


class FashnBackend(TryOnBackend):
    """FASHN virtual try-on API (https://docs.fashn.ai). Licensed for commercial use."""

    name = "fashn"

    def __init__(self) -> None:
        if not config.FASHN_API_KEY:
            raise TryOnError("FASHN_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {config.FASHN_API_KEY}"}

    async def run(self, req: TryOnRequest) -> TryOnResult:
        body = {
            "model_name": "tryon-v1.6",
            "inputs": {
                "model_image": req.person_image,
                "garment_image": req.garment.product_data_uri(),
                "category": req.garment.category,  # our categories match FASHN's vocabulary
                "mode": config.FASHN_MODE,
                "output_format": "jpeg",
                "return_base64": True,
            },
        }
        deadline = time.monotonic() + config.TRYON_TIMEOUT
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.post(f"{API}/run", json=body)
            if r.status_code >= 400:
                raise TryOnError(f"FASHN /run failed ({r.status_code}): {r.text}")
            pred_id = r.json()["id"]

            while time.monotonic() < deadline:
                await asyncio.sleep(POLL_INTERVAL)
                s = (await client.get(f"{API}/status/{pred_id}")).json()
                if s["status"] == "completed":
                    out = s["output"][0]
                    if not out.startswith(("http", "data:")):
                        out = f"data:image/jpeg;base64,{out}"
                    return TryOnResult(image=out, backend=self.name)
                if s["status"] == "failed":
                    err = s.get("error") or {}
                    raise TryOnError(f"FASHN failed: {err.get('name')}: {err.get('message')}")
        raise TryOnError("FASHN timed out")
