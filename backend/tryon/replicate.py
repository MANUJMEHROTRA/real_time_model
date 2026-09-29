import asyncio
import time

import httpx

from .. import config
from .base import TryOnBackend, TryOnError, TryOnRequest, TryOnResult

API = "https://api.replicate.com/v1"
POLL_INTERVAL = 1.0

# IDM-VTON's category vocabulary
IDM_CATEGORY = {"tops": "upper_body", "bottoms": "lower_body", "one-pieces": "dresses"}


class ReplicateBackend(TryOnBackend):
    """Runs a try-on model hosted on Replicate. Defaults to IDM-VTON, which is licensed for
    NON-COMMERCIAL use only - fine for prototyping, not for a store deployment."""

    name = "replicate"

    def __init__(self) -> None:
        if not config.REPLICATE_API_TOKEN:
            raise TryOnError("REPLICATE_API_TOKEN is not set")
        self._headers = {"Authorization": f"Bearer {config.REPLICATE_API_TOKEN}"}
        self._version: str | None = None

    async def _latest_version(self, client: httpx.AsyncClient) -> str:
        if self._version is None:
            r = await client.get(f"{API}/models/{config.REPLICATE_MODEL}")
            if r.status_code >= 400:
                raise TryOnError(f"Replicate model lookup failed ({r.status_code}): {r.text}")
            self._version = r.json()["latest_version"]["id"]
        return self._version

    async def run(self, req: TryOnRequest) -> TryOnResult:
        deadline = time.monotonic() + config.TRYON_TIMEOUT
        async with httpx.AsyncClient(headers=self._headers, timeout=60) as client:
            body = {
                "version": await self._latest_version(client),
                "input": {
                    "human_img": req.person_image,
                    "garm_img": req.garment.product_data_uri(),
                    "garment_des": req.garment.description,
                    "category": IDM_CATEGORY[req.garment.category],
                    "crop": False,
                },
            }
            r = await client.post(f"{API}/predictions", json=body, headers={"Prefer": "wait=30"})
            if r.status_code >= 400:
                raise TryOnError(f"Replicate prediction failed ({r.status_code}): {r.text}")
            pred = r.json()

            while pred["status"] not in {"succeeded", "failed", "canceled"}:
                if time.monotonic() > deadline:
                    raise TryOnError("Replicate timed out")
                await asyncio.sleep(POLL_INTERVAL)
                pred = (await client.get(pred["urls"]["get"])).json()

        if pred["status"] != "succeeded":
            raise TryOnError(f"Replicate {pred['status']}: {pred.get('error')}")
        out = pred["output"]
        return TryOnResult(image=out[0] if isinstance(out, list) else out, backend=self.name)
